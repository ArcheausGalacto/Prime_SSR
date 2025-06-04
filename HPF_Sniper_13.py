#!/usr/bin/env python3
import argparse
import math
import sys
import random
import numpy as np
from tqdm import tqdm
from sympy import primerange
from sympy.ntheory.primetest import mr, is_strong_lucas_prp

try:
    from numba import cuda  # type: ignore
    HAVE_CUDA = cuda.is_available()
except Exception:
    HAVE_CUDA = False

try:
    import gmpy2  # type: ignore
    HAVE_GMPY2 = True
except Exception:
    HAVE_GMPY2 = False

# For parallel processing
import os
from multiprocessing import Pool

# ─────────────────────────────────────────────────────────────────────────────
# Lift Python’s big‐int→str limit (3.11+)
# ─────────────────────────────────────────────────────────────────────────────
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(100000000)

# ─────────────────────────────────────────────────────────────────────────────
# 1) Parse and format scientific notation → exact int
# ─────────────────────────────────────────────────────────────────────────────
def parse_sci(s: str) -> int:
    s = s.strip().lower().replace('+','')
    if 'e' in s:
        coeff, expo = s.split('e',1)
        exp = int(expo)
        if '.' in coeff:
            a, b = coeff.split('.',1)
            coeff = int(a+b)
            exp  -= len(b)
        else:
            coeff = int(coeff)
        return coeff * (10**exp) if exp >= 0 else coeff // (10**-exp)
    return int(s)

def sci_pow_plus(n: int) -> str:
    ss  = str(n)
    exp = len(ss) - 1
    base = 10**exp
    return f"1e+{exp} + {n - base}"

# ─────────────────────────────────────────────────────────────────────────────
# Chunk based utilities for enormous integers
# ─────────────────────────────────────────────────────────────────────────────
def split_to_chunks(num_str: str, chunk_size: int = 9) -> list[int]:
    """Split a base-10 string into big-endian integer chunks."""
    num_str = num_str.lstrip().lstrip('+')
    if num_str.startswith('-'):
        raise ValueError("Negative numbers not supported in chunked ops")

    chunks = []
    start = len(num_str) % chunk_size
    if start:
        chunks.append(int(num_str[:start]))
    for i in range(start, len(num_str), chunk_size):
        chunks.append(int(num_str[i:i+chunk_size]))
    return chunks

def chunks_mod(chunks: list[int], divisor: int, base: int) -> int:
    """Return the modulus of a large number expressed as chunks."""
    rem = 0
    for ch in reversed(chunks):
        rem = (ch + rem * base) % divisor
    return rem

def is_divisible_chunked(num_str: str, divisor: int, chunk_size: int = 9) -> bool:
    base = 10 ** chunk_size
    chunks = split_to_chunks(num_str, chunk_size)
    return chunks_mod(chunks, divisor, base) == 0

def trial_division_chunked(num_str: str, limit: int = 100000, chunk_size: int = 9) -> bool:
    """Return False if num_str is divisible by any prime <= limit."""
    for p in primerange(2, limit + 1):
        if is_divisible_chunked(num_str, p, chunk_size):
            return False
    return True

# ─────────────────────────────────────────────────────────────────────────────
# 2) Fast primality: Baillie–PSW + Miller–Rabin fallback
# ─────────────────────────────────────────────────────────────────────────────
def miller_rabin(n: int, rounds: int = 7) -> bool:
    if n < 2: return False
    # Checks for n % 2 == 0 or n % 3 == 0 are already in fast_isprime's initial part
    # So n here will be odd and not a multiple of 3, and > 3 if those checks are done prior.
    # For robustness if called directly:
    if n == 2 or n == 3: return True
    if n % 2 == 0 or n % 3 == 0: return False

    d, s = n-1, 0
    while not (d & 1):
        d >>= 1
        s += 1
    for _ in range(rounds):
        a = random.randrange(2, n-2) # n-1 is not chosen, n is already > 3
        x = pow(a, d, n)
        if x == 1 or x == n-1:
            continue
        for _ in range(s-1):
            x = pow(x, 2, n)
            if x == n-1:
                break
        else:
            return False
    return True

if HAVE_CUDA:
    @cuda.jit
    def _mr_kernel(n, d, s, bases, results):
        idx = cuda.grid(1)
        if idx >= bases.size:
            return
        a = bases[idx]
        x = 1 % n
        base = a % n
        exp = d
        while exp > 0:
            if exp & 1:
                x = (x * base) % n
            base = (base * base) % n
            exp >>= 1
        if x == 1 or x == n - 1:
            results[idx] = 1
            return
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                results[idx] = 1
                return
        results[idx] = 0

    def miller_rabin_gpu(n: int, rounds: int = 7):
        if n.bit_length() > 63:
            return None
        d, s = n - 1, 0
        while (d & 1) == 0:
            d >>= 1
            s += 1
        bases = np.random.randint(2, n - 2, size=rounds, dtype=np.uint64)
        d_device = np.uint64(d)
        s_device = np.uint64(s)
        bases_device = cuda.to_device(bases)
        results_device = cuda.device_array(bases.shape[0], dtype=np.uint8)
        threads_per_block = 32
        blocks = (bases.shape[0] + threads_per_block - 1) // threads_per_block
        _mr_kernel[blocks, threads_per_block](n, d_device, s_device, bases_device, results_device)
        results = results_device.copy_to_host()
        return bool(results.all())

def fast_isprime(n: int) -> bool:
    """Fast probabilistic primality test."""
    if n < 2:
        return False

    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    if n in small_primes:
        return True
    for p in small_primes:
        if n % p == 0:
            return False

    if HAVE_CUDA and n.bit_length() <= 63:
        try:
            result_gpu = miller_rabin_gpu(n, rounds=7)
            if result_gpu is not None:
                return result_gpu
        except Exception:
            pass

    if HAVE_GMPY2:
        return bool(gmpy2.is_prime(n))

    if not mr(n, [2]):  # Part of Baillie-PSW
        return False
    if not is_strong_lucas_prp(n):
        return False

    if n.bit_length() > 64:
        return miller_rabin(n, rounds=7)
    return True

def fast_isprime_big(num_str: str, trial_limit: int = 100000, chunk_size: int = 9) -> bool:
    """Primality test for extremely large numbers represented as strings."""
    if not trial_division_chunked(num_str, trial_limit, chunk_size):
        return False

    if len(num_str) <= 10000:
        try:
            return fast_isprime(int(num_str))
        except Exception:
            pass

    if HAVE_GMPY2:
        try:
            return bool(gmpy2.is_prime(gmpy2.mpz(num_str)))
        except Exception:
            pass

    return True

# ─────────────────────────────────────────────────────────────────────────────
# 3) √k‐enhanced oscillator fit parameters
# ─────────────────────────────────────────────────────────────────────────────
ω_osc = 2 * math.pi / 6 
m_osc, b_osc, D_osc = 0.000856281, -0.211299, 0.0705661
A_osc, B_osc    = -7.7839e-05, 0.0011218

def f(k: int) -> float:
    safe_k = k if k >= 0 else 0 
    return (m_osc*safe_k + b_osc
            + D_osc*math.sqrt(safe_k)
            + A_osc*safe_k*math.sin(ω_osc*safe_k)
            + B_osc*safe_k*math.cos(ω_osc*safe_k))

# ─────────────────────────────────────────────────────────────────────────────
# 4) FFT‐build ab_table[x] = # unordered ways x = p1+p2 up to N_max_offset
# ─────────────────────────────────────────────────────────────────────────────
def build_ab_table(N_max_offset: int) -> np.ndarray:
    p0 = np.zeros(N_max_offset + 1, dtype=int)
    for p_val in primerange(2, N_max_offset + 1):
        p0[p_val] = 1
        
    L_fft = 1 << int(math.ceil(math.log2(2 * N_max_offset + 1)))
    pad = np.zeros(L_fft, dtype=int)
    pad[:N_max_offset + 1] = p0
    
    P_fft = np.fft.rfft(pad)
    conv_full = np.fft.irfft(P_fft * P_fft, n=L_fft).round().astype(int)
    conv = conv_full[:N_max_offset + 1] 
    
    diag = np.zeros_like(conv)
    for j in range(2, N_max_offset + 1): 
        if p0[j]:
            idx = 2 * j
            if idx <= N_max_offset:
                diag[idx] = 1
                
    return (conv + diag) // 2

# ─────────────────────────────────────────────────────────────────────────────
# 5) Try only local minima & survivors; on fail, return None
# ─────────────────────────────────────────────────────────────────────────────
def find_candidate(center: int,
                   b_range_fc: int,
                   ab: np.ndarray,
                   current_delta0_fc: int):
    ks = list(range(current_delta0_fc, b_range_fc + 1, 6))
    if not ks:
        return None, None, None

    fs_vals = [f(k_val) for k_val in ks]
    cnt = [ab[k_val] if k_val < len(ab) else 0 for k_val in ks]

    # a) largest‐k local minima with cnt>0
    local_minima_offsets = []
    if len(ks) >= 3:
        for i in range(1, len(ks)-1):
            if cnt[i] > 0 and fs_vals[i] <= fs_vals[i-1] and fs_vals[i] <= fs_vals[i+1]:
                local_minima_offsets.append(ks[i])
    
    if local_minima_offsets:
        for k_offset in tqdm(sorted(local_minima_offsets, reverse=True),
                             desc=f"Testing local minima (δ₀={current_delta0_fc})", unit="k", leave=False):
            cand = center + k_offset
            if cand != 2 and cand % 2 == 0: continue
            if cand != 3 and cand % 3 == 0: continue
            
            if fast_isprime(cand):
                return k_offset, cand, cnt[ks.index(k_offset)]

    # b) survivors with refined selection, re-ranking, and parallel testing
    positive_counts_data = [] 
    for i_idx, k_val_loop in enumerate(ks):
        if cnt[i_idx] > 0:
            positive_counts_data.append((k_val_loop, cnt[i_idx]))

    final_survivor_k_offsets = []
    actual_target_min_c = -1 
    desc_string_detail = "no_viable_ab_counts"

    if positive_counts_data:
        counts_gt_2_data = [(k_val, c_val) for k_val, c_val in positive_counts_data if c_val > 2]
        if counts_gt_2_data:
            actual_target_min_c = min(c_val for k_val, c_val in counts_gt_2_data)
            final_survivor_k_offsets = [k_val for k_val, c_val in counts_gt_2_data if c_val == actual_target_min_c]
            desc_string_detail = f"target_ab>{2}≈{actual_target_min_c}"
        else:
            actual_target_min_c = min(c_val for k_val, c_val in positive_counts_data)
            final_survivor_k_offsets = [k_val for k_val, c_val in positive_counts_data if c_val == actual_target_min_c]
            desc_string_detail = f"target_ab>{0}≈{actual_target_min_c}"

    if final_survivor_k_offsets:
        # Re-rank these survivors by f(k) then k
        k_to_f_score_map = {k_val: fs_vals[i] for i, k_val in enumerate(ks)}
        survivors_to_rank_by_f = []
        for k_cand_offset in final_survivor_k_offsets:
            f_score = k_to_f_score_map.get(k_cand_offset, float('inf'))
            survivors_to_rank_by_f.append((f_score, k_cand_offset))
        
        ranked_survivor_tuples = sorted(survivors_to_rank_by_f)
        ordered_k_offsets_to_test = [k_off for f_s, k_off in ranked_survivor_tuples]

        # Optional: Limit the number of survivors to test
        # MAX_SURVIVORS_TO_TEST = 100 
        # if len(ordered_k_offsets_to_test) > MAX_SURVIVORS_TO_TEST:
        #     print(f"INFO: Limiting survivor tests from {len(ordered_k_offsets_to_test)} to {MAX_SURVIVORS_TO_TEST}")
        #     ordered_k_offsets_to_test = ordered_k_offsets_to_test[:MAX_SURVIVORS_TO_TEST]
        
        if not ordered_k_offsets_to_test:
            return None, None, None

        candidate_values = [center + k_off for k_off in ordered_k_offsets_to_test]
        
        # Determine number of processes for the pool
        # Leave one core for system, max out at a reasonable number like 8 for this task
        num_processes = min(max(1, os.cpu_count() - 1 if os.cpu_count() and os.cpu_count() > 1 else 1), 8)

        # Heuristic: if the list is too small, sequential might be faster due to overhead
        if len(candidate_values) < num_processes * 2 or len(candidate_values) < 5:
            # print(f"INFO: Testing {len(candidate_values)} survivors sequentially (small list or few processes). δ₀={current_delta0_fc}, {desc_string_detail}")
            for i, k_offset in enumerate(ordered_k_offsets_to_test):
                cand = candidate_values[i]
                if cand != 2 and cand % 2 == 0: continue
                if cand != 3 and cand % 3 == 0: continue
                if fast_isprime(cand):
                    return k_offset, cand, actual_target_min_c
        else:
            # print(f"INFO: Testing {len(candidate_values)} survivors in parallel with {num_processes} processes. δ₀={current_delta0_fc}, {desc_string_detail}")
            try:
                with Pool(processes=num_processes) as pool:
                    # `map` preserves order, returns list of booleans
                    chunk_size = max(1, len(candidate_values) // (num_processes * 4) +1) # Heuristic for chunksize
                    primality_results = pool.map(fast_isprime, candidate_values, chunksize=chunk_size)
                
                for i, is_p in enumerate(primality_results):
                    if is_p:
                        k_offset_prime = ordered_k_offsets_to_test[i]
                        prime_found_val = candidate_values[i]
                        return k_offset_prime, prime_found_val, actual_target_min_c
            except Exception as e:
                print(f"ERROR: Exception during parallel survivor testing: {e}")
                print("INFO: Falling back to sequential survivor testing due to error.")
                for i, k_offset_seq in enumerate(ordered_k_offsets_to_test): # Use a different loop var name
                    cand_seq = candidate_values[i]
                    if cand_seq != 2 and cand_seq % 2 == 0: continue
                    if cand_seq != 3 and cand_seq % 3 == 0: continue
                    if fast_isprime(cand_seq):
                        return k_offset_seq, cand_seq, actual_target_min_c
    return None, None, None

# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="HPF_Sniper: find a prime near 1e+N by local‐dip offsets"
    )
    parser.add_argument(
        'center',
        help="Center in sci‐notation, e.g. 1e+100"
    )
    parser.add_argument(
        '--range',
        type=int,
        default=10000,
        help="Max value for offsets k (k_max = range)"
    )
    parser.add_argument(
        '--delta0',
        type=int,
        help="Override δ₀ mod6 search stream. WARNING: May lead to non-prime candidates if not chosen carefully."
    )
    args = parser.parse_args()

    b_search_range = args.range 
    ab = build_ab_table(b_search_range)
    current_center = parse_sci(args.center)

    while True: 
        center_mod_6 = current_center % 6
        k_start_options = []

        if args.delta0 is not None:
            chosen_delta0 = args.delta0 % 6
            if chosen_delta0 == 0: 
                chosen_delta0 = 6
            print(f"User specified --delta0 {args.delta0} (using {chosen_delta0} as δ₀ stream start). This will be the only δ₀ stream tested.")
            k_start_options = [chosen_delta0]
            
            candidate_val_at_chosen_delta0 = current_center + chosen_delta0
            cand_start_mod_6 = candidate_val_at_chosen_delta0 % 6
            
            if not (cand_start_mod_6 == 1 or cand_start_mod_6 == 5):
                print(f"WARNING: With current center ({sci_pow_plus(current_center)}, center % 6 = {center_mod_6}) and user-supplied δ₀ resulting in stream start {chosen_delta0}:")
                print(f"         Initial candidates (center+k) like {sci_pow_plus(candidate_val_at_chosen_delta0)} will be {cand_start_mod_6} mod 6.")
                print(f"         Such candidates are unlikely to be prime. Primality generally requires (center+k) % 6 to be 1 or 5.")
        else:
            opt1_mod_6 = (1 - center_mod_6 + 6) % 6
            opt2_mod_6 = (5 - center_mod_6 + 6) % 6
            
            processed_options_set = set()
            processed_options_set.add(opt1_mod_6 if opt1_mod_6 != 0 else 6)
            processed_options_set.add(opt2_mod_6 if opt2_mod_6 != 0 else 6)
            k_start_options = sorted(list(processed_options_set))
            print(f"Center {sci_pow_plus(current_center)} (center % 6 = {center_mod_6}). Valid δ₀ start options for k: {k_start_options}")

        found_prime_for_current_center = False
        for δ0_choice in k_start_options:
            search_display_end_offset = current_center + b_search_range 
            print(f"Searching candidates from {sci_pow_plus(current_center + δ0_choice)} to {sci_pow_plus(search_display_end_offset)}, using δ₀={δ0_choice} stream.")
            
            k_offset_found, prime_candidate, score = find_candidate(current_center, b_search_range, ab, δ0_choice)
            
            if prime_candidate is not None:
                print(f"\n--- Successfully Found Prime ---")
                print(f"Center was: {sci_pow_plus(current_center)}")
                print(f"Selected offset k = {k_offset_found} (using δ₀={δ0_choice} stream)")
                print(f"ab_table value for this k (score) = {score}")
                print(f"Prime Candidate = {sci_pow_plus(prime_candidate)}")
                print(f"Value: {prime_candidate}")
                print(f"Primality test: ✔️ Prime")
                found_prime_for_current_center = True
                return 

        if not found_prime_for_current_center:
            old_center = current_center
            current_log10 = math.log10(current_center) if current_center > 0 else -1 # Avoid log10(0)
            next_power = math.floor(current_log10) + 1
            # Ensure next_power is reasonable, esp. if current_center was small
            if current_center > 0 and next_power <= current_log10 : # Should only happen if current_log10 is already an integer
                 next_power = math.floor(current_log10) + 1
            if current_center == 0 : # Special case for starting at 0, though parse_sci likely prevents this.
                 next_power = 1 # e.g. 10^1

            current_center = 10**next_power
            
            if current_center <= old_center and old_center > 0 : 
                print(f"Error: Center is not increasing or stuck. Old: {sci_pow_plus(old_center)}, New: {sci_pow_plus(current_center)}. Exiting to prevent infinite loop.")
                return

            print(f"\nNo candidate found for previous center. Raising center from {sci_pow_plus(old_center)} to {sci_pow_plus(current_center)} and retrying.\n")

if __name__ == '__main__':
    # This check ensures that Pool is not created recursively on Windows when script is imported.
    # For scripts that use multiprocessing, it's good practice.
    main()
