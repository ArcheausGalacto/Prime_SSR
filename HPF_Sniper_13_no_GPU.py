#!/usr/bin/env python3
"""HPF Sniper 13 (no GPU)

Simplified prime search using chunk based arithmetic only.  This
version drops all GPU logic but keeps the search space reduction logic
present in the GPU script so that extremely large numbers like
``10**1_000_000 + k`` can be explored.
"""

import argparse
import math
import sys
import numpy as np
from sympy import primerange

# Increase Python int->str limit (Python 3.11+)
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(100000000)

# ─────────────────────────────────────────────────────────────────────────────
# Chunk utilities
# ─────────────────────────────────────────────────────────────────────────────

def split_to_chunks(num_str: str, chunk_size: int = 9) -> list[int]:
    """Split ``num_str`` into big-endian integer chunks of ``chunk_size`` digits."""
    num_str = num_str.lstrip().lstrip('+')
    if num_str.startswith('-'):
        raise ValueError("Negative numbers not supported")
    chunks: list[int] = []
    start = len(num_str) % chunk_size
    if start:
        chunks.append(int(num_str[:start]))
    for i in range(start, len(num_str), chunk_size):
        chunks.append(int(num_str[i : i + chunk_size]))
    return chunks


def chunks_mod(chunks: list[int], divisor: int, base: int) -> int:
    """Return the modulus of a large number expressed as ``chunks``."""
    rem = 0
    for ch in chunks:
        rem = (rem * base + ch) % divisor
    return rem


def trial_division_chunked(num_str: str, limit: int = 100000, chunk_size: int = 9) -> bool:
    """Return ``False`` if ``num_str`` has a prime factor ``<= limit``."""
    base = 10 ** chunk_size
    chunks = split_to_chunks(num_str, chunk_size)
    for p in primerange(2, limit + 1):
        if chunks_mod(chunks, p, base) == 0:
            return False
    return True


def number_str_from_power_offset(exp: int, offset: int) -> str:
    """Return decimal string for ``10**exp + offset`` (``offset`` >= 0)."""
    offset_str = str(offset)
    zeros = exp - len(offset_str)
    if zeros < 0:
        raise ValueError("offset must be smaller than 10**exp")
    return "1" + "0" * zeros + offset_str


def fast_isprime_big(num_str: str, trial_limit: int = 100000, chunk_size: int = 9) -> bool:
    """Probabilistic primality test for huge numbers represented as strings."""
    return trial_division_chunked(num_str, trial_limit, chunk_size)

# ─────────────────────────────────────────────────────────────────────────────
# √k-enhanced oscillator for ranking offsets
# ─────────────────────────────────────────────────────────────────────────────
ω_osc = 2 * math.pi / 6
m_osc, b_osc, D_osc = 0.000856281, -0.211299, 0.0705661
A_osc, B_osc = -7.7839e-05, 0.0011218


def f(k: int) -> float:
    safe_k = k if k >= 0 else 0
    return (
        m_osc * safe_k
        + b_osc
        + D_osc * math.sqrt(safe_k)
        + A_osc * safe_k * math.sin(ω_osc * safe_k)
        + B_osc * safe_k * math.cos(ω_osc * safe_k)
    )

# ─────────────────────────────────────────────────────────────────────────────
# Build ab-table via FFT (counts of representations k = p1 + p2)
# ─────────────────────────────────────────────────────────────────────────────

def build_ab_table(N_max_offset: int) -> np.ndarray:
    p0 = np.zeros(N_max_offset + 1, dtype=int)
    for p_val in primerange(2, N_max_offset + 1):
        p0[p_val] = 1
    L_fft = 1 << int(math.ceil(math.log2(2 * N_max_offset + 1)))
    pad = np.zeros(L_fft, dtype=int)
    pad[: N_max_offset + 1] = p0
    P_fft = np.fft.rfft(pad)
    conv_full = np.fft.irfft(P_fft * P_fft, n=L_fft).round().astype(int)
    conv = conv_full[: N_max_offset + 1]
    diag = np.zeros_like(conv)
    for j in range(2, N_max_offset + 1):
        if p0[j]:
            idx = 2 * j
            if idx <= N_max_offset:
                diag[idx] = 1
    return (conv + diag) // 2

# ─────────────────────────────────────────────────────────────────────────────
# Candidate search using ab-table and oscillator heuristic
# ─────────────────────────────────────────────────────────────────────────────

def find_candidate(exp: int, b_range_fc: int, ab: np.ndarray, current_delta0_fc: int, trial_limit: int) -> tuple[int | None, str | None, int | None]:
    ks = list(range(current_delta0_fc, b_range_fc + 1, 6))
    if not ks:
        return None, None, None
    fs_vals = [f(k_val) for k_val in ks]
    cnt = [ab[k_val] if k_val < len(ab) else 0 for k_val in ks]

    # a) test local minima first
    local_minima_offsets = []
    if len(ks) >= 3:
        for i in range(1, len(ks) - 1):
            if cnt[i] > 0 and fs_vals[i] <= fs_vals[i - 1] and fs_vals[i] <= fs_vals[i + 1]:
                local_minima_offsets.append(ks[i])

    for k_offset in sorted(local_minima_offsets, reverse=True):
        if k_offset % 2 == 0:
            continue
        if (1 + k_offset) % 3 == 0:
            continue
        cand_str = number_str_from_power_offset(exp, k_offset)
        if fast_isprime_big(cand_str, trial_limit):
            return k_offset, cand_str, cnt[ks.index(k_offset)]

    # b) survivors with smallest ab-count and best f(k)
    positive_counts = [(k_val, c_val) for k_val, c_val in zip(ks, cnt) if c_val > 0]
    if not positive_counts:
        return None, None, None

    counts_gt_2 = [(k, c) for k, c in positive_counts if c > 2]
    if counts_gt_2:
        target_min_c = min(c for k, c in counts_gt_2)
        survivor_offsets = [k for k, c in counts_gt_2 if c == target_min_c]
    else:
        target_min_c = min(c for k, c in positive_counts)
        survivor_offsets = [k for k, c in positive_counts if c == target_min_c]

    ranked = sorted((f(k), k) for k in survivor_offsets)
    for _, k_offset in ranked:
        if k_offset % 2 == 0:
            continue
        if (1 + k_offset) % 3 == 0:
            continue
        cand_str = number_str_from_power_offset(exp, k_offset)
        if fast_isprime_big(cand_str, trial_limit):
            return k_offset, cand_str, target_min_c

    return None, None, None

# ─────────────────────────────────────────────────────────────────────────────
# Command-line interface
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Find primes near 10^N without a GPU")
    parser.add_argument("exp", type=int, help="Exponent N for the base 10^N")
    parser.add_argument("--range", type=int, default=1000, help="Max offset k to examine")
    parser.add_argument("--delta0", type=int, help="Override δ₀ mod6 search stream")
    parser.add_argument("--limit", type=int, default=10000, help="Trial division limit")
    args = parser.parse_args()

    b_range = args.range
    ab = build_ab_table(b_range)
    center_mod_6 = pow(10, args.exp, 6)

    if args.delta0 is not None:
        delta_options = [args.delta0 % 6 or 6]
    else:
        opt1 = (1 - center_mod_6) % 6
        opt2 = (5 - center_mod_6) % 6
        delta_options = sorted({opt1 or 6, opt2 or 6})

    for δ0_choice in delta_options:
        k_offset, prime_str, score = find_candidate(args.exp, b_range, ab, δ0_choice, args.limit)
        if prime_str:
            print("--- Successfully Found Prime ---")
            print(f"Exponent N = {args.exp}")
            print(f"Selected offset k = {k_offset} (using δ₀={δ0_choice} stream)")
            print(f"ab_table value (score) = {score}")
            print(f"Prime Candidate = 1e+{args.exp} + {k_offset}")
            print(f"Digits: {len(prime_str)}")
            return

    print("No prime found in the given range.")


if __name__ == "__main__":
    main()
