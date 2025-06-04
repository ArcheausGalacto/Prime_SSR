#!/usr/bin/env python3
import argparse
import math
import sys
import numpy as np
from sympy import primerange

# ─────────────────────────────────────────────────────────────────────────────
# Lift Python’s default limit on big‐int→str conversions (for sci_pow_plus)
# ─────────────────────────────────────────────────────────────────────────────
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(100000)

# ─────────────────────────────────────────────────────────────────────────────
# Parse “1e+<exp>” or “1.23e<exp>” into an exact Python int (no float)
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
        return coeff * (10**exp) if exp>=0 else coeff // (10**-exp)
    return int(s)

# ─────────────────────────────────────────────────────────────────────────────
# Format a big integer as "1e+<exp> + <offset>"
# ─────────────────────────────────────────────────────────────────────────────
def sci_pow_plus(n: int) -> str:
    s   = str(n)
    exp = len(s)-1
    base= 10**exp
    return f"1e+{exp} + {n-base}"

# ─────────────────────────────────────────────────────────────────────────────
# Updated oscillator parameters from your 6000-sieve fit:
#   m ≈ 0.0038692452460753366
#   b ≈ 1.6414300780568594
#   A ≈ -0.0003479502254430944
#   B ≈ 0.002938647125866848
# and ω = 2π/6
# ─────────────────────────────────────────────────────────────────────────────
m, b = 0.0038692452460753366, 1.6414300780568594
A, B = -0.0003479502254430944, 0.002938647125866848
ω    = 2*math.pi/6

def f(k: int) -> float:
    """Divergent‐harmonic oscillator predictor at shift k."""
    return m*k + b + A*k*math.sin(ω*k) + B*k*math.cos(ω*k)

# ─────────────────────────────────────────────────────────────────────────────
# Build ab_table[x] = #(unordered p1+p2=x) via FFT for x=0..N_max
# ─────────────────────────────────────────────────────────────────────────────
def build_ab_table(N_max: int) -> np.ndarray:
    p0 = np.zeros(N_max+1, dtype=int)
    for p in primerange(2, N_max+1):
        p0[p] = 1

    L   = 1 << int(math.ceil(math.log2(2*N_max+1)))
    pad = np.zeros(L, dtype=int)
    pad[:N_max+1] = p0
    P = np.fft.rfft(pad)
    conv = np.fft.irfft(P*P, n=L).round().astype(int)[:N_max+1]

    # correct for p+p (the diagonal)
    diag = np.zeros_like(conv)
    for j in range(2, N_max+1):
        if p0[j]:
            idx = 2*j
            if idx < len(diag):
                diag[idx] = 1

    return (conv + diag)//2

# ─────────────────────────────────────────────────────────────────────────────
# Find the largest k≤b_range, k≡δ0(mod6), where
# 1) f(k) is a local minimum among its ±1 neighbors,
# 2) ab_table[k]>0,
# fallback to largest k with minimal ab_table[k].
# ─────────────────────────────────────────────────────────────────────────────
def find_best_k(b_range: int,
                ab: np.ndarray,
                δ0: int):
    ks     = list(range(δ0, b_range+1, 6))
    fs     = [f(k)       for k in ks]
    counts = [ab[k]      for k in ks]

    # collect local minima where ab[k]>0
    local_minima = []
    for i in range(1, len(ks)-1):
        if counts[i]>0 and fs[i]<=fs[i-1] and fs[i]<=fs[i+1]:
            local_minima.append((ks[i], counts[i], fs[i]))

    if local_minima:
        # pick the maximum k among those dips
        return max(local_minima, key=lambda t: t[0])

    # fallback: largest k among those with globally minimal ab[k]>0
    filtered = [(k,c) for k,c in zip(ks,counts) if c>0]
    if not filtered:
        return None, None, None
    rmin     = min(c for k,c in filtered)
    candidates = [k for k,c in filtered if c==rmin]
    k_best   = max(candidates)
    return k_best, rmin, f(k_best)

# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Use updated oscillator fit to pick large‐k local dip in ab_table."
    )
    parser.add_argument('center',
                        help="Center in sci notation, e.g. 1e+100")
    parser.add_argument('--range', type=int, default=10_000,
                        help="Max k to search above center (default 10000)")
    parser.add_argument('--delta0', type=int, default=15,
                        help="Phase offset δ₀ mod6 (default 15→3 mod6)")
    args = parser.parse_args()

    center  = parse_sci(args.center)
    b_range = args.range
    δ0      = args.delta0 % 6

    print(f"Searching for k in [0…{b_range}], k≡{δ0} mod6")
    print("Building ab_table…", end="", flush=True)
    ab = build_ab_table(b_range)
    print(" done.\n")

    k, matches, fk = find_best_k(b_range, ab, δ0)
    if k is None:
        print("No valid k found (all ab_table[k]==0).")
        return

    cand = center + k
    print(f"Selected k = {k}  (ab_table[k] = {matches}, f(k) = {fk:.4f})")
    print(f"→ Candidate (unverified): {sci_pow_plus(cand)}")

if __name__ == '__main__':
    main()
