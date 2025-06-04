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
    sys.set_int_max_str_digits(10000)

# ─────────────────────────────────────────────────────────────────────────────
# Parse “1e+<exp>” or “1.23e<exp>” into an exact Python int (no float)
# ─────────────────────────────────────────────────────────────────────────────
def parse_sci(s: str) -> int:
    s = s.strip().lower().replace('+','')
    if 'e' in s:
        coeff_str, exp_str = s.split('e', 1)
        exp = int(exp_str)
        if '.' in coeff_str:
            a,b = coeff_str.split('.', 1)
            coeff = int(a + b)
            exp  -= len(b)
        else:
            coeff = int(coeff_str)
        return coeff * (10**exp) if exp>=0 else coeff // (10**-exp)
    return int(s)

# ─────────────────────────────────────────────────────────────────────────────
# Format a big integer as "1e+<exp> + <offset>"
# ─────────────────────────────────────────────────────────────────────────────
def sci_pow_plus(n: int) -> str:
    s = str(n)
    exp = len(s) - 1
    base = 10**exp
    return f"1e+{exp} + {n - base}"

# ─────────────────────────────────────────────────────────────────────────────
# Divergent‐harmonic oscillator model f(k)
# ─────────────────────────────────────────────────────────────────────────────
m, b = 0.0077972, 0.23527
A, B = -0.0019369, 0.0032648
ω, φ = 2*math.pi/6, 2.1382
def f(k: int) -> float:
    return m*k + b + A*k*math.sin(ω*k) + B*k*math.cos(ω*k)

def compute_delta0() -> int:
    δ = (1/ω)*(1.5*math.pi - φ)
    return int(round(δ)) % 6

# ─────────────────────────────────────────────────────────────────────────────
# Build ab_table[x] = #ways x = p1 + p2, for x=0..N_max via FFT
# ─────────────────────────────────────────────────────────────────────────────
def build_ab_table(N_max: int) -> np.ndarray:
    p0 = np.zeros(N_max+1, dtype=int)
    for p in primerange(2, N_max+1):
        p0[p] = 1

    L = 1 << int(math.ceil(math.log2(2*N_max + 1)))
    pad = np.zeros(L, dtype=int)
    pad[:N_max+1] = p0
    P = np.fft.rfft(pad)
    conv = np.fft.irfft(P*P, n=L).round().astype(int)[:N_max+1]

    diag = np.zeros_like(conv)
    for j in range(2, N_max+1):
        if p0[j]:
            idx = 2*j
            if idx < len(diag):
                diag[idx] = 1

    return (conv + diag)//2

# ─────────────────────────────────────────────────────────────────────────────
# Find the largest k in [0..b_range] with k ≡ δ0 mod 6,
# such that:
#   1) f(k) is a local minimum among its neighbors,
#   2) ab_table[k] > 0.
# If none, falls back to the largest k with minimal ab_table[k].
# ─────────────────────────────────────────────────────────────────────────────
def find_best_k(b_range: int,
                ab: np.ndarray,
                δ0: int):
    # generate all k ≡ δ0 (mod 6)
    ks = list(range(δ0, b_range+1, 6))
    # precompute f(k) and ab_table counts
    fs = [f(k) for k in ks]
    counts = [ab[k] for k in ks]

    # identify local minima in f: f[i] <= f[i-1] and f[i] <= f[i+1]
    local_minima = []
    for i in range(1, len(ks)-1):
        if fs[i] <= fs[i-1] and fs[i] <= fs[i+1] and counts[i] > 0:
            local_minima.append((ks[i], counts[i], fs[i]))

    if local_minima:
        # pick the one with the largest k
        k_best, c_best, f_best = max(local_minima, key=lambda t: t[0])
        return k_best, c_best, f_best

    # fallback: minimal count among all k with count>0
    filtered = [(k, c) for k,c in zip(ks,counts) if c>0]
    if not filtered:
        return None, None, None
    # find global minimal count
    rmin = min(c for k,c in filtered)
    candidates = [k for k,c in filtered if c==rmin]
    k_best = max(candidates)
    return k_best, rmin, f(k_best)

# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Pick k ≡ δ₀ mod6 with local minima in f(k) and ab_table[k]>0, largest possible."
    )
    parser.add_argument('center',
                        help="Base number in sci-notation, e.g. 1e+100")
    parser.add_argument('--range', type=int, default=10_000,
                        help="Max k to search (default 10000)")
    args = parser.parse_args()

    center = parse_sci(args.center)
    b_range = args.range

    δ0 = 15#compute_delta0()
    print(f"Searching k in [0 … {b_range}] with k≡{δ0} mod6")
    print(f"Building ab_table up to {b_range}…", end="", flush=True)
    ab = build_ab_table(b_range)
    print(" done.\n")

    k, matches, fk = find_best_k(b_range, ab, δ0)
    if k is None:
        print("No valid k found (all ab_table[k] == 0).")
        return

    candidate = center + k
    print(f"Selected k = {k}  (f(k) = {fk:.4f}, ab_table[k] = {matches})")
    print(f"→ Candidate prime (unverified): {sci_pow_plus(candidate)}")

if __name__ == '__main__':
    main()
