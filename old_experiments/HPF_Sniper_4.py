#!/usr/bin/env python3
import argparse
import math
import numpy as np
from sympy import primerange, isprime, nextprime

# ─────────────────────────────────────────────────────────────────────────────
# Helper: format a big integer as "1e+<exp> + <offset>"
# ─────────────────────────────────────────────────────────────────────────────
def sci_pow_plus(n: int) -> str:
    s = str(n)
    exp = len(s) - 1
    base = 10 ** exp
    offset = n - base
    return f"1e+{exp} + {offset}"

# ─────────────────────────────────────────────────────────────────────────────
# Parse strings like "1e+102" → int(10**102)
# ─────────────────────────────────────────────────────────────────────────────
def parse_sci(s: str) -> int:
    if 'e+' in s:
        base, exp = s.split('e+')
        return int(float(base) * (10 ** int(exp)))
    elif 'e' in s:
        base, exp = s.split('e')
        return int(float(base) * (10 ** int(exp)))
    else:
        return int(s)

# ─────────────────────────────────────────────────────────────────────────────
# Build ab_table[x] = #(p₁+p₂=x) via FFT, for x up to N_max
# ─────────────────────────────────────────────────────────────────────────────
def build_ab_table(N_max: int) -> np.ndarray:
    limit = N_max
    # prime indicator up to limit
    p0 = np.zeros(limit+1, dtype=int)
    for p in primerange(2, limit+1):
        p0[p] = 1

    L = 1 << int(math.ceil(math.log2(2*limit + 1)))
    pad = np.zeros(L, dtype=int)
    pad[:limit+1] = p0
    P = np.fft.rfft(pad)
    conv = np.fft.irfft(P * P, n=L).round().astype(int)[:limit+1]

    # account for p+p cases
    diag = np.zeros_like(conv)
    for j in range(2, limit+1):
        if p0[j]:
            idx = 2*j
            if idx < len(diag):
                diag[idx] = 1

    ab_table = (conv + diag) // 2
    return ab_table

# ─────────────────────────────────────────────────────────────────────────────
# Compute dip-offset δ₀ mod 6 using the oscillator parameters
# ─────────────────────────────────────────────────────────────────────────────
def compute_delta0() -> int:
    m, b =  0.0077972, 0.23527
    A, B = -0.0019369, 0.0032648
    ω, φ =  2*math.pi/6, 2.1382
    delta = (1/ω)*(1.5*math.pi - φ)
    return int(round(delta)) % 6

# ─────────────────────────────────────────────────────────────────────────────
# Given center, half-range, ab_table and δ₀, find the best prime
# ─────────────────────────────────────────────────────────────────────────────
def find_best_prime(center: int,
                    b_range: int,
                    ab_table: np.ndarray,
                    delta0: int):
    lo, hi = -b_range, b_range
    # first x ≥ lo with x ≡ δ₀ (mod 6)
    start = lo + ((delta0 - lo) % 6)
    xs = list(range(start, hi+1, 6))

    # evaluate ab_table at |x|
    vals = [ab_table[abs(x)] for x in xs]
    rmin = min(vals)
    survivors = [x for x,v in zip(xs, vals) if v == rmin]

    # pick the first survivor that is prime
    for x in survivors:
        candidate = center + x
        if isprime(candidate):
            return candidate, x, rmin

    return None, None, None

# ─────────────────────────────────────────────────────────────────────────────
# Main: parse args, run one search, print result
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Find the 'best' prime near a big power-of-ten using the ab_table heuristic."
    )
    parser.add_argument(
        'center',
        help="Center value in scientific notation (e.g. 1e+102)"
    )
    parser.add_argument(
        '--range',
        type=int,
        default=10_000,
        help="Half-width of search window (default: 10000)"
    )
    args = parser.parse_args()

    center = parse_sci(args.center)
    b_range = args.range

    print(f"Searching for best prime in [{sci_pow_plus(center - b_range)} … {sci_pow_plus(center + b_range)}]")
    print("Building ab_table up to ±", b_range, "…", end="", flush=True)
    ab = build_ab_table(b_range)
    print(" done.")
    δ0 = compute_delta0()

    prime_val, offset, score = find_best_prime(center, b_range, ab, δ0)
    if prime_val is not None:
        print(f"→ Best prime: {sci_pow_plus(prime_val)}  (offset {offset}, ab_table score {score})")
    else:
        fallback = nextprime(center)
        print("No prime found among minimal-score survivors.")
        print(f"→ Falling back to nextprime({sci_pow_plus(center)}) = {sci_pow_plus(fallback)}")

if __name__ == '__main__':
    main()
