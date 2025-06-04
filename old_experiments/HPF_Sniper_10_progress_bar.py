#!/usr/bin/env python3
import argparse
import math
import sys
import random

import numpy as np
from tqdm import tqdm
from sympy import primerange
from sympy.ntheory.primetest import mr, is_strong_lucas_prp

# ─────────────────────────────────────────────────────────────────────────────
# Lift Python’s big‐int→str safety limit (3.11+)
# ─────────────────────────────────────────────────────────────────────────────
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(1000000)

# ─────────────────────────────────────────────────────────────────────────────
# 1) Parse and format scientific notation → exact int
# ─────────────────────────────────────────────────────────────────────────────
def parse_sci(s: str) -> int:
    s = s.strip().lower().replace('+','')
    if 'e' in s:
        coeff_str, exp_str = s.split('e', 1)
        exp = int(exp_str)
        if '.' in coeff_str:
            a, b = coeff_str.split('.', 1)
            coeff = int(a + b)
            exp -= len(b)
        else:
            coeff = int(coeff_str)
        return coeff * (10**exp) if exp >= 0 else coeff // (10**-exp)
    return int(s)

def sci_pow_plus(n: int) -> str:
    ss  = str(n)
    exp = len(ss) - 1
    base = 10**exp
    offset = n - base
    return f"1e+{exp} + {offset}"

# ─────────────────────────────────────────────────────────────────────────────
# 2) Fast primality: Baillie–PSW + MR fallback
# ─────────────────────────────────────────────────────────────────────────────
def miller_rabin(n: int, rounds: int = 7) -> bool:
    if n < 2:
        return False
    d, s = n - 1, 0
    while not (d & 1):
        d >>= 1
        s += 1
    for _ in range(rounds):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        if x not in (1, n - 1):
            for _ in range(s - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
    return True

def fast_isprime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29):
        if n == p:
            return True
        if n % p == 0:
            return False
    if not mr(n, [2]):
        return False
    if not is_strong_lucas_prp(n):
        return False
    if n.bit_length() > 64:
        return miller_rabin(n, rounds=7)
    return True

def next_prime(n: int) -> int:
    cand = n + 1 if n % 2 == 0 else n + 2
    wheel = [4, 2, 4, 2, 4, 6, 2, 6]
    i = 0
    while True:
        if fast_isprime(cand):
            return cand
        cand += wheel[i]
        i = (i + 1) % len(wheel)

# ─────────────────────────────────────────────────────────────────────────────
# 3) √k-enhanced oscillator fit parameters
# ─────────────────────────────────────────────────────────────────────────────
ω = 2 * math.pi / 6
m, b, D = 0.000856281, -0.211299, 0.0705661
A, B    = -7.7839e-05, 0.0011218

def f(k: int) -> float:
    return (
        m * k
        + b
        + D * math.sqrt(k)
        + A * k * math.sin(ω * k)
        + B * k * math.cos(ω * k)
    )

def compute_delta0() -> int:
    φ = 2.1382
    δ = (1 / ω) * (1.5 * math.pi - φ)
    return int(round(δ)) % 6

# ─────────────────────────────────────────────────────────────────────────────
# 4) FFT build of ab_table[x] = # of unordered ways x = p1+p2
# ─────────────────────────────────────────────────────────────────────────────
def build_ab_table(N_max: int) -> np.ndarray:
    p0 = np.zeros(N_max + 1, dtype=int)
    for p in primerange(2, N_max + 1):
        p0[p] = 1

    L = 1 << int(math.ceil(math.log2(2 * N_max + 1)))
    pad = np.zeros(L, dtype=int)
    pad[: N_max + 1] = p0
    P = np.fft.rfft(pad)
    conv = np.fft.irfft(P * P, n=L).round().astype(int)[: N_max + 1]

    diag = np.zeros_like(conv)
    for j in range(2, N_max + 1):
        if p0[j]:
            idx = 2 * j
            if idx < len(diag):
                diag[idx] = 1

    return (conv + diag) // 2

# ─────────────────────────────────────────────────────────────────────────────
# 5) Pick and verify the best offset k with progress bars
# ─────────────────────────────────────────────────────────────────────────────
def find_candidate(center: int,
                   b_range: int,
                   ab: np.ndarray,
                   δ0: int):
    ks = list(range(δ0, b_range + 1, 6))
    fs = [f(k) for k in ks]
    counts = [ab[k] for k in ks]

    # 5a) Try largest-k local minima
    local = [
        ks[i]
        for i in range(1, len(ks) - 1)
        if counts[i] > 0 and fs[i] <= fs[i - 1] and fs[i] <= fs[i + 1]
    ]
    if local:
        for k in tqdm(sorted(local, reverse=True),
                      desc="Testing local minima", unit="k"):
            cand = center + k
            if fast_isprime(cand):
                return k, cand, counts[ks.index(k)]

    # 5b) Fallback: minimal-count survivors
    positive = [(k, c) for k, c in zip(ks, counts) if c > 0]
    if positive:
        rmin = min(c for _, c in positive)
        surv = [k for k, c in positive if c == rmin]
        for k in tqdm(sorted(surv),
                      desc="Testing survivors", unit="k"):
            cand = center + k
            if fast_isprime(cand):
                return k, cand, rmin

    # 5c) Ultimate fallback: next prime > center
    cand = next_prime(center)
    return cand - center, cand, None

# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="HPF_Sniper: pick & verify a huge prime near 1e+N"
    )
    parser.add_argument(
        'center',
        help="Center in scientific notation (e.g. 1e+5000)"
    )
    parser.add_argument(
        '--range',
        type=int,
        default=10000,
        help="Half-width above center (default 10000)"
    )
    parser.add_argument(
        '--delta0',
        type=int,
        help="Override δ₀ mod6 (default from fit)"
    )
    args = parser.parse_args()

    center = parse_sci(args.center)
    b_range = args.range
    δ0 = (args.delta0 % 6) if (args.delta0 is not None) else compute_delta0()

    print(f"Searching in [{sci_pow_plus(center)} … {sci_pow_plus(center + b_range)}], δ₀={δ0} mod6")
    print("Building ab_table…", end="", flush=True)
    ab = build_ab_table(b_range)
    print(" done.")

    k, candidate, score = find_candidate(center, b_range, ab, δ0)
    print(f"\nSelected offset k = {k}")
    if score is not None:
        print(f"ab_table[k] = {score}")
    print(f"Candidate = {sci_pow_plus(candidate)}")
    print(f"Primality test: {'✔️ prime' if fast_isprime(candidate) else '❌ composite'}")

if __name__ == '__main__':
    main()
