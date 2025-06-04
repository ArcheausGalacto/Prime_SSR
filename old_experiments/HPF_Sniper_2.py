#!/usr/bin/env python3
import sys, math, numpy as np
from sympy import primerange

# ─────────────────────────────────────────────────────────────────────────────
# 1) Build ab_table[x] for x ≤ N_max via FFT
# ─────────────────────────────────────────────────────────────────────────────
N_max = 1_000_000
limit = N_max // 2

p0 = np.zeros(limit+1, int)
for p in primerange(2, limit+1):
    p0[p] = 1

L   = 1 << int(math.ceil(math.log2(2*limit + 1)))
pad = np.zeros(L, int); pad[:limit+1] = p0
P   = np.fft.rfft(pad)
conv= np.fft.irfft(P*P, n=L).round().astype(int)[:N_max+1]

diag = np.zeros_like(conv)
for x in range(0, N_max+1, 2):
    j = x//2
    if j <= limit and p0[j]:
        diag[x] = 1

ab_table = (conv + diag) // 2

# ─────────────────────────────────────────────────────────────────────────────
# 2) Divergent‐harmonic fit params & compute δ₀ mod 6
# ─────────────────────────────────────────────────────────────────────────────
m, b =  0.0077972, 0.23527
A, B = -0.0019369, 0.0032648
ω, φ =  2*math.pi/6,   2.1382

delta = (1/ω)*(1.5*math.pi - φ)
δ0    = int(round(delta)) % 6

# ─────────────────────────────────────────────────────────────────────────────
# 3) Main routine—no huge ints, just compute offset
# ─────────────────────────────────────────────────────────────────────────────
def find_offset(sci_str):
    # parse "1e+123" → mant=1, exp=123
    mant, exp = sci_str.lower().split('e+')
    mant, exp = int(mant), int(exp)
    if mant != 1:
        raise ValueError("Mantissa must be 1 (e.g. '1e+1000')")
    print(f"Target base = 1e+{exp}")

    # compute rem so that k ≡ δ₀ mod 6
    # 10^exp mod 6 gives base_mod6
    base_mod6 = pow(10, exp, 6)
    rem       = (base_mod6 - δ0) % 6
    print(f"Shift alignment rem = {rem}  (so k ≡ δ₀ mod 6)")

    # find minimal ab_table[x] for x in 1..N_max
    # these are all small loops—no big ints at all
    xs   = range(1, N_max+1)
    vals = ab_table[1:]  # index 0 unused
    rmin = int(vals.min())

    # survivors with minimal count AND x>rem (to ensure offset>0)
    survivors = [x for x in xs if ab_table[x]==rmin and x>rem]
    if not survivors:
        raise RuntimeError(f"No survivors with x>rem in [1..{N_max}]")

    x0     = survivors[0]
    offset = x0 - rem
    print(f"Minimal ab_table = {rmin}, first survivor x = {x0}")
    print(f"\nCandidate prime ≈ 1e+{exp} + {offset}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 HPF_Sniper.py 1e+<exp>")
        sys.exit(1)
    find_offset(sys.argv[1])
