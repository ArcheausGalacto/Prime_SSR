#!/usr/bin/env python3
import sys, math
import numpy as np
from sympy import primerange

# ─────────────────────────────────────────────────────────────────────────────
# 1) Build ab_table[x] = # unordered prime‐pairs summing to x, for 1≤x≤N_max
# ─────────────────────────────────────────────────────────────────────────────
N_max = 1_000_000
limit = N_max // 2

p0 = np.zeros(limit+1, int)
for p in primerange(2, limit+1):
    p0[p] = 1

L   = 1 << int(math.ceil(math.log2(2*limit + 1)))
pad = np.zeros(L, int)
pad[:limit+1] = p0
P   = np.fft.rfft(pad)
conv= np.fft.irfft(P*P, n=L).round().astype(int)[:N_max+1]

diag = np.zeros_like(conv)
for x in range(0, N_max+1, 2):
    j = x // 2
    if j <= limit and p0[j]:
        diag[x] = 1

ab_table = (conv + diag) // 2

# ─────────────────────────────────────────────────────────────────────────────
# 2) Divergent‐harmonic fit params & dip offset δ₀ mod 6
# ─────────────────────────────────────────────────────────────────────────────
m, b =  0.0077972, 0.23527
A, B = -0.0019369, 0.0032648
ω, φ =  2*math.pi/6,   2.1382

# solve ω·δ₀ + φ ≡ 3π/2  ⇒ δ₀
delta = (1/ω)*(1.5*math.pi - φ)
δ0    = int(round(delta)) % 6

# ─────────────────────────────────────────────────────────────────────────────
# 3) Main: given "1e+<exp>", find candidate "1e+exp + offset"
# ─────────────────────────────────────────────────────────────────────────────
def find_offset(sci_str):
    # parse input
    try:
        mant, exp = sci_str.lower().split('e+')
        mant = int(mant)
        exp  = int(exp)
    except:
        raise ValueError("Usage: script.py 1e+<exp> (mantissa must be integer)")
    if mant != 1:
        raise ValueError("Mantissa must be 1 for this script.")
    print(f"Target base = 1e+{exp}")

    # compute rem so that k ≡ δ₀ mod 6: rem ≡ base_mod6 - δ₀
    base_mod6 = pow(10, exp, 6)
    rem       = (base_mod6 - δ0) % 6
    print(f"Shift k = base - {rem}  (k ≡ δ₀ mod6)")

    # find minimal ab_table[x] over 1≤x≤N_max
    vals = ab_table[1:]  # index 0 is unused
    rmin = vals.min()

    # survivors: x>rem with ab_table[x]==rmin
    survivors = [x for x in range(rem+1, N_max+1) if ab_table[x]==rmin]
    if not survivors:
        raise RuntimeError(f"No survivors x>rem up to {N_max}; increase N_max")

    x = survivors[0]
    offset = x - rem
    print(f"Minimal ab_table = {rmin}; first survivor x = {x}")
    print(f"Candidate prime ≈ 1e+{exp} + {offset}")

if __name__=="__main__":
    if len(sys.argv)!=2:
        print("Usage: python3 script.py 1e+<exp>")
        sys.exit(1)
    find_offset(sys.argv[1])
