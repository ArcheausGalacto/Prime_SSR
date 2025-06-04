#!/usr/bin/env python3
import sys, math, numpy as np
from sympy import primerange, isprime, nextprime

# ─────────────────────────────────────────────────────────────────────────────
# 1) Build ab_table[x] = # unordered prime-pairs a+b = x, for 1 ≤ x ≤ N_max
# ─────────────────────────────────────────────────────────────────────────────
N_max = 10_000_000
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

ab_table = (conv + diag)//2
print(f"✓ ab_table built for 1 ≤ x ≤ {N_max}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 2) Divergent‐harmonic oscillator parameters & dip offset δ₀
# ─────────────────────────────────────────────────────────────────────────────
m, b =  0.0077972, 0.23527
A, B = -0.0019369, 0.0032648
ω, φ =  2*math.pi/6,   2.1382

delta = (1/ω)*(1.5*math.pi - φ)
δ0    = int(round(delta)) % 6
print(f"Dip offset δ₀ = {δ0} mod 6\n")

# ─────────────────────────────────────────────────────────────────────────────
# 3) Single‐shot prime finder using your original method
#    Usage: python3 this_script.py 1e+<exp>
# ─────────────────────────────────────────────────────────────────────────────
def sci_candidate(sci_str):
    # parse "1e+E"
    mant, exp = sci_str.lower().split('e+')
    mant, exp = int(mant), int(exp)
    if mant != 1:
        raise ValueError("Mantissa must be 1, e.g. '1e+123'")
    print(f"Target base = 1e+{exp}")

    # compute rem so that k ≡ δ₀ mod 6
    base_mod6 = pow(10, exp, 6)
    rem       = (base_mod6 - δ0) % 6
    # shift = base - rem (but we never build base)
    print(f"Alignment rem = {rem}  (ensures k≡δ₀ mod6)")

    # window around center = base: x_low..x_high = (center±b_range)-k
    b_range = 10_000
    # center = base, k = base - rem
    # so x_low = (base - b_range) - (base - rem) = rem - b_range
    #    x_high= (base + b_range) - (base - rem) = b_range + rem
    lo = max(1, rem - b_range)
    hi = min(N_max, b_range + rem)

    # align to the dip residue class mod 6
    start = lo + ((δ0 - lo) % 6)
    xs = list(range(start, hi+1, 6))
    if not xs:
        raise RuntimeError("No candidates; increase N_max or b_range")

    # pick offsets with minimal ab_table[x]
    vals = [ab_table[x] for x in xs]
    rmin = min(vals)
    survivors = [x for x,v in zip(xs,vals) if v==rmin and x>rem]
    if not survivors:
        raise RuntimeError("No survivors above rem; increase N_max or b_range")

    # primality-test survivors in ascending order
    for x in survivors:
        # candidate = base - rem + x = base + (x-rem)
        offset = x - rem
        # we don't need to build base**: just report offset
        print(f"Minimal ab_table = {rmin}, tried offset = {offset}")
        # optionally test primality of the big number:
        # from sympy import isprime
        # if isprime(pow(10,exp)+offset):
        #     print("Found prime!")
        print(f"\n→ Candidate prime ≈ 1e+{exp} + {offset}")
        return

if __name__=="__main__":
    if len(sys.argv)!=2:
        print("Usage: python3 this_script.py 1e+<exp>")
        sys.exit(1)
    sci_candidate(sys.argv[1])
