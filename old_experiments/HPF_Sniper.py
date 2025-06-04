#!/usr/bin/env python3
import sys, math, numpy as np
from sympy import primerange, isprime, nextprime

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def parse_sci(s: str) -> int:
    mant, exp = s.lower().split('e+')
    return int(mant) * (10**int(exp))

def sci_pow_plus(n: int) -> str:
    s = str(n)
    e = len(s) - 1
    base = 10**e
    offset = n - base
    return f"1e+{e} + {offset}"

# ─────────────────────────────────────────────────────────────────────────────
# 1) Build ab_table up to N_max
# ─────────────────────────────────────────────────────────────────────────────
N_max = 2_000_000
limit = N_max//2

p0 = np.zeros(limit+1, int)
for p in primerange(2, limit+1):
    p0[p] = 1

L   = 1 << int(math.ceil(math.log2(2*limit+1)))
pad = np.zeros(L, int); pad[:limit+1] = p0
P   = np.fft.rfft(pad)
conv= np.fft.irfft(P*P, n=L).round().astype(int)[:N_max+1]

diag = np.zeros_like(conv)
for x in range(0, N_max+1, 2):
    j = x//2
    if j<=limit and p0[j]:
        diag[x] = 1

ab_table = (conv + diag)//2

# ─────────────────────────────────────────────────────────────────────────────
# 2) Oscillator + compute δ₀ mod 6
# ─────────────────────────────────────────────────────────────────────────────
m, b =  0.0077972, 0.23527
A, B = -0.0019369, 0.0032648
ω, φ =  2*math.pi/6,   2.1382
delta = (1/ω)*(1.5*math.pi - φ)
δ0    = int(round(delta)) % 6

# ─────────────────────────────────────────────────────────────────────────────
# 3) Single‐shot find
# ─────────────────────────────────────────────────────────────────────────────
def find_prime_near(sci_str: str):
    e    = int(sci_str.split('e+')[1])
    base = 10**e
    print(f"Target base = 1e+{e}")

    # 3.a) shift k ≡ δ₀ mod 6
    rem = (pow(10, e, 6) - δ0) % 6
    k   = base - rem
    print(f"Using shift k = 1e+{e} - {rem}")

    # 3.b) scan offsets 1…N_max
    xs   = list(range(1, N_max+1))
    vals = [ab_table[x] for x in xs]
    rmin = min(vals)

    # 3.c) survivors with minimal count *and* x>rem
    surv = [x for x,v in zip(xs,vals) if v==rmin and x>rem]
    print(f"Minimal count {rmin}, survivors={len(surv)} offsets (x>rem)")

    # 3.d) primality‐test in order
    for x in surv:
        n = k + x
        if isprime(n):
            print(f"→ Found prime: {sci_pow_plus(k)}  +  {x}  =  {sci_pow_plus(n)}")
            return
    print("⚠️ None found in table, fallback to nextprime")
    p = nextprime(base)
    print(f"  → {sci_pow_plus(p)}")

if __name__=="__main__":
    sci = sys.argv[1]
    find_prime_near(sci)
