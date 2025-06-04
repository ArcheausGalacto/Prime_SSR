import numpy as np, math
from sympy import primerange, isprime, nextprime

# ─────────────────────────────────────────────────────────────────────────────
# Helper: compact scientific‐notation formatting
def sci(n, sig=4):
    s = str(int(n))
    sign = "-" if s.startswith("-") else ""
    s = s.lstrip("-")
    exp = len(s) - 1
    if exp < sig:
        return sign + s
    mant = s[0] + "." + s[1:sig]
    return f"{sign}{mant}e+{exp}"

# ─────────────────────────────────────────────────────────────────────────────
# 1) FFT‐build ab_table[x] = # of unordered prime‐pairs summing to x, for x≤N_max
# ─────────────────────────────────────────────────────────────────────────────
N_max = 2_000_000
limit = N_max // 2

# prime‐indicator up to `limit`
p0 = np.zeros(limit+1, int)
for p in primerange(2, limit+1):
    p0[p] = 1

# convolve via FFT
L   = 1 << int(math.ceil(math.log2(2*limit + 1)))
pad = np.zeros(L, int); pad[:limit+1] = p0
P   = np.fft.rfft(pad)
conv= np.fft.irfft(P*P, n=L).round().astype(int)[:N_max+1]

# diagonal correction (a=a) and /2 for unordered counts
diag    = np.zeros_like(conv)
for x in range(0, N_max+1, 2):
    j = x//2
    if j<=limit and p0[j]:
        diag[x] = 1
ab_table = (conv + diag) // 2

print(f"✓ ab_table built for 0 ≤ x ≤ {N_max}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 2) Divergent‐harmonic oscillator params + compute dip offset δ₀ mod 6
# ─────────────────────────────────────────────────────────────────────────────
m, b =  0.0077972, 0.23527
A, B = -0.0019369, 0.0032648
ω, φ =  2*math.pi/6, 2.1382

delta = (1/ω)*(1.5*math.pi - φ)
δ0    = int(round(delta)) % 6
print("Dip offset δ₀ =", δ0, "mod 6\n")

def f(k):
    return m*k + b + A*k*math.sin(ω*k) + B*k*math.cos(ω*k)

# ─────────────────────────────────────────────────────────────────────────────
# 3) Bootstrap: 10× jumps, fixed half‐width b_range, safe clamps
# ─────────────────────────────────────────────────────────────────────────────
k        = 101        # initial shift
shifts   = [k]
orders   = 1000          # climb to ~10^6
b_range  = 10_000     # half‐width of each search window

print("Starting k₀ =", k)
for order in range(1, orders+1):
    # next power‐of‐10 center:
    power  = math.floor(math.log10(k)) + 1
    center = 10**power

    # define [center−b…center+b]
    low_n, high_n = center - b_range, center + b_range

    # translate to x = n - k
    x_low  = low_n  - k
    x_high = high_n - k

    # clamp into [0..N_max]
    lo = max(0, int(math.ceil(x_low)))
    hi = min(N_max, int(math.floor(x_high)))

    # if there’s literally no overlap, fallback to nextprime
    if lo > hi:
        print(f"\nOrder {order}: no overlap (x_low={sci(x_low)}, x_high={sci(x_high)})")
        k = nextprime(center)
        print(" → fallback nextprime(center) =", sci(k))
        shifts.append(k)
        continue

    # only the δ₀‐residue class mod 6
    start = lo + ((δ0 - lo) % 6)
    xs = list(range(start, hi+1, 6))
    if not xs:
        print(f"\nOrder {order}: no x≡{δ0} mod6 in [{lo}…{hi}]: fallback")
        k = nextprime(center)
        print(" → fallback nextprime(center) =", sci(k))
        shifts.append(k)
        continue

    # O(1) lookups of ab_table[x], find minimal count
    vals = [ab_table[x] for x in xs]
    rmin = min(vals)
    survivors = [x+k for x,v in zip(xs,vals) if v==rmin]

    print(f"\nOrder {order}: center={sci(center)}, window=[{sci(low_n)}…{sci(high_n)}]")
    print(f"  x∈[{sci(lo)}…{sci(hi)}], candidates={len(xs)}, r_min={rmin}")

    # primality-test survivors
    found = next((n for n in survivors if isprime(int(n))), None)
    if found:
        print(" → picked prime k =", sci(found))
        k = int(found)
    else:
        k = nextprime(center)
        print(" — no survivor prime; fallback nextprime(center) =", sci(k))

    shifts.append(k)

print("\n✅ Bootstrap complete. shifts:", [sci(s) for s in shifts])
