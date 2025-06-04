import numpy as np, math, time, csv
from sympy import primerange, isprime, nextprime

# ─────────────────────────────────────────────────────────────────────────────
# Helper: format a big integer as "1e+<exp> + <offset>"
# ─────────────────────────────────────────────────────────────────────────────
def sci_pow_plus(n: int):
    s = str(n)
    exp = len(s) - 1
    base = 10 ** exp
    offset = n - base
    return f"1e+{exp} + {offset}"

# ─────────────────────────────────────────────────────────────────────────────
# 1) Build ab_table[x] = # unordered prime‐pairs summing to x, for x ≤ N_max
# ─────────────────────────────────────────────────────────────────────────────
N_max = 2_000_000
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
print(f"✓ ab_table built for 0 ≤ x ≤ {N_max}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 2) Divergent‐harmonic oscillator params and compute dip offset δ₀ mod 6
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
# 3) Bootstrap: 10× jumps, fixed half‐width b_range, safe clamps,
#    log each k to CSV with elapsed time
# ─────────────────────────────────────────────────────────────────────────────
k        = 101
shifts   = [k]
orders   = 500        # number of decades to climb
b_range  = 10_000

start_time = time.time()
csv_file    = open("primes_log.csv", "w", newline="")
writer      = csv.writer(csv_file)
writer.writerow(["order", "prime_k", "elapsed_s"])

# log initial shift
writer.writerow([0, k, f"{time.time()-start_time:.6f}"])
csv_file.flush()

print("Starting k₀ =", k)
for order in range(1, orders+1):
    power  = math.floor(math.log10(k)) + 1
    center = 10 ** power

    low_n, high_n = center - b_range, center + b_range
    x_low, x_high = low_n - k, high_n - k

    lo = max(0, int(math.ceil(x_low)))
    hi = min(N_max, int(math.floor(x_high)))

    if lo > hi:
        # no overlap: fallback
        k = nextprime(center)
        print(f"\nOrder {order}: no overlap, fallback to nextprime → k = {sci_pow_plus(k)}")
    else:
        # only residue class δ₀ mod 6
        start = lo + ((δ0 - lo) % 6)
        xs = list(range(start, hi+1, 6))
        if not xs:
            # no hits: fallback
            k = nextprime(center)
            print(f"\nOrder {order}: no residue-class hits, fallback → k = {sci_pow_plus(k)}")
        else:
            vals      = [ab_table[x] for x in xs]
            rmin      = min(vals)
            survivors = [x+k for x,v in zip(xs,vals) if v==rmin]
            print(f"\nOrder {order}: center={sci_pow_plus(center)}, "
                  f"window=[{sci_pow_plus(low_n)}…{sci_pow_plus(high_n)}], "
                  f"candidates={len(xs)}, r_min={rmin}")
            # primality-test survivors
            found = next((n for n in survivors if isprime(n)), None)
            if found:
                k = found
                print(" → picked prime k =", sci_pow_plus(k))
            else:
                k = nextprime(center)
                print(" — no survivor prime; fallback → k =", sci_pow_plus(k))

    shifts.append(k)
    elapsed = time.time() - start_time
    # write actual integer k to CSV
    writer.writerow([order, k, f"{elapsed:.6f}"])
    if order % 50 == 0:
        csv_file.flush()
    print(f"   elapsed = {elapsed:.2f}s")

csv_file.close()
print("\n✅ Done. Logged primes in primes_log.csv")
