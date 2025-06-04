import matplotlib.pyplot as plt
from statistics import mean

# 1. Generate primes up to 300
def sieve(n):
    is_prime = [False, False] + [True] * (n - 1)
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i, prime in enumerate(is_prime) if prime]

primes = sieve(3000)
prime_set = set(primes)

# 2. Compute average absolute deviation for odd shifts k up to 149
shifts = list(range(1, 1500, 2))
avg_devs = []
for k in shifts:
    # dynamic domain: primes p <= 2k+1
    domain_primes = [p for p in primes if p <= 2*k + 1]
    # representation counts at those primes
    counts = [sum(1 for a in primes if (p - k - a) in prime_set) for p in domain_primes]
    # successive absolute differences
    diffs = [abs(counts[i+1] - counts[i]) for i in range(len(counts) - 1)]
    avg_devs.append(mean(diffs) if diffs else 0)

# 3. Plot
plt.figure(figsize=(10, 5))
plt.plot(shifts, avg_devs, linewidth=1, color='blue')
plt.title("Average |Δ r_k(p)| vs odd shift k\n(domain p ≤ 2k+1)")
plt.xlabel("Shift k (odd, up to 149)")
plt.ylabel("Average absolute deviation")
plt.grid(True)
plt.tight_layout()
plt.show()
