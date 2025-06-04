import numpy as np
import matplotlib.pyplot as plt

# 1. Generate primes and compute avg_devs as before
def sieve(n):
    is_prime = [False, False] + [True] * (n - 1)
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i, p in enumerate(is_prime) if p]

primes = sieve(3000)
prime_set = set(primes)

# Odd shifts up to 149
shifts = np.array(list(range(1, 1500, 2)))
avg_devs = []
for k in shifts:
    domain_primes = [p for p in primes if p <= 2*k + 1]
    counts = [sum(1 for a in primes if (p - k - a) in prime_set) for p in domain_primes]
    diffs = np.abs(np.diff(counts))
    avg_devs.append(np.mean(diffs) if len(diffs) > 0 else 0)
avg_devs = np.array(avg_devs)

# 2. Fit model: f(k) = m*k + b + A*k*sin(omega k) + B*k*cos(omega k)
omega = 2*np.pi / 6
X = np.column_stack([
    shifts,                                 # m*k
    np.ones_like(shifts),                   # b
    shifts * np.sin(omega * shifts),        # A * k * sin(omega*k)
    shifts * np.cos(omega * shifts)         # B * k * cos(omega*k)
])
# Least squares
coef, *_ = np.linalg.lstsq(X, avg_devs, rcond=None)
m_fit, b_fit, A_fit, B_fit = coef

# Compute fitted values
fit_vals = X.dot(coef)

# Pearson r
r_value = np.corrcoef(avg_devs, fit_vals)[0,1]

# Plotting
plt.figure(figsize=(10,5))
plt.plot(shifts, avg_devs, label="Actual", color='blue')
plt.plot(shifts, fit_vals, label="Divergent Harmonic Fit", color='orange')
plt.title("Avg |Δrₖ(p)| vs shift k\nwith Divergent Harmonic Oscillator Fit")
plt.xlabel("Shift k (odd)")
plt.ylabel("Average absolute deviation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Output fit parameters and r
print(m_fit, b_fit, A_fit, B_fit, r_value)
