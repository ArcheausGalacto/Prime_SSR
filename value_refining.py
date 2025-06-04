#!/usr/bin/env python3
import argparse, math, sys
import numpy as np
import matplotlib.pyplot as plt
from sympy import primerange

def build_ab_table(N):
    p0 = np.zeros(N+1, int)
    for p in primerange(2, N+1):
        p0[p] = 1
    L = 1 << int(math.ceil(math.log2(2*N+1)))
    pad = np.zeros(L, int); pad[:N+1] = p0
    P    = np.fft.rfft(pad)
    conv = np.fft.irfft(P*P, n=L).round().astype(int)[:N+1]
    diag = np.zeros_like(conv)
    for j in range(2, N+1):
        if p0[j]:
            idx = 2*j
            if idx < len(diag):
                diag[idx] = 1
    return (conv + diag)//2

def compute_avg_devs(ab, max_k):
    primes = list(primerange(2, 2*max_k+2))
    shifts = np.arange(1, max_k+1, 2)
    out = []
    for k in shifts:
        dom = [p for p in primes if p <= 2*k+1]
        cnt = [(ab[p-k] if 0 <= p-k < len(ab) else 0) for p in dom]
        diffs = np.abs(np.diff(cnt))
        out.append(diffs.mean() if diffs.size>0 else 0.0)
    return shifts, np.array(out)

def fit_models(shifts, avg_devs):
    ω = 2*math.pi/6

    # 1) oscillator
    X0 = np.column_stack([
        shifts,
        np.ones_like(shifts),
        shifts*np.sin(ω*shifts),
        shifts*np.cos(ω*shifts),
    ])
    c0, *_ = np.linalg.lstsq(X0, avg_devs, rcond=None)
    fit0   = X0.dot(c0)
    r0     = np.corrcoef(avg_devs, fit0)[0,1]

    # 2) + log
    X1 = np.column_stack([
        shifts,
        np.ones_like(shifts),
        np.log(shifts+1),
        shifts*np.sin(ω*shifts),
        shifts*np.cos(ω*shifts),
    ])
    c1, *_ = np.linalg.lstsq(X1, avg_devs, rcond=None)
    fit1   = X1.dot(c1)
    r1     = np.corrcoef(avg_devs, fit1)[0,1]

    # 3) + sqrt
    X2 = np.column_stack([
        shifts,
        np.ones_like(shifts),
        np.sqrt(shifts),
        shifts*np.sin(ω*shifts),
        shifts*np.cos(ω*shifts),
    ])
    c2, *_ = np.linalg.lstsq(X2, avg_devs, rcond=None)
    fit2   = X2.dot(c2)
    r2     = np.corrcoef(avg_devs, fit2)[0,1]

    return (c0,r0,fit0), (c1,r1,fit1), (c2,r2,fit2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_k', type=int, default=3000,
                        help="Max odd shift k (default 3000)")
    args = parser.parse_args()

    max_k = args.max_k
    if hasattr(sys, "set_int_max_str_digits"):
        sys.set_int_max_str_digits(1000000)

    print(f"Building ab_table up to {max_k+1}…", end="", flush=True)
    ab = build_ab_table(max_k+1)
    print(" done.")

    print("Computing ⟨|Δrₖ(p)|⟩…", end="", flush=True)
    shifts, avg_devs = compute_avg_devs(ab, max_k)
    print(" done.\n")

    (c0,r0,fit0), (c1,r1,fit1), (c2,r2,fit2) = fit_models(shifts, avg_devs)

    m0,b0,A0,B0 = c0
    m1,b1,C,A1,B1 = c1
    m2,b2,D,A2,B2 = c2

    print("Oscillator fit:      ", f"m={m0:.6g}", f"b={b0:.6g}", f"A={A0:.6g}", f"B={B0:.6g}", f"r={r0:.4f}")
    print("+ log fit:           ", f"m={m1:.6g}", f"b={b1:.6g}", f"C={C:.6g}", f"A={A1:.6g}", f"B={B1:.6g}", f"r={r1:.4f}")
    print("+ sqrt fit (new):    ", f"m={m2:.6g}", f"b={b2:.6g}", f"D={D:.6g}", f"A={A2:.6g}", f"B={B2:.6g}", f"r={r2:.4f}\n")

    plt.figure(figsize=(12,5))
    plt.plot(shifts, avg_devs, label="Actual", color='blue',   linewidth=1)
    plt.plot(shifts, fit0,     label="Oscillator", color='orange',linewidth=1)
    plt.plot(shifts, fit1,     label="+ log",      color='green', linewidth=1)
    plt.plot(shifts, fit2,     label="+ sqrt",     color='purple',linewidth=1)
    plt.title(f"Avg |Δrₖ(p)| vs odd shift k (up to {max_k})")
    plt.xlabel("Shift k (odd)")
    plt.ylabel("Average absolute deviation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
