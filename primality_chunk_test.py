#!/usr/bin/env python3
"""Exhaustive primality testing for huge numbers using chunk arithmetic.

This script demonstrates trial division on extremely large integers such as
``10**1_000_000 + 61`` represented as strings.  The candidate is split into
base-``10**chunk_size`` chunks so that modulus operations can be computed
incrementally without ever forming a single gigantic integer object.

Given a center of ``10**exp`` and an offset ``k``, the program reports an
approximate time to fully confirm primality using pure trial division.
During testing, it prints a rough confidence based on how far it has
progressed toward the dynamic stopping point ``candidate // divisor``.

This approach is illustrative only.  Exhaustively checking values near
``10**1_000_000`` would take far longer than the age of the universe.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import List, Tuple


if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(100_000_000)


# ---------------------------------------------------------------------------
# Chunk utilities
# ---------------------------------------------------------------------------

def split_to_chunks(num_str: str, chunk_size: int = 9) -> List[str]:
    """Split ``num_str`` into big-endian chunks of ``chunk_size`` digits."""
    num_str = num_str.lstrip().lstrip("+")
    if num_str.startswith("-"):
        raise ValueError("Negative numbers not supported")
    chunks: List[str] = []
    start = len(num_str) % chunk_size
    if start:
        chunks.append(num_str[:start])
    for i in range(start, len(num_str), chunk_size):
        chunks.append(num_str[i : i + chunk_size])
    return chunks


def chunks_mod(chunks: List[str], divisor_str: str, base: int) -> int:
    """Return ``number % int(divisor_str)`` where ``number`` is represented by ``chunks``."""
    divisor = int(divisor_str)
    rem = 0
    for ch in chunks:
        rem = (rem * base + int(ch)) % divisor
    return rem


def chunks_divide(chunks: List[str], divisor_str: str, base: int) -> Tuple[List[str], str]:
    """Return ``(quotient_chunks, remainder_str)`` for ``number // int(divisor_str)``."""
    divisor = int(divisor_str)
    q: List[str] = []
    rem = 0
    for ch in chunks:
        rem = rem * base + int(ch)
        q_digit = rem // divisor
        rem -= q_digit * divisor
        if q or q_digit:
            q.append(str(q_digit))
    if not q:
        q.append("0")
    return q, str(rem)


def compare_chunks_str_int(chunks: List[str], value_str: str, chunk_size: int) -> int:
    """Compare a chunked number with an integer represented as a string.

    Returns 1 if ``chunks`` > ``value_str``, 0 if equal, -1 if smaller.
    """
    other_chunks = split_to_chunks(value_str, chunk_size)
    if len(chunks) != len(other_chunks):
        return 1 if len(chunks) > len(other_chunks) else -1
    for a, b in zip(chunks, other_chunks):
        a_int = int(a)
        b_int = int(b)
        if a_int != b_int:
            return 1 if a_int > b_int else -1
    return 0


def number_str_from_power_offset(exp: int, offset: int) -> str:
    """Return decimal string for ``10**exp + offset`` (``offset`` >= 0)."""
    offset_str = str(offset)
    zeros = exp - len(offset_str)
    if zeros < 0:
        raise ValueError("offset must be smaller than 10**exp")
    return "1" + "0" * zeros + offset_str


# ---------------------------------------------------------------------------
# Helper routines
# ---------------------------------------------------------------------------

def estimate_time(num_str: str, ops_per_sec: float) -> str:
    """Return a textual estimate of trial division time for ``num_str``."""
    digits = len(num_str)
    # sqrt(candidate) has roughly half as many digits
    log10_divisors = (digits + 1) // 2
    # Estimated seconds = 10**log10_divisors / ops_per_sec
    log10_seconds = log10_divisors - math.log10(ops_per_sec)
    if log10_seconds > 6:
        return f"~10^{log10_seconds:.1f} seconds"
    seconds = 10 ** log10_seconds
    if seconds > 3600:
        return f"~{seconds/3600:.2f} hours"
    if seconds > 60:
        return f"~{seconds/60:.2f} minutes"
    return f"~{seconds:.2f} seconds"


def measure_ops_per_sec(candidate: str, chunk_size: int = 9, samples: int = 1000) -> float:
    """Roughly measure modulus operations per second."""
    base = 10 ** chunk_size
    chunks = split_to_chunks(candidate, chunk_size)
    start = time.perf_counter()
    for i in range(2, 2 + samples):
        chunks_mod(chunks, str(i), base)
    end = time.perf_counter()
    return samples / (end - start)


# ---------------------------------------------------------------------------
# Exhaustive primality test
# ---------------------------------------------------------------------------

def exhaustive_isprime(num_str: str, chunk_size: int = 9) -> bool:
    base = 10 ** chunk_size
    chunks = split_to_chunks(num_str, chunk_size)

    ops_sec = measure_ops_per_sec(num_str, chunk_size, samples=100)
    print("Estimated trial division speed:", f"{ops_sec:.2f} ops/sec")
    print("Estimated time to complete:", estimate_time(num_str, ops_sec))

    divisor = "2"
    limit_chunks, _ = chunks_divide(chunks, divisor, base)
    while True:
        if chunks_mod(chunks, divisor, base) == 0:
            print(f"Found factor {divisor}")
            return False
        limit_chunks, _ = chunks_divide(chunks, divisor, base)
        if compare_chunks_str_int(limit_chunks, divisor, chunk_size) <= 0:
            break
        divisor = str(int(divisor) + 1)
        if int(divisor) % 1000 == 0:
            progress = math.log10(int(divisor)) / ((len(num_str) + 1) // 2)
            print(f"Checked up to {divisor}, confidence ~{progress*100:.2f}%")
    return True


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exhaustive primality check using chunk arithmetic")
    parser.add_argument("exp", type=int, help="Exponent in 10**exp + k")
    parser.add_argument("k", type=int, help="Offset added to 10**exp")
    parser.add_argument("--chunk", type=int, default=9, help="Chunk size")
    args = parser.parse_args()

    candidate = number_str_from_power_offset(args.exp, args.k)
    print("Testing candidate:", f"1e+{args.exp} + {args.k}")
    isprime = exhaustive_isprime(candidate, args.chunk)
    if isprime:
        print("Likely prime (no factors found)")
    else:
        print("Composite")
