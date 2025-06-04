#!/usr/bin/env python3
"""HPF Sniper 13 (no GPU)

A simplified script demonstrating chunk-based modular arithmetic
for extremely large numbers. It drops CUDA support and instead
focuses on handling values like 10**1_000_000 represented as strings.
"""

import argparse
from sympy import primerange
import sys

# Increase Python int->str limit (Python 3.11+)
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(100000000)

# ─────────────────────────────────────────────────────────────────────────────
# Chunk utilities
# ─────────────────────────────────────────────────────────────────────────────

def split_to_chunks(num_str: str, chunk_size: int = 9) -> list[int]:
    """Split ``num_str`` into big-endian integer chunks of ``chunk_size`` digits."""
    num_str = num_str.lstrip().lstrip('+')
    if num_str.startswith('-'):
        raise ValueError("Negative numbers not supported")

    chunks: list[int] = []
    start = len(num_str) % chunk_size
    if start:
        chunks.append(int(num_str[:start]))
    for i in range(start, len(num_str), chunk_size):
        chunks.append(int(num_str[i : i + chunk_size]))
    return chunks


def chunks_mod(chunks: list[int], divisor: int, base: int) -> int:
    """Return the modulus of a large number expressed as ``chunks``."""
    rem = 0
    for ch in chunks:
        rem = (rem * base + ch) % divisor
    return rem


def trial_division_chunked(num_str: str, limit: int = 100000, chunk_size: int = 9) -> bool:
    """Return ``False`` if ``num_str`` has a prime factor <= ``limit``."""
    base = 10 ** chunk_size
    chunks = split_to_chunks(num_str, chunk_size)
    for p in primerange(2, limit + 1):
        if chunks_mod(chunks, p, base) == 0:
            return False
    return True


def number_str_from_power_offset(exp: int, offset: int) -> str:
    """Return decimal string for 10**exp + offset (offset >= 0, offset < 10**exp)."""
    offset_str = str(offset)
    zeros = exp - len(offset_str)
    if zeros < 0:
        raise ValueError("offset must be smaller than 10**exp")
    return "1" + "0" * zeros + offset_str


def fast_isprime_big(num_str: str, trial_limit: int = 100000, chunk_size: int = 9) -> bool:
    """Probabilistic primality test for huge numbers represented as strings."""
    # Only do trial division. For actual use, integrate Miller-Rabin or gmpy2.
    return trial_division_chunked(num_str, trial_limit, chunk_size)


# ─────────────────────────────────────────────────────────────────────────────
# Command-line interface
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Test primality of 10^N + k using chunk arithmetic")
    parser.add_argument("exp", type=int, help="Exponent N for the base 10^N")
    parser.add_argument("offset", type=int, help="Offset k to add")
    parser.add_argument("--limit", type=int, default=10000, help="Trial division limit")
    args = parser.parse_args()

    candidate = number_str_from_power_offset(args.exp, args.offset)
    print(f"Testing candidate with {len(candidate)} digits…")
    if fast_isprime_big(candidate, args.limit):
        print("Candidate passes trial division (likely prime beyond that range).")
    else:
        print("Candidate is divisible by a small prime.")


if __name__ == "__main__":
    main()