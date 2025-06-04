#!/usr/bin/env python3
import sys

# lift Python 3.11+ limit on big‐int→str conversions
sys.set_int_max_str_digits(60_000)
import re

def expand(s: str) -> str:
    # split off the last '+' (the offset)
    try:
        left, offset_str = s.rsplit('+', 1)
    except ValueError:
        raise ValueError("Input must be of the form '<coeff>e+<exp> + <offset>'")
    left = left.strip()
    offset_str = offset_str.strip()
    # parse offset
    offset = int(offset_str)
    # parse coeff and exponent from left, e.g. '1e+451'
    m = re.fullmatch(r'(\d+)e\+(\d+)', left)
    if not m:
        raise ValueError("Left part must be of the form '<coeff>e+<exp>'")
    coeff = int(m.group(1))
    exp   = int(m.group(2))
    # compute expansion
    return str(coeff * pow(10, exp) + offset)

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} \"1e+451 + 451\"", file=sys.stderr)
        sys.exit(1)
    s = sys.argv[1]
    try:
        full = expand(s)
    except ValueError as e:
        print("Error:", e, file=sys.stderr)
        sys.exit(1)
    with open("expanded.txt", "w") as f:
        f.write(full)
    print(f"Expanded integer ({len(full)} digits) written to expanded.txt")

if __name__ == "__main__":
    main()
