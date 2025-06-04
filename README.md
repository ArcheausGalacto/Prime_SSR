# Prime_SSR

This project contains scripts for experimenting with large prime searching via the "HPF Sniper" algorithm. The latest script `HPF_Sniper_12.py` can optionally use `gmpy2` for faster big integer primality checks and will try to leverage GPU acceleration with `numba` if available.

## Dependencies

- Python 3.12 or later
- `numpy`
- `sympy`
- `tqdm`
- `gmpy2` (optional but recommended)
- `numba` (optional, for GPU acceleration)

Install the required packages via:

```bash
pip install numpy sympy tqdm gmpy2 numba
```

GPU acceleration requires a CUDA-capable GPU and the appropriate CUDA toolkit installed. The script automatically falls back to CPU methods when CUDA is unavailable.

 ## Dependencies
 
 - Python 3.12 or later
 - `numpy`
 - `sympy`
 - `tqdm`
 - `gmpy2` (optional but recommended)
 - `numba` (optional, for GPU acceleration)
 
 Install the required packages via:
 
 ```bash
 pip install numpy sympy tqdm gmpy2 numba
 ```
 
 GPU acceleration requires a CUDA-capable GPU and the appropriate CUDA toolkit installed. The script automatically falls back to CPU methods when CUDA is unavailable.

## Theoretical Notes

`HPF_Sniper` explores prime values near huge powers of ten.  The search space is narrowed by modelling local oscillations in the distribution of primes and by skipping large ranges known to contain composites.  A set of √k‑enhanced oscillator fit parameters guides which offsets are most promising, dramatically reducing the number of candidates that must be tested.  Version 13 introduces a chunk-based big integer representation.  Each candidate value is stored as an array of base‑10 digits grouped into wide chunks.  Modular reductions are then performed by propagating remainders from the most significant chunk downward.  This avoids creating a single massive integer object and allows quick trial division on numbers with millions of digits.

These utilities provide the groundwork for further refinements.  By combining trial division over the pre‑chunked representation with probabilistic tests and parallel scanning of candidate offsets, the script aims to reach extremely large primes far quicker than a naive search.

`HPF_Sniper_13_no_GPU.py` can test numbers such as `10^1_000_000 + k` entirely with chunk arithmetic.  The number is constructed as a string, split into manageable pieces, and trial division is propagated chunk by chunk.

## Example

Running a short search near `1e+6` demonstrates the output format.  The prime
is printed using the `1e+N + k` notation:

```bash
$ python3 HPF_Sniper_13.py 1e+6 --range 1000
Center 1e+6 + 0 (center % 6 = 4). Valid δ₀ start options for k: [1, 3]
Searching candidates from 1e+6 + 1 to 1e+6 + 1000, using δ₀=1 stream.
--- Successfully Found Prime ---
Center was: 1e+6 + 0
Selected offset k = 133 (using δ₀=1 stream)
Prime Candidate = 1e+6 + 133
Value: 1000133
```
