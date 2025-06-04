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
