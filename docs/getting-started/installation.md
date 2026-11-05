# Installation

## Requirements

- Python 3.12, 3.13, or 3.14
- NumPy >= 2.0
- Numba >= 0.60

## Standard Installation

Install Quatham from PyPI using pip:

```bash
pip install quatham
```

## Development Installation

If you want to contribute to Quatham or use the latest development version:

```bash
# Clone the repository
git clone https://github.com/jpmorr/quatham.git
cd quatham

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e ".[dev]"
```

## Verification

Verify your installation by running a simple test:

```python
from quatham import Quaternion
import numpy as np

q = Quaternion.from_axisangle([0, 0, 1], np.pi/2)
print(q)
```

Expected output:
```
Quaternion(1.570796; [0. 0. 1.])
```

## Troubleshooting

### Numba Compilation

On first use, Numba will JIT-compile the low-level functions. This may take a few seconds. Subsequent calls will be fast as the compiled code is cached.

If you encounter compilation errors:
1. Ensure you have a compatible C compiler installed
2. Clear the Numba cache: `rm -rf __pycache__ ~/.cache/numba`
3. Update Numba to the latest version

### NumPy Compatibility

Quatham requires NumPy 2.0+. Check your version:

```python
import numpy as np
print(np.__version__)
```

Update if necessary:
```bash
pip install --upgrade numpy
```
