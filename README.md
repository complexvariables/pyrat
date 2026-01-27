# PyRat: Python Rational Function Approximation

PyRat provides a Python interface to Julia's RationalFunctionApproximation library, enabling efficient computation of rational function approximations and interpolations.

## Installation

```bash
pip install pyrat
```

**Note**: Requires Julia to be installed. The required Julia packages will be installed automatically on first use.

## Quick Start

```python
import numpy as np
from pyrat import approximate, unitinterval, AAA

# Approximate a function on the unit interval [-1, 1]
f = approximate(np.sin, unitinterval, method=AAA)

# Evaluate the approximation
y = f(0.5)

# Get poles and residues
poles = f.poles()
poles, residues = f.residues()
```

## Documentation

Full documentation is available at: https://complexvariables.github.io/pyrat

## License

MIT License

## Author

Toby Driscoll (driscoll@udel.edu)
