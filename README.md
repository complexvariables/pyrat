# pyratapprox: Approximation by rational functions 

pyratapprox provides a Python interface to Julia's [RationalFunctionApproximation](https://complexvariables.github.io/RationalFunctionApproximation.jl/) library, enabling efficient computation of rational function approximations and interpolations.

## Installation

```bash
pip install pyratapprox
```

**Note**: Requires Julia to be installed. The required Julia packages will be installed automatically on first use.

## Quick Start

```python
import numpy as np
from pyratapprox import approximate, unitinterval, AAA

# Approximate a function on the unit interval [-1, 1]
f = approximate(np.sin, unitinterval, method=AAA)

# Evaluate the approximation
y = f(0.5)

# Get poles and residues
poles = f.poles()
poles, residues = f.residues()
```

## Documentation

Full documentation is available at: https://complexvariables.github.io/pyratapprox

## License

MIT License

## Author

Toby Driscoll (driscoll@udel.edu)
