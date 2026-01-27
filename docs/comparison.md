---
kernelspec: 
    name: python3
---
# Comparison to AAA in SciPy

```{python}
from pyratapprox import *
import numpy as np
from scipy.interpolate import AAA

from scipy.special import gamma
x = np.linspace(-1.5, 5, 500)
f = gamma(x)
r_pyrat = approximate(f, x, method=AAA)
r_scipy = AAA(x, f)
```

```{python}
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
test_pts = np.linspace(-1.5, 5, 1000)
ax.plot(test_pts, r_pyrat(test_pts) - r_scipy(test_pts), 'k-', label='gamma(x)', lw=2)
```