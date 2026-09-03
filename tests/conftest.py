"""Test configuration.

juliacall gets its Julia project from `CONDA_PREFIX` whenever `sys.prefix` equals
`sys.base_prefix`, which is the case for a pixi (conda-style) environment. Anything that
runs pytest outside of `pixi run` -- the VS Code test pane, for instance -- therefore
inherits whatever conda environment happens to be active in the shell, and loads the
wrong version of RationalFunctionApproximation. Pin the project to the one belonging to
this interpreter before any test module imports pyratapprox.
"""

import os
import sys

_project = os.path.join(sys.prefix, "julia_env")
if os.path.isdir(_project):
    os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", _project)
