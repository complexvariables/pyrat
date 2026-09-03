"""
pyratapprox: Python interface for Rational Function Approximation

This package provides Python wrappers for Julia's RationalFunctionApproximation
library, enabling efficient computation of rational function approximations and
interpolations in Python.

The package supports:
- AAA (Adaptive Antoulas-Anderson) algorithm
- TCF (Thiele continued fraction) method
- Approximation on continuum domains (curves, regions)
- Approximation on discrete point sets
"""

import juliacall
import numpy as np
import cxregions as cr

# Initialize Julia module and load dependencies
jl = juliacall.newmodule("pyratapprox")
jl.seval('import Pkg')
installed = False
for v in jl.Pkg.dependencies().values():
    if v.name == "RationalFunctionApproximation":
        installed = True
        break
if not installed:
    jl.seval('Pkg.add("RationalFunctionApproximation")')
    jl.seval('Pkg.add("ComplexRegions")')
    
jl.seval("using RationalFunctionApproximation, ComplexRegions, PythonCall")
RFA = jl.RationalFunctionApproximation
CR = jl.ComplexRegions

# The interface here follows RationalFunctionApproximation v0.4, which changed how the
# approximation method is selected and what approximate() records about convergence.
if jl.seval('pkgversion(RationalFunctionApproximation) < v"0.4"'):
    raise ImportError(
        "pyratapprox requires RationalFunctionApproximation v0.4 or later, but v"
        f"{jl.seval('string(pkgversion(RationalFunctionApproximation))')} is installed. "
        "Update it with Pkg.update() in the Julia environment used by juliacall."
        )

__all__ = ['Thiele', 'Bary', 'JuliaApprox', 'ContinuumApprox', 'DiscreteApprox', 'ConvergenceStatus', 'approximate', 'unitcircle', 'unitinterval', 'unitdisk', 'RFA', 'CR', 'TCF', 'AAA', 'PartialFractions']

# Predefined domains for common use cases
unitcircle = cr.Circle(0.0, 1.0)  # Unit circle in the complex plane
unitinterval = cr.Segment(-1.0, 1.0)  # Real interval [-1, 1]
unitdisk = jl.unit_disk  # Unit disk in the complex plane

# Approximation methods. As of RationalFunctionApproximation v0.4.0, Julia selects the
# method by dispatching on an *instance* passed as the last positional argument of
# approximate(); the types below are instantiated by approximate() as needed.
TCF = jl.TCF  # Thiele continued fraction (default)
AAA = jl.AAA  # Adaptive Antoulas-Anderson algorithm (barycentric)
PartialFractions = jl.PartialFractions  # partial fractions with prescribed poles

# Predicate for "is this Julia object a rational function type?"
_is_ratfun_type = jl.seval(
    "T -> (T isa Type) && (T <: RationalFunctionApproximation.AbstractRationalFunction)"
    )

# Method names accepted as strings, mapped to their Julia types.
_method_aliases = {
    "tcf": "Thiele",
    "thiele": "Thiele",
    "aaa": "Barycentric",
    "bary": "Barycentric",
    "barycentric": "Barycentric",
    "partialfractions": "PartialFractions",
    }

class JuliaRatfun:
    """
    Python wrapper for Julia rational functions.
    
    This class wraps Julia's AbstractRationalFunction type, providing a Python
    interface for rational function operations including evaluation, arithmetic,
    and analysis of poles, zeros, and residues.
    
    Attributes:
        julia: The underlying Julia rational function object.
    """
    
    def __init__(self, julia_obj):
        """
        Initialize a JuliaRatfun wrapper.
        
        Args:
            julia_obj: A Julia object of type AbstractRationalFunction.
            
        Raises:
            ValueError: If julia_obj is not a valid AbstractRationalFunction.
        """
        if isinstance(julia_obj, juliacall.AnyValue):  # type: ignore
            if jl.isa(julia_obj, RFA.AbstractRationalFunction):
                self.julia = julia_obj
        else:
            raise ValueError("Invalid argument to constructor")

    def get(self, field):
        """
        Get a property from the underlying Julia object.
        
        Args:
            field: Name of the property to retrieve.
            
        Returns:
            The value of the requested property.
        """
        return jl.getproperty(self.julia, jl.Symbol(field))
    
    def __call__(self, z):
        """
        Evaluate the rational function at point(s) z.
        
        Args:
            z: Scalar or array-like of evaluation points.
            
        Returns:
            np.complex128 or np.ndarray: Function value(s) at z.
        """
        if np.ndim(z) > 0:
            vec_z = juliacall.convert(jl.Vector, z)
            result = jl.map(self.julia, vec_z)
            return np.array(result)
        else:
            return np.complex128(self.julia(z))
        
    def degree(self):
        """
        Get the total degree of the rational function.
        
        Returns:
            int: Maximum of numerator and denominator degrees.
        """
        return jl.degree(self.julia)
    
    def degrees(self):
        """
        Get the numerator and denominator degrees.
        
        Returns:
            tuple: (numerator_degree, denominator_degree).
        """
        return tuple(jl.degrees(self.julia))
    
    def poles(self):
        """
        Compute the poles of the rational function.
        
        Returns:
            np.ndarray: Array of pole locations in the complex plane.
        """
        zp = jl.poles(self.julia)
        return np.array(zp)
    
    def residues(self):
        """
        Compute poles and their residues.
        
        Returns:
            tuple: (poles, residues) as numpy arrays.
        """
        zp, res = jl.residues(self.julia)
        return np.array(zp), np.array(res)
    
    def roots(self):
        """
        Compute the zeros of the rational function.
        
        Returns:
            np.ndarray: Array of zero locations in the complex plane.
        """
        rt = jl.roots(self.julia)
        return np.array(rt)
    
    def isreal(self):
        """
        Check if the rational function has real coefficients.
        
        Returns:
            bool: True if all coefficients are real.
        """
        return jl.isreal(self.julia)
    
    def isempty(self):
        """
        Check if the rational function is empty/undefined.
        
        Returns:
            bool: True if the function is empty.
        """
        return jl.isempty(self.julia)
    
    def __repr__(self):
        """String representation showing the rational function type."""
        return f"Rational function of type {self.degrees()}"
    
    def __add__(self, other):
        """Add two rational functions or a rational function and a scalar."""
        julia_add = getattr(jl, "+")
        if isinstance(other, JuliaRatfun):
            other = other.julia
        t = julia_add(self.julia, other)
        return type(self)(t)

    def __radd__(self, other):
        """Right addition for scalar + rational function."""
        julia_add = getattr(jl, "+")
        t = julia_add(other, self.julia)
        return type(self)(t)

    def __neg__(self):
        """Negate the rational function."""
        julia_neg = getattr(jl, "-")
        t = julia_neg(self.julia)
        return type(self)(t)

    def __sub__(self, other):
        """Subtract two rational functions or a scalar from a rational function."""
        julia_sub = getattr(jl, "-")
        t = julia_sub(self.julia, other)
        return type(self)(t)

    def __rsub__(self, other):
        """Right subtraction for scalar - rational function."""
        julia_sub = getattr(jl, "-")
        t = julia_sub(other, self.julia)
        return type(self)(t)
    
    def __mul__(self, other):
        """Multiply two rational functions or a rational function by a scalar."""
        julia_mul = getattr(jl, "*")
        t = julia_mul(self.julia, other)
        return type(self)(t)

    def __rmul__(self, other):
        """Right multiplication for scalar * rational function."""
        julia_mul = getattr(jl, "*")
        t = julia_mul(other, self.julia)
        return type(self)(t)

    def __truediv__(self, other):
        """Divide a rational function by another or by a scalar."""
        julia_div = getattr(jl, "/")
        t = julia_div(self.julia, other)
        return type(self)(t)

class JuliaRatinterp(JuliaRatfun):
    """
    Python wrapper for Julia rational interpolants.
    
    Extends JuliaRatfun to include interpolation-specific functionality such as
    accessing interpolation nodes and values. This is the base class for specific
    interpolation methods like Thiele and Barycentric.
    
    Attributes:
        julia: The underlying Julia rational interpolant object.
    """
    
    def __init__(self, julia_obj):
        """
        Initialize a JuliaRatinterp wrapper.
        
        Args:
            julia_obj: A Julia object of type AbstractRationalInterpolant.
            
        Raises:
            ValueError: If julia_obj is not a valid AbstractRationalInterpolant.
        """
        if isinstance(julia_obj, juliacall.AnyValue):  # type: ignore
            if jl.isa(julia_obj, RFA.AbstractRationalInterpolant):
                self.julia = julia_obj
        else:
            raise ValueError("Invalid argument to constructor")

    def __repr__(self):
        """String representation showing the interpolant type."""
        return f"Rational interpolant of type {self.degrees()}"
    
    def nodes(self):
        """
        Get the interpolation nodes.
        
        Returns:
            np.ndarray: Array of interpolation node locations.
        """
        nds = jl.nodes(self.julia)
        return np.array(nds)
    
    def values(self):
        """
        Get the interpolation values at the nodes.
        
        Returns:
            np.ndarray: Array of function values at interpolation nodes.
        """
        vals = jl.values(self.julia)
        return np.array(vals)
    
    def length(self):
        """
        Get the number of interpolation points.
        
        Returns:
            int: Number of nodes in the interpolant.
        """
        return jl.length(self.julia)

class Thiele(JuliaRatinterp):
    """
    Thiele continued fraction interpolant.
    
    Implements rational interpolation using Thiele's continued fraction method,
    which provides a stable representation for rational interpolants.
    
    Attributes:
        julia: The underlying Julia Thiele object.
        weights: Numpy array of interpolation weights.
    """
    
    def __init__(self, nodes, values=None, weights=None):
        """
        Create a Thiele continued fraction interpolant.
        
        Args:
            nodes: Array-like of interpolation nodes, or a Julia Thiele object.
            values: Array-like of function values at nodes (required if nodes is array).
            weights: Optional array-like of interpolation weights.
            
        Raises:
            ValueError: If arguments are invalid or incompatible.
            
        Examples:
            >>> nodes = [0.0, 1.0, 2.0]
            >>> values = [1.0, 2.0, 4.0]
            >>> f = Thiele(nodes, values)
        """
        if isinstance(nodes, juliacall.AnyValue): # type: ignore
            if jl.isa(nodes, RFA.Thiele):
                self.julia = nodes
            else:
                raise ValueError("Invalid argument to Thiele constructor")
        else:
            vn = juliacall.convert(jl.Vector, nodes)
            vv = juliacall.convert(jl.Vector, values)
            if weights is None:
                self.julia = RFA.Thiele(vn, vv)
            else:
                vw = juliacall.convert(jl.Vector, weights)
                self.julia = RFA.Thiele(vn, vv, vw)

        self.weights = np.array(JuliaRatfun.get(self, "weights"))

    def __repr__(self):
        """String representation showing the Thiele interpolant type."""
        return f"Thiele continued fraction of type {self.degrees()}"
    
class Bary(JuliaRatinterp):
    """
    Barycentric rational interpolant.
    
    Implements rational interpolation using the barycentric formula, which
    provides numerical stability and efficient evaluation.
    
    Attributes:
        julia: The underlying Julia Barycentric object.
        weights: Numpy array of barycentric weights.
    """
    
    def __init__(self, nodes, values=None, weights=None):
        """
        Create a barycentric rational interpolant.
        
        Args:
            nodes: Array-like of interpolation nodes, or a Julia Barycentric object.
            values: Array-like of function values at nodes (required if nodes is array).
            weights: Array-like of barycentric weights (required if nodes is array).
            
        Raises:
            ValueError: If arguments are invalid or weights are missing.
            
        Examples:
            >>> nodes = [0.0, 1.0, 2.0]
            >>> values = [1.0, 2.0, 4.0]
            >>> weights = [1.0, -2.0, 1.0]
            >>> f = Bary(nodes, values, weights)
        """
        if isinstance(nodes, juliacall.AnyValue): # type: ignore
            if jl.isa(nodes, RFA.Barycentric):
                self.julia = nodes
            else:
                raise ValueError("Invalid argument to Bary constructor")
        elif weights is None:
            raise ValueError("Weights must be provided for Barycentric constructor")
        else:
            vn = juliacall.convert(jl.Vector, nodes)
            vv = juliacall.convert(jl.Vector, values)
            vw = juliacall.convert(jl.Vector, weights)
            self.julia = RFA.Barycentric(vn, vv, vw)

        self.weights = np.array(JuliaRatfun.get(self, "weights"))
    
    def __repr__(self):
        """String representation showing the barycentric interpolant type."""
        return f"Barycentric rational function of type {self.degrees()}"

def wrap_jl_ratfun(julia_obj):
    """
    Wrap a Julia rational function object in the appropriate Python class.
    
    Args:
        julia_obj: A Julia rational function or interpolant object.
        
    Returns:
        Thiele, Bary, JuliaRatinterp, or JuliaRatfun: Appropriate wrapper instance.
        
    Raises:
        ValueError: If julia_obj is not a recognized rational function type.
    """
    if jl.isa(julia_obj, RFA.Thiele):
        return Thiele(julia_obj)
    elif jl.isa(julia_obj, RFA.Barycentric):
        return Bary(julia_obj)
    elif jl.isa(julia_obj, RFA.AbstractRationalInterpolant):
        return JuliaRatinterp(julia_obj)
    elif jl.isa(julia_obj, RFA.AbstractRationalFunction):
        return JuliaRatfun(julia_obj)
    else:
        raise ValueError("Unknown rational function type")

class ConvergenceStatus:
    """
    Explanation of why an approximation iteration stopped.

    Wraps Julia's ConvergenceStatus, which is recorded by approximate() as of
    RationalFunctionApproximation v0.4.0.

    Attributes:
        julia: The underlying Julia ConvergenceStatus object.
        reason: str, the cause of termination. One of "converged", "stagnated",
            "max_degree", "node_failure", "nan_weight", "refinement",
            "exhausted", or "rewound".
        best: int, index into the history of the interpolant that was returned.
        iterations: int, number of iterations completed.
        error: float, estimated error of the returned interpolant.
    """

    def __init__(self, julia_obj):
        """
        Initialize a ConvergenceStatus wrapper.

        Args:
            julia_obj: A Julia object of type ConvergenceStatus.

        Raises:
            ValueError: If julia_obj is not a valid ConvergenceStatus.
        """
        if not (isinstance(julia_obj, juliacall.AnyValue)  # type: ignore
                and jl.isa(julia_obj, RFA.ConvergenceStatus)):
            raise ValueError("Invalid argument to constructor")
        self.julia = julia_obj
        self.reason = str(jl.getproperty(julia_obj, jl.Symbol("reason")))
        self.best = int(jl.getproperty(julia_obj, jl.Symbol("best")))
        self.iterations = int(jl.getproperty(julia_obj, jl.Symbol("iterations")))
        self.error = float(jl.getproperty(julia_obj, jl.Symbol("error")))

    def isconverged(self):
        """
        Check whether the iteration reached the requested tolerance.

        Returns:
            bool: True if the reason for stopping was convergence.
        """
        return self.reason == "converged"

    def __repr__(self):
        """String representation showing why the iteration stopped."""
        return (f"Stopped by {self.reason} after {self.iterations} iterations "
                f"with estimated error {self.error:.4g}")

def wrap_status(julia_obj):
    """
    Wrap a Julia ConvergenceStatus, if one was recorded.

    Args:
        julia_obj: A Julia ConvergenceStatus object, or None.

    Returns:
        ConvergenceStatus or None: None if no status was recorded.
    """
    return None if julia_obj is None else ConvergenceStatus(julia_obj)

class JuliaApprox:
    """
    Base class for rational function approximations.
    
    Wraps Julia's AbstractApproximation type, providing methods for evaluating
    approximations, analyzing their properties, and managing the approximation
    history.
    
    Attributes:
        julia: The underlying Julia approximation object.
    """
    
    def __init__(self, julia_obj):
        """
        Initialize a JuliaApprox wrapper.
        
        Args:
            julia_obj: A Julia object of type AbstractApproximation.
            
        Raises:
            ValueError: If julia_obj is not a valid AbstractApproximation.
        """
        if isinstance(julia_obj, juliacall.AnyValue):  # type: ignore
            if jl.isa(julia_obj, RFA.AbstractApproximation):
                self.julia = julia_obj
        else:
            raise ValueError("Invalid argument to constructor")

    def get(self, field):
        """
        Get a property from the underlying Julia object.
        
        Args:
            field: Name of the property to retrieve.
            
        Returns:
            The value of the requested property.
        """
        return jl.getproperty(self.julia, jl.Symbol(field))
    
    def __call__(self, z):
        """
        Evaluate the approximation at point(s) z.
        
        Args:
            z: Scalar or array-like of evaluation points.
            
        Returns:
            np.complex128 or np.ndarray: Approximation value(s) at z.
        """
        if np.ndim(z) > 0:
            vec_z = juliacall.convert(jl.Vector, z)
            result = self.julia(vec_z)
            return np.array(result)
        else:
            return np.complex128(self.julia(z))

    def degree(self):
        """
        Get the total degree of the approximation.
        
        Returns:
            int: Maximum of numerator and denominator degrees.
        """
        return jl.degree(self.julia)
    
    def degrees(self):
        """
        Get the numerator and denominator degrees.
        
        Returns:
            tuple: (numerator_degree, denominator_degree).
        """
        return tuple(jl.degrees(self.julia))
    
    def poles(self):
        """
        Compute the poles of the approximation.
        
        Returns:
            np.ndarray: Array of pole locations in the complex plane.
        """
        zp = jl.poles(self.julia)
        return np.array(zp)
    
    def residues(self):
        """
        Compute poles and their residues.
        
        Returns:
            tuple: (poles, residues) as numpy arrays.
        """
        zp, res = jl.residues(self.julia)
        return np.array(zp), np.array(res)
    
    def roots(self):
        """
        Compute the zeros of the approximation.
        
        Returns:
            np.ndarray: Array of zero locations in the complex plane.
        """
        rt = jl.roots(self.julia)
        return np.array(rt)
    
    def isempty(self):
        """
        Check if the approximation is empty/undefined.
        
        Returns:
            bool: True if the approximation is empty.
        """
        return jl.isempty(self.julia)
    
    def check(self):
        """
        Check the quality of the approximation.
        
        Returns:
            Approximation quality metrics from Julia.
        """
        return jl.check(self.julia)
    
    def isconverged(self):
        """
        Check whether the iteration reached the requested tolerance.
        
        Returns:
            bool: True if the approximation stopped by converging, and False if it
                stagnated, exhausted its degree budget, failed, or recorded no status.
        """
        return bool(jl.isconverged(self.julia))
    
    def rewind(self, n=1):
        """
        Rewind the approximation history by n steps.
        
        Args:
            n: Number of steps to rewind (default: 1).
        """
        jl.rewind(self.julia, n)
    
    def __repr__(self):
        """String representation showing the approximation type."""
        return f"Rational function of type {self.degrees()}"
 
def wrap_approx_domain(domain):
    """
    Wrap a Julia domain object in the appropriate Python class.
    
    Args:
        domain: A Julia region, curve, or path object.
        
    Returns:
        Wrapped Python object from cxregions package.
        
    Raises:
        ValueError: If domain is not a recognized type.
    """
    if jl.isa(domain, CR.AbstractRegion):
        return cr.wrap_jl_region(domain)
    elif jl.isa(domain, CR.AbstractCurve) or jl.isa(domain, CR.AbstractPath):
        return cr.wrap_jl_curve(domain)
    else:
        raise ValueError("Unknown domain type")
    
class ContinuumApprox(JuliaApprox):
    """
    Rational approximation on a continuum domain.
    
    Represents a rational function approximation computed on a continuous domain
    such as a curve, path, or region in the complex plane. The approximation is
    constructed adaptively using methods like AAA or TCF.
    
    Attributes:
        julia: The underlying Julia ContinuumApproximation object.
        original: The original function being approximated.
        domain: The continuum domain (curve, path, or region).
        fun: The rational function approximation (Thiele, Bary, etc.).
        allowed: Allowed pole locations.
        path: Integration path used in approximation.
        history: History of the approximation process.
        status: ConvergenceStatus saying why the iteration stopped, or None if
            none was recorded.
    """
    
    def __init__(self, julia_obj):
        """
        Initialize a ContinuumApprox wrapper.
        
        Args:
            julia_obj: A Julia object of type ContinuumApproximation.
            
        Raises:
            ValueError: If julia_obj is not a valid ContinuumApproximation.
        """
        if isinstance(julia_obj, juliacall.AnyValue):  # type: ignore
            if jl.isa(julia_obj, RFA.ContinuumApproximation):
                self.julia = julia_obj
        else:
            raise ValueError("Invalid argument to constructor")
        
        self.original = JuliaApprox.get(self, "original")
        self.domain = wrap_approx_domain(JuliaApprox.get(self, "domain"))
        self.fun = wrap_jl_ratfun(JuliaApprox.get(self, "fun"))
        self.allowed = JuliaApprox.get(self, "allowed")
        self.path = JuliaApprox.get(self, "path")
        self.history = JuliaApprox.get(self, "history")
        self.status = wrap_status(JuliaApprox.get(self, "status"))

    def __repr__(self):
        """String representation showing approximation type and domain."""
        return f"Rational approximation of type {self.degrees()} on {self.domain}"
    
    def getfunction(self):
        """
        Get the underlying rational function.
        
        Returns:
            Thiele, Bary, or JuliaRatfun: The rational function approximation.
        """
        return self.fun
    
    def testpoints(self):
        """
        Get the test points used to validate the approximation.
        
        Returns:
            np.ndarray: Array of test point locations.
        """
        pts = jl.test_points(self.julia)
        return np.array(pts)
    
    def nodes(self):
        """
        Get the interpolation nodes of the approximation.
        
        Returns:
            np.ndarray: Array of node locations.
        """
        nds = jl.nodes(self.julia)
        return np.array(nds)
    
    def values(self):
        """
        Get the function values at the interpolation nodes.
        
        Returns:
            np.ndarray: Array of function values.
        """
        vals = jl.values(self.julia)
        return np.array(vals)
    
    def isapprox(self, other):
        """
        Check if this approximation is close to another function.
        
        Args:
            other: Another JuliaApprox or callable function.
            
        Returns:
            bool: True if approximations are close on test points.
        """
        if isinstance(other, JuliaApprox):
            return jl.isapprox(self.julia, other.julia)
        else:
            x = self.testpoints()
            return np.all([np.isclose(self(xk), other(xk)) for xk in x])
    
class DiscreteApprox(JuliaApprox):
    """
    Rational approximation on a discrete point set.
    
    Represents a rational function approximation computed on a discrete set of
    points in the complex plane. The approximation is constructed adaptively
    using methods like AAA or TCF.
    
    Attributes:
        julia: The underlying Julia DiscreteApproximation object.
        data: The discrete data points and values.
        domain: The discrete point set (numpy array).
        fun: The rational function approximation (Thiele, Bary, etc.).
        test_index: Indices of points used for testing.
        allowed: Allowed pole locations.
        history: History of the approximation process.
        status: ConvergenceStatus saying why the iteration stopped, or None if
            none was recorded.
    """
    
    def __init__(self, julia_obj):
        """
        Initialize a DiscreteApprox wrapper.
        
        Args:
            julia_obj: A Julia object of type DiscreteApproximation.
            
        Raises:
            ValueError: If julia_obj is not a valid DiscreteApproximation.
        """
        if isinstance(julia_obj, juliacall.AnyValue):  # type: ignore
            if jl.isa(julia_obj, RFA.DiscreteApproximation):
                self.julia = julia_obj
        else:
            raise ValueError("Invalid argument to constructor")
        
        self.data = np.array(JuliaApprox.get(self, "data"))
        self.domain = np.array(JuliaApprox.get(self, "domain"))
        self.fun = wrap_jl_ratfun(JuliaApprox.get(self, "fun"))
        self.test_index = np.array(JuliaApprox.get(self, "test_index"))
        self.allowed = JuliaApprox.get(self, "allowed")
        self.history = JuliaApprox.get(self, "history")
        self.status = wrap_status(JuliaApprox.get(self, "status"))

    def __repr__(self):
        """String representation showing approximation type and discrete domain."""
        return f"Rational approximation of type {self.degrees()} on a discrete domain"
    
    def getfunction(self):
        """
        Get the underlying rational function.
        
        Returns:
            Thiele, Bary, or JuliaRatfun: The rational function approximation.
        """
        return self.fun
    
    def testpoints(self):
        """
        Get the test points used to validate the approximation.
        
        Returns:
            np.ndarray: Array of test point locations from the domain.
        """
        pts = np.array(self.domain)
        return pts[self.test_index]
 
    def nodes(self):
        """
        Get the interpolation nodes of the approximation.
        
        Returns:
            np.ndarray: Array of node locations.
        """
        nds = jl.nodes(self.julia)
        return np.array(nds)
    
    def values(self):
        """
        Get the function values at the interpolation nodes.
        
        Returns:
            np.ndarray: Array of function values.
        """
        vals = jl.values(self.julia)
        return np.array(vals)

    def isapprox(self, other):
        """
        Check if this approximation is close to another function.
        
        Args:
            other: Another JuliaApprox or callable function.
            
        Returns:
            bool: True if approximations are close on all domain points.
        """
        if isinstance(other, JuliaApprox):
            return jl.isapprox(self.julia, other.julia)
        else:
            x = self.domain
            return np.all([np.isclose(self(xk), other(xk)) for xk in x])

def _method_instance(method):
    """
    Convert a method selector into a Julia rational function instance.

    As of RationalFunctionApproximation v0.4.0, Julia selects the type of interpolant
    by dispatching on an instance passed as the last positional argument of
    approximate(). Accepted selectors are a Julia type (e.g. TCF, AAA,
    PartialFractions) or instance of one, the Python classes Thiele and Bary or an
    instance of one, a name such as "thiele", or None for the Julia default.

    Args:
        method: A method selector, or None.

    Returns:
        A Julia rational function instance, or None if method is None.

    Raises:
        ValueError: If method is not a recognized selector.
    """
    if method is None:
        return None
    if isinstance(method, str):
        name = _method_aliases.get(method.lower().replace("_", ""))
        if name is None:
            raise ValueError(f"Unknown approximation method '{method}'")
        return jl.getproperty(RFA, jl.Symbol(name))()
    if method is Thiele:
        return RFA.Thiele()
    if method is Bary:
        return RFA.Barycentric()
    if isinstance(method, JuliaRatfun):
        return method.julia
    if isinstance(method, juliacall.AnyValue):  # type: ignore
        if _is_ratfun_type(method):
            return method()    # empty instance, used only as a selector
        if jl.isa(method, RFA.AbstractRationalFunction):
            return method
    raise ValueError(f"Invalid approximation method: {method}")

def _is_method(arg):
    """
    Check whether an argument can be interpreted as a method selector.

    Args:
        arg: Any value.

    Returns:
        bool: True if arg names or is a rational function type.
    """
    if arg is None:
        return False
    try:
        _method_instance(arg)
    except ValueError:
        return False
    return True

def approximate(fun, domain=unitinterval, zeta=None, method=None, **kwargs):
    """
    Compute a rational function approximation.
    
    This is the main entry point for creating rational approximations. It
    automatically selects between continuum and discrete approximation based
    on the domain type, and uses adaptive algorithms (TCF or AAA) to construct
    high-quality approximations.
    
    Args:
        fun: Callable function to approximate, or array of function values.
        domain: Approximation domain (default: the interval [-1, 1]) - can be:
            - A continuum domain (Circle, Segment, Region, Curve, Path)
            - A discrete array of points
        zeta: Optional array of prescribed poles, which switches the default method
            to partial fractions.
        method: Type of rational interpolant, given either positionally (in place of
            zeta, as in Julia) or by keyword. It may be a Julia type or instance
            (TCF, AAA, PartialFractions), the Python class Thiele or Bary or an
            instance of one, or a name such as "thiele". The default is TCF, or
            PartialFractions when zeta is given.
        **kwargs: Additional keyword arguments passed to Julia's approximate():
            - tol: Relative tolerance for stopping
            - max_degree: Maximum degree of the approximation (on a discrete domain,
              TCF takes max_iter instead)
            - allowed: True to accept all poles (default), "strict" to require poles
              off the curve or outside the region, or a predicate on pole locations
            - refinement: Number of test points between adjacent nodes (continuum only)
            - stagnation: Number of iterations used to detect stagnation
            - float_type: Floating point type used in the computation
    
    Returns:
        ContinuumApprox or DiscreteApprox: The computed approximation.
        
    Raises:
        ValueError: If the method or the approximation type is not recognized.
        
    Examples:
        >>> # Approximate on the unit interval, using the default method
        >>> f = approximate(np.sin)
        >>>
        >>> # Choose a method positionally, as in Julia
        >>> f = approximate(np.sin, unitinterval, AAA)
        >>>
        >>> # Approximate on discrete points
        >>> x = np.linspace(-1, 1, 100)
        >>> f = approximate(np.exp, x, method=TCF)
        >>>
        >>> # Keep the poles away from the domain
        >>> f = approximate(lambda x: np.abs(x), unitinterval, allowed="strict")
        >>>
        >>> # Evaluate the approximation
        >>> y = f(0.5)
        >>>
        >>> # Get poles and residues
        >>> poles = f.poles()
        >>> poles, residues = f.residues()
        >>>
        >>> # Find out why the iteration stopped
        >>> f.isconverged(), f.status.error
    """
    # Julia takes the method as the last positional argument, so it may show up here
    # in the zeta slot.
    if method is None and _is_method(zeta):
        zeta, method = None, zeta

    if not callable(fun):
        fun = np.array(fun).flatten()

    if isinstance(domain, cr.JuliaRegion) or isinstance(domain, cr.JuliaCurve): # type: ignore
        domain = domain.julia
    else:
        domain = np.array(domain).flatten()

    # A policy for allowed poles is named by a Julia symbol, e.g. allowed="strict".
    if isinstance(kwargs.get("allowed"), str):
        kwargs["allowed"] = jl.Symbol(kwargs["allowed"].lstrip(":"))

    args = [fun, domain]
    if zeta is not None:
        args.append(np.array(zeta).flatten())
    method = _method_instance(method)
    if method is not None:
        args.append(method)

    julia_approx = RFA.approximate(*args, **kwargs)
    
    if jl.isa(julia_approx, RFA.ContinuumApproximation):
        return ContinuumApprox(julia_approx)
    elif jl.isa(julia_approx, RFA.DiscreteApproximation):
        return DiscreteApprox(julia_approx)
    else:
        raise ValueError("Unknown approximation type returned")
