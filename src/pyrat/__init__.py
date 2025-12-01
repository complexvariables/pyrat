import juliacall
import numpy as np
import cxregions as cr
jl = juliacall.newmodule("pyrat")
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

__all__ = ['Thiele', 'Bary', 'ContinuumApprox', 'DiscreteApprox', 'approximate', 'unitcircle', 'unitinterval', 'unitdisk', 'RFA', 'CR', 'TCF', 'AAA']

unitcircle = cr.Circle(0.0, 1.0)
unitinterval = cr.Segment(-1.0, 1.0)
unitdisk = jl.unit_disk
TCF = jl.TCF
AAA = jl.AAA

class JuliaRatfun:
    def __init__(self, julia_obj):
        if isinstance(julia_obj, juliacall.AnyValue):  # type: ignore
            if jl.isa(julia_obj, RFA.AbstractRationalFunction):
                self.julia = julia_obj
        else:
            raise ValueError("Invalid argument to constructor")

    def get(self, field):
        return jl.getproperty(self.julia, jl.Symbol(field))
    
    def __call__(self, z):
        if np.ndim(z) > 0:
            vec_z = juliacall.convert(jl.Vector, z)
            result = self.julia(vec_z)
            return np.array(result)
        else:
            return np.complex128(self.julia(z))
        
    def degree(self):
        return jl.degree(self.julia)
    
    def degrees(self):
        return tuple(jl.degrees(self.julia))
    
    def poles(self):
        zp = jl.poles(self.julia)
        return np.array(zp)
    
    def residues(self):
        zp, res = jl.residues(self.julia)
        return np.array(zp), np.array(res)
    
    def roots(self):
        rt = jl.roots(self.julia)
        return np.array(rt)
    
    def isreal(self):
        return jl.isreal(self.julia)
    
    def isempty(self):
        return jl.isempty(self.julia)
    
    def __repr__(self):
        return f"Rational function of type {self.degrees()}"
    
    def __add__(self, other):
        julia_add = getattr(jl, "+")
        t = julia_add(self.julia, other)
        return type(self)(t)

    def __radd__(self, other):
        julia_add = getattr(jl, "+")
        t = julia_add(other, self.julia)
        return type(self)(t)

    def __neg__(self):
        julia_neg = getattr(jl, "-")
        t = julia_neg(self.julia)
        return type(self)(t)

    def __sub__(self, other):
        julia_sub = getattr(jl, "-")
        t = julia_sub(self.julia, other)
        return type(self)(t)

    def __rsub__(self, other):
        julia_sub = getattr(jl, "-")
        t = julia_sub(other, self.julia)
        return type(self)(t)
    
    def __mul__(self, other):
        julia_mul = getattr(jl, "*")
        t = julia_mul(self.julia, other)
        return type(self)(t)

    def __rmul__(self, other):
        julia_mul = getattr(jl, "*")
        t = julia_mul(other, self.julia)
        return type(self)(t)

    def __truediv__(self, other):
        julia_div = getattr(jl, "/")
        t = julia_div(self.julia, other)
        return type(self)(t)

class JuliaRatinterp(JuliaRatfun):
    def __init__(self, julia_obj):
        if isinstance(julia_obj, juliacall.AnyValue):  # type: ignore
            if jl.isa(julia_obj, RFA.AbstractRationalInterpolant):
                self.julia = julia_obj
        else:
            raise ValueError("Invalid argument to constructor")

    def __repr__(self):
        return f"Rational interpolant of type {self.degrees()}"
    
    def nodes(self):
        nds = jl.nodes(self.julia)
        return np.array(nds)
    
    def values(self):
        vals = jl.values(self.julia)
        return np.array(vals)
    
    def length(self):
        return jl.length(self.julia)

class Thiele(JuliaRatfun):
    def __init__(self, nodes, values, weights=None):
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

        self.nodes = JuliaRatfun.get(self, "nodes")
        self.values = JuliaRatfun.get(self, "values")
        self.weights = JuliaRatfun.get(self, "weights")

    def __repr__(self):
        return f"Thiele continued fraction of type {self.degrees()}"
    
class Bary(JuliaRatfun):
    def __init__(self, nodes, values, weights=None):
        if isinstance(nodes, juliacall.AnyValue): # type: ignore
            if jl.isa(nodes, RFA.Bary):
                self.julia = nodes
            else:
                raise ValueError("Invalid argument to Bary constructor")
        elif weights is not None:
            self.julia = RFA.Bary(nodes, values, weights)
        else:
            self.julia = RFA.Bary(nodes, values)

        self.nodes = JuliaRatfun.get(self, "nodes")
        self.values = JuliaRatfun.get(self, "values")
        self.weights = JuliaRatfun.get(self, "weights")
    
    def __repr__(self):
        return f"Barycentric rational function of type {self.degrees()}"

class JuliaApprox:
    def __init__(self, julia_obj):
        if isinstance(julia_obj, juliacall.AnyValue):  # type: ignore
            if jl.isa(julia_obj, RFA.AbstractApproximation):
                self.julia = julia_obj
        else:
            raise ValueError("Invalid argument to constructor")

    def get(self, field):
        return jl.getproperty(self.julia, jl.Symbol(field))
    
    def __call__(self, z):
        if np.ndim(z) > 0:
            return np.array([self.julia(zi) for zi in z])
        else:
            f = self.julia(z)
            return f
        
    def degree(self):
        return jl.degree(self.julia)
    
    def degrees(self):
        return tuple(jl.degrees(self.julia))
    
    def poles(self):
        zp = jl.poles(self.julia)
        return np.array(zp)
    
    def residues(self):
        zp, res = jl.residues(self.julia)
        return np.array(zp), np.array(res)
    
    def roots(self):
        rt = jl.roots(self.julia)
        return np.array(rt)
    
    def isapprox(self, other):
        return jl.isapprox(self.julia, other.julia)
    
    def isempty(self):
        return jl.isempty(self.julia)
    
    def check(self):
        return jl.check(self.julia)
    
    def rewind(self, n=1):
        jl.rewind(self.julia, n)
    
    def __repr__(self):
        return f"Rational function of type {self.degrees()}"
 
class ContinuumApprox(JuliaApprox):
    def __init__(self, julia_obj):
        if isinstance(julia_obj, juliacall.AnyValue):  # type: ignore
            if jl.isa(julia_obj, RFA.ContinuumApproximation):
                self.julia = julia_obj
        else:
            raise ValueError("Invalid argument to constructor")
        
        self.original = JuliaApprox.get(self, "original")
        self.domain = JuliaApprox.get(self, "domain")
        self.fun = JuliaApprox.get(self, "fun")
        self.allowed = JuliaApprox.get(self, "allowed")
        self.path = JuliaApprox.get(self, "path")
        self.history = JuliaApprox.get(self, "history")

    def __repr__(self):
        return f"Rational approximation of type {self.degrees()} on {self.domain}"
    
    def getfunction(self):
        f = jl.get_function(self.julia)
        return JuliaRatfun(f)
    
    def testpoints(self):
        pts = jl.test_points(self.julia)
        return np.array(pts)
    
class DiscreteApprox(JuliaApprox):
    def __init__(self, julia_obj):
        if isinstance(julia_obj, juliacall.AnyValue):  # type: ignore
            if jl.isa(julia_obj, RFA.DiscreteApproximation):
                self.julia = julia_obj
        else:
            raise ValueError("Invalid argument to constructor")
        
        self.data = JuliaApprox.get(self, "data")
        self.domain = JuliaApprox.get(self, "domain")
        self.fun = JuliaApprox.get(self, "fun")
        self.test_index = JuliaApprox.get(self, "test_index")
        self.allowed = JuliaApprox.get(self, "allowed")
        self.history = JuliaApprox.get(self, "history")

    def __repr__(self):
        return f"Rational approximation of type {self.degrees()} on a discrete domain"
    
    def getfunction(self):
        f = jl.get_function(self.julia)
        return JuliaRatfun(f)
 
def approximate(fun, domain, zeta=None, **kwargs):
    if not callable(fun):
        fun = np.array(fun).flatten()

    if isinstance(domain, cr.JuliaRegion) or isinstance(domain, cr.JuliaCurve): # type: ignore
        domain = domain.julia
    else:
        domain = np.array(domain).flatten()

    if zeta is None:
        julia_approx = RFA.approximate(fun, domain, **kwargs)
    else:
        julia_approx = RFA.approximate(fun, domain, zeta, **kwargs)
    
    if jl.isa(julia_approx, RFA.ContinuumApproximation):
        return ContinuumApprox(julia_approx)
    elif jl.isa(julia_approx, RFA.DiscreteApproximation):
        return DiscreteApprox(julia_approx)