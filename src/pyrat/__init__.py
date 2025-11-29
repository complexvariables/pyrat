import juliacall
import numpy as np
jl = juliacall.newmodule("PyRat")
# jl.seval('import Pkg; Pkg.add("RationalFunctionApproximation")')
# jl.seval('import Pkg; Pkg.add("ComplexRegions")')
# jl.seval('Pkg.compat("RationalFunctionApproximation", "0.3.0")')

jl.seval("using RationalFunctionApproximation, ComplexRegions, PythonCall")

__all__ = ['Segment', 'TCF', 'Approximation']

class Segment:
    def __init__(self, a, b):
        self.julia = jl.ComplexRegions.Segment(a, b)
        self.first = a
        self.last = b

    def ends(self):
        return (self.first, self.last)
    
    def __repr__(self):
        return f"Segment from {self.first} to {self.last}"

class TCF:
    def __init__(self, nodes, values, weights):
        self.nodes = nodes
        self.values = values
        self.weights = weights
        self.julia = jl.Thiele(nodes, values, weights)
        # if callable(func):
        #     self.bary = jl.aaa(func)
        #     self.original = func
    def __call__(self, z):
        return self.julia(z)
    def poles(self):
        return np.array(jl.poles(self.julia))
    def residues(self):
        zp, res = jl.residues(self.julia)
        return np.array(zp), np.array(res)
    def roots(self):
        return np.array(jl.roots(self.julia))
    def degrees(self):
        return tuple(jl.degrees(self.julia))
    def __repr__(self):
        return f"Thiele continued fraction of type {self.degrees()}"

class Approximation:
    def __init__(self, func, domain=Segment(-1, 1)):
        if callable(func):
            self.julia = jl.approximate(func, domain.julia, method=jl.TCF)
            self.original = func
        self.domain = domain
    def __call__(self, z):
        return self.julia(z)
    def nodes(self):
        return np.array(jl.nodes(self.julia))
    def values(self):
        return np.array(jl.values(self.julia))
    def poles(self):
        return np.array(jl.poles(self.julia))
    def residues(self):
        zp, res = jl.residues(self.julia)
        return np.array(zp), np.array(res)
    def roots(self):
        return np.array(jl.roots(self.julia))
    def __repr__(self):
        return f"Rational approximation on {self.domain}"

class DiscreteApproximation:
    def __init__(self, y, domain):
        self.julia = jl.approximate(y, domain, method=jl.TCF)
        self.values = y
        self.domain = domain
    def __call__(self, z):
        return self.julia(z)
    def nodes(self):
        return np.array(jl.nodes(self.julia))
    def poles(self):
        return np.array(jl.poles(self.julia))
    def residues(self):
        zp, res = jl.residues(self.julia)
        return np.array(zp), np.array(res)
    def roots(self):
        return np.array(jl.roots(self.julia))
    def __repr__(self):
        return f"Rational approximation on {self.domain}"
