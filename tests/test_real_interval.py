import pytest
import numpy as np
from pyrat import *
import cxregions as cr
import juliacall

def test_continuum_build():
    assert isinstance(unitinterval, cr.Segment)
    for method in [TCF, AAA]:
        f = approximate(np.sin, unitinterval, method=method)
        assert isinstance(f, ContinuumApprox)
        assert callable(f)
        assert isinstance(f.original, juliacall.AnyValue) # type: ignore
        assert isinstance(f.domain, juliacall.AnyValue) # type: ignore
        assert isinstance(f.fun, juliacall.AnyValue) # type: ignore
        assert isinstance(f.allowed, juliacall.AnyValue) # type: ignore
        assert isinstance(f.path, juliacall.AnyValue) # type: ignore
        assert isinstance(f.history, juliacall.AnyValue) # type: ignore

def test_continuum_accuracy():
    pts = np.linspace(-1, 1, 2000)
    for fun in [
        np.exp,
        lambda x: np.cos(x) + 1j * np.sin(x),
        lambda x: 1j*x + np.exp(-1 / (x + 1e-20)**2),
        lambda x: 1 / (1.1 - x),
        lambda x: np.log(1.1 - x),
        lambda x: np.sin(1 / (21/20 - x)),
        lambda x: np.abs(x + 1/2 + 1j/60),
        lambda x: np.exp(-100 * x**2),
        lambda x: np.sin(40*x) * np.exp(-8*x**2),
        lambda x: 1j + np.exp(-10 / (6/5 - x)),
        lambda x: 10j*x + np.tanh(100*(x - 1/5)),
        lambda x: x + np.tanh(100*x),
    ]:
        for method in [TCF, AAA]:
            f = approximate(fun, unitinterval, method=method)
            y = [fun(x) for x in pts]
            u = [f(x) for x in pts]
            assert y == pytest.approx(u, rel=1e-10, abs=1e-10)
            assert f.isapprox(fun)

def test_continuum_nodes_values():
    f = approximate(np.sin, unitinterval, method=AAA)
    nds = f.nodes()
    vals = f.values()
    for n, v in zip(nds, vals):
        assert v == pytest.approx(np.sin(n))

def test_discrete_build():
    x = np.linspace(-1, 1, 20)
    for method in [TCF, AAA]:
        f = approximate(np.sin, x, method=method)
        assert callable(f)
        assert hasattr(f, 'data')
        assert hasattr(f, 'domain')
        assert hasattr(f, 'fun')
        assert hasattr(f, 'allowed')
        assert hasattr(f, 'test_index')
        assert hasattr(f, 'history')


def test_discrete_accuracy():
    pts = np.linspace(-1, 1, 4000)
    for fun in [
        np.exp,
        lambda x: np.cos(x) + 1j * np.sin(x),
        lambda x: 1j*x + np.exp(-1 / (x + 1e-20)**2),
        lambda x: 1 / (1.1 - x),
        lambda x: np.log(1.1 - x),
        lambda x: np.sin(1 / (21/20 - x)),
        lambda x: np.abs(x + 1/2 + 1j/60),
        lambda x: np.exp(-100 * x**2),
        lambda x: np.sin(40*x) * np.exp(-8*x**2),
        lambda x: 1j + np.exp(-10 / (6/5 - x)),
        lambda x: 10j*x + np.tanh(100*(x - 1/5)),
        lambda x: x + np.tanh(100*x),
    ]:
        for method in [TCF, AAA]:
            f = approximate(fun, pts, method=method)
            y = [fun(x) for x in pts]
            u = [f(x) for x in pts]
            assert y == pytest.approx(u, rel=1e-10, abs=1e-10)
            assert f.isapprox(fun)

def test_discrete_nodes_values():
    f = approximate(np.sin, np.linspace(-1, 1, 20), method=AAA)
    nds = f.nodes()
    vals = f.values()
    for n, v in zip(nds, vals):
        assert v == pytest.approx(np.sin(n))