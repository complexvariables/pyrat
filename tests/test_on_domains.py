import pytest
import numpy as np
from pyrat import *
import cxregions as cr
import juliacall

def test_continuum_segment():
    pts = np.linspace(-1 -1j, 2 + 0.5j, 2000)
    for fun in [
        np.exp,
        lambda x: np.cos(x) + 1j * np.sin(x),
        lambda x: 1 / (1.1 - x),
        lambda x: np.abs(x + 1/2 + 1j/60),
        lambda x: 10j*x + np.tanh((x - 1/5)),
    ]:
        for method in [TCF, AAA]:
            f = approximate(fun, cr.Segment(-1 -1j, 2 + 0.5j), method=method)
            y = [fun(x) for x in pts]
            u = f(pts)
            assert y == pytest.approx(u, rel=1e-10, abs=1e-10)
            assert f.isapprox(fun)

def test_discrete_segment():
    domain = cr.Segment(-1 -1j, 2 + 0.5j)
    pts = domain.point(np.linspace(0, 1, 2000))
    for fun in [
        np.exp,
        lambda x: np.cos(x) + 1j * np.sin(x),
        lambda x: 1 / (1.1 - x),
        lambda x: np.abs(x + 1/2 + 1j/60),
        lambda x: 10j*x + np.tanh((x - 1/5)),
    ]:
        for method in [TCF, AAA]:
            f = approximate(fun, pts, method=method)
            y = [fun(x) for x in pts]
            u = f(pts)
            assert y == pytest.approx(u, rel=1e-10, abs=1e-10)
            assert f.isapprox(fun)

def test_continuum_circle():
    domain = cr.Circle(0 + 1j, 1.2)
    pts = domain.point(np.linspace(0, 1, 2000))
    for fun in [
        np.exp,
        lambda x: np.cos(x) + 1j * np.sin(x),
        lambda x: 1 / (1.1 - x),
        lambda x: np.abs(x + 2 + 1j/60),
        lambda x: 10j*x + np.tanh((x - 1/5)),
    ]:
        for method in [TCF, AAA]:
            f = approximate(fun, domain, method=method)
            y = [fun(x) for x in pts]
            u = f(pts)
            assert y == pytest.approx(u, rel=1e-10, abs=1e-10)
            assert f.isapprox(fun)

def test_discrete_circle():
    domain = cr.Circle(0 + 1j, 1.2)
    pts = domain.point(np.linspace(0, 1, 2000))
    for fun in [
        np.exp,
        lambda x: np.cos(x) + 1j * np.sin(x),
        lambda x: 1 / (1.1 - x),
        lambda x: np.abs(x + 2 + 1j/60),
        lambda x: 10j*x + np.tanh((x - 1/5)),
    ]:
        for method in [TCF, AAA]:
            f = approximate(fun, pts, method=method)
            y = [fun(x) for x in pts]
            u = f(pts)
            assert y == pytest.approx(u, rel=1e-10, abs=1e-10)
            assert f.isapprox(fun)
