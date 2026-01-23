import pytest
import numpy as np
from pyrat import *
import pyrat
import cxregions as cr

def test_continuum_segment():
    for fun in [
        np.exp,
        lambda x: np.cos(x) + 1j * np.sin(x),
        lambda x: 1 / (1.1 - x),
        lambda x: np.log(4 + x),
        lambda x: np.abs(x + 1/2),
    ]:
        for method in [TCF, AAA]:
            domain = cr.Segment(-2, 3+2j)
            pts = np.linspace(-2, 3+2j, 1000)
            f = approximate(fun, domain, method=method)
            y = [fun(x) for x in pts]
            u = [f(x) for x in pts]
            assert y == pytest.approx(u, rel=1e-10, abs=1e-10)
            assert f.isapprox(fun)

def test_continuum_circle():
    for fun in [
        np.exp,
        lambda x: np.cos(x) + 1j * np.sin(x),
        lambda x: 1 / (1.1 - x),
        lambda x: np.log(4 + x),
        lambda x: np.abs(x + 1/2),
    ]:
        for method in [TCF, AAA]:
            domain = cr.Circle(1, 2)
            pts = np.linspace(0, 2*np.pi, 1000)
            pts = 1 + 2 * np.exp(1j * pts)
            f = approximate(fun, domain, method=method)
            y = [fun(x) for x in pts]
            u = [f(x) for x in pts]
            assert y == pytest.approx(u, rel=1e-10, abs=1e-10)
            assert f.isapprox(fun)

def test_discrete_segment():
    for fun in [
        np.exp,
        lambda x: np.cos(x) + 1j * np.sin(x),
        lambda x: 1 / (1.1 - x),
        lambda x: np.log(4 + x),
        lambda x: np.abs(x + 1/2),
    ]:
        for method in [TCF, AAA]:
            domain = cr.Segment(-2, 3+2j)
            pts = np.linspace(-2, 3+2j, 1000)
            f = approximate(fun, pts, method=method)
            y = [fun(x) for x in pts]
            u = [f(x) for x in pts]
            assert y == pytest.approx(u, rel=1e-10, abs=1e-10)
            assert f.isapprox(fun)

def test_discrete_circle():
    for fun in [
        np.exp,
        lambda x: np.cos(x) + 1j * np.sin(x),
        lambda x: 1 / (1.1 - x),
        lambda x: np.log(4 + x),
        lambda x: np.abs(x + 1/2),
    ]:
        for method in [TCF, AAA]:
            domain = cr.Circle(1, 2)
            pts = np.linspace(0, 2*np.pi, 1000)
            pts = 1 + 2 * np.exp(1j * pts)
            f = approximate(fun, pts, method=method)
            y = [fun(x) for x in pts]
            u = [f(x) for x in pts]
            assert y == pytest.approx(u, rel=1e-10, abs=1e-10)
            assert f.isapprox(fun)