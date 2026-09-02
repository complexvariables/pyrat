"""Tests for the approximate() interface introduced in RFA v0.4.0."""

import pytest
import numpy as np
from pyratapprox import *
import pyratapprox
import cxregions as cr

def test_default_domain_and_method():
    f = approximate(np.sin)
    assert isinstance(f, ContinuumApprox)
    assert isinstance(f.domain, cr.Segment)
    assert f(0.3) == pytest.approx(np.sin(0.3))
    # the default method is the Thiele continued fraction
    assert isinstance(f.fun, Thiele)

def test_method_selectors():
    x = 0.3
    for selector, cls in [
        (TCF, Thiele), ("tcf", Thiele), ("thiele", Thiele), (Thiele, Thiele),
        (RFA.Thiele(), Thiele),
        (AAA, Bary), ("aaa", Bary), ("barycentric", Bary), (Bary, Bary),
        (RFA.Barycentric(), Bary),
    ]:
        # positional, as in Julia
        f = approximate(np.exp, unitinterval, selector)
        assert isinstance(f.fun, cls)
        assert f(x) == pytest.approx(np.exp(x))
        # by keyword
        g = approximate(np.exp, unitinterval, method=selector)
        assert isinstance(g.fun, cls)
        assert g(x) == pytest.approx(np.exp(x))

def test_method_selector_from_interpolant():
    f = approximate(np.sin, unitinterval, AAA)
    g = approximate(np.cos, unitinterval, f.fun)
    assert isinstance(g.fun, Bary)
    assert g(0.3) == pytest.approx(np.cos(0.3))

def test_discrete_method_selection():
    x = np.linspace(-1, 1, 200)
    for selector, cls in [(TCF, Thiele), (AAA, Bary)]:
        f = approximate(np.exp, x, selector)
        assert isinstance(f, DiscreteApprox)
        assert isinstance(f.fun, cls)
        assert f(0.3) == pytest.approx(np.exp(0.3))

def test_bad_method():
    with pytest.raises(ValueError):
        approximate(np.sin, unitinterval, method="no such method")
    with pytest.raises(ValueError):
        approximate(np.sin, unitinterval, method=np.sin)

def test_prescribed_poles():
    fun = lambda z: 1 / (z**2 + 4)
    zeta = [2j, -2j]
    for f in [
        approximate(fun, unitinterval, zeta),          # positional
        approximate(fun, unitinterval, zeta=zeta),     # by keyword
        approximate(fun, unitinterval, zeta, PartialFractions),
    ]:
        assert f(0.3) == pytest.approx(fun(0.3))
        # partial fractions do not iterate, so no status is recorded
        assert f.status is None
        assert not f.isconverged()

def test_status_on_convergence():
    f = approximate(np.sin, unitinterval)
    assert f.isconverged()
    assert f.status.reason == "converged"
    assert f.status.error < 1e-10
    assert f.status.best <= f.status.iterations

def test_status_on_degree_budget():
    f = approximate(lambda x: np.abs(x - 0.1), unitinterval, AAA, max_degree=12)
    assert f.degree() <= 12
    assert not f.isconverged()
    assert f.status.reason in ("max_degree", "stagnated")
    assert f.status.isconverged() is False

def test_allowed_policy():
    # as of RFA v0.4.0, poles are unrestricted by default
    f = approximate(np.exp, unitinterval)
    assert f.allowed is True
    # "strict" keeps the poles off the curve
    fun = lambda z: 1 / (z - 0.5)
    g = approximate(fun, cr.Circle(0, 1), allowed="strict")
    assert g.allowed is not True
    assert g(2.0) == pytest.approx(fun(2.0))
    assert np.all(np.abs(np.abs(g.poles()) - 1) > 1e-8)
