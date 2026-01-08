import pytest
import numpy as np
from pyrat import *
import cxregions as cr
import juliacall

@pytest.fixture
def data():
    nodes = np.array([0.0, 1.0, 2.0, 3.0])
    values = (nodes**2 - 2*nodes + 5) / (nodes**3 + 2)
    weights = np.array([1.0, -1.0, 1.0, -1.0])
    return nodes, values, weights

@pytest.fixture
def r_real(data):
    nodes, values, weights = data
    return Bary(nodes, values, weights)

def test_bary_build(r_real, data):
    nodes, values, weights = data
    assert isinstance(r_real, Bary)
    assert np.allclose(r_real.nodes(), nodes)
    assert np.allclose(r_real.values(), values)
    assert r_real.length() == len(nodes)
    assert r_real.weights == pytest.approx(weights)
    assert r_real.degrees() == (3, 3)
    assert r_real.degree() == 3
    assert r_real.isreal() == True
    assert r_real.isempty() == False
    assert r_real.__repr__() == "Barycentric rational function of type (3, 3)"

def test_bary_evaluate(r_real, data):
    # Test evaluating a bary continued fraction
    nodes, values, weights = data
    ba = r_real
    assert ba(2.0) == pytest.approx(values[2])
    assert ba(0.0) == pytest.approx(values[0])
    assert ba(nodes) == pytest.approx(values)
    
    x = np.array([0.5, 1.5, 2.5])
    result = ba(x)
    expected = np.array([1.9938423645320196, 0.6810344827586207, 0.4454022988505747])
    assert np.allclose(result, expected, atol=1e-5)

def test_evaluate_complex():
    # Test evaluating a complex Thiele continued fraction
    nodes = np.array([0.0, 1.0 + 1.0j, 2.0 - 1.0j, 3.0])
    values = np.array([1.0, 2.0 + 1.0j, 0.5 - 0.5j, 0.0])
    weights = np.array([1.0, -1.0, 1.0, -1.0])
    ba = Bary(nodes, values, weights)
    
    assert ba(nodes[2]) == pytest.approx(values[2])
    assert ba(nodes) == pytest.approx(values)

    test_points = np.array([0.5 + 0.5j, 1.5, 2.5 + 0.5j])
    result = ba(test_points)
    expected = np.array([1.29411764705882 + 0.448529411764706j, 1.25 - 0.375j, -1.0 - 1.375j])
    assert np.allclose(result, expected, atol=1e-11)

def test_roots_poles(r_real):
    ba = r_real
    
    z = ba.roots()
    assert np.allclose(np.sort(z.real), np.array([1.644006179 , 1.644006179, 3.546698385]), atol=1e-7)
    assert np.allclose(np.sort(z.imag), np.array([-0.58149367935, 0, 0.58149367935]), atol=1e-7)
    p = ba.poles()
    assert np.allclose(np.sort(p.real), np.array([1.5, 1.5]), atol=1e-10)
    assert np.allclose(np.sort(p.imag), np.array([-1, 1])*np.sqrt(3)/2, atol=1e-7)

def test_add_subract(r_real, data):
    nodes, values, weights = data

    r_neg = -r_real
    assert np.allclose(r_neg(nodes), -values, atol=1e-10)
    
    r_sum = r_real + 2.0j
    r_rsum = 2.0j + r_real
    r_diff = r_real - 2.0j
    assert np.allclose(r_sum(nodes), values + 2.0j, atol=1e-10)
    assert np.allclose(r_rsum(nodes), values + 2.0j, atol=1e-10)
    assert np.allclose(r_diff(nodes), values - 2.0j, atol=1e-10)

def test_multiply_divide(r_real, data):
    nodes, values, weights = data

    r_mul = r_real * 2.0j
    r_rmul = 2.0j * r_real
    assert np.allclose(r_mul(nodes), values * 2.0j, atol=1e-10)
    assert np.allclose(r_rmul(nodes), values * 2.0j, atol=1e-10)
    
    r_div = r_real / 2.0j
    assert np.allclose(r_div(nodes), values / 2.0j, atol=1e-10)