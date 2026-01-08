import pytest
import numpy as np
from pyrat import *
import cxregions as cr
import juliacall

def test_thiele_build():
    # Test building a Thiele continued fraction
    nodes = np.array([0.0, 1.0, 2.0, 3.0])
    values = np.array([1.0, 2.0, 0.5, 4.0])
    cf = Thiele(nodes, values)
    
    assert isinstance(cf, Thiele)
    assert np.allclose(cf.nodes(), nodes)
    assert np.allclose(cf.values(), values)
    assert cf.length() == len(nodes)
    assert len(cf.weights) == 4
    assert cf.degrees() == (2, 1)
    assert cf.degree() == 1
    assert cf.isreal() == True
    assert cf.isempty() == False
    assert cf.__repr__() == "Thiele continued fraction of type (2, 1)"

def test_thiele_evaluate():
    # Test evaluating a Thiele continued fraction
    nodes = np.array([0.0, 1.0, 2.0, 3.0])
    values = (nodes**2 - 2*nodes + 5) / (nodes + 2)
    cf = Thiele(nodes, values)
    assert cf(2.0) == pytest.approx(values[2])
    assert cf(0.0) == pytest.approx(values[0])
    assert cf(nodes) == pytest.approx(values)
    
    x = np.array([0.5, 1.5, 2.5])
    result = cf(x)
    expected = (x**2 - 2*x + 5) / (x + 2)
    assert np.allclose(result, expected, atol=1e-5)

def test_evaluate_complex_thiele():
    # Test evaluating a complex Thiele continued fraction
    nodes = np.array([0.0, 1.0 + 1.0j, 2.0 - 1.0j, 3.0])
    values = np.array([1.0, 2.0 + 1.0j, 0.5 - 0.5j, 0.0])
    cf = Thiele(nodes, values)
    
    assert cf(nodes[2]) == pytest.approx(values[2])
    assert cf(nodes) == pytest.approx(values)

    test_points = np.array([0.5 + 0.5j, 1.5, 2.5 + 0.5j])
    result = cf(test_points)
    expected = np.array([1.436+ 0.148j, 0.98 - 1.36j, -0.8034482758620684 + 0.14137931034482762j])
    assert np.allclose(result, expected, atol=1e-11)

def test_roots_poles():
    nodes = np.array([0.0, 1.0, 2.0, 3.0])
    values = (nodes**2 - 2*nodes + 5) / (nodes + 2)
    cf = Thiele(nodes, values)
    
    z = cf.roots()
    assert np.allclose(z.real, (np.array([1.0 , 1.0])), atol=1e-10)
    assert np.allclose(z.imag, (np.array([-2.0 , 2.0])), atol=1e-10)
    p = cf.poles()
    assert np.allclose(p, np.array([-2.0]), atol=1e-10)

def test_add_subract():
    nodes = np.array([0.0, 1.0, 2.0, 3.0])
    f1 = lambda x: (x**2 - 2*x + 5) / (x + 2)
    f2 = lambda x: (x + 3) / (x**2 + 3*x + 2)
    v1 = f1(nodes)
    v2 = f2(nodes)
    cf1 = Thiele(nodes, v1)
    cf2 = Thiele(nodes, v2)
    test_points = np.array([0.5, 1.5, 2.5])

    cf_neg = -cf1
    assert np.allclose(cf_neg(test_points), -f1(test_points), atol=1e-10)
    
    # cf_sum = cf1 + cf2
    # cf_diff = cf1 - cf2  
    # assert np.allclose(cf_sum(test_points), f1(test_points) + f2(test_points), atol=1e-10)
    # assert np.allclose(cf_diff(test_points), f1(test_points) - f2(test_points), atol=1e-10)

    cf_sum = cf1 + 2.0j
    cf_rsum = 2.0j + cf1
    cf_diff = cf1 - 2.0j
    assert np.allclose(cf_sum(test_points), f1(test_points) + 2.0j, atol=1e-10)
    assert np.allclose(cf_rsum(test_points), f1(test_points) + 2.0j, atol=1e-10)
    assert np.allclose(cf_diff(test_points), f1(test_points) - 2.0j, atol=1e-10)

def test_multiply_divide():
    nodes = np.array([0.0, 1.0, 2.0, 3.0])
    f1 = lambda x: (x**2 - 2*x + 5) / (x + 2)
    v1 = f1(nodes)
    cf1 = Thiele(nodes, v1)
    test_points = np.array([0.5, 1.5, 2.5])

    cf_mul = cf1 * 2.0j
    cf_rmul = 2.0j * cf1
    assert np.allclose(cf_mul(test_points), f1(test_points) * 2.0j, atol=1e-10)
    assert np.allclose(cf_rmul(test_points), f1(test_points) * 2.0j, atol=1e-10)
    
    cf_div = cf1 / 2.0j
    assert np.allclose(cf_div(test_points), f1(test_points) / 2.0j, atol=1e-10)