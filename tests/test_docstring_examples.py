"""
Unit tests for examples from docstring documentation.

This test file validates all the code examples provided in the docstrings
of the pyratapprox package to ensure they work correctly.
"""

import pytest
import numpy as np
from pyratapprox import *
from pyratapprox import JuliaRatfun, JuliaRatinterp  # Import internal classes for type checking


class TestThieleExamples:
    """Test examples from Thiele class docstring."""
    
    def test_thiele_basic_construction(self):
        """Test basic Thiele construction example."""
        nodes = [0.0, 1.0, 2.0]
        values = [1.0, 2.0, 4.0]
        f = Thiele(nodes, values)
        
        assert isinstance(f, Thiele)
        assert callable(f)
        # Verify interpolation at nodes
        for n, v in zip(nodes, values):
            assert f(n) == pytest.approx(v, abs=1e-10)


class TestBaryExamples:
    """Test examples from Bary class docstring."""
    
    def test_bary_basic_construction(self):
        """Test basic Barycentric construction example."""
        nodes = [0.0, 1.0, 2.0]
        values = [1.0, 2.0, 4.0]
        weights = [1.0, -2.0, 1.0]
        f = Bary(nodes, values, weights)
        
        assert isinstance(f, Bary)
        assert callable(f)
        # Verify interpolation at nodes
        for n, v in zip(nodes, values):
            assert f(n) == pytest.approx(v, abs=1e-10)


class TestApproximateExamples:
    """Test examples from approximate function docstring."""
    
    def test_approximate_default_domain_and_method(self):
        """Test approximation with the default domain and method."""
        # Approximate on the unit interval, using the default method
        f = approximate(np.sin)
        
        assert isinstance(f, ContinuumApprox)
        assert callable(f)
        
        # Verify approximation quality
        test_points = np.linspace(-1, 1, 100)
        for x in test_points:
            assert f(x) == pytest.approx(np.sin(x), rel=1e-10, abs=1e-10)
    
    def test_approximate_on_unit_interval(self):
        """Test approximation on the unit interval."""
        # Choose a method positionally, as in Julia
        f = approximate(np.sin, unitinterval, AAA)
        
        assert isinstance(f, ContinuumApprox)
        assert callable(f)
        
        # Verify approximation quality
        test_points = np.linspace(-1, 1, 100)
        for x in test_points:
            assert f(x) == pytest.approx(np.sin(x), rel=1e-10, abs=1e-10)
    
    def test_approximate_with_strict_poles(self):
        """Test keeping the poles away from the domain."""
        f = approximate(lambda x: np.abs(x), unitinterval, allowed="strict")
        
        assert isinstance(f, ContinuumApprox)
        assert f.allowed is not True
        assert f(0.5) == pytest.approx(0.5, rel=1e-4, abs=1e-4)
    
    def test_approximate_convergence_status(self):
        """Test finding out why the iteration stopped."""
        f = approximate(np.sin, unitinterval, AAA)
        
        assert f.isconverged()
        assert isinstance(f.status, ConvergenceStatus)
        assert f.status.error < 1e-10
    
    def test_approximate_on_discrete_points(self):
        """Test approximation on discrete points."""
        # Approximate on discrete points
        x = np.linspace(-1, 1, 100)
        f = approximate(np.exp, x, method=TCF)
        
        assert isinstance(f, DiscreteApprox)
        assert callable(f)
        
        # Verify approximation quality
        for xi in x:
            assert f(xi) == pytest.approx(np.exp(xi), rel=1e-10, abs=1e-10)
    
    def test_evaluate_approximation(self):
        """Test evaluating the approximation."""
        f = approximate(np.sin, unitinterval, method=AAA)
        
        # Evaluate the approximation
        y = f(0.5)
        
        assert isinstance(y, (float, complex, np.number))
        assert y == pytest.approx(np.sin(0.5), rel=1e-10, abs=1e-10)
    
    def test_get_poles(self):
        """Test getting poles from approximation."""
        f = approximate(lambda x: 1 / (1.1 - x), unitinterval, method=AAA)
        
        # Get poles and residues
        poles = f.poles()
        
        assert isinstance(poles, np.ndarray)
        # The function has a pole at x = 1.1
        assert len(poles) > 0
    
    def test_get_poles_and_residues(self):
        """Test getting poles and residues from approximation."""
        f = approximate(lambda x: 1 / (1.1 - x), unitinterval, method=AAA)
        
        # Get poles and residues
        poles, residues = f.residues()
        
        assert isinstance(poles, np.ndarray)
        assert isinstance(residues, np.ndarray)
        assert len(poles) == len(residues)


class TestRationalFunctionOperations:
    """Test rational function operations and methods."""
    
    def test_callable_evaluation_scalar(self):
        """Test that rational functions can be called with scalar arguments."""
        f = approximate(np.sin, unitinterval, method=AAA)
        
        result = f(0.0)
        assert isinstance(result, (float, complex, np.number))
        assert result == pytest.approx(0.0, abs=1e-10)
    
    def test_callable_evaluation_array(self):
        """Test that rational functions can be called with array arguments."""
        f = approximate(np.sin, unitinterval, method=AAA)
        
        x = np.array([0.0, 0.5, 1.0])
        result = f(x)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)
        for i, xi in enumerate(x):
            assert result[i] == pytest.approx(np.sin(xi), rel=1e-10, abs=1e-10)
    
    def test_degrees_method(self):
        """Test getting degrees of rational function."""
        f = approximate(np.sin, unitinterval, method=AAA)
        
        degrees = f.degrees()
        assert isinstance(degrees, tuple)
        assert len(degrees) == 2
        assert all(isinstance(d, int) for d in degrees)
    
    def test_nodes_and_values(self):
        """Test getting nodes and values from approximation."""
        f = approximate(np.sin, unitinterval, method=AAA)
        
        nodes = f.nodes()
        values = f.values()
        
        assert isinstance(nodes, np.ndarray)
        assert isinstance(values, np.ndarray)
        assert len(nodes) == len(values)
        
        # Verify interpolation at nodes
        for n, v in zip(nodes, values):
            assert v == pytest.approx(np.sin(n), rel=1e-10, abs=1e-10)
    
    def test_isapprox_method(self):
        """Test isapprox method for comparing approximations."""
        f = approximate(np.sin, unitinterval, method=AAA)
        
        # Should be approximately equal to the original function
        assert f.isapprox(np.sin)


class TestContinuumApproxMethods:
    """Test ContinuumApprox-specific methods."""
    
    def test_getfunction_method(self):
        """Test getting the underlying rational function."""
        f = approximate(np.sin, unitinterval, method=AAA)
        
        ratfun = f.getfunction()
        assert isinstance(ratfun, (Thiele, Bary, JuliaRatinterp, JuliaRatfun))
        assert callable(ratfun)
    
    def test_testpoints_method(self):
        """Test getting test points from approximation."""
        f = approximate(np.sin, unitinterval, method=AAA)
        
        test_pts = f.testpoints()
        assert isinstance(test_pts, np.ndarray)
        assert len(test_pts) > 0


class TestDiscreteApproxMethods:
    """Test DiscreteApprox-specific methods."""
    
    def test_discrete_getfunction_method(self):
        """Test getting the underlying rational function from discrete approx."""
        x = np.linspace(-1, 1, 50)
        f = approximate(np.sin, x, method=AAA)
        
        ratfun = f.getfunction()
        assert isinstance(ratfun, (Thiele, Bary, JuliaRatinterp, JuliaRatfun))
        assert callable(ratfun)
    
    def test_discrete_testpoints_method(self):
        """Test getting test points from discrete approximation."""
        x = np.linspace(-1, 1, 50)
        f = approximate(np.sin, x, method=AAA)
        
        test_pts = f.testpoints()
        assert isinstance(test_pts, np.ndarray)
        assert len(test_pts) > 0
        # Test points should be a subset of the domain
        assert len(test_pts) <= len(x)


class TestPredefinedDomains:
    """Test predefined domain constants."""
    
    def test_unitinterval_exists(self):
        """Test that unitinterval is defined and usable."""
        assert unitinterval is not None
        f = approximate(np.sin, unitinterval, method=AAA)
        assert isinstance(f, ContinuumApprox)
    
    def test_unitcircle_exists(self):
        """Test that unitcircle is defined."""
        assert unitcircle is not None
        # unitcircle should be a Circle object
        assert hasattr(unitcircle, 'center')
        assert hasattr(unitcircle, 'radius')
    
    def test_unitdisk_exists(self):
        """Test that unitdisk is defined."""
        assert unitdisk is not None


class TestApproximationMethods:
    """Test approximation method constants."""
    
    def test_aaa_method_exists(self):
        """Test that AAA method constant exists."""
        assert AAA is not None
        f = approximate(np.sin, unitinterval, method=AAA)
        assert isinstance(f, ContinuumApprox)
    
    def test_tcf_method_exists(self):
        """Test that TCF method constant exists."""
        assert TCF is not None
        f = approximate(np.sin, unitinterval, method=TCF)
        assert isinstance(f, ContinuumApprox)


class TestArithmeticOperations:
    """Test arithmetic operations on rational functions."""
    
    def test_scalar_addition(self):
        """Test adding a scalar to a rational function."""
        f = approximate(np.sin, unitinterval, method=AAA)
        r = f.getfunction()
        
        # Test scalar addition
        r_shifted = r + 1.0
        assert callable(r_shifted)
        
        x = 0.5
        expected = np.sin(x) + 1.0
        assert r_shifted(x) == pytest.approx(expected, rel=1e-9, abs=1e-9)
    
    def test_scalar_subtraction(self):
        """Test subtracting a scalar from a rational function."""
        f = approximate(np.sin, unitinterval, method=AAA)
        r = f.getfunction()
        
        # Test scalar subtraction
        r_shifted = r - 0.5
        assert callable(r_shifted)
        
        x = 0.5
        expected = np.sin(x) - 0.5
        assert r_shifted(x) == pytest.approx(expected, rel=1e-9, abs=1e-9)
    
    def test_scalar_multiplication(self):
        """Test multiplying rational function by scalar."""
        f = approximate(np.sin, unitinterval, method=AAA)
        r = f.getfunction()
        
        r_scaled = r * 2.0
        assert callable(r_scaled)
        
        x = 0.5
        expected = 2.0 * np.sin(x)
        assert r_scaled(x) == pytest.approx(expected, rel=1e-9, abs=1e-9)
    
    def test_negation(self):
        """Test negating a rational function."""
        f = approximate(np.sin, unitinterval, method=AAA)
        r = f.getfunction()
        
        r_neg = -r
        assert callable(r_neg)
        
        x = 0.5
        expected = -np.sin(x)
        assert r_neg(x) == pytest.approx(expected, rel=1e-9, abs=1e-9)
