#!/usr/bin/env python3
"""
Test suite for heat equation methods comparison project
File: test_heat_equation_methods.py

This test suite validates both reference solution and student implementation
for the heat equation solver using multiple numerical methods.
"""

import unittest
import numpy as np
import sys
import os
from typing import Tuple, Dict
import warnings

# Import reference solution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'solution'))
from heat_equation_methods_solution import (
    create_initial_condition,
    solve_ftcs,
    solve_backward_euler, 
    solve_crank_nicolson,
    solve_with_scipy,
    calculate_errors,
    compare_methods,
    ALPHA, L, T_FINAL
)

# Import student template
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
try:
    from heat_equation_methods_student import (
        create_initial_condition as student_create_initial_condition,
        solve_ftcs as student_solve_ftcs,
        solve_backward_euler as student_solve_backward_euler,
        solve_crank_nicolson as student_solve_crank_nicolson,
        solve_with_scipy as student_solve_with_scipy,
        calculate_errors as student_calculate_errors,
        compare_methods as student_compare_methods
    )
except ImportError as e:
    print(f"Warning: Could not import student functions: {e}")
    # Create dummy functions to prevent test failures
    def dummy_func(*args, **kwargs):
        raise NotImplementedError("Student function not implemented")
    
    student_create_initial_condition = dummy_func
    student_solve_ftcs = dummy_func
    student_solve_backward_euler = dummy_func
    student_solve_crank_nicolson = dummy_func
    student_solve_with_scipy = dummy_func
    student_calculate_errors = dummy_func
    student_compare_methods = dummy_func

class TestHeatEquationMethods(unittest.TestCase):
    """Test suite for heat equation methods"""
    
    def setUp(self):
        """Set up test parameters"""
        self.nx = 51
        self.nt = 200
        self.total_time = 1.0  # Shorter time for testing
        self.tolerance = 1e-10
        self.x_test = np.linspace(0, L, self.nx)
        
        # Suppress warnings during testing
        warnings.filterwarnings('ignore')
    
    def tearDown(self):
        """Clean up after tests"""
        warnings.resetwarnings()
    
    # Reference solution validation tests (0 points - for verification)
    
    def test_reference_initial_condition(self):
        """Verify reference initial condition implementation"""
        u0 = create_initial_condition(self.x_test)
        
        # Check shape
        self.assertEqual(u0.shape, self.x_test.shape)
        
        # Check boundary conditions
        self.assertAlmostEqual(u0[0], 0.0, places=10)
        self.assertAlmostEqual(u0[-1], 0.0, places=10)
        
        # Check that there's a non-zero region
        self.assertTrue(np.any(u0 > 0))
        
        # Check that the non-zero region is approximately in [10, 11]
        nonzero_indices = np.where(u0 > 0.5)[0]
        if len(nonzero_indices) > 0:
            x_nonzero = self.x_test[nonzero_indices]
            self.assertTrue(np.all(x_nonzero >= 9.5))
            self.assertTrue(np.all(x_nonzero <= 11.5))
    
    def test_reference_ftcs_basic(self):
        """Verify reference FTCS implementation"""
        x, t, u = solve_ftcs(self.nx, self.nt, self.total_time)
        
        # Check shapes
        self.assertEqual(x.shape, (self.nx,))
        self.assertEqual(t.shape, (self.nt,))
        self.assertEqual(u.shape, (self.nt, self.nx))
        
        # Check boundary conditions
        np.testing.assert_allclose(u[:, 0], 0.0, atol=1e-12)
        np.testing.assert_allclose(u[:, -1], 0.0, atol=1e-12)
        
        # Check conservation (total heat should decrease)
        total_heat = np.trapz(u, x, axis=1)
        self.assertTrue(np.all(np.diff(total_heat) <= 1e-10))  # Non-increasing
    
    def test_reference_backward_euler_basic(self):
        """Verify reference backward Euler implementation"""
        x, t, u = solve_backward_euler(self.nx, self.nt, self.total_time)
        
        # Check shapes
        self.assertEqual(x.shape, (self.nx,))
        self.assertEqual(t.shape, (self.nt,))
        self.assertEqual(u.shape, (self.nt, self.nx))
        
        # Check boundary conditions
        np.testing.assert_allclose(u[:, 0], 0.0, atol=1e-12)
        np.testing.assert_allclose(u[:, -1], 0.0, atol=1e-12)
    
    def test_reference_crank_nicolson_basic(self):
        """Verify reference Crank-Nicolson implementation"""
        x, t, u = solve_crank_nicolson(self.nx, self.nt, self.total_time)
        
        # Check shapes
        self.assertEqual(x.shape, (self.nx,))
        self.assertEqual(t.shape, (self.nt,))
        self.assertEqual(u.shape, (self.nt, self.nx))
        
        # Check boundary conditions
        np.testing.assert_allclose(u[:, 0], 0.0, atol=1e-12)
        np.testing.assert_allclose(u[:, -1], 0.0, atol=1e-12)
    
    def test_reference_scipy_basic(self):
        """Verify reference scipy implementation"""
        x, t, u = solve_with_scipy(self.nx, self.total_time)
        
        # Check shapes
        self.assertEqual(x.shape, (self.nx,))
        self.assertTrue(len(t) > 0)
        self.assertEqual(u.shape[1], self.nx)
        
        # Check boundary conditions
        np.testing.assert_allclose(u[:, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(u[:, -1], 0.0, atol=1e-10)
    
    def test_reference_error_calculation(self):
        """Verify reference error calculation"""
        # Create test data
        u1 = np.array([[1.0, 0.5, 0.0], [0.8, 0.4, 0.0]])
        u2 = np.array([[0.9, 0.6, 0.0], [0.7, 0.5, 0.0]])
        dx = 0.1
        
        errors = calculate_errors(u1, u2, dx)
        
        # Check that all expected keys are present
        expected_keys = ['l2_error', 'l2_relative', 'max_error', 'max_relative', 'rms_error']
        for key in expected_keys:
            self.assertIn(key, errors)
            self.assertIsInstance(errors[key], (int, float))
            self.assertGreaterEqual(errors[key], 0.0)
    
    # Student implementation tests
    
    def test_student_initial_condition_5pts(self):
        """Test student initial condition implementation (5 points)"""
        try:
            u0_student = student_create_initial_condition(self.x_test)
            u0_reference = create_initial_condition(self.x_test)
            
            np.testing.assert_allclose(u0_student, u0_reference, 
                                     rtol=self.tolerance, atol=1e-12)
        except NotImplementedError:
            self.fail("Student has not implemented create_initial_condition")
        except Exception as e:
            self.fail(f"Student initial condition failed: {e}")
    
    def test_student_ftcs_basic_8pts(self):
        """Test student FTCS basic functionality (8 points)"""
        try:
            x_s, t_s, u_s = student_solve_ftcs(self.nx, self.nt, self.total_time)
            x_r, t_r, u_r = solve_ftcs(self.nx, self.nt, self.total_time)
            
            np.testing.assert_allclose(x_s, x_r, rtol=self.tolerance)
            np.testing.assert_allclose(t_s, t_r, rtol=self.tolerance)
            np.testing.assert_allclose(u_s, u_r, rtol=1e-8, atol=1e-10)
        except NotImplementedError:
            self.fail("Student has not implemented solve_ftcs")
        except Exception as e:
            self.fail(f"Student FTCS failed: {e}")
    
    def test_student_backward_euler_basic_8pts(self):
        """Test student backward Euler basic functionality (8 points)"""
        try:
            x_s, t_s, u_s = student_solve_backward_euler(self.nx, self.nt, self.total_time)
            x_r, t_r, u_r = solve_backward_euler(self.nx, self.nt, self.total_time)
            
            np.testing.assert_allclose(x_s, x_r, rtol=self.tolerance)
            np.testing.assert_allclose(t_s, t_r, rtol=self.tolerance)
            np.testing.assert_allclose(u_s, u_r, rtol=1e-8, atol=1e-10)
        except NotImplementedError:
            self.fail("Student has not implemented solve_backward_euler")
        except Exception as e:
            self.fail(f"Student backward Euler failed: {e}")
    
    def test_student_crank_nicolson_basic_8pts(self):
        """Test student Crank-Nicolson basic functionality (8 points)"""
        try:
            x_s, t_s, u_s = student_solve_crank_nicolson(self.nx, self.nt, self.total_time)
            x_r, t_r, u_r = solve_crank_nicolson(self.nx, self.nt, self.total_time)
            
            np.testing.assert_allclose(x_s, x_r, rtol=self.tolerance)
            np.testing.assert_allclose(t_s, t_r, rtol=self.tolerance)
            np.testing.assert_allclose(u_s, u_r, rtol=1e-8, atol=1e-10)
        except NotImplementedError:
            self.fail("Student has not implemented solve_crank_nicolson")
        except Exception as e:
            self.fail(f"Student Crank-Nicolson failed: {e}")
    
    def test_student_scipy_basic_6pts(self):
        """Test student scipy basic functionality (6 points)"""
        try:
            x_s, t_s, u_s = student_solve_with_scipy(self.nx, self.total_time)
            x_r, t_r, u_r = solve_with_scipy(self.nx, self.total_time)
            
            np.testing.assert_allclose(x_s, x_r, rtol=self.tolerance)
            # Note: scipy solutions may have different time grids
            # We'll check the final solution instead
            np.testing.assert_allclose(u_s[-1], u_r[-1], rtol=1e-6, atol=1e-8)
        except NotImplementedError:
            self.fail("Student has not implemented solve_with_scipy")
        except Exception as e:
            self.fail(f"Student scipy method failed: {e}")
    
    def test_student_error_calculation_3pts(self):
        """Test student error calculation (3 points)"""
        try:
            # Create test data
            u1 = np.array([[1.0, 0.5, 0.0], [0.8, 0.4, 0.0]])
            u2 = np.array([[0.9, 0.6, 0.0], [0.7, 0.5, 0.0]])
            dx = 0.1
            
            errors_student = student_calculate_errors(u1, u2, dx)
            errors_reference = calculate_errors(u1, u2, dx)
            
            for key in errors_reference:
                self.assertIn(key, errors_student)
                np.testing.assert_allclose(errors_student[key], errors_reference[key],
                                         rtol=1e-10)
        except NotImplementedError:
            self.fail("Student has not implemented calculate_errors")
        except Exception as e:
            self.fail(f"Student error calculation failed: {e}")
    
    def test_student_boundary_conditions_4pts(self):
        """Test that student implementations satisfy boundary conditions (4 points)"""
        methods = [
            ('FTCS', student_solve_ftcs),
            ('Backward Euler', student_solve_backward_euler),
            ('Crank-Nicolson', student_solve_crank_nicolson)
        ]
        
        for method_name, method_func in methods:
            try:
                x, t, u = method_func(self.nx, self.nt, self.total_time)
                
                # Check boundary conditions
                np.testing.assert_allclose(u[:, 0], 0.0, atol=1e-10,
                                         err_msg=f"{method_name} left boundary condition failed")
                np.testing.assert_allclose(u[:, -1], 0.0, atol=1e-10,
                                         err_msg=f"{method_name} right boundary condition failed")
            except NotImplementedError:
                self.fail(f"Student has not implemented {method_name}")
            except Exception as e:
                self.fail(f"Student {method_name} boundary condition test failed: {e}")
    
    def test_student_stability_ftcs_3pts(self):
        """Test FTCS stability condition handling (3 points)"""
        try:
            # Test with parameters that should trigger stability warning
            nx_unstable = 101
            nt_unstable = 50  # This should create r > 0.5
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                x, t, u = student_solve_ftcs(nx_unstable, nt_unstable, 0.1)
                
                # Check that solution doesn't blow up (basic stability check)
                self.assertTrue(np.all(np.isfinite(u)))
                self.assertTrue(np.all(u >= -1e-10))  # Physical constraint
                
        except NotImplementedError:
            self.fail("Student has not implemented solve_ftcs")
        except Exception as e:
            self.fail(f"Student FTCS stability test failed: {e}")
    
    def test_student_conservation_property_3pts(self):
        """Test conservation properties (3 points)"""
        methods = [
            ('FTCS', student_solve_ftcs),
            ('Backward Euler', student_solve_backward_euler),
            ('Crank-Nicolson', student_solve_crank_nicolson)
        ]
        
        for method_name, method_func in methods:
            try:
                x, t, u = method_func(self.nx, self.nt, self.total_time)
                
                # Check that total heat is non-increasing (due to boundary conditions)
                total_heat = np.trapz(u, x, axis=1)
                heat_diff = np.diff(total_heat)
                
                # Allow small numerical errors
                self.assertTrue(np.all(heat_diff <= 1e-8),
                              f"{method_name} violates conservation property")
                
            except NotImplementedError:
                self.fail(f"Student has not implemented {method_name}")
            except Exception as e:
                self.fail(f"Student {method_name} conservation test failed: {e}")
    
    def test_student_compare_methods_2pts(self):
        """Test student compare_methods function (2 points)"""
        try:
            results = student_compare_methods(nx=31, nt=100)
            
            # Check that results is a dictionary
            self.assertIsInstance(results, dict)
            
            # Check for expected keys
            expected_keys = ['ftcs', 'backward_euler', 'crank_nicolson', 'scipy']
            for key in expected_keys:
                self.assertIn(key, results)
                
        except NotImplementedError:
            self.fail("Student has not implemented compare_methods")
        except Exception as e:
            self.fail(f"Student compare_methods failed: {e}")
    
    def test_student_method_consistency_5pts(self):
        """Test consistency between different methods (5 points)"""
        try:
            # All methods should give similar results for the same problem
            nx, nt = 51, 400
            total_time = 0.5
            
            x1, t1, u1 = student_solve_ftcs(nx, nt, total_time)
            x2, t2, u2 = student_solve_backward_euler(nx, nt, total_time)
            x3, t3, u3 = student_solve_crank_nicolson(nx, nt, total_time)
            
            # Check that final solutions are reasonably close
            # (allowing for method differences)
            np.testing.assert_allclose(u1[-1], u2[-1], rtol=0.1, atol=1e-6,
                                     err_msg="FTCS and Backward Euler final solutions differ too much")
            np.testing.assert_allclose(u2[-1], u3[-1], rtol=0.05, atol=1e-6,
                                     err_msg="Backward Euler and Crank-Nicolson final solutions differ too much")
            
        except NotImplementedError:
            self.fail("Student has not implemented required methods")
        except Exception as e:
            self.fail(f"Student method consistency test failed: {e}")

class TestAdvancedFeatures(unittest.TestCase):
    """Advanced tests for bonus features"""
    
    def test_student_scipy_methods_bonus_2pts(self):
        """Test different scipy integration methods (2 bonus points)"""
        try:
            methods = ['RK45', 'DOP853', 'Radau', 'BDF']
            results = []
            
            for method in methods:
                try:
                    x, t, u = student_solve_with_scipy(51, 1.0, method=method)
                    results.append((method, u[-1]))  # Store final solution
                except Exception:
                    continue  # Skip if method not implemented
            
            # If multiple methods work, check they give similar results
            if len(results) >= 2:
                for i in range(1, len(results)):
                    np.testing.assert_allclose(results[0][1], results[i][1], 
                                             rtol=0.01, atol=1e-8,
                                             err_msg=f"Methods {results[0][0]} and {results[i][0]} give different results")
                
        except NotImplementedError:
            pass  # Bonus feature, no penalty
        except Exception as e:
            pass  # Bonus feature, no penalty

if __name__ == '__main__':
    # Create a test suite with proper ordering
    suite = unittest.TestSuite()
    
    # Add reference tests first (for validation)
    reference_tests = [
        'test_reference_initial_condition',
        'test_reference_ftcs_basic',
        'test_reference_backward_euler_basic',
        'test_reference_crank_nicolson_basic',
        'test_reference_scipy_basic',
        'test_reference_error_calculation'
    ]
    
    for test in reference_tests:
        suite.addTest(TestHeatEquationMethods(test))
    
    # Add student tests
    student_tests = [
        'test_student_initial_condition_5pts',
        'test_student_ftcs_basic_8pts',
        'test_student_backward_euler_basic_8pts',
        'test_student_crank_nicolson_basic_8pts',
        'test_student_scipy_basic_6pts',
        'test_student_error_calculation_3pts',
        'test_student_boundary_conditions_4pts',
        'test_student_stability_ftcs_3pts',
        'test_student_conservation_property_3pts',
        'test_student_compare_methods_2pts',
        'test_student_method_consistency_5pts'
    ]
    
    for test in student_tests:
        suite.addTest(TestHeatEquationMethods(test))
    
    # Add bonus tests
    suite.addTest(TestAdvancedFeatures('test_student_scipy_methods_bonus_2pts'))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors))/result.testsRun*100:.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}")