#!/usr/bin/env python3
"""
Test suite for heat equation methods
File: test_heat_equation_methods.py

Tests all 6 numerical methods for solving 1D heat equation
"""

import unittest
import numpy as np
import sys
import os
from typing import Tuple

# Import reference solution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'solution'))
from heat_equation_methods_solution import heat_equation_solver as reference_solver
from heat_equation_methods_solution import initial_condition_function as reference_initial

# Import student template
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from heat_equation_methods_student import heat_equation_solver as student_solver
from heat_equation_methods_student import initial_condition_function as student_initial

class TestHeatEquationMethods(unittest.TestCase):
    """
    Test suite for heat equation numerical methods
    """
    
    def setUp(self):
        """Set up test parameters"""
        self.L = 20.0
        self.T = 5.0  # Shorter time for faster testing
        self.a_squared = 10.0
        self.nx = 50  # Smaller grid for faster testing
        self.nt = 200
        self.tolerance = 1e-3  # Tolerance for numerical comparison
        
        # Create reference solver
        self.ref_solver = reference_solver(self.L, self.T, self.a_squared, self.nx, self.nt)
        self.ref_solver.set_initial_condition(reference_initial)
    
    def test_reference_initial_condition(self):
        """Verify reference initial condition (0 points - validation)"""
        x = np.linspace(0, 20, 100)
        phi = reference_initial(x)
        
        # Check that phi = 1 for 10 <= x <= 11
        mask_inside = (x >= 10) & (x <= 11)
        mask_outside = (x < 10) | (x > 11)
        
        self.assertTrue(np.allclose(phi[mask_inside], 1.0))
        self.assertTrue(np.allclose(phi[mask_outside], 0.0))
    
    def test_reference_solver_initialization(self):
        """Verify reference solver initialization (0 points - validation)"""
        solver = reference_solver(self.L, self.T, self.a_squared, self.nx, self.nt)
        
        self.assertEqual(solver.L, self.L)
        self.assertEqual(solver.T, self.T)
        self.assertEqual(solver.a_squared, self.a_squared)
        self.assertEqual(solver.nx, self.nx)
        self.assertEqual(solver.nt, self.nt)
        
        expected_dx = self.L / (self.nx - 1)
        expected_dt = self.T / self.nt
        expected_r = self.a_squared * expected_dt / (expected_dx**2)
        
        self.assertAlmostEqual(solver.dx, expected_dx, places=10)
        self.assertAlmostEqual(solver.dt, expected_dt, places=10)
        self.assertAlmostEqual(solver.r, expected_r, places=10)
    
    def test_reference_ftcs_method(self):
        """Verify reference FTCS method (0 points - validation)"""
        t_array, u_solution = self.ref_solver.ftcs_method()
        
        # Check dimensions
        self.assertEqual(len(t_array), self.nt + 1)
        self.assertEqual(u_solution.shape, (self.nt + 1, self.nx))
        
        # Check boundary conditions
        np.testing.assert_allclose(u_solution[:, 0], 0, atol=1e-10)
        np.testing.assert_allclose(u_solution[:, -1], 0, atol=1e-10)
        
        # Check that solution decreases over time (heat dissipation)
        total_heat_initial = np.sum(u_solution[0])
        total_heat_final = np.sum(u_solution[-1])
        self.assertLess(total_heat_final, total_heat_initial)
    
    def test_reference_analytical_solution(self):
        """Verify reference analytical solution (0 points - validation)"""
        x = self.ref_solver.x
        u_analytical_t0 = self.ref_solver.analytical_solution(x, 0.0)
        u_analytical_tf = self.ref_solver.analytical_solution(x, self.T)
        
        # At t=0, should approximate initial condition
        u_initial = reference_initial(x)
        # Allow some tolerance due to Fourier series approximation
        self.assertLess(np.max(np.abs(u_analytical_t0 - u_initial)), 0.1)
        
        # At final time, should be smaller than initial
        self.assertLess(np.max(u_analytical_tf), np.max(u_analytical_t0))
    
    def test_student_solver_initialization_15pts(self):
        """Test student solver initialization (15 points)"""
        try:
            solver = student_solver(self.L, self.T, self.a_squared, self.nx, self.nt)
            
            self.assertEqual(solver.L, self.L)
            self.assertEqual(solver.T, self.T)
            self.assertEqual(solver.a_squared, self.a_squared)
            self.assertEqual(solver.nx, self.nx)
            self.assertEqual(solver.nt, self.nt)
            
            expected_dx = self.L / (self.nx - 1)
            expected_dt = self.T / self.nt
            expected_r = self.a_squared * expected_dt / (expected_dx**2)
            
            self.assertAlmostEqual(solver.dx, expected_dx, places=8)
            self.assertAlmostEqual(solver.dt, expected_dt, places=8)
            self.assertAlmostEqual(solver.r, expected_r, places=8)
            
        except NotImplementedError:
            self.fail("Student has not implemented solver initialization")
    
    def test_student_initial_condition_10pts(self):
        """Test student initial condition function (10 points)"""
        try:
            x = np.linspace(0, 20, 100)
            phi = student_initial(x)
            
            # Check that phi = 1 for 10 <= x <= 11
            mask_inside = (x >= 10) & (x <= 11)
            mask_outside = (x < 10) | (x > 11)
            
            self.assertTrue(np.allclose(phi[mask_inside], 1.0, atol=1e-10))
            self.assertTrue(np.allclose(phi[mask_outside], 0.0, atol=1e-10))
            
        except NotImplementedError:
            self.fail("Student has not implemented initial condition function")
    
    def test_student_set_initial_condition_10pts(self):
        """Test student set_initial_condition method (10 points)"""
        try:
            solver = student_solver(self.L, self.T, self.a_squared, self.nx, self.nt)
            solver.set_initial_condition(student_initial)
            
            # Check that initial condition is set
            self.assertIsNotNone(solver.u0)
            self.assertEqual(len(solver.u0), self.nx)
            
            # Check boundary conditions
            self.assertAlmostEqual(solver.u0[0], 0.0, places=10)
            self.assertAlmostEqual(solver.u0[-1], 0.0, places=10)
            
        except NotImplementedError:
            self.fail("Student has not implemented set_initial_condition method")
    
    def test_student_ftcs_method_15pts(self):
        """Test student FTCS method (15 points)"""
        try:
            solver = student_solver(self.L, self.T, self.a_squared, self.nx, self.nt)
            solver.set_initial_condition(student_initial)
            
            t_array, u_solution = solver.ftcs_method()
            
            # Check dimensions
            self.assertEqual(len(t_array), self.nt + 1)
            self.assertEqual(u_solution.shape, (self.nt + 1, self.nx))
            
            # Check boundary conditions
            np.testing.assert_allclose(u_solution[:, 0], 0, atol=1e-10)
            np.testing.assert_allclose(u_solution[:, -1], 0, atol=1e-10)
            
            # Compare with reference solution
            ref_t, ref_u = self.ref_solver.ftcs_method()
            np.testing.assert_allclose(u_solution, ref_u, rtol=self.tolerance)
            
        except NotImplementedError:
            self.fail("Student has not implemented FTCS method")
    
    def test_student_laplace_explicit_method_10pts(self):
        """Test student Laplace explicit method (10 points)"""
        try:
            solver = student_solver(self.L, self.T, self.a_squared, self.nx, self.nt)
            solver.set_initial_condition(student_initial)
            
            t_array, u_solution = solver.laplace_explicit_method()
            
            # Check dimensions
            self.assertEqual(len(t_array), self.nt + 1)
            self.assertEqual(u_solution.shape, (self.nt + 1, self.nx))
            
            # Compare with reference solution
            ref_t, ref_u = self.ref_solver.laplace_explicit_method()
            np.testing.assert_allclose(u_solution, ref_u, rtol=self.tolerance)
            
        except NotImplementedError:
            self.fail("Student has not implemented Laplace explicit method")
    
    def test_student_btcs_method_15pts(self):
        """Test student BTCS method (15 points)"""
        try:
            solver = student_solver(self.L, self.T, self.a_squared, self.nx, self.nt)
            solver.set_initial_condition(student_initial)
            
            t_array, u_solution = solver.btcs_method()
            
            # Check dimensions
            self.assertEqual(len(t_array), self.nt + 1)
            self.assertEqual(u_solution.shape, (self.nt + 1, self.nx))
            
            # Compare with reference solution
            ref_t, ref_u = self.ref_solver.btcs_method()
            np.testing.assert_allclose(u_solution, ref_u, rtol=self.tolerance)
            
        except NotImplementedError:
            self.fail("Student has not implemented BTCS method")
    
    def test_student_crank_nicolson_method_15pts(self):
        """Test student Crank-Nicolson method (15 points)"""
        try:
            solver = student_solver(self.L, self.T, self.a_squared, self.nx, self.nt)
            solver.set_initial_condition(student_initial)
            
            t_array, u_solution = solver.crank_nicolson_method()
            
            # Check dimensions
            self.assertEqual(len(t_array), self.nt + 1)
            self.assertEqual(u_solution.shape, (self.nt + 1, self.nx))
            
            # Compare with reference solution
            ref_t, ref_u = self.ref_solver.crank_nicolson_method()
            np.testing.assert_allclose(u_solution, ref_u, rtol=self.tolerance)
            
        except NotImplementedError:
            self.fail("Student has not implemented Crank-Nicolson method")
    
    def test_student_modified_crank_nicolson_method_10pts(self):
        """Test student modified Crank-Nicolson method (10 points)"""
        try:
            solver = student_solver(self.L, self.T, self.a_squared, self.nx, self.nt)
            solver.set_initial_condition(student_initial)
            
            t_array, u_solution = solver.modified_crank_nicolson_method()
            
            # Check dimensions
            self.assertEqual(len(t_array), self.nt + 1)
            self.assertEqual(u_solution.shape, (self.nt + 1, self.nx))
            
            # Compare with reference solution
            ref_t, ref_u = self.ref_solver.modified_crank_nicolson_method()
            np.testing.assert_allclose(u_solution, ref_u, rtol=self.tolerance)
            
        except NotImplementedError:
            self.fail("Student has not implemented modified Crank-Nicolson method")
    
    def test_student_solve_ivp_method_10pts(self):
        """Test student solve_ivp method (10 points)"""
        try:
            solver = student_solver(self.L, self.T, self.a_squared, self.nx, self.nt)
            solver.set_initial_condition(student_initial)
            
            t_array, u_solution = solver.solve_ivp_method()
            
            # Check dimensions
            self.assertEqual(len(t_array), self.nt + 1)
            self.assertEqual(u_solution.shape, (self.nt + 1, self.nx))
            
            # Compare with reference solution (allow larger tolerance for ODE solver)
            ref_t, ref_u = self.ref_solver.solve_ivp_method()
            np.testing.assert_allclose(u_solution, ref_u, rtol=0.01)
            
        except NotImplementedError:
            self.fail("Student has not implemented solve_ivp method")
    
    def test_student_analytical_solution_5pts(self):
        """Test student analytical solution (5 points)"""
        try:
            solver = student_solver(self.L, self.T, self.a_squared, self.nx, self.nt)
            solver.set_initial_condition(student_initial)
            
            x = solver.x
            u_analytical = solver.analytical_solution(x, self.T/2)
            
            # Compare with reference
            ref_analytical = self.ref_solver.analytical_solution(x, self.T/2)
            np.testing.assert_allclose(u_analytical, ref_analytical, rtol=self.tolerance)
            
        except NotImplementedError:
            self.fail("Student has not implemented analytical solution")
    
    def test_student_compare_methods_5pts(self):
        """Test student compare_methods function (5 points)"""
        try:
            solver = student_solver(self.L, self.T, self.a_squared, self.nx, self.nt)
            solver.set_initial_condition(student_initial)
            
            results = solver.compare_methods()
            
            # Check that results is a dictionary
            self.assertIsInstance(results, dict)
            
            # Check that it contains expected methods
            expected_methods = ['FTCS', 'Laplace_Explicit', 'BTCS', 
                              'Crank_Nicolson', 'Modified_CN', 'solve_ivp']
            
            for method in expected_methods:
                if method in results and results[method].get('status') == 'success':
                    self.assertIn('l2_error', results[method])
                    self.assertIn('max_error', results[method])
                    self.assertIn('computation_time', results[method])
            
        except NotImplementedError:
            self.fail("Student has not implemented compare_methods function")
    
    def test_student_boundary_conditions_5pts(self):
        """Test that student methods properly enforce boundary conditions (5 points)"""
        try:
            solver = student_solver(self.L, self.T, self.a_squared, self.nx, self.nt)
            solver.set_initial_condition(student_initial)
            
            # Test FTCS method boundary conditions
            t_array, u_solution = solver.ftcs_method()
            
            # All time steps should have zero boundary values
            np.testing.assert_allclose(u_solution[:, 0], 0, atol=1e-10)
            np.testing.assert_allclose(u_solution[:, -1], 0, atol=1e-10)
            
        except NotImplementedError:
            self.fail("Student has not implemented FTCS method")
    
    def test_student_stability_awareness_5pts(self):
        """Test that student solver calculates stability parameter correctly (5 points)"""
        try:
            # Create solver with potentially unstable parameters
            nx_small = 20
            nt_large = 1000
            solver = student_solver(self.L, self.T, self.a_squared, nx_small, nt_large)
            
            expected_r = self.a_squared * (self.T/nt_large) / ((self.L/(nx_small-1))**2)
            self.assertAlmostEqual(solver.r, expected_r, places=6)
            
        except NotImplementedError:
            self.fail("Student has not implemented solver initialization")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)