#!/usr/bin/env python3
"""
Test suite for Heat Equation Methods
File: test_heat_equation_methods.py
"""

import unittest
import numpy as np
import os
import sys
import warnings

# Add parent directory to module search path for importing student code
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from student code (switch to solution for testing reference)
# from solution.heat_equation_methods_solution import HeatEquationSolver
from heat_equation_methods_student import HeatEquationSolver

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

class TestHeatEquationSolver(unittest.TestCase):
    """
    Test cases for HeatEquationSolver class.
    """
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.solver = HeatEquationSolver(L=20.0, alpha=10.0, nx=21, T_final=5.0)
        self.tolerance = 1e-6
        self.plot_times = [0, 1, 2.5, 5]
        
    def test_initialization_5pts(self):
        """Test solver initialization (5 points)"""
        try:
            # Test basic attributes
            self.assertEqual(self.solver.L, 20.0)
            self.assertEqual(self.solver.alpha, 10.0)
            self.assertEqual(self.solver.nx, 21)
            self.assertEqual(self.solver.T_final, 5.0)
            
            # Test grid setup
            self.assertAlmostEqual(self.solver.dx, 1.0, places=6)
            self.assertEqual(len(self.solver.x), 21)
            self.assertAlmostEqual(self.solver.x[0], 0.0, places=6)
            self.assertAlmostEqual(self.solver.x[-1], 20.0, places=6)
            
            # Test initial condition
            self.assertEqual(len(self.solver.u_initial), 21)
            self.assertAlmostEqual(self.solver.u_initial[0], 0.0, places=6)  # Boundary
            self.assertAlmostEqual(self.solver.u_initial[-1], 0.0, places=6)  # Boundary
            
            # Check initial condition in [10, 11] region
            mask = (self.solver.x >= 10) & (self.solver.x <= 11)
            self.assertTrue(np.all(self.solver.u_initial[mask] == 1.0))
            
        except NotImplementedError:
            self.fail("Student has not implemented the initialization")
    
    def test_explicit_method_basic_15pts(self):
        """Test explicit finite difference method basic functionality (15 points)"""
        try:
            results = self.solver.solve_explicit(dt=0.001, plot_times=self.plot_times)
            
            # Check result structure
            self.assertIn('times', results)
            self.assertIn('solutions', results)
            self.assertIn('method', results)
            self.assertIn('computation_time', results)
            self.assertIn('stability_parameter', results)
            
            # Check solution properties
            self.assertEqual(len(results['times']), len(results['solutions']))
            self.assertTrue(len(results['solutions']) > 0)
            
            # Check solution array dimensions
            for sol in results['solutions']:
                self.assertEqual(len(sol), self.solver.nx)
                
            # Check boundary conditions
            for sol in results['solutions']:
                self.assertAlmostEqual(sol[0], 0.0, places=6)
                self.assertAlmostEqual(sol[-1], 0.0, places=6)
                
            # Check stability parameter
            expected_r = self.solver.alpha * 0.001 / (self.solver.dx**2)
            self.assertAlmostEqual(results['stability_parameter'], expected_r, places=6)
            
        except NotImplementedError:
            self.fail("Student has not implemented solve_explicit method")
    
    def test_implicit_method_basic_15pts(self):
        """Test implicit finite difference method basic functionality (15 points)"""
        try:
            results = self.solver.solve_implicit(dt=0.1, plot_times=self.plot_times)
            
            # Check result structure
            self.assertIn('times', results)
            self.assertIn('solutions', results)
            self.assertIn('method', results)
            self.assertIn('computation_time', results)
            self.assertIn('stability_parameter', results)
            
            # Check solution properties
            self.assertEqual(len(results['times']), len(results['solutions']))
            self.assertTrue(len(results['solutions']) > 0)
            
            # Check solution array dimensions
            for sol in results['solutions']:
                self.assertEqual(len(sol), self.solver.nx)
                
            # Check boundary conditions
            for sol in results['solutions']:
                self.assertAlmostEqual(sol[0], 0.0, places=6)
                self.assertAlmostEqual(sol[-1], 0.0, places=6)
                
        except NotImplementedError:
            self.fail("Student has not implemented solve_implicit method")
    
    def test_crank_nicolson_method_basic_15pts(self):
        """Test Crank-Nicolson method basic functionality (15 points)"""
        try:
            results = self.solver.solve_crank_nicolson(dt=0.5, plot_times=self.plot_times)
            
            # Check result structure
            self.assertIn('times', results)
            self.assertIn('solutions', results)
            self.assertIn('method', results)
            self.assertIn('computation_time', results)
            self.assertIn('stability_parameter', results)
            
            # Check solution properties
            self.assertEqual(len(results['times']), len(results['solutions']))
            self.assertTrue(len(results['solutions']) > 0)
            
            # Check solution array dimensions
            for sol in results['solutions']:
                self.assertEqual(len(sol), self.solver.nx)
                
            # Check boundary conditions
            for sol in results['solutions']:
                self.assertAlmostEqual(sol[0], 0.0, places=6)
                self.assertAlmostEqual(sol[-1], 0.0, places=6)
                
        except NotImplementedError:
            self.fail("Student has not implemented solve_crank_nicolson method")
    
    def test_solve_ivp_method_basic_15pts(self):
        """Test solve_ivp method basic functionality (15 points)"""
        try:
            results = self.solver.solve_with_solve_ivp(method='BDF', plot_times=self.plot_times)
            
            # Check result structure
            self.assertIn('times', results)
            self.assertIn('solutions', results)
            self.assertIn('method', results)
            self.assertIn('computation_time', results)
            
            # Check solution properties
            self.assertEqual(len(results['times']), len(results['solutions']))
            self.assertTrue(len(results['solutions']) > 0)
            
            # Check solution array dimensions
            for sol in results['solutions']:
                self.assertEqual(len(sol), self.solver.nx)
                
            # Check boundary conditions
            for sol in results['solutions']:
                self.assertAlmostEqual(sol[0], 0.0, places=6)
                self.assertAlmostEqual(sol[-1], 0.0, places=6)
                
        except NotImplementedError:
            self.fail("Student has not implemented solve_with_solve_ivp method")
    
    def test_heat_equation_ode_helper_10pts(self):
        """Test ODE helper function (10 points)"""
        try:
            # Test with simple internal state
            u_internal = np.ones(self.solver.nx - 2) * 0.5
            
            # Call the ODE function
            du_dt = self.solver._heat_equation_ode(0.0, u_internal)
            
            # Check output dimensions
            self.assertEqual(len(du_dt), len(u_internal))
            
            # Check that it returns finite values
            self.assertTrue(np.all(np.isfinite(du_dt)))
            
        except NotImplementedError:
            self.fail("Student has not implemented _heat_equation_ode method")
    
    def test_compare_methods_10pts(self):
        """Test method comparison functionality (10 points)"""
        try:
            # Use smaller time domain for faster testing
            solver_small = HeatEquationSolver(L=20.0, alpha=10.0, nx=11, T_final=1.0)
            
            results = solver_small.compare_methods(
                dt_explicit=0.001,
                dt_implicit=0.1,
                dt_cn=0.1,
                ivp_method='BDF',
                plot_times=[0, 0.5, 1.0]
            )
            
            # Check that all methods are included
            expected_methods = ['explicit', 'implicit', 'crank_nicolson', 'solve_ivp']
            for method in expected_methods:
                self.assertIn(method, results)
                
            # Check each method has required fields
            for method_name, method_results in results.items():
                self.assertIn('times', method_results)
                self.assertIn('solutions', method_results)
                self.assertIn('method', method_results)
                self.assertIn('computation_time', method_results)
                
        except NotImplementedError:
            self.fail("Student has not implemented compare_methods method")
    
    def test_physical_behavior_10pts(self):
        """Test physical behavior of solutions (10 points)"""
        try:
            # Test with explicit method (most straightforward)
            results = self.solver.solve_explicit(dt=0.001, plot_times=[0, 1, 5])
            
            initial_sol = results['solutions'][0]
            final_sol = results['solutions'][-1]
            
            # Check heat diffusion: maximum should decrease over time
            initial_max = np.max(initial_sol)
            final_max = np.max(final_sol)
            self.assertLess(final_max, initial_max, 
                          "Temperature maximum should decrease due to diffusion")
            
            # Check conservation: total "heat" should decrease (due to boundary conditions)
            initial_sum = np.sum(initial_sol)
            final_sum = np.sum(final_sol)
            self.assertLess(final_sum, initial_sum,
                          "Total heat should decrease due to boundary heat loss")
            
            # Check smoothing: solution should become smoother
            initial_gradient = np.max(np.abs(np.diff(initial_sol)))
            final_gradient = np.max(np.abs(np.diff(final_sol)))
            self.assertLess(final_gradient, initial_gradient,
                          "Solution should become smoother over time")
            
        except NotImplementedError:
            self.fail("Student has not implemented methods for physical behavior test")
    
    def test_stability_explicit_5pts(self):
        """Test stability warning for explicit method (5 points)"""
        try:
            # Test with unstable parameters (large dt)
            unstable_dt = 1.0  # This should violate stability condition
            
            # Capture output to check for stability warning
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                results = self.solver.solve_explicit(dt=unstable_dt, plot_times=[0, 1])
            
            output = f.getvalue()
            
            # Check that stability parameter is calculated
            self.assertIn('stability_parameter', results)
            
            # Check that r > 0.5 (unstable)
            r = results['stability_parameter']
            self.assertGreater(r, 0.5, "Should detect unstable condition")
            
        except NotImplementedError:
            self.fail("Student has not implemented stability checking")
    
    def test_error_handling_5pts(self):
        """Test error handling (5 points)"""
        try:
            # Test with invalid parameters
            solver_invalid = HeatEquationSolver(L=0, alpha=-1, nx=2, T_final=-1)
            
            # Should still create object but may have issues in solving
            self.assertIsInstance(solver_invalid, HeatEquationSolver)
            
            # Test with empty plot_times
            results = self.solver.solve_explicit(dt=0.001, plot_times=[])
            self.assertIsInstance(results, dict)
            
        except NotImplementedError:
            self.fail("Student has not implemented basic error handling")
        except Exception as e:
            # Some errors are acceptable for invalid parameters
            pass


class TestIntegration(unittest.TestCase):
    """
    Integration tests for the complete workflow.
    """
    
    def test_complete_workflow_5pts(self):
        """Test complete workflow from initialization to analysis (5 points)"""
        try:
            # Create solver
            solver = HeatEquationSolver(L=20.0, alpha=10.0, nx=11, T_final=2.0)
            
            # Run comparison
            results = solver.compare_methods(
                dt_explicit=0.001,
                dt_implicit=0.1,
                dt_cn=0.1,
                ivp_method='BDF',
                plot_times=[0, 1, 2]
            )
            
            # Test plotting (should not raise errors)
            try:
                solver.plot_comparison(results, save_figure=False)
            except Exception:
                pass  # Plotting errors are acceptable in test environment
            
            # Test accuracy analysis
            accuracy = solver.analyze_accuracy(results, reference_method='solve_ivp')
            self.assertIsInstance(accuracy, dict)
            
        except NotImplementedError:
            self.fail("Student has not implemented complete workflow")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)