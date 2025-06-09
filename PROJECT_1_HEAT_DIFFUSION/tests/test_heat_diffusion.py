#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
铝棒热传导方程显式差分法数值解 - 测试文件

测试显式差分法的正确性、稳定性和精度

作者: 测试框架
日期: 2024
"""

import unittest
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import reference solution for validation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'solution'))
from heat_diffusion_solution import HeatDiffusionSolver as ReferenceSolver
from heat_diffusion_solution import analyze_stability, calculate_errors, compare_with_analytical, demonstrate_instability

# Import student solution
try:
    from heat_diffusion_student import HeatDiffusionSolver as StudentSolver
    from heat_diffusion_student import analyze_stability as student_analyze_stability
    from heat_diffusion_student import calculate_errors as student_calculate_errors
    from heat_diffusion_student import compare_with_analytical as student_compare_with_analytical
    from heat_diffusion_student import demonstrate_instability as student_demonstrate_instability
except ImportError:
    print("Warning: Could not import student solution")
    StudentSolver = None

class TestHeatDiffusionReference(unittest.TestCase):
    """
    Reference solution validation tests (0 points - for verification only)
    """
    
    def setUp(self):
        """Setup test environment"""
        self.solver = ReferenceSolver(L=1.0, nx=21, nt=100, total_time=0.01)
        self.tolerance = 1e-10
    
    def test_reference_initialization(self):
        """Verify reference solution initialization"""
        self.assertEqual(self.solver.L, 1.0)
        self.assertEqual(self.solver.nx, 21)
        self.assertEqual(self.solver.nt, 100)
        self.assertEqual(self.solver.total_time, 0.01)
        
        # Check grid parameters
        expected_dx = 1.0 / 20
        expected_dt = 0.01 / 100
        
        self.assertAlmostEqual(self.solver.dx, expected_dx, places=10)
        self.assertAlmostEqual(self.solver.dt, expected_dt, places=10)
    
    def test_reference_explicit_method(self):
        """Verify reference explicit finite difference method"""
        t, T = self.solver.explicit_finite_difference()
        
        # Check return value shapes
        self.assertEqual(len(t), self.solver.nt + 1)
        self.assertEqual(T.shape, (self.solver.nx, self.solver.nt + 1))
        
        # Check boundary conditions
        for j in range(T.shape[1]):
            self.assertAlmostEqual(T[0, j], 0.0, places=10)
            self.assertAlmostEqual(T[-1, j], 0.0, places=10)
    
    def test_reference_analytical_solution(self):
        """Verify reference analytical solution"""
        t, T_analytical = self.solver.analytical_solution(n_terms=20)
        
        # Check return value shapes
        self.assertEqual(len(t), self.solver.nt + 1)
        self.assertEqual(T_analytical.shape, (self.solver.nx, self.solver.nt + 1))
        
        # Check boundary conditions
        for j in range(T_analytical.shape[1]):
            self.assertAlmostEqual(T_analytical[0, j], 0.0, places=8)
            self.assertAlmostEqual(T_analytical[-1, j], 0.0, places=8)

class TestHeatDiffusionStudent(unittest.TestCase):
    """
    Student solution tests
    """
    
    def setUp(self):
        """Setup test environment"""
        if StudentSolver is None:
            self.skipTest("Student solution not available")
        
        self.solver = StudentSolver(L=1.0, nx=21, nt=100, total_time=0.01)
        self.tolerance = 1e-10
    
    def test_student_initialization_5pts(self):
        """
        Test solver initialization (5 points)
        """
        self.assertEqual(self.solver.L, 1.0)
        self.assertEqual(self.solver.nx, 21)
        self.assertEqual(self.solver.nt, 100)
        self.assertEqual(self.solver.total_time, 0.01)
        
        # Check grid parameters
        expected_dx = 1.0 / 20
        expected_dt = 0.01 / 100
        
        self.assertAlmostEqual(self.solver.dx, expected_dx, places=10)
        self.assertAlmostEqual(self.solver.dt, expected_dt, places=10)
        
        # Check spatial and temporal arrays
        self.assertEqual(len(self.solver.x), 21)
        self.assertEqual(len(self.solver.t), 101)
        self.assertAlmostEqual(self.solver.x[0], 0.0, places=10)
        self.assertAlmostEqual(self.solver.x[-1], 1.0, places=10)
    
    def test_student_initial_condition_5pts(self):
        """
        Test initial condition setup (5 points)
        """
        try:
            T0 = self.solver.initial_condition()
            
            # Check array length
            self.assertEqual(len(T0), self.solver.nx)
            
            # Check boundary conditions
            self.assertEqual(T0[0], 0.0)
            self.assertEqual(T0[-1], 0.0)
            
            # Check internal initial temperature
            for i in range(1, len(T0) - 1):
                self.assertEqual(T0[i], 100.0)
                
        except NotImplementedError:
            self.fail("Initial condition method not implemented")
    
    def test_student_explicit_method_basic_15pts(self):
        """
        Test explicit finite difference method basic functionality (15 points)
        """
        try:
            t, T = self.solver.explicit_finite_difference()
            
            # Check return value shapes
            self.assertEqual(len(t), self.solver.nt + 1)
            self.assertEqual(T.shape, (self.solver.nx, self.solver.nt + 1))
            
            # Check boundary conditions at all time steps
            for j in range(T.shape[1]):
                self.assertAlmostEqual(T[0, j], 0.0, places=10)
                self.assertAlmostEqual(T[-1, j], 0.0, places=10)
            
            # Check initial condition
            np.testing.assert_array_almost_equal(T[:, 0], self.solver.initial_condition())
            
            # Check temperature monotonic decrease (due to boundary conditions being 0)
            center_idx = self.solver.nx // 2
            center_temps = T[center_idx, :]
            
            # Center temperature should decrease over time
            for i in range(1, len(center_temps)):
                self.assertLessEqual(center_temps[i], center_temps[i-1] + self.tolerance)
            
        except NotImplementedError:
            self.fail("Explicit finite difference method not implemented")
    
    def test_student_analytical_solution_10pts(self):
        """
        Test analytical solution calculation (10 points)
        """
        try:
            t, T_analytical = self.solver.analytical_solution(n_terms=20)
            
            # Check return value shapes
            self.assertEqual(len(t), self.solver.nt + 1)
            self.assertEqual(T_analytical.shape, (self.solver.nx, self.solver.nt + 1))
            
            # Check boundary conditions
            for j in range(T_analytical.shape[1]):
                self.assertAlmostEqual(T_analytical[0, j], 0.0, places=8)
                self.assertAlmostEqual(T_analytical[-1, j], 0.0, places=8)
            
            # Check initial condition approximation (Fourier series approximation)
            initial_analytical = T_analytical[:, 0]
            initial_exact = self.solver.initial_condition()
            
            # Check initial condition approximation at internal points
            for i in range(1, len(initial_analytical) - 1):
                self.assertAlmostEqual(initial_analytical[i], initial_exact[i], delta=5.0)
            
            # Check temperature monotonicity
            center_idx = self.solver.nx // 2
            center_temps = T_analytical[center_idx, :]
            
            for i in range(1, len(center_temps)):
                self.assertLessEqual(center_temps[i], center_temps[i-1] + self.tolerance)
            
        except NotImplementedError:
            self.fail("Analytical solution not implemented")
    
    def test_student_stability_condition_10pts(self):
        """
        Test explicit method stability condition (10 points)
        """
        # Test stable case (η < 0.25)
        stable_solver = StudentSolver(L=1.0, nx=21, nt=50, total_time=0.001)
        
        try:
            t_stable, T_stable = stable_solver.explicit_finite_difference()
            
            # Check that solution remains bounded
            self.assertTrue(np.all(np.isfinite(T_stable)))
            self.assertTrue(np.all(T_stable >= -self.tolerance))
            self.assertTrue(np.all(T_stable <= 100 + self.tolerance))
            
            # Test unstable case (η > 0.25)
            # Use larger time step to make η > 0.25
            unstable_solver = StudentSolver(L=1.0, nx=21, nt=10, total_time=0.01)
            
            # This should either fail or produce unstable results
            try:
                t_unstable, T_unstable = unstable_solver.explicit_finite_difference()
                
                # If it runs, check if results show instability
                if np.any(np.abs(T_unstable) > 1000):  # Large values indicate instability
                    pass  # Expected instability
                else:
                    # If stable, η should be <= 0.25
                    self.assertLessEqual(unstable_solver.eta, 0.25 + 1e-10)
                    
            except (ValueError, RuntimeError):
                pass  # Expected failure due to instability
                
        except NotImplementedError:
            self.fail("Explicit finite difference method not implemented")
    
    def test_student_accuracy_comparison_10pts(self):
        """
        Test accuracy by comparing with analytical solution (10 points)
        """
        try:
            # Use finer grid for better accuracy
            fine_solver = StudentSolver(L=1.0, nx=41, nt=200, total_time=0.005)
            
            t_num, T_num = fine_solver.explicit_finite_difference()
            t_ana, T_ana = fine_solver.analytical_solution(n_terms=50)
            
            # Calculate errors
            error_l2 = np.linalg.norm(T_num - T_ana) / np.linalg.norm(T_ana)
            error_max = np.max(np.abs(T_num - T_ana))
            
            # Check that errors are reasonable
            self.assertLess(error_l2, 0.1)  # L2 relative error < 10%
            self.assertLess(error_max, 10.0)  # Max absolute error < 10K
            
        except NotImplementedError:
            self.fail("Required methods not implemented")
    
    def test_student_visualization_methods_5pts(self):
        """
        Test visualization methods (5 points)
        """
        try:
            t, T = self.solver.explicit_finite_difference()
            
            # Test plot_evolution method
            try:
                self.solver.plot_evolution(t, T, "Test Evolution")
            except NotImplementedError:
                self.fail("plot_evolution method not implemented")
            except Exception as e:
                # Allow other exceptions (e.g., display issues in testing)
                pass
            
            # Test plot_3d_surface method
            try:
                self.solver.plot_3d_surface(t, T, "Test 3D Surface")
            except NotImplementedError:
                self.fail("plot_3d_surface method not implemented")
            except Exception as e:
                # Allow other exceptions (e.g., display issues in testing)
                pass
            
            # Test create_animation method
            try:
                self.solver.create_animation(t, T, "test_animation.gif")
            except NotImplementedError:
                self.fail("create_animation method not implemented")
            except Exception as e:
                # Allow other exceptions (e.g., display issues in testing)
                pass
                
        except NotImplementedError:
            self.fail("Explicit finite difference method not implemented")

class TestAnalysisFunctions(unittest.TestCase):
    """
    Test analysis functions
    """
    
    def test_analyze_stability_function_5pts(self):
        """
        Test analyze_stability function (5 points)
        """
        try:
            student_analyze_stability()
        except NotImplementedError:
            self.fail("analyze_stability function not implemented")
        except Exception as e:
            # Allow other exceptions (e.g., display issues)
            pass
    
    def test_calculate_errors_function_5pts(self):
        """
        Test calculate_errors function (5 points)
        """
        try:
            student_calculate_errors()
        except NotImplementedError:
            self.fail("calculate_errors function not implemented")
        except Exception as e:
            # Allow other exceptions (e.g., display issues)
            pass
    
    def test_compare_with_analytical_function_5pts(self):
        """
        Test compare_with_analytical function (5 points)
        """
        try:
            student_compare_with_analytical()
        except NotImplementedError:
            self.fail("compare_with_analytical function not implemented")
        except Exception as e:
            # Allow other exceptions (e.g., display issues)
            pass
    
    def test_demonstrate_instability_function_5pts(self):
        """
        Test demonstrate_instability function (5 points)
        """
        try:
            student_demonstrate_instability()
        except NotImplementedError:
            self.fail("demonstrate_instability function not implemented")
        except Exception as e:
            # Allow other exceptions (e.g., display issues)
            pass

if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add reference tests (for validation)
    suite.addTest(unittest.makeSuite(TestHeatDiffusionReference))
    
    # Add student tests (for grading)
    suite.addTest(unittest.makeSuite(TestHeatDiffusionStudent))
    suite.addTest(unittest.makeSuite(TestAnalysisFunctions))
    
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
    print(f"{'='*60}")