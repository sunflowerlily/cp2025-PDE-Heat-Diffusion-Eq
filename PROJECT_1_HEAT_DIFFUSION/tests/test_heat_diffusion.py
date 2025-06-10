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
from heat_diffusion_solution import (
    solve_heat_diffusion_problem_1 as reference_solve_problem_1,
    solve_heat_diffusion_analytical_comparison as reference_solve_analytical_comparison,
    solve_heat_diffusion_newton_cooling as reference_solve_newton_cooling,
    # analyze_stability, calculate_errors, compare_with_analytical, demonstrate_instability # Keep if used by other tests
)

# Import student solution
try:
    from heat_diffusion_student import HeatDiffusionSolver as StudentSolver
    from heat_diffusion_student import (
        solve_heat_diffusion_problem_1 as student_solve_problem_1,
        solve_heat_diffusion_analytical_comparison as student_solve_analytical_comparison,
        solve_heat_diffusion_newton_cooling as student_solve_newton_cooling
    )
except ImportError:
    print("Warning: Could not import student solution or one of its main functions.")
    StudentSolver = None
    student_solve_problem_1 = None
    student_solve_analytical_comparison = None
    student_solve_newton_cooling = None

class TestHeatDiffusionReference(unittest.TestCase):
    """
    Reference solution validation tests (0 points - for verification only)
    These tests verify the core HeatDiffusionSolver class and helper functions if they are still used directly.
    With the new structure, most validation will happen via the task-specific solve_... functions.
    """
    
    def setUp(self):
        """Setup test environment"""
        # Basic parameters for solver instantiation if needed for direct tests
        self.default_params = {
            'L': 0.1, 'nx': 21, 'total_time': 10.0, 'dt': 0.01,
            'K': 237.0, 'C': 900.0, 'rho': 2700.0,
            'initial_condition_type': 'constant', 'T0_val': 100.0,
            'boundary_condition_left_type': 'dirichlet', 'bc_left_val': 0.0,
            'boundary_condition_right_type': 'dirichlet', 'bc_right_val': 0.0,
        }
        self.solver = ReferenceSolver(self.default_params)
        self.tolerance = 1e-9 # Adjusted tolerance

    def test_reference_solver_initialization(self):
        """Verify reference solver initialization with new params structure"""
        self.assertEqual(self.solver.L, self.default_params['L'])
        self.assertEqual(self.solver.nx, self.default_params['nx'])
        self.assertAlmostEqual(self.solver.dx, self.default_params['L'] / (self.default_params['nx'] - 1), places=10)
        # dt and nt are now calculated internally based on total_time and dt
        # self.assertEqual(self.solver.nt, calculated_nt)
        # self.assertAlmostEqual(self.solver.dt, self.default_params['dt'], places=10)

    def test_reference_task1_basic_simulation(self):
        """Verify reference solution for Task 1: Basic Simulation"""
        params = {
            'L': 0.1, 'nx': 21, 'total_time': 100.0, 'dt': 0.1,
            'K': 237.0, 'C': 900.0, 'rho': 2700.0,
            'initial_condition_type': 'constant', 'T0_val': 100.0,
            'boundary_condition_left_type': 'dirichlet', 'bc_left_val': 0.0,
            'boundary_condition_right_type': 'dirichlet', 'bc_right_val': 0.0,
            'plot_times': [] # No plotting for test
        }
        try:
            solver, t_array, T_matrix = reference_solve_problem_1(params, save_plot=False)
            self.assertIsNotNone(solver)
            self.assertEqual(T_matrix.shape[0], params['nx'])
            # Check boundary conditions at final time step
            self.assertAlmostEqual(T_matrix[0, -1], params['bc_left_val'], delta=self.tolerance)
            self.assertAlmostEqual(T_matrix[-1, -1], params['bc_right_val'], delta=self.tolerance)
            # Check if temperature decreased in the center (qualitative)
            self.assertLess(T_matrix[params['nx']//2, -1], params['T0_val'])
        except Exception as e:
            self.fail(f"Reference solve_heat_diffusion_problem_1 failed: {e}")

    def test_reference_task2_analytical_comparison(self):
        """Verify reference solution for Task 2: Analytical Comparison"""
        params = {
            'L': 0.1, 'nx': 51, 'total_time': 100.0, 'dt': 0.02, # Smaller dt for better accuracy
            'K': 237.0, 'C': 900.0, 'rho': 2700.0,
            'initial_condition_type': 'sine_half_period', 'T_max_sine': 100.0,
            'boundary_condition_left_type': 'dirichlet', 'bc_left_val': 0.0,
            'boundary_condition_right_type': 'dirichlet', 'bc_right_val': 0.0,
            'plot_time_analytical': 50.0, 'analytical_terms': 50
        }
        try:
            # This function in solution might primarily plot, so we check if it runs without error
            # And if it returns values, we can check them.
            # For now, just ensure it runs.
            reference_solve_analytical_comparison(params, save_plot=False)
            # If it returned data, e.g., numerical and analytical solutions at plot_time_analytical:
            # np.testing.assert_allclose(numerical_at_t, analytical_at_t, rtol=1e-2) # Example assertion
        except Exception as e:
            self.fail(f"Reference solve_heat_diffusion_analytical_comparison failed: {e}")

    def test_reference_task5_newton_cooling(self):
        """Verify reference solution for Task 5: Newton's Cooling"""
        params = {
            'L': 0.1, 'nx': 21, 'total_time': 200.0, 'dt': 0.2,
            'K': 237.0, 'C': 900.0, 'rho': 2700.0,
            'initial_condition_type': 'constant', 'T0_val': 100.0,
            'boundary_condition_left_type': 'newton', 'h_coeff_left': 10.0, 'T_env_left': 20.0,
            'boundary_condition_right_type': 'newton', 'h_coeff_right': 10.0, 'T_env_right': 20.0,
            'plot_times': [] # No plotting for test
        }
        try:
            solver, t_array, T_matrix = reference_solve_newton_cooling(params, save_plot=False)
            self.assertIsNotNone(solver)
            self.assertEqual(T_matrix.shape[0], params['nx'])
            # Check if temperature approaches T_env (qualitative)
            # For long enough time, internal points should be > T_env and < T0_val
            center_temp_final = T_matrix[params['nx']//2, -1]
            self.assertGreater(center_temp_final, params['T_env_left'] - self.tolerance) # Should be above T_env
            self.assertLess(center_temp_final, params['T0_val'] + self.tolerance) # Should be below T0
        except Exception as e:
            self.fail(f"Reference solve_heat_diffusion_newton_cooling failed: {e}")

# It's good practice to keep the old TestHeatDiffusionStudent class and adapt it,
# or create new classes for new task-based testing structure.
# For now, let's adapt TestHeatDiffusionStudent for the new structure.

class TestHeatDiffusionStudentTasks(unittest.TestCase):
    """
    Student solution tests for new task-based structure.
    """
    def setUp(self):
        """Setup test environment"""
        self.tolerance = 1e-9
        if StudentSolver is None or student_solve_problem_1 is None or \
           student_solve_analytical_comparison is None or student_solve_newton_cooling is None:
            self.skipTest("Student solution or one of its main functions is not available or not imported correctly.")

    # Task 1: Basic Simulation (20 points)
    def test_student_task1_basic_simulation_20pts(self):
        """Test student solution for Task 1: Basic Simulation (20 points)"""
        params = {
            'L': 0.1, 'nx': 21, 'total_time': 100.0, 'dt': 0.1,
            'K': 237.0, 'C': 900.0, 'rho': 2700.0,
            'initial_condition_type': 'constant', 'T0_val': 100.0,
            'boundary_condition_left_type': 'dirichlet', 'bc_left_val': 0.0,
            'boundary_condition_right_type': 'dirichlet', 'bc_right_val': 0.0,
            'plot_times': []
        }
        ref_solver, ref_t, ref_T = reference_solve_problem_1(params, save_plot=False)
        try:
            student_solver, student_t, student_T = student_solve_problem_1(params, save_plot=False)
            self.assertIsNotNone(student_solver, "Student solver object should not be None")
            self.assertEqual(student_T.shape, ref_T.shape, "Student T_matrix shape mismatch")
            np.testing.assert_allclose(student_T, ref_T, rtol=1e-5, atol=self.tolerance, err_msg="Student Task 1 basic simulation results differ from reference.")
        except NotImplementedError:
            self.fail("Student solve_heat_diffusion_problem_1 for Task 1 is not implemented.")
        except Exception as e:
            self.fail(f"Student Task 1 basic simulation failed: {e}")

    # Task 2: Analytical Comparison (20 points)
    def test_student_task2_analytical_comparison_20pts(self):
        """Test student solution for Task 2: Analytical Comparison (20 points)"""
        params = {
            'L': 0.1, 'nx': 21, 'total_time': 100.0, 'dt': 0.02, # Use nx=21 for faster test, solution uses 51
            'K': 237.0, 'C': 900.0, 'rho': 2700.0,
            'initial_condition_type': 'sine_half_period', 'T_max_sine': 100.0,
            'boundary_condition_left_type': 'dirichlet', 'bc_left_val': 0.0,
            'boundary_condition_right_type': 'dirichlet', 'bc_right_val': 0.0,
            'plot_time_analytical': 50.0, 'analytical_terms': 30 # Fewer terms for faster test
        }
        # Reference function might primarily plot. We need to get data from it if possible.
        # For now, we assume the student's function should produce comparable numerical results
        # to a direct solver call from the reference for the numerical part.
        
        # Get reference numerical solution at plot_time_analytical
        ref_solver_params = params.copy()
        ref_solver_params.pop('plot_time_analytical', None)
        ref_solver_params.pop('analytical_terms', None)
        ref_solver_instance = ReferenceSolver(ref_solver_params)
        ref_t_num, ref_T_num_matrix = ref_solver_instance.explicit_finite_difference()
        time_idx_ref = np.argmin(np.abs(ref_t_num - params['plot_time_analytical']))
        ref_numerical_at_t = ref_T_num_matrix[:, time_idx_ref]
        ref_analytical_at_t = ref_solver_instance.analytical_solution_at_time(params['plot_time_analytical'], params['analytical_terms'])

        try:
            # Student's function might also primarily plot. We need to see what it returns.
            # Assuming it might return (solver, numerical_solution_at_t, analytical_solution_at_t)
            # Or we might need to call student's solver and analytical methods directly.
            # For now, let's assume student_solve_analytical_comparison is expected to perform the comparison.
            # This test might need significant adjustment based on student function's actual return.
            student_solver, student_numerical_at_t, student_analytical_at_t = student_solve_analytical_comparison(params, save_plot=False)
            self.assertIsNotNone(student_solver, "Student solver object should not be None for Task 2")
            np.testing.assert_allclose(student_numerical_at_t, ref_numerical_at_t, rtol=1e-3, atol=self.tolerance, err_msg="Student Task 2 numerical result at comparison time differs from reference.")
            np.testing.assert_allclose(student_analytical_at_t, ref_analytical_at_t, rtol=1e-3, atol=self.tolerance, err_msg="Student Task 2 analytical result at comparison time differs from reference.")
        except NotImplementedError:
            self.fail("Student solve_heat_diffusion_analytical_comparison for Task 2 is not implemented.")
        except TypeError: # If student function doesn't return 3 values
            self.fail("Student solve_heat_diffusion_analytical_comparison for Task 2 does not return the expected (solver, numerical_data, analytical_data) tuple or direct data for comparison.")
        except Exception as e:
            self.fail(f"Student Task 2 analytical comparison failed: {e}")

    # Task 3: Stability Analysis (10 points) - This is more conceptual, tested by observation.
    # No direct autogradable test here, but we can check if a known unstable case fails as expected if student implements checks.
    # For now, this is a placeholder or can be a check for a report item.

    # Task 4: Different Initial/Boundary Conditions (Two Rods) (20 points)
    def test_student_task4_two_rods_20pts(self):
        """Test student solution for Task 4: Two Rods (20 points)"""
        params = {
            'L': 0.2, 'nx': 41, 'total_time': 200.0, 'dt': 0.1, # Adjusted for faster test
            'K': 237.0, 'C': 900.0, 'rho': 2700.0,
            'initial_condition_type': 'two_rods', 'T_left_rod': 100.0, 'T_right_rod': 0.0, 'contact_point_ratio': 0.5,
            'boundary_condition_left_type': 'dirichlet', 'bc_left_val': 100.0,
            'boundary_condition_right_type': 'dirichlet', 'bc_right_val': 0.0,
            'plot_times': []
        }
        ref_solver, ref_t, ref_T = reference_solve_problem_1(params, save_plot=False) # Using problem_1 as it's a general solver
        try:
            student_solver, student_t, student_T = student_solve_problem_1(params, save_plot=False)
            self.assertIsNotNone(student_solver, "Student solver object should not be None for Task 4")
            self.assertEqual(student_T.shape, ref_T.shape, "Student T_matrix shape mismatch for Task 4")
            np.testing.assert_allclose(student_T, ref_T, rtol=1e-5, atol=self.tolerance, err_msg="Student Task 4 (Two Rods) results differ from reference.")
        except NotImplementedError:
            self.fail("Student solve_heat_diffusion_problem_1 for Task 4 (Two Rods) is not implemented.")
        except Exception as e:
            self.fail(f"Student Task 4 (Two Rods) failed: {e}")

    # Task 5: Newton's Cooling (20 points)
    def test_student_task5_newton_cooling_20pts(self):
        """Test student solution for Task 5: Newton's Cooling (20 points)"""
        params = {
            'L': 0.1, 'nx': 21, 'total_time': 200.0, 'dt': 0.2,
            'K': 237.0, 'C': 900.0, 'rho': 2700.0,
            'initial_condition_type': 'constant', 'T0_val': 100.0,
            'boundary_condition_left_type': 'newton', 'h_coeff_left': 10.0, 'T_env_left': 20.0,
            'boundary_condition_right_type': 'newton', 'h_coeff_right': 10.0, 'T_env_right': 20.0,
            'plot_times': []
        }
        ref_solver, ref_t, ref_T = reference_solve_newton_cooling(params, save_plot=False)
        try:
            student_solver, student_t, student_T = student_solve_newton_cooling(params, save_plot=False)
            self.assertIsNotNone(student_solver, "Student solver object should not be None for Task 5")
            self.assertEqual(student_T.shape, ref_T.shape, "Student T_matrix shape mismatch for Task 5")
            np.testing.assert_allclose(student_T, ref_T, rtol=1e-5, atol=self.tolerance, err_msg="Student Task 5 (Newton's Cooling) results differ from reference.")
        except NotImplementedError:
            self.fail("Student solve_heat_diffusion_newton_cooling for Task 5 is not implemented.")
        except Exception as e:
            self.fail(f"Student Task 5 (Newton's Cooling) failed: {e}")

# Remove or adapt old TestHeatDiffusionStudent and other classes if they are now redundant.
# For example, TestHeatDiffusionProblem1Student might be merged or removed.

# Keep TestAnalysisFunctions if still relevant for helper functions like stability, errors.
class TestAnalysisFunctions(unittest.TestCase):
    """
    Tests for analysis helper functions (stability, errors).
    These might need to be adapted if the main solver functions change significantly.
    """
    def setUp(self):
        self.params_stable = {
            'L': 0.1, 'nx': 21, 'dt': 0.01, # total_time removed as it's not direct input to analyze_stability
            'K': 237.0, 'C': 900.0, 'rho': 2700.0,
            # Other params for a full solver run if needed by student's analyze_stability
            'initial_condition_type': 'constant', 'T0_val': 100.0,
            'boundary_condition_left_type': 'dirichlet', 'bc_left_val': 0.0,
            'boundary_condition_right_type': 'dirichlet', 'bc_right_val': 0.0,
        }
        self.params_unstable = self.params_stable.copy()
        # Calculate alpha for stability condition: alpha = K / (C * rho)
        alpha = self.params_stable['K'] / (self.params_stable['C'] * self.params_stable['rho'])
        dx = self.params_stable['L'] / (self.params_stable['nx'] -1)
        # Make dt clearly unstable: dt > dx^2 / (2*alpha)
        self.params_unstable['dt'] = (dx**2 / (2*alpha)) * 1.5 

        if StudentSolver is None or student_analyze_stability is None:
            self.skipTest("StudentSolver or student_analyze_stability not available.")

    def test_student_analyze_stability_5pts(self):
        """Test student's stability analysis function (5 points)"""
        try:
            # Test stable case
            stable_flag_student, critical_dt_student = student_analyze_stability(self.params_stable)
            self.assertTrue(stable_flag_student, "Student's analyze_stability should report stable for stable parameters.")
            # Test unstable case
            unstable_flag_student, _ = student_analyze_stability(self.params_unstable)
            self.assertFalse(unstable_flag_student, "Student's analyze_stability should report unstable for unstable parameters.")
            
            # Compare critical_dt with reference if possible
            # alpha = K / (C * rho)
            # dx = L / (nx - 1)
            # critical_dt_ref = dx**2 / (2 * alpha)
            # self.assertAlmostEqual(critical_dt_student, critical_dt_ref, delta=1e-5, msg="Student's critical_dt differs from reference.")

        except NotImplementedError:
            self.fail("Student analyze_stability method not implemented.")
        except Exception as e:
            self.fail(f"Student analyze_stability failed: {e}")

    # Add tests for calculate_errors, compare_with_analytical (if these are separate, testable functions)
    # and demonstrate_instability if they are part of the student's required deliverables.

# The old TestHeatDiffusionProblem1Student might be entirely replaced by TestHeatDiffusionStudentTasks
# or specific tests from it can be merged if they test unique aspects not covered by the new tasks.
# For now, let's comment it out to avoid duplicate tests or conflicts.

# class TestHeatDiffusionProblem1Student(unittest.TestCase):
#     """
#     Tests for the specific solve_heat_diffusion_problem_1 function.
#     """
#     def setUp(self):
#         self.default_params = {
#             'L': 1.0, 'nx': 51, 'total_time': 100.0, 'dt': 0.1,
#             'K': 237.0, 'C': 900.0, 'rho': 2700.0,
#             'T0': 100.0, 'bc_left_val': 0.0, 'bc_right_val': 0.0,
#             'plot_times': [0, 10.0, 50.0, 100.0]
#         }
#         self.tolerance = 1e-9
#         if student_solve_problem_1 is None or reference_solve_problem_1 is None:
#             self.skipTest("student_solve_problem_1 or reference_solve_problem_1 not available.")

#     def test_reference_solve_problem_1(self): # 0 points
#         """Test reference solve_heat_diffusion_problem_1 for basic execution."""
#         try:
#             solver, t_array, T_matrix = reference_solve_problem_1(self.default_params, save_plot=False)
#             self.assertIsNotNone(solver)
#             self.assertEqual(T_matrix.shape[0], self.default_params['nx'])
#         except Exception as e:
#             self.fail(f"Reference solve_heat_diffusion_problem_1 failed: {e}")

#     def test_student_solve_problem_1_20pts(self):
#         """Test student's solve_heat_diffusion_problem_1 against reference (20 points)."""
#         ref_solver, ref_t_array, ref_T_matrix = reference_solve_problem_1(self.default_params, save_plot=False)
#         try:
#             student_solver, student_t_array, student_T_matrix = student_solve_problem_1(self.default_params, save_plot=False)
#             self.assertIsNotNone(student_solver, "Student solver object should not be None.")
#             self.assertEqual(student_T_matrix.shape, ref_T_matrix.shape, "Student T_matrix shape mismatch.")
#             np.testing.assert_allclose(student_T_matrix, ref_T_matrix, rtol=1e-5, atol=self.tolerance, err_msg="Student solve_heat_diffusion_problem_1 results differ from reference.")
#         except NotImplementedError:
#             self.fail("Student solve_heat_diffusion_problem_1 is not implemented.")
#         except Exception as e:
#             self.fail(f"Student solve_heat_diffusion_problem_1 failed: {e}")


if __name__ == '__main__':
    # Create a test suite
    suite = unittest.TestSuite()
    # Add tests from new task-based class
    suite.addTest(unittest.makeSuite(TestHeatDiffusionStudentTasks))
    # Add reference tests (optional, but good for sanity check during development)
    suite.addTest(unittest.makeSuite(TestHeatDiffusionReference))
    # Add analysis function tests if they are still relevant and separate
    suite.addTest(unittest.makeSuite(TestAnalysisFunctions))

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Optional: exit with a non-zero code if tests failed, for CI/CD
    if not result.wasSuccessful():
        sys.exit(1)


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

class TestHeatDiffusionProblem1Student(unittest.TestCase):
    """
    Tests for the solve_heat_diffusion_problem_1 function based on Markdown Problem 1.
    """
    def setUp(self):
        """Setup test environment"""
        self.tolerance = 1e-5 # Tolerance for comparing floating point numbers
        # Parameters from the markdown's first code block, used in solve_heat_diffusion_problem_1
        self.L_md = 1.0       # Length of aluminum rod (m)
        self.dx_md = 0.01   # Spatial step (m)
        self.dt_md = 0.5    # Time step (s)
        self.total_time_md = 1000.0 # s
        self.Nx_md = int(self.L_md / self.dx_md) + 1
        self.Nt_md = int(self.total_time_md / self.dt_md)

    def test_reference_solve_problem_1(self):
        """Verify reference solution for solve_heat_diffusion_problem_1 (0 points - validation)"""
        try:
            # Suppress plots during test
            import matplotlib.pyplot as plt
            plt.ioff()
            u_md_ref = reference_solve_problem_1()
            plt.ion()
            
            self.assertIsNotNone(u_md_ref, "Reference solution should return a matrix.")
            self.assertEqual(u_md_ref.shape, (self.Nx_md, self.Nt_md + 1), "Reference solution matrix shape is incorrect.")
            
            # Check boundary conditions
            np.testing.assert_array_almost_equal(u_md_ref[0, :], 0.0, decimal=6, err_msg="Reference BC at x=0 failed.")
            np.testing.assert_array_almost_equal(u_md_ref[-1, :], 0.0, decimal=6, err_msg="Reference BC at x=L failed.")
            
            # Check initial condition (excluding boundaries)
            np.testing.assert_array_almost_equal(u_md_ref[1:-1, 0], 100.0, decimal=6, err_msg="Reference IC failed.")
            self.assertAlmostEqual(u_md_ref[0,0], 0.0, msg="Reference IC at x=0, t=0 failed")
            self.assertAlmostEqual(u_md_ref[-1,0], 0.0, msg="Reference IC at x=L, t=0 failed")

            # Check a few points for expected behavior (e.g. center point cooling)
            center_x_idx = self.Nx_md // 2
            self.assertTrue(u_md_ref[center_x_idx, -1] < u_md_ref[center_x_idx, 0], "Center point should cool down.")
            self.assertTrue(np.all(u_md_ref[:, -1] <= 100.0) and np.all(u_md_ref[:, -1] >= 0.0), "Final temps out of expected range [0,100]")

        except Exception as e:
            self.fail(f"Reference solution for solve_heat_diffusion_problem_1 raised an exception: {e}")

    def test_student_solve_problem_1_20pts(self):
        """Test student's implementation of solve_heat_diffusion_problem_1 (20 points)"""
        if student_solve_problem_1 is None:
            self.skipTest("Student function solve_heat_diffusion_problem_1 not available or not imported.")
        
        try:
            # Suppress plots during test
            import matplotlib.pyplot as plt
            plt.ioff()
            u_md_student = student_solve_problem_1()
            plt.ion()

            self.assertIsNotNone(u_md_student, "Student solution should return a matrix.")
            self.assertEqual(u_md_student.shape, (self.Nx_md, self.Nt_md + 1), "Student solution matrix shape is incorrect.")

            # For student, we compare against the reference if available, or check basic properties
            # This assumes reference_solve_problem_1 works and is tested by test_reference_solve_problem_1
            plt.ioff() # Suppress plot from reference again if it's called here
            u_md_ref = reference_solve_problem_1()
            plt.ion()

            np.testing.assert_allclose(u_md_student, u_md_ref, rtol=self.tolerance, atol=self.tolerance,
                                        err_msg="Student solution does not match reference solution for Problem 1.")

        except NotImplementedError:
            self.fail("Student has not implemented solve_heat_diffusion_problem_1")
        except Exception as e:
            self.fail(f"Student's solve_heat_diffusion_problem_1 raised an exception: {e}")

if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add reference tests (for validation)
    suite.addTest(unittest.makeSuite(TestHeatDiffusionReference))
    
    # Add student tests (for grading)
    suite.addTest(unittest.makeSuite(TestHeatDiffusionStudent))
    suite.addTest(unittest.makeSuite(TestAnalysisFunctions))
    suite.addTest(unittest.makeSuite(TestHeatDiffusionProblem1Student)) # Add new test suite
    
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