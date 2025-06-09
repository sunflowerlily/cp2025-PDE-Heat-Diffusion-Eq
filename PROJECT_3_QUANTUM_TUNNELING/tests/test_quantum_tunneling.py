#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Module: Quantum Tunneling Effect Numerical Simulation
File: test_quantum_tunneling.py

Comprehensive unit tests for quantum tunneling solver implementation.
Tests both reference solution and student implementation.
"""

import unittest
import numpy as np
import sys
import os
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Import reference solution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'solution'))
from quantum_tunneling_solution import (
    QuantumTunnelingSolver, 
    compare_different_energies, 
    verify_conservation_laws,
    HBAR, M_E, EV_TO_AU
)

# Import student template
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from quantum_tunneling_student import (
    QuantumTunnelingSolver as StudentSolver,
    compare_different_energies as student_compare_energies,
    verify_conservation_laws as student_verify_conservation
)

class TestQuantumTunneling(unittest.TestCase):
    """Test cases for quantum tunneling simulation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_params = {
            'L': 20.0,
            'nx': 200,  # Smaller grid for faster testing
            't_final': 10.0,  # Shorter time for testing
            'nt': 400,
            'barrier_height': 0.18 * EV_TO_AU,
            'barrier_width': 2.0,
            'barrier_center': 10.0
        }
        
        self.tolerance = 1e-6
        self.loose_tolerance = 1e-3  # For numerical integration comparisons
    
    def test_reference_solver_initialization(self):
        """Test reference solver initialization (0 points - validation)"""
        solver = QuantumTunnelingSolver(**self.test_params)
        
        # Check basic parameters
        self.assertEqual(solver.L, self.test_params['L'])
        self.assertEqual(solver.nx, self.test_params['nx'])
        self.assertEqual(solver.nt, self.test_params['nt'])
        
        # Check grid setup
        expected_dx = self.test_params['L'] / (self.test_params['nx'] - 1)
        expected_dt = self.test_params['t_final'] / (self.test_params['nt'] - 1)
        
        self.assertAlmostEqual(solver.dx, expected_dx, places=10)
        self.assertAlmostEqual(solver.dt, expected_dt, places=10)
        
        # Check array shapes
        self.assertEqual(len(solver.x), self.test_params['nx'])
        self.assertEqual(len(solver.t), self.test_params['nt'])
        self.assertEqual(solver.psi.shape, (self.test_params['nx'], self.test_params['nt']))
    
    def test_reference_potential_setup(self):
        """Test reference potential function setup (0 points - validation)"""
        solver = QuantumTunnelingSolver(**self.test_params)
        V = solver.setup_potential()
        
        # Check potential array shape
        self.assertEqual(len(V), self.test_params['nx'])
        
        # Check barrier region
        barrier_left = self.test_params['barrier_center'] - self.test_params['barrier_width'] / 2
        barrier_right = self.test_params['barrier_center'] + self.test_params['barrier_width'] / 2
        
        # Find indices
        idx_left = np.argmin(np.abs(solver.x - barrier_left))
        idx_right = np.argmin(np.abs(solver.x - barrier_right))
        
        # Check potential values
        self.assertAlmostEqual(V[0], 0.0, places=10)  # Left boundary
        self.assertAlmostEqual(V[-1], 0.0, places=10)  # Right boundary
        self.assertAlmostEqual(V[idx_left:idx_right].max(), self.test_params['barrier_height'], places=10)
    
    def test_reference_initial_wavepacket(self):
        """Test reference initial wave packet setup (0 points - validation)"""
        solver = QuantumTunnelingSolver(**self.test_params)
        
        x0, sigma, k0 = 5.0, 1.0, 2.0
        psi_initial = solver.setup_initial_wavepacket(x0, sigma, k0)
        
        # Check normalization
        norm = np.trapz(np.abs(psi_initial)**2, solver.x)
        self.assertAlmostEqual(norm, 1.0, places=6)
        
        # Check wave packet center (approximately)
        prob_density = np.abs(psi_initial)**2
        center_of_mass = np.trapz(solver.x * prob_density, solver.x)
        self.assertAlmostEqual(center_of_mass, x0, places=1)
        
        # Check complex nature
        self.assertTrue(np.iscomplexobj(psi_initial))
    
    def test_reference_crank_nicolson_matrices(self):
        """Test reference Crank-Nicolson matrix construction (0 points - validation)"""
        solver = QuantumTunnelingSolver(**self.test_params)
        solver.V = solver.setup_potential()
        
        A_matrix, B_matrix = solver.build_crank_nicolson_matrices()
        
        # Check matrix shapes
        expected_shape = (self.test_params['nx'], self.test_params['nx'])
        self.assertEqual(A_matrix.shape, expected_shape)
        self.assertEqual(B_matrix.shape, expected_shape)
        
        # Check boundary conditions
        self.assertEqual(A_matrix[0, 0], 1.0)
        self.assertEqual(A_matrix[-1, -1], 1.0)
        self.assertEqual(B_matrix[0, 0], 1.0)
        self.assertEqual(B_matrix[-1, -1], 1.0)
    
    def test_student_solver_initialization_15pts(self):
        """Test student solver initialization (15 points)"""
        try:
            solver = StudentSolver(**self.test_params)
            
            # Check basic parameters
            self.assertEqual(solver.L, self.test_params['L'])
            self.assertEqual(solver.nx, self.test_params['nx'])
            self.assertEqual(solver.nt, self.test_params['nt'])
            
            # Check grid setup
            expected_dx = self.test_params['L'] / (self.test_params['nx'] - 1)
            expected_dt = self.test_params['t_final'] / (self.test_params['nt'] - 1)
            
            self.assertAlmostEqual(solver.dx, expected_dx, places=10)
            self.assertAlmostEqual(solver.dt, expected_dt, places=10)
            
            # Check array shapes
            self.assertEqual(len(solver.x), self.test_params['nx'])
            self.assertEqual(len(solver.t), self.test_params['nt'])
            self.assertEqual(solver.psi.shape, (self.test_params['nx'], self.test_params['nt']))
            
        except NotImplementedError:
            self.fail("Student has not implemented the __init__ method")
        except Exception as e:
            self.fail(f"Student implementation failed with error: {e}")
    
    def test_student_potential_setup_10pts(self):
        """Test student potential function setup (10 points)"""
        try:
            solver = StudentSolver(**self.test_params)
            V = solver.setup_potential()
            
            # Check potential array shape
            self.assertEqual(len(V), self.test_params['nx'])
            
            # Check barrier region
            barrier_left = self.test_params['barrier_center'] - self.test_params['barrier_width'] / 2
            barrier_right = self.test_params['barrier_center'] + self.test_params['barrier_width'] / 2
            
            # Find indices
            idx_left = np.argmin(np.abs(solver.x - barrier_left))
            idx_right = np.argmin(np.abs(solver.x - barrier_right))
            
            # Check potential values
            self.assertAlmostEqual(V[0], 0.0, places=10)
            self.assertAlmostEqual(V[-1], 0.0, places=10)
            self.assertAlmostEqual(V[idx_left:idx_right].max(), self.test_params['barrier_height'], places=10)
            
        except NotImplementedError:
            self.fail("Student has not implemented the setup_potential method")
        except Exception as e:
            self.fail(f"Student implementation failed with error: {e}")
    
    def test_student_initial_wavepacket_10pts(self):
        """Test student initial wave packet setup (10 points)"""
        try:
            solver = StudentSolver(**self.test_params)
            
            x0, sigma, k0 = 5.0, 1.0, 2.0
            psi_initial = solver.setup_initial_wavepacket(x0, sigma, k0)
            
            # Check normalization
            norm = np.trapz(np.abs(psi_initial)**2, solver.x)
            self.assertAlmostEqual(norm, 1.0, places=5)
            
            # Check wave packet center (approximately)
            prob_density = np.abs(psi_initial)**2
            center_of_mass = np.trapz(solver.x * prob_density, solver.x)
            self.assertAlmostEqual(center_of_mass, x0, places=1)
            
            # Check complex nature
            self.assertTrue(np.iscomplexobj(psi_initial))
            
        except NotImplementedError:
            self.fail("Student has not implemented the setup_initial_wavepacket method")
        except Exception as e:
            self.fail(f"Student implementation failed with error: {e}")
    
    def test_student_crank_nicolson_matrices_15pts(self):
        """Test student Crank-Nicolson matrix construction (15 points)"""
        try:
            solver = StudentSolver(**self.test_params)
            solver.V = solver.setup_potential()
            
            A_matrix, B_matrix = solver.build_crank_nicolson_matrices()
            
            # Check matrix shapes
            expected_shape = (self.test_params['nx'], self.test_params['nx'])
            self.assertEqual(A_matrix.shape, expected_shape)
            self.assertEqual(B_matrix.shape, expected_shape)
            
            # Check boundary conditions
            self.assertAlmostEqual(A_matrix[0, 0], 1.0, places=10)
            self.assertAlmostEqual(A_matrix[-1, -1], 1.0, places=10)
            self.assertAlmostEqual(B_matrix[0, 0], 1.0, places=10)
            self.assertAlmostEqual(B_matrix[-1, -1], 1.0, places=10)
            
            # Check matrix structure (tridiagonal)
            A_dense = A_matrix.toarray()
            B_dense = B_matrix.toarray()
            
            # Check that matrices are approximately tridiagonal (except boundaries)
            for i in range(1, self.test_params['nx']-1):
                for j in range(self.test_params['nx']):
                    if abs(i - j) > 1:
                        self.assertAlmostEqual(A_dense[i, j], 0.0, places=10)
                        self.assertAlmostEqual(B_dense[i, j], 0.0, places=10)
            
        except NotImplementedError:
            self.fail("Student has not implemented the build_crank_nicolson_matrices method")
        except Exception as e:
            self.fail(f"Student implementation failed with error: {e}")
    
    def test_student_time_evolution_20pts(self):
        """Test student time evolution solver (20 points)"""
        try:
            # Use smaller problem for faster testing
            small_params = self.test_params.copy()
            small_params.update({'nx': 100, 'nt': 200, 't_final': 5.0})
            
            solver = StudentSolver(**small_params)
            solver.V = solver.setup_potential()
            
            # Solve time evolution
            psi_result = solver.solve_time_evolution()
            
            # Check result shape
            self.assertEqual(psi_result.shape, (small_params['nx'], small_params['nt']))
            
            # Check that wavefunction is complex
            self.assertTrue(np.iscomplexobj(psi_result))
            
            # Check boundary conditions
            for n in range(small_params['nt']):
                self.assertAlmostEqual(abs(psi_result[0, n]), 0.0, places=10)
                self.assertAlmostEqual(abs(psi_result[-1, n]), 0.0, places=10)
            
            # Check approximate probability conservation
            for n in range(0, small_params['nt'], 50):
                prob_density = np.abs(psi_result[:, n])**2
                total_prob = np.trapz(prob_density, solver.x)
                self.assertAlmostEqual(total_prob, 1.0, places=2)
            
        except NotImplementedError:
            self.fail("Student has not implemented the solve_time_evolution method")
        except Exception as e:
            self.fail(f"Student implementation failed with error: {e}")
    
    def test_student_probability_calculations_10pts(self):
        """Test student probability density and current calculations (10 points)"""
        try:
            # Use very small problem for testing
            small_params = self.test_params.copy()
            small_params.update({'nx': 50, 'nt': 100, 't_final': 2.0})
            
            solver = StudentSolver(**small_params)
            solver.V = solver.setup_potential()
            
            # Set up a simple test wavefunction
            psi_test = np.zeros((small_params['nx'], small_params['nt']), dtype=complex)
            x0, sigma = 5.0, 1.0
            for n in range(small_params['nt']):
                psi_test[:, n] = np.exp(-(solver.x - x0)**2 / (2 * sigma**2))
                norm = np.sqrt(np.trapz(np.abs(psi_test[:, n])**2, solver.x))
                psi_test[:, n] /= norm
            
            solver.psi = psi_test
            
            # Test probability density calculation
            prob_density = solver.calculate_probability_density()
            self.assertEqual(prob_density.shape, (small_params['nx'], small_params['nt']))
            self.assertTrue(np.all(prob_density >= 0))
            
            # Test probability current calculation
            prob_current = solver.calculate_probability_current()
            self.assertEqual(prob_current.shape, (small_params['nx'], small_params['nt']))
            self.assertTrue(np.all(np.isreal(prob_current)))
            
        except NotImplementedError:
            self.fail("Student has not implemented probability calculation methods")
        except Exception as e:
            self.fail(f"Student implementation failed with error: {e}")
    
    def test_student_tunneling_analysis_15pts(self):
        """Test student tunneling effect analysis (15 points)"""
        try:
            # Use reference solution to generate test data
            ref_solver = QuantumTunnelingSolver(**self.test_params)
            ref_solver.V = ref_solver.setup_potential()
            ref_solver.solve_time_evolution()
            
            # Test student analysis with reference data
            student_solver = StudentSolver(**self.test_params)
            student_solver.V = ref_solver.V
            student_solver.psi = ref_solver.psi
            student_solver.x = ref_solver.x
            student_solver.t = ref_solver.t
            
            # Analyze tunneling effect
            results = student_solver.analyze_tunneling_effect()
            
            # Check result structure
            required_keys = ['transmission_coefficient', 'reflection_coefficient', 
                           'barrier_probability', 'total_probability']
            for key in required_keys:
                self.assertIn(key, results)
            
            # Check physical constraints
            self.assertGreaterEqual(results['transmission_coefficient'], 0.0)
            self.assertLessEqual(results['transmission_coefficient'], 1.0)
            self.assertGreaterEqual(results['reflection_coefficient'], 0.0)
            self.assertLessEqual(results['reflection_coefficient'], 1.0)
            
            # Check approximate probability conservation
            total_prob = results['total_probability']
            self.assertAlmostEqual(total_prob, 1.0, places=1)
            
        except NotImplementedError:
            self.fail("Student has not implemented the analyze_tunneling_effect method")
        except Exception as e:
            self.fail(f"Student implementation failed with error: {e}")
    
    def test_student_visualization_methods_5pts(self):
        """Test student visualization methods (5 points)"""
        try:
            # Use very small problem for testing
            small_params = self.test_params.copy()
            small_params.update({'nx': 50, 'nt': 100})
            
            solver = StudentSolver(**small_params)
            solver.V = solver.setup_potential()
            
            # Set up simple test data
            solver.psi = np.random.random((small_params['nx'], small_params['nt'])) + \
                        1j * np.random.random((small_params['nx'], small_params['nt']))
            
            # Test plot method (should not raise exception)
            try:
                solver.plot_wavefunction_evolution(time_indices=[0, 50, 99], save_fig=False)
            except Exception as e:
                # Plotting might fail in headless environment, but method should exist
                if "display" not in str(e).lower():
                    raise e
            
        except NotImplementedError:
            self.fail("Student has not implemented visualization methods")
        except AttributeError:
            self.fail("Student visualization methods not found")
    
    def test_student_full_simulation_10pts(self):
        """Test student complete simulation workflow (10 points)"""
        try:
            # Use small parameters for faster testing
            small_params = self.test_params.copy()
            small_params.update({'nx': 100, 'nt': 200, 't_final': 5.0})
            
            solver = StudentSolver(**small_params)
            
            # Run full simulation
            results = solver.run_full_simulation()
            
            # Check result structure
            required_sections = ['solver_parameters', 'physical_parameters', 'tunneling_analysis']
            for section in required_sections:
                self.assertIn(section, results)
            
            # Check tunneling analysis results
            tunneling_results = results['tunneling_analysis']
            self.assertIn('transmission_coefficient', tunneling_results)
            self.assertIn('reflection_coefficient', tunneling_results)
            
            # Check physical reasonableness
            transmission = tunneling_results['transmission_coefficient']
            reflection = tunneling_results['reflection_coefficient']
            
            self.assertGreaterEqual(transmission, 0.0)
            self.assertLessEqual(transmission, 1.0)
            self.assertGreaterEqual(reflection, 0.0)
            self.assertLessEqual(reflection, 1.0)
            
        except NotImplementedError:
            self.fail("Student has not implemented the run_full_simulation method")
        except Exception as e:
            self.fail(f"Student implementation failed with error: {e}")
    
    def test_student_energy_comparison_5pts(self):
        """Test student energy comparison function (5 points)"""
        try:
            # Test with small energy range
            energies = [0.1 * EV_TO_AU, 0.15 * EV_TO_AU, 0.2 * EV_TO_AU]
            
            results = student_compare_energies(energies, barrier_height=0.18*EV_TO_AU)
            
            # Check result structure
            self.assertIn('energies', results)
            self.assertIn('transmission_coefficients', results)
            
            # Check data consistency
            self.assertEqual(len(results['energies']), len(energies))
            self.assertEqual(len(results['transmission_coefficients']), len(energies))
            
            # Check physical reasonableness
            for coeff in results['transmission_coefficients']:
                self.assertGreaterEqual(coeff, 0.0)
                self.assertLessEqual(coeff, 1.0)
            
        except NotImplementedError:
            self.fail("Student has not implemented the compare_different_energies function")
        except Exception as e:
            self.fail(f"Student implementation failed with error: {e}")
    
    def test_student_conservation_verification_5pts(self):
        """Test student conservation law verification (5 points)"""
        try:
            # Use reference solution for test data
            ref_solver = QuantumTunnelingSolver(**self.test_params)
            ref_solver.V = ref_solver.setup_potential()
            ref_solver.solve_time_evolution()
            
            # Test student verification
            results = student_verify_conservation(ref_solver)
            
            # Check result structure
            self.assertIn('probability_conservation', results)
            self.assertIn('continuity_equation', results)
            
            # Check probability conservation
            prob_conservation = results['probability_conservation']
            self.assertIn('mean_total_probability', prob_conservation)
            self.assertIn('max_deviation', prob_conservation)
            
            # Check continuity equation
            continuity = results['continuity_equation']
            self.assertIn('max_error', continuity)
            self.assertIn('rms_error', continuity)
            
        except NotImplementedError:
            self.fail("Student has not implemented the verify_conservation_laws function")
        except Exception as e:
            self.fail(f"Student implementation failed with error: {e}")

def run_all_tests_and_calculate_score():
    """Run all tests and calculate total score"""
    # Test mapping: test_method -> points
    test_scores = {
        'test_student_solver_initialization_15pts': 15,
        'test_student_potential_setup_10pts': 10,
        'test_student_initial_wavepacket_10pts': 10,
        'test_student_crank_nicolson_matrices_15pts': 15,
        'test_student_time_evolution_20pts': 20,
        'test_student_probability_calculations_10pts': 10,
        'test_student_tunneling_analysis_15pts': 15,
        'test_student_visualization_methods_5pts': 5,
        'test_student_full_simulation_10pts': 10,
        'test_student_energy_comparison_5pts': 5,
        'test_student_conservation_verification_5pts': 5
    }
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add student tests
    for test_name in test_scores.keys():
        suite.addTest(TestQuantumTunneling(test_name))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Calculate score
    total_possible = sum(test_scores.values())
    total_earned = 0
    
    print("\n" + "="*60)
    print("QUANTUM TUNNELING PROJECT SCORING REPORT")
    print("="*60)
    
    for test_name, points in test_scores.items():
        # Check if test passed
        test_passed = True
        for failure in result.failures + result.errors:
            if test_name in str(failure[0]):
                test_passed = False
                break
        
        if test_passed:
            total_earned += points
            status = "PASS"
        else:
            status = "FAIL"
        
        print(f"{test_name:<50} {points:>3} pts [{status}]")
    
    print("-" * 60)
    print(f"Total Score: {total_earned}/{total_possible} points ({total_earned/total_possible*100:.1f}%)")
    print("="*60)
    
    return total_earned, total_possible

if __name__ == '__main__':
    # Run individual test or full scoring
    if len(sys.argv) > 1 and sys.argv[1] == '--score':
        run_all_tests_and_calculate_score()
    else:
        unittest.main(verbosity=2)