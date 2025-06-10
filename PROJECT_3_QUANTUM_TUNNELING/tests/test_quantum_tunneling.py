import unittest
import numpy as np
import sys
import os

# Import reference solution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'solution'))
from quantum_tunneling_solution import QuantumTunnelingSolver

# Import student template
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from quantum_tunneling_student import QuantumTunnelingSolver as StudentSolver

class TestQuantumTunneling(unittest.TestCase):
    def setUp(self):
        """Set up test parameters"""
        self.m = 220
        self.N = 100  # Reduced for faster testing
        self.dt = 1.0
        self.dx = 1.0
        
        # Create reference solver
        self.ref_solver = QuantumTunnelingSolver(self.m, self.N, self.dt, self.dx)
        
        # Test parameters for wave packet
        self.x0 = 40
        self.k0 = 0.5
        self.d = 10
    
    def test_reference_solution(self):
        """Verify reference solution works (0 points - validation)"""
        # Test initialization
        self.assertEqual(self.ref_solver.m, self.m)
        self.assertEqual(self.ref_solver.N, self.N)
        
        # Test Gaussian wave packet generation
        psi_initial = self.ref_solver.gaussian_wave_packet(self.x0, self.k0, self.d)
        self.assertEqual(len(psi_initial), self.m)
        self.assertTrue(np.iscomplexobj(psi_initial))
        
        # Test solving
        self.ref_solver.solve(self.x0, self.k0, self.d)
        self.assertIsNotNone(self.ref_solver.psi)
        
        # Test probability density calculation
        prob_density = self.ref_solver.get_probability_density()
        self.assertEqual(prob_density.shape, (self.m, self.N))
        self.assertTrue(np.all(prob_density >= 0))
        
        # Test transmission coefficient
        T = self.ref_solver.calculate_transmission_coefficient()
        self.assertIsInstance(T, float)
        self.assertTrue(0 <= T <= 1)
        
        # Test reflection coefficient
        R = self.ref_solver.calculate_reflection_coefficient()
        self.assertIsInstance(R, float)
        self.assertTrue(0 <= R <= 1)
        
        # Test probability conservation
        total_probs = self.ref_solver.verify_probability_conservation()
        self.assertEqual(len(total_probs), self.N)
        # Check conservation (should be close to 1)
        self.assertTrue(np.all(np.abs(total_probs - 1.0) < 0.1))
    
    def test_student_initialization_5pts(self):
        """Test student solver initialization (5 points)"""
        try:
            student_solver = StudentSolver(self.m, self.N, self.dt, self.dx)
            self.assertEqual(student_solver.m, self.m)
            self.assertEqual(student_solver.N, self.N)
            self.assertEqual(student_solver.dt, self.dt)
            self.assertEqual(student_solver.dx, self.dx)
        except NotImplementedError:
            self.fail("Student has not implemented initialization")
    
    def test_student_gaussian_wave_packet_10pts(self):
        """Test student Gaussian wave packet generation (10 points)"""
        try:
            student_solver = StudentSolver(self.m, self.N, self.dt, self.dx)
            psi_student = student_solver.gaussian_wave_packet(self.x0, self.k0, self.d)
            psi_ref = self.ref_solver.gaussian_wave_packet(self.x0, self.k0, self.d)
            
            # Check shape and type
            self.assertEqual(len(psi_student), self.m)
            self.assertTrue(np.iscomplexobj(psi_student))
            
            # Check similarity to reference (allowing some numerical differences)
            np.testing.assert_allclose(psi_student, psi_ref, rtol=1e-10, atol=1e-12)
        except NotImplementedError:
            self.fail("Student has not implemented gaussian_wave_packet")
    
    def test_student_solve_15pts(self):
        """Test student solve method (15 points)"""
        try:
            student_solver = StudentSolver(self.m, self.N, self.dt, self.dx)
            student_solver.solve(self.x0, self.k0, self.d)
            
            # Check that psi array exists and has correct shape
            self.assertIsNotNone(student_solver.psi)
            self.assertEqual(student_solver.psi.shape, (self.m, self.N))
            self.assertTrue(np.iscomplexobj(student_solver.psi))
            
            # Check that solution is not trivial (not all zeros)
            self.assertFalse(np.allclose(student_solver.psi, 0))
            
        except NotImplementedError:
            self.fail("Student has not implemented solve method")
    
    def test_student_probability_density_5pts(self):
        """Test student probability density calculation (5 points)"""
        try:
            student_solver = StudentSolver(self.m, self.N, self.dt, self.dx)
            student_solver.solve(self.x0, self.k0, self.d)
            
            prob_density = student_solver.get_probability_density()
            self.assertEqual(prob_density.shape, (self.m, self.N))
            self.assertTrue(np.all(prob_density >= 0))
            
            # Test specific time index
            prob_t0 = student_solver.get_probability_density(0)
            self.assertEqual(len(prob_t0), self.m)
            self.assertTrue(np.all(prob_t0 >= 0))
            
        except NotImplementedError:
            self.fail("Student has not implemented get_probability_density")
    
    def test_student_transmission_coefficient_5pts(self):
        """Test student transmission coefficient calculation (5 points)"""
        try:
            student_solver = StudentSolver(self.m, self.N, self.dt, self.dx)
            student_solver.solve(self.x0, self.k0, self.d)
            
            T = student_solver.calculate_transmission_coefficient()
            self.assertIsInstance(T, float)
            self.assertTrue(0 <= T <= 1)
            
        except NotImplementedError:
            self.fail("Student has not implemented calculate_transmission_coefficient")
    
    def test_student_reflection_coefficient_5pts(self):
        """Test student reflection coefficient calculation (5 points)"""
        try:
            student_solver = StudentSolver(self.m, self.N, self.dt, self.dx)
            student_solver.solve(self.x0, self.k0, self.d)
            
            R = student_solver.calculate_reflection_coefficient()
            self.assertIsInstance(R, float)
            self.assertTrue(0 <= R <= 1)
            
        except NotImplementedError:
            self.fail("Student has not implemented calculate_reflection_coefficient")
    
    def test_student_probability_conservation_5pts(self):
        """Test student probability conservation verification (5 points)"""
        try:
            student_solver = StudentSolver(self.m, self.N, self.dt, self.dx)
            student_solver.solve(self.x0, self.k0, self.d)
            
            total_probs = student_solver.verify_probability_conservation()
            self.assertEqual(len(total_probs), self.N)
            # Check conservation (should be close to 1)
            self.assertTrue(np.all(np.abs(total_probs - 1.0) < 0.2))  # Allow larger tolerance
            
        except NotImplementedError:
            self.fail("Student has not implemented verify_probability_conservation")
    
    def test_student_plotting_methods_5pts(self):
        """Test student plotting methods exist (5 points)"""
        try:
            student_solver = StudentSolver(self.m, self.N, self.dt, self.dx)
            student_solver.solve(self.x0, self.k0, self.d)
            
            # Test that methods exist (don't actually create plots in tests)
            self.assertTrue(hasattr(student_solver, 'plot_evolution'))
            self.assertTrue(hasattr(student_solver, 'create_animation'))
            self.assertTrue(hasattr(student_solver, 'plot_probability_conservation'))
            
        except NotImplementedError:
            self.fail("Student has not implemented plotting methods")
    
    def test_student_matrix_building_5pts(self):
        """Test student matrix building method (5 points)"""
        try:
            student_solver = StudentSolver(self.m, self.N, self.dt, self.dx)
            
            # Test that _build_matrix method exists
            self.assertTrue(hasattr(student_solver, '_build_matrix'))
            
        except NotImplementedError:
            self.fail("Student has not implemented _build_matrix method")
    
    def test_student_demonstration_function_5pts(self):
        """Test student demonstration function exists (5 points)"""
        try:
            # Import the demonstration function
            from quantum_tunneling_student import demonstrate_quantum_tunneling
            
            # Test that function exists
            self.assertTrue(callable(demonstrate_quantum_tunneling))
            
        except ImportError:
            self.fail("Student has not implemented demonstrate_quantum_tunneling function")
        except NotImplementedError:
            self.fail("Student has not implemented demonstrate_quantum_tunneling function")

if __name__ == '__main__':
    unittest.main()