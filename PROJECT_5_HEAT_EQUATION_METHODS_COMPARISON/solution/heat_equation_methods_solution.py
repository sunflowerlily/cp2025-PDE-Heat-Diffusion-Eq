#!/usr/bin/env python3
"""
Module: Heat Equation Methods Solution
File: heat_equation_methods_solution.py

Implements 6 different numerical methods for solving 1D heat equation:
1. FTCS (Forward Time Central Space)
2. Laplace operator explicit method
3. BTCS (Backward Time Central Space)
4. Crank-Nicolson method
5. Modified Crank-Nicolson method
6. solve_ivp method
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve_banded
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import time
from typing import Callable, Tuple, Dict, List

class heat_equation_solver:
    """
    Solver for 1D heat equation using multiple numerical methods
    
    Solves: u_t = a^2 * u_xx with boundary conditions u(0,t) = u(L,t) = 0
    """
    
    def __init__(self, L: float = 20.0, T: float = 25.0, a_squared: float = 10.0, 
                 nx: int = 100, nt: int = 1000):
        """
        Initialize the heat equation solver
        
        Args:
            L: Length of the rod
            T: Total time
            a_squared: Thermal diffusivity coefficient
            nx: Number of spatial grid points
            nt: Number of time steps
        """
        self.L = L
        self.T = T
        self.a_squared = a_squared
        self.nx = nx
        self.nt = nt
        
        # Grid setup
        self.dx = L / (nx - 1)
        self.dt = T / nt
        self.x = np.linspace(0, L, nx)
        self.t = np.linspace(0, T, nt + 1)
        
        # Stability parameter
        self.r = a_squared * self.dt / (self.dx**2)
        
        # Initial condition function
        self.phi_func = None
        self.u0 = None
        
        print(f"Grid setup: nx={nx}, nt={nt}, dx={self.dx:.4f}, dt={self.dt:.4f}")
        print(f"Stability parameter r = {self.r:.4f}")
        if self.r > 0.5:
            print("WARNING: r > 0.5, explicit methods may be unstable!")
    
    def set_initial_condition(self, phi_func: Callable[[np.ndarray], np.ndarray]):
        """
        Set the initial condition function
        
        Args:
            phi_func: Function that takes x array and returns initial temperature
        """
        self.phi_func = phi_func
        self.u0 = phi_func(self.x)
        # Apply boundary conditions
        self.u0[0] = 0
        self.u0[-1] = 0
    
    def ftcs_method(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward Time Central Space (FTCS) explicit method
        
        Returns:
            Tuple of (time_array, solution_matrix)
        """
        if self.u0 is None:
            raise ValueError("Initial condition not set")
        
        u = np.zeros((self.nt + 1, self.nx))
        u[0] = self.u0.copy()
        
        for n in range(self.nt):
            for i in range(1, self.nx - 1):
                u[n+1, i] = u[n, i] + self.r * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
            # Boundary conditions
            u[n+1, 0] = 0
            u[n+1, -1] = 0
        
        return self.t, u
    
    def laplace_explicit_method(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Explicit method using Laplace operator matrix
        
        Returns:
            Tuple of (time_array, solution_matrix)
        """
        if self.u0 is None:
            raise ValueError("Initial condition not set")
        
        # Create Laplace operator matrix
        diagonals = [np.ones(self.nx-1), -2*np.ones(self.nx), np.ones(self.nx-1)]
        L_matrix = diags(diagonals, [-1, 0, 1], shape=(self.nx, self.nx)).toarray()
        L_matrix = L_matrix / (self.dx**2)
        
        # Apply boundary conditions to matrix
        L_matrix[0, :] = 0
        L_matrix[-1, :] = 0
        
        u = np.zeros((self.nt + 1, self.nx))
        u[0] = self.u0.copy()
        
        for n in range(self.nt):
            u[n+1] = u[n] + self.a_squared * self.dt * L_matrix @ u[n]
            # Ensure boundary conditions
            u[n+1, 0] = 0
            u[n+1, -1] = 0
        
        return self.t, u
    
    def btcs_method(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward Time Central Space (BTCS) implicit method
        
        Returns:
            Tuple of (time_array, solution_matrix)
        """
        if self.u0 is None:
            raise ValueError("Initial condition not set")
        
        # Create coefficient matrix for implicit scheme
        # (I - r*L) * u^{n+1} = u^n
        main_diag = np.ones(self.nx) * (1 + 2*self.r)
        upper_diag = np.ones(self.nx-1) * (-self.r)
        lower_diag = np.ones(self.nx-1) * (-self.r)
        
        # Apply boundary conditions
        main_diag[0] = 1
        main_diag[-1] = 1
        upper_diag[0] = 0
        lower_diag[-1] = 0
        
        # Create banded matrix for solve_banded
        ab = np.zeros((3, self.nx))
        ab[0, 1:] = upper_diag
        ab[1, :] = main_diag
        ab[2, :-1] = lower_diag
        
        u = np.zeros((self.nt + 1, self.nx))
        u[0] = self.u0.copy()
        
        for n in range(self.nt):
            rhs = u[n].copy()
            rhs[0] = 0  # Boundary condition
            rhs[-1] = 0  # Boundary condition
            
            u[n+1] = solve_banded((1, 1), ab, rhs)
            # Ensure boundary conditions
            u[n+1, 0] = 0
            u[n+1, -1] = 0
        
        return self.t, u
    
    def crank_nicolson_method(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crank-Nicolson method (theta = 0.5)
        
        Returns:
            Tuple of (time_array, solution_matrix)
        """
        if self.u0 is None:
            raise ValueError("Initial condition not set")
        
        r_half = self.r / 2
        
        # Left hand side matrix: (I + r/2 * L)
        main_diag_lhs = np.ones(self.nx) * (1 + r_half)
        upper_diag_lhs = np.ones(self.nx-1) * (-r_half/2)
        lower_diag_lhs = np.ones(self.nx-1) * (-r_half/2)
        
        # Right hand side matrix: (I - r/2 * L)
        main_diag_rhs = np.ones(self.nx) * (1 - r_half)
        upper_diag_rhs = np.ones(self.nx-1) * (r_half/2)
        lower_diag_rhs = np.ones(self.nx-1) * (r_half/2)
        
        # Apply boundary conditions
        main_diag_lhs[0] = main_diag_lhs[-1] = 1
        upper_diag_lhs[0] = lower_diag_lhs[-1] = 0
        main_diag_rhs[0] = main_diag_rhs[-1] = 1
        upper_diag_rhs[0] = lower_diag_rhs[-1] = 0
        
        # Create banded matrix for LHS
        ab_lhs = np.zeros((3, self.nx))
        ab_lhs[0, 1:] = upper_diag_lhs
        ab_lhs[1, :] = main_diag_lhs
        ab_lhs[2, :-1] = lower_diag_lhs
        
        # Create RHS matrix
        rhs_matrix = diags([lower_diag_rhs, main_diag_rhs, upper_diag_rhs], 
                          [-1, 0, 1], shape=(self.nx, self.nx))
        
        u = np.zeros((self.nt + 1, self.nx))
        u[0] = self.u0.copy()
        
        for n in range(self.nt):
            rhs = rhs_matrix @ u[n]
            rhs[0] = 0  # Boundary condition
            rhs[-1] = 0  # Boundary condition
            
            u[n+1] = solve_banded((1, 1), ab_lhs, rhs)
            # Ensure boundary conditions
            u[n+1, 0] = 0
            u[n+1, -1] = 0
        
        return self.t, u
    
    def modified_crank_nicolson_method(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Modified Crank-Nicolson method with theta = 0.6 for better stability
        
        Returns:
            Tuple of (time_array, solution_matrix)
        """
        if self.u0 is None:
            raise ValueError("Initial condition not set")
        
        theta = 0.6  # Modified parameter
        
        # Left hand side matrix: (I + theta*r*L)
        main_diag_lhs = np.ones(self.nx) * (1 + theta*self.r)
        upper_diag_lhs = np.ones(self.nx-1) * (-theta*self.r/2)
        lower_diag_lhs = np.ones(self.nx-1) * (-theta*self.r/2)
        
        # Right hand side matrix: (I - (1-theta)*r*L)
        main_diag_rhs = np.ones(self.nx) * (1 - (1-theta)*self.r)
        upper_diag_rhs = np.ones(self.nx-1) * ((1-theta)*self.r/2)
        lower_diag_rhs = np.ones(self.nx-1) * ((1-theta)*self.r/2)
        
        # Apply boundary conditions
        main_diag_lhs[0] = main_diag_lhs[-1] = 1
        upper_diag_lhs[0] = lower_diag_lhs[-1] = 0
        main_diag_rhs[0] = main_diag_rhs[-1] = 1
        upper_diag_rhs[0] = lower_diag_rhs[-1] = 0
        
        # Create banded matrix for LHS
        ab_lhs = np.zeros((3, self.nx))
        ab_lhs[0, 1:] = upper_diag_lhs
        ab_lhs[1, :] = main_diag_lhs
        ab_lhs[2, :-1] = lower_diag_lhs
        
        # Create RHS matrix
        rhs_matrix = diags([lower_diag_rhs, main_diag_rhs, upper_diag_rhs], 
                          [-1, 0, 1], shape=(self.nx, self.nx))
        
        u = np.zeros((self.nt + 1, self.nx))
        u[0] = self.u0.copy()
        
        for n in range(self.nt):
            rhs = rhs_matrix @ u[n]
            rhs[0] = 0  # Boundary condition
            rhs[-1] = 0  # Boundary condition
            
            u[n+1] = solve_banded((1, 1), ab_lhs, rhs)
            # Ensure boundary conditions
            u[n+1, 0] = 0
            u[n+1, -1] = 0
        
        return self.t, u
    
    def solve_ivp_method(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use scipy.integrate.solve_ivp to solve the heat equation as ODE system
        
        Returns:
            Tuple of (time_array, solution_matrix)
        """
        if self.u0 is None:
            raise ValueError("Initial condition not set")
        
        def heat_ode(t, u):
            """
            ODE system for heat equation: du/dt = a^2 * d^2u/dx^2
            """
            dudt = np.zeros_like(u)
            
            # Interior points
            for i in range(1, len(u) - 1):
                dudt[i] = self.a_squared * (u[i+1] - 2*u[i] + u[i-1]) / (self.dx**2)
            
            # Boundary conditions
            dudt[0] = 0
            dudt[-1] = 0
            
            return dudt
        
        # Solve ODE system
        sol = solve_ivp(heat_ode, [0, self.T], self.u0, 
                       t_eval=self.t, method='RK45', rtol=1e-8)
        
        return sol.t, sol.y.T
    
    def analytical_solution(self, x: np.ndarray, t: float, n_terms: int = 100) -> np.ndarray:
        """
        Analytical solution using Fourier series (for validation)
        
        Args:
            x: Spatial coordinates
            t: Time
            n_terms: Number of Fourier terms
            
        Returns:
            Analytical solution at given x and t
        """
        u_analytical = np.zeros_like(x)
        
        for n in range(1, n_terms + 1):
            # Fourier coefficient for the given initial condition
            # phi(x) = 1 for 10 <= x <= 11, 0 otherwise
            if n % 2 == 0:  # Even terms are zero for this symmetric problem
                continue
                
            # Calculate Fourier coefficient
            bn = (2/self.L) * (np.sin(n*np.pi*11/self.L) - np.sin(n*np.pi*10/self.L)) / (n*np.pi/self.L)
            
            # Add term to solution
            lambda_n = (n * np.pi / self.L)**2
            u_analytical += bn * np.sin(n * np.pi * x / self.L) * np.exp(-self.a_squared * lambda_n * t)
        
        return u_analytical
    
    def compare_methods(self) -> Dict[str, Dict]:
        """
        Compare all 6 methods in terms of accuracy and computational time
        
        Returns:
            Dictionary containing results and timing for each method
        """
        if self.u0 is None:
            raise ValueError("Initial condition not set")
        
        methods = {
            'FTCS': self.ftcs_method,
            'Laplace_Explicit': self.laplace_explicit_method,
            'BTCS': self.btcs_method,
            'Crank_Nicolson': self.crank_nicolson_method,
            'Modified_CN': self.modified_crank_nicolson_method,
            'solve_ivp': self.solve_ivp_method
        }
        
        results = {}
        
        for name, method in methods.items():
            print(f"Running {name}...")
            start_time = time.time()
            
            try:
                t_array, u_solution = method()
                end_time = time.time()
                
                # Calculate error against analytical solution at final time
                u_analytical_final = self.analytical_solution(self.x, self.T)
                u_numerical_final = u_solution[-1]
                
                l2_error = np.sqrt(np.sum((u_numerical_final - u_analytical_final)**2) * self.dx)
                max_error = np.max(np.abs(u_numerical_final - u_analytical_final))
                
                results[name] = {
                    'time_array': t_array,
                    'solution': u_solution,
                    'computation_time': end_time - start_time,
                    'l2_error': l2_error,
                    'max_error': max_error,
                    'final_solution': u_numerical_final,
                    'status': 'success'
                }
                
                print(f"  Completed in {end_time - start_time:.4f}s")
                print(f"  L2 error: {l2_error:.6e}")
                print(f"  Max error: {max_error:.6e}")
                
            except Exception as e:
                print(f"  Failed: {str(e)}")
                results[name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return results
    
    def plot_comparison(self, results: Dict[str, Dict], save_fig: bool = True):
        """
        Plot comparison of all methods
        
        Args:
            results: Results from compare_methods()
            save_fig: Whether to save the figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot analytical solution for reference
        u_analytical_final = self.analytical_solution(self.x, self.T)
        
        successful_methods = [name for name, result in results.items() 
                            if result.get('status') == 'success']
        
        for i, method_name in enumerate(successful_methods[:6]):
            if i >= 6:
                break
                
            result = results[method_name]
            
            axes[i].plot(self.x, u_analytical_final, 'r-', linewidth=2, 
                        label='Analytical', alpha=0.7)
            axes[i].plot(self.x, result['final_solution'], 'b--', linewidth=2, 
                        label=method_name)
            
            axes[i].set_title(f'{method_name}\nL2 Error: {result["l2_error"]:.2e}\n'
                            f'Time: {result["computation_time"]:.3f}s')
            axes[i].set_xlabel('x')
            axes[i].set_ylabel('Temperature')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(successful_methods), 6):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig('heat_equation_methods_comparison.png', dpi=300, bbox_inches='tight')
            print("Comparison plot saved as 'heat_equation_methods_comparison.png'")
        
        plt.show()
    
    def plot_error_analysis(self, results: Dict[str, Dict], save_fig: bool = True):
        """
        Plot error analysis for all methods
        
        Args:
            results: Results from compare_methods()
            save_fig: Whether to save the figure
        """
        successful_methods = [name for name, result in results.items() 
                            if result.get('status') == 'success']
        
        method_names = []
        l2_errors = []
        max_errors = []
        comp_times = []
        
        for method_name in successful_methods:
            result = results[method_name]
            method_names.append(method_name)
            l2_errors.append(result['l2_error'])
            max_errors.append(result['max_error'])
            comp_times.append(result['computation_time'])
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # L2 Error comparison
        axes[0].bar(method_names, l2_errors, color='skyblue', alpha=0.7)
        axes[0].set_ylabel('L2 Error')
        axes[0].set_title('L2 Error Comparison')
        axes[0].set_yscale('log')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # Max Error comparison
        axes[1].bar(method_names, max_errors, color='lightcoral', alpha=0.7)
        axes[1].set_ylabel('Max Error')
        axes[1].set_title('Maximum Error Comparison')
        axes[1].set_yscale('log')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        # Computation Time comparison
        axes[2].bar(method_names, comp_times, color='lightgreen', alpha=0.7)
        axes[2].set_ylabel('Computation Time (s)')
        axes[2].set_title('Computation Time Comparison')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig('heat_equation_error_analysis.png', dpi=300, bbox_inches='tight')
            print("Error analysis plot saved as 'heat_equation_error_analysis.png'")
        
        plt.show()


def initial_condition_function(x: np.ndarray) -> np.ndarray:
    """
    Initial condition: phi(x) = 1 for 10 <= x <= 11, 0 otherwise
    
    Args:
        x: Spatial coordinate array
        
    Returns:
        Initial temperature distribution
    """
    phi = np.zeros_like(x)
    mask = (x >= 10) & (x <= 11)
    phi[mask] = 1.0
    return phi


if __name__ == "__main__":
    # Problem parameters
    L = 20.0      # Rod length
    T = 25.0      # Total time
    a_squared = 10.0  # Thermal diffusivity
    nx = 100      # Spatial grid points
    nt = 1000     # Time steps
    
    print("=" * 60)
    print("Heat Equation Solver - Multiple Methods Comparison")
    print("=" * 60)
    
    # Create solver
    solver = heat_equation_solver(L, T, a_squared, nx, nt)
    
    # Set initial condition
    solver.set_initial_condition(initial_condition_function)
    
    print("\nInitial condition set: phi(x) = 1 for 10 <= x <= 11, 0 otherwise")
    print(f"Stability parameter r = {solver.r:.4f}")
    
    # Compare all methods
    print("\nRunning comparison of all 6 methods...")
    results = solver.compare_methods()
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY OF RESULTS")
    print("=" * 60)
    
    successful_methods = [(name, result) for name, result in results.items() 
                         if result.get('status') == 'success']
    
    if successful_methods:
        # Sort by L2 error
        successful_methods.sort(key=lambda x: x[1]['l2_error'])
        
        print(f"{'Method':<20} {'L2 Error':<12} {'Max Error':<12} {'Time (s)':<10}")
        print("-" * 60)
        
        for name, result in successful_methods:
            print(f"{name:<20} {result['l2_error']:<12.2e} "
                  f"{result['max_error']:<12.2e} {result['computation_time']:<10.3f}")
        
        print(f"\nMost accurate method: {successful_methods[0][0]}")
        print(f"Fastest method: {min(successful_methods, key=lambda x: x[1]['computation_time'])[0]}")
    
    # Plot results
    print("\nGenerating comparison plots...")
    solver.plot_comparison(results)
    solver.plot_error_analysis(results)
    
    print("\nAnalysis complete!")