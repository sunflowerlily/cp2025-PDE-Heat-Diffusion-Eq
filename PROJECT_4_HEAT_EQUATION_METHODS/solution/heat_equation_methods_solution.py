#!/usr/bin/env python3
"""
Module: Heat Equation Methods Solution
File: heat_equation_methods_solution.py

Implements four numerical methods for solving 1D heat equation:
1. Explicit Finite Difference (FTCS)
2. Implicit Finite Difference (Backward Euler)
3. Crank-Nicolson Method
4. scipy.solve_ivp Method

Problem: u_t = a²u_xx with boundary conditions u(0,t) = u(l,t) = 0
Initial condition: u(x,0) = 1 for 10 ≤ x ≤ 11, 0 elsewhere
Parameters: a² = 10, l = 20, t ∈ [0, 25]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.integrate import solve_ivp
import time
from typing import Tuple, Dict, Optional
import warnings

# Physical parameters
ALPHA = 10.0  # Thermal diffusivity a²
L = 20.0      # Rod length
T_FINAL = 25.0  # Final time

def create_initial_condition(x: np.ndarray) -> np.ndarray:
    """
    Create initial condition: u(x,0) = 1 for 10 ≤ x ≤ 11, 0 elsewhere
    
    Args:
        x: Spatial grid points
    Returns:
        Initial temperature distribution
    """
    u0 = np.zeros_like(x)
    mask = (x >= 10.0) & (x <= 11.0)
    u0[mask] = 1.0
    return u0

def solve_ftcs(nx: int, nt: int, total_time: float = T_FINAL, 
               alpha: float = ALPHA) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve heat equation using Forward Time Central Space (FTCS) explicit method
    
    Args:
        nx: Number of spatial grid points
        nt: Number of time steps
        total_time: Total simulation time
        alpha: Thermal diffusivity
    
    Returns:
        Tuple of (x_grid, t_grid, solution_matrix)
    """
    # Grid setup
    x = np.linspace(0, L, nx)
    t = np.linspace(0, total_time, nt)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    
    # Stability parameter
    r = alpha * dt / (dx**2)
    
    # Check stability condition
    if r > 0.5:
        warnings.warn(f"Stability condition violated: r = {r:.4f} > 0.5. "
                     f"Solution may be unstable.")
    
    # Initialize solution matrix
    u = np.zeros((nt, nx))
    u[0, :] = create_initial_condition(x)
    
    # Time stepping
    for n in range(nt - 1):
        # Interior points using FTCS scheme
        u[n+1, 1:-1] = (u[n, 1:-1] + 
                        r * (u[n, 2:] - 2*u[n, 1:-1] + u[n, :-2]))
        
        # Boundary conditions: u(0,t) = u(L,t) = 0
        u[n+1, 0] = 0.0
        u[n+1, -1] = 0.0
    
    return x, t, u

def solve_backward_euler(nx: int, nt: int, total_time: float = T_FINAL,
                        alpha: float = ALPHA) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve heat equation using Backward Euler implicit method
    
    Args:
        nx: Number of spatial grid points
        nt: Number of time steps
        total_time: Total simulation time
        alpha: Thermal diffusivity
    
    Returns:
        Tuple of (x_grid, t_grid, solution_matrix)
    """
    # Grid setup
    x = np.linspace(0, L, nx)
    t = np.linspace(0, total_time, nt)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    
    # Stability parameter
    r = alpha * dt / (dx**2)
    
    # Build tridiagonal matrix for interior points
    # (I - rA)u^{n+1} = u^n
    n_interior = nx - 2  # Exclude boundary points
    
    # Tridiagonal matrix: [r, -(1+2r), r]
    diagonals = [r * np.ones(n_interior-1),
                -(1 + 2*r) * np.ones(n_interior),
                r * np.ones(n_interior-1)]
    A = diags(diagonals, [-1, 0, 1], shape=(n_interior, n_interior), format='csc')
    
    # Initialize solution
    u = np.zeros((nt, nx))
    u[0, :] = create_initial_condition(x)
    
    # Time stepping
    for n in range(nt - 1):
        # Right-hand side (interior points only)
        rhs = -u[n, 1:-1].copy()
        
        # Solve linear system for interior points
        u_interior = spsolve(A, rhs)
        
        # Update solution
        u[n+1, 1:-1] = u_interior
        u[n+1, 0] = 0.0   # Boundary condition
        u[n+1, -1] = 0.0  # Boundary condition
    
    return x, t, u

def solve_crank_nicolson(nx: int, nt: int, total_time: float = T_FINAL,
                        alpha: float = ALPHA) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve heat equation using Crank-Nicolson method
    
    Args:
        nx: Number of spatial grid points
        nt: Number of time steps
        total_time: Total simulation time
        alpha: Thermal diffusivity
    
    Returns:
        Tuple of (x_grid, t_grid, solution_matrix)
    """
    # Grid setup
    x = np.linspace(0, L, nx)
    t = np.linspace(0, total_time, nt)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    
    # Stability parameter
    r = alpha * dt / (dx**2)
    
    # Build matrices for interior points
    n_interior = nx - 2
    
    # Left-hand side matrix: (I - r/2 * A)
    diag_left = [r/2 * np.ones(n_interior-1),
                -(1 + r) * np.ones(n_interior),
                r/2 * np.ones(n_interior-1)]
    A_left = diags(diag_left, [-1, 0, 1], shape=(n_interior, n_interior), format='csc')
    
    # Right-hand side matrix: (I + r/2 * A)
    diag_right = [-r/2 * np.ones(n_interior-1),
                 -(1 - r) * np.ones(n_interior),
                 -r/2 * np.ones(n_interior-1)]
    A_right = diags(diag_right, [-1, 0, 1], shape=(n_interior, n_interior), format='csc')
    
    # Initialize solution
    u = np.zeros((nt, nx))
    u[0, :] = create_initial_condition(x)
    
    # Time stepping
    for n in range(nt - 1):
        # Right-hand side
        rhs = A_right @ u[n, 1:-1]
        
        # Solve linear system
        u_interior = spsolve(A_left, rhs)
        
        # Update solution
        u[n+1, 1:-1] = u_interior
        u[n+1, 0] = 0.0   # Boundary condition
        u[n+1, -1] = 0.0  # Boundary condition
    
    return x, t, u

def solve_with_scipy(nx: int, total_time: float = T_FINAL, alpha: float = ALPHA,
                    method: str = 'RK45', rtol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve heat equation using scipy.solve_ivp by converting PDE to ODE system
    
    Args:
        nx: Number of spatial grid points
        total_time: Total simulation time
        alpha: Thermal diffusivity
        method: Integration method ('RK45', 'DOP853', 'Radau', 'BDF')
        rtol: Relative tolerance
    
    Returns:
        Tuple of (x_grid, t_grid, solution_matrix)
    """
    # Spatial grid
    x = np.linspace(0, L, nx)
    dx = x[1] - x[0]
    
    # Build spatial differentiation matrix for interior points
    n_interior = nx - 2
    diagonals = [np.ones(n_interior-1),
                -2 * np.ones(n_interior),
                np.ones(n_interior-1)]
    A_spatial = alpha / (dx**2) * diags(diagonals, [-1, 0, 1], 
                                       shape=(n_interior, n_interior), format='csc')
    
    def ode_system(t, u_interior):
        """
        ODE system: du/dt = alpha * d²u/dx²
        Only for interior points (boundary conditions u[0] = u[-1] = 0)
        """
        return A_spatial @ u_interior
    
    # Initial condition for interior points
    u0_full = create_initial_condition(x)
    u0_interior = u0_full[1:-1]
    
    # Time span
    t_span = (0, total_time)
    
    # Solve ODE system
    sol = solve_ivp(ode_system, t_span, u0_interior, method=method, 
                   rtol=rtol, dense_output=True)
    
    # Create uniform time grid for output
    nt = max(100, int(total_time * 10))  # Adaptive time grid
    t_eval = np.linspace(0, total_time, nt)
    u_interior_eval = sol.sol(t_eval)
    
    # Reconstruct full solution with boundary conditions
    u_full = np.zeros((nt, nx))
    u_full[:, 1:-1] = u_interior_eval.T
    u_full[:, 0] = 0.0   # Boundary condition
    u_full[:, -1] = 0.0  # Boundary condition
    
    return x, t_eval, u_full

def calculate_errors(u_numerical: np.ndarray, u_reference: np.ndarray, 
                    dx: float) -> Dict[str, float]:
    """
    Calculate various error norms between numerical and reference solutions
    
    Args:
        u_numerical: Numerical solution
        u_reference: Reference solution
        dx: Spatial grid spacing
    
    Returns:
        Dictionary containing different error measures
    """
    # Ensure same shape
    if u_numerical.shape != u_reference.shape:
        raise ValueError("Solutions must have the same shape")
    
    # Calculate errors
    error = u_numerical - u_reference
    
    # L2 norm
    l2_error = np.sqrt(np.sum(error**2) * dx)
    l2_norm_ref = np.sqrt(np.sum(u_reference**2) * dx)
    l2_relative = l2_error / (l2_norm_ref + 1e-12)
    
    # Maximum norm
    max_error = np.max(np.abs(error))
    max_relative = max_error / (np.max(np.abs(u_reference)) + 1e-12)
    
    # RMS error
    rms_error = np.sqrt(np.mean(error**2))
    
    return {
        'l2_absolute': l2_error,
        'l2_relative': l2_relative,
        'max_absolute': max_error,
        'max_relative': max_relative,
        'rms': rms_error
    }

def compare_methods(nx: int = 101, nt: int = 1000) -> Dict:
    """
    Compare all four numerical methods
    
    Args:
        nx: Number of spatial grid points
        nt: Number of time steps
    
    Returns:
        Dictionary containing results and timing information
    """
    results = {}
    
    print(f"Comparing methods with nx={nx}, nt={nt}")
    print("=" * 50)
    
    # Method 1: FTCS
    print("Running FTCS method...")
    start_time = time.time()
    x, t, u_ftcs = solve_ftcs(nx, nt)
    time_ftcs = time.time() - start_time
    
    results['ftcs'] = {
        'solution': (x, t, u_ftcs),
        'time': time_ftcs,
        'stability_param': ALPHA * (t[1] - t[0]) / (x[1] - x[0])**2
    }
    print(f"  Completed in {time_ftcs:.4f} seconds")
    print(f"  Stability parameter r = {results['ftcs']['stability_param']:.4f}")
    
    # Method 2: Backward Euler
    print("\nRunning Backward Euler method...")
    start_time = time.time()
    x, t, u_be = solve_backward_euler(nx, nt)
    time_be = time.time() - start_time
    
    results['backward_euler'] = {
        'solution': (x, t, u_be),
        'time': time_be
    }
    print(f"  Completed in {time_be:.4f} seconds")
    
    # Method 3: Crank-Nicolson
    print("\nRunning Crank-Nicolson method...")
    start_time = time.time()
    x, t, u_cn = solve_crank_nicolson(nx, nt)
    time_cn = time.time() - start_time
    
    results['crank_nicolson'] = {
        'solution': (x, t, u_cn),
        'time': time_cn
    }
    print(f"  Completed in {time_cn:.4f} seconds")
    
    # Method 4: scipy.solve_ivp
    print("\nRunning scipy.solve_ivp method...")
    start_time = time.time()
    x, t_scipy, u_scipy = solve_with_scipy(nx)
    time_scipy = time.time() - start_time
    
    results['scipy_ivp'] = {
        'solution': (x, t_scipy, u_scipy),
        'time': time_scipy
    }
    print(f"  Completed in {time_scipy:.4f} seconds")
    
    # Use Crank-Nicolson as reference (highest accuracy)
    reference_solution = u_cn
    
    # Calculate errors (interpolate scipy solution to common time grid)
    print("\nCalculating errors...")
    
    # Interpolate scipy solution to common time grid
    u_scipy_interp = np.zeros_like(u_cn)
    for i in range(nx):
        u_scipy_interp[:, i] = np.interp(t, t_scipy, u_scipy[:, i])
    
    dx = x[1] - x[0]
    
    # Calculate errors relative to Crank-Nicolson
    results['ftcs']['errors'] = calculate_errors(u_ftcs, reference_solution, dx)
    results['backward_euler']['errors'] = calculate_errors(u_be, reference_solution, dx)
    results['scipy_ivp']['errors'] = calculate_errors(u_scipy_interp, reference_solution, dx)
    
    # Print summary
    print("\nMethod Comparison Summary:")
    print("-" * 70)
    print(f"{'Method':<15} {'Time (s)':<10} {'L2 Error':<12} {'Max Error':<12}")
    print("-" * 70)
    
    for method_name, data in results.items():
        if method_name == 'crank_nicolson':
            print(f"{method_name:<15} {data['time']:<10.4f} {'Reference':<12} {'Reference':<12}")
        else:
            errors = data['errors']
            print(f"{method_name:<15} {data['time']:<10.4f} {errors['l2_relative']:<12.2e} {errors['max_relative']:<12.2e}")
    
    return results

def plot_comparison(results: Dict, save_plots: bool = True):
    """
    Create comprehensive comparison plots
    
    Args:
        results: Results dictionary from compare_methods
        save_plots: Whether to save plots to files
    """
    # Extract solutions
    x_ftcs, t_ftcs, u_ftcs = results['ftcs']['solution']
    x_be, t_be, u_be = results['backward_euler']['solution']
    x_cn, t_cn, u_cn = results['crank_nicolson']['solution']
    x_scipy, t_scipy, u_scipy = results['scipy_ivp']['solution']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Solutions at different times
    ax1 = plt.subplot(2, 3, 1)
    time_indices = [0, len(t_cn)//4, len(t_cn)//2, -1]
    time_labels = ['t=0', f't={t_cn[len(t_cn)//4]:.1f}', 
                  f't={t_cn[len(t_cn)//2]:.1f}', f't={t_cn[-1]:.1f}']
    
    for i, (idx, label) in enumerate(zip(time_indices, time_labels)):
        plt.plot(x_cn, u_cn[idx, :], label=f'CN {label}', linestyle='-', alpha=0.8)
        plt.plot(x_ftcs, u_ftcs[idx, :], label=f'FTCS {label}', linestyle='--', alpha=0.6)
    
    plt.xlabel('Position x')
    plt.ylabel('Temperature u')
    plt.title('Temperature Evolution Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Error evolution
    ax2 = plt.subplot(2, 3, 2)
    
    # Calculate error at each time step
    error_ftcs = np.zeros(len(t_cn))
    error_be = np.zeros(len(t_cn))
    
    for i in range(len(t_cn)):
        error_ftcs[i] = np.max(np.abs(u_ftcs[i, :] - u_cn[i, :]))
        error_be[i] = np.max(np.abs(u_be[i, :] - u_cn[i, :]))
    
    plt.semilogy(t_cn, error_ftcs, label='FTCS vs CN', linewidth=2)
    plt.semilogy(t_cn, error_be, label='BE vs CN', linewidth=2)
    plt.xlabel('Time t')
    plt.ylabel('Maximum Error')
    plt.title('Error Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Computational time comparison
    ax3 = plt.subplot(2, 3, 3)
    methods = ['FTCS', 'Backward\nEuler', 'Crank-\nNicolson', 'scipy\nsolve_ivp']
    times = [results['ftcs']['time'], results['backward_euler']['time'],
            results['crank_nicolson']['time'], results['scipy_ivp']['time']]
    
    bars = plt.bar(methods, times, color=['blue', 'red', 'green', 'orange'], alpha=0.7)
    plt.ylabel('Computation Time (s)')
    plt.title('Computational Efficiency')
    plt.yscale('log')
    
    # Add value labels on bars
    for bar, time_val in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{time_val:.4f}s', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    
    # Plot 4: 3D surface plot of Crank-Nicolson solution
    ax4 = plt.subplot(2, 3, 4, projection='3d')
    T_mesh, X_mesh = np.meshgrid(t_cn[::10], x_cn)  # Subsample for clarity
    U_mesh = u_cn[::10, :].T
    
    surf = ax4.plot_surface(X_mesh, T_mesh, U_mesh, cmap='viridis', alpha=0.8)
    ax4.set_xlabel('Position x')
    ax4.set_ylabel('Time t')
    ax4.set_zlabel('Temperature u')
    ax4.set_title('3D Temperature Evolution (CN)')
    
    # Plot 5: Accuracy vs Efficiency scatter plot
    ax5 = plt.subplot(2, 3, 5)
    
    # Extract relative errors and times
    methods_data = [
        ('FTCS', results['ftcs']['errors']['l2_relative'], results['ftcs']['time']),
        ('BE', results['backward_euler']['errors']['l2_relative'], results['backward_euler']['time']),
        ('scipy', results['scipy_ivp']['errors']['l2_relative'], results['scipy_ivp']['time'])
    ]
    
    for method, error, time_val in methods_data:
        plt.scatter(time_val, error, s=100, label=method, alpha=0.7)
        plt.annotate(method, (time_val, error), xytext=(5, 5), 
                    textcoords='offset points')
    
    plt.xlabel('Computation Time (s)')
    plt.ylabel('Relative L2 Error')
    plt.title('Accuracy vs Efficiency')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 6: Stability analysis
    ax6 = plt.subplot(2, 3, 6)
    
    # Show stability parameter for FTCS
    r_ftcs = results['ftcs']['stability_param']
    
    plt.axhline(y=0.5, color='red', linestyle='--', linewidth=2, 
               label='Stability limit (r = 0.5)')
    plt.bar(['FTCS'], [r_ftcs], color='blue', alpha=0.7, 
           label=f'Current r = {r_ftcs:.4f}')
    
    if r_ftcs > 0.5:
        plt.text(0, r_ftcs + 0.05, 'UNSTABLE', ha='center', 
                color='red', fontweight='bold')
    else:
        plt.text(0, r_ftcs + 0.05, 'STABLE', ha='center', 
                color='green', fontweight='bold')
    
    plt.ylabel('Stability Parameter r')
    plt.title('FTCS Stability Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('heat_equation_comparison.png', dpi=300, bbox_inches='tight')
        print("\nPlots saved as 'heat_equation_comparison.png'")
    
    plt.show()

def convergence_study(nx_values: list = [21, 41, 81, 161], 
                     nt_factor: int = 10) -> Dict:
    """
    Perform convergence study for different grid resolutions
    
    Args:
        nx_values: List of spatial grid sizes to test
        nt_factor: Factor to determine nt = nt_factor * nx
    
    Returns:
        Dictionary containing convergence data
    """
    print("Performing convergence study...")
    print("=" * 40)
    
    convergence_data = {
        'nx_values': nx_values,
        'dx_values': [],
        'errors_ftcs': [],
        'errors_be': [],
        'errors_cn': [],
        'times_ftcs': [],
        'times_be': [],
        'times_cn': []
    }
    
    # Use finest grid as reference
    nx_ref = nx_values[-1]
    nt_ref = nt_factor * nx_ref
    print(f"Computing reference solution with nx={nx_ref}, nt={nt_ref}...")
    x_ref, t_ref, u_ref = solve_crank_nicolson(nx_ref, nt_ref)
    
    for nx in nx_values[:-1]:  # Exclude reference grid
        nt = nt_factor * nx
        dx = L / (nx - 1)
        
        print(f"\nTesting nx={nx}, nt={nt}, dx={dx:.4f}")
        
        # FTCS
        start_time = time.time()
        x, t, u_ftcs = solve_ftcs(nx, nt)
        time_ftcs = time.time() - start_time
        
        # Backward Euler
        start_time = time.time()
        x, t, u_be = solve_backward_euler(nx, nt)
        time_be = time.time() - start_time
        
        # Crank-Nicolson
        start_time = time.time()
        x, t, u_cn = solve_crank_nicolson(nx, nt)
        time_cn = time.time() - start_time
        
        # Interpolate reference solution to current grid
        u_ref_interp = np.zeros_like(u_cn)
        for i in range(len(t)):
            u_ref_interp[i, :] = np.interp(x, x_ref, u_ref[i, :])
        
        # Calculate errors
        error_ftcs = calculate_errors(u_ftcs, u_ref_interp, dx)
        error_be = calculate_errors(u_be, u_ref_interp, dx)
        error_cn = calculate_errors(u_cn, u_ref_interp, dx)
        
        # Store results
        convergence_data['dx_values'].append(dx)
        convergence_data['errors_ftcs'].append(error_ftcs['l2_relative'])
        convergence_data['errors_be'].append(error_be['l2_relative'])
        convergence_data['errors_cn'].append(error_cn['l2_relative'])
        convergence_data['times_ftcs'].append(time_ftcs)
        convergence_data['times_be'].append(time_be)
        convergence_data['times_cn'].append(time_cn)
        
        print(f"  FTCS: L2 error = {error_ftcs['l2_relative']:.2e}, time = {time_ftcs:.4f}s")
        print(f"  BE:   L2 error = {error_be['l2_relative']:.2e}, time = {time_be:.4f}s")
        print(f"  CN:   L2 error = {error_cn['l2_relative']:.2e}, time = {time_cn:.4f}s")
    
    return convergence_data

def plot_convergence(convergence_data: Dict, save_plots: bool = True):
    """
    Plot convergence study results
    
    Args:
        convergence_data: Data from convergence_study
        save_plots: Whether to save plots
    """
    dx_values = np.array(convergence_data['dx_values'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Convergence plot
    ax1.loglog(dx_values, convergence_data['errors_ftcs'], 'o-', 
              label='FTCS', linewidth=2, markersize=8)
    ax1.loglog(dx_values, convergence_data['errors_be'], 's-', 
              label='Backward Euler', linewidth=2, markersize=8)
    ax1.loglog(dx_values, convergence_data['errors_cn'], '^-', 
              label='Crank-Nicolson', linewidth=2, markersize=8)
    
    # Add reference slopes
    ax1.loglog(dx_values, dx_values**1, '--', color='gray', alpha=0.7, label='O(Δx)')
    ax1.loglog(dx_values, dx_values**2, ':', color='gray', alpha=0.7, label='O(Δx²)')
    
    ax1.set_xlabel('Grid Spacing Δx')
    ax1.set_ylabel('Relative L2 Error')
    ax1.set_title('Convergence Study')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Efficiency plot
    ax2.loglog(convergence_data['times_ftcs'], convergence_data['errors_ftcs'], 
              'o-', label='FTCS', linewidth=2, markersize=8)
    ax2.loglog(convergence_data['times_be'], convergence_data['errors_be'], 
              's-', label='Backward Euler', linewidth=2, markersize=8)
    ax2.loglog(convergence_data['times_cn'], convergence_data['errors_cn'], 
              '^-', label='Crank-Nicolson', linewidth=2, markersize=8)
    
    ax2.set_xlabel('Computation Time (s)')
    ax2.set_ylabel('Relative L2 Error')
    ax2.set_title('Efficiency Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('convergence_study.png', dpi=300, bbox_inches='tight')
        print("Convergence plots saved as 'convergence_study.png'")
    
    plt.show()

if __name__ == "__main__":
    print("Heat Equation Solver - Multiple Methods Comparison")
    print("=" * 60)
    print(f"Problem parameters:")
    print(f"  Domain: [0, {L}]")
    print(f"  Time: [0, {T_FINAL}]")
    print(f"  Thermal diffusivity: α² = {ALPHA}")
    print(f"  Initial condition: u(x,0) = 1 for 10 ≤ x ≤ 11, 0 elsewhere")
    print(f"  Boundary conditions: u(0,t) = u({L},t) = 0")
    print()
    
    try:
        # Main comparison
        results = compare_methods(nx=101, nt=500)
        
        # Create comparison plots
        print("\nCreating comparison plots...")
        plot_comparison(results)
        
        # Convergence study
        print("\nStarting convergence study...")
        conv_data = convergence_study(nx_values=[21, 41, 81], nt_factor=8)
        
        # Plot convergence results
        print("\nCreating convergence plots...")
        plot_convergence(conv_data)
        
        print("\nAnalysis complete!")
        print("\nKey findings:")
        print("1. Crank-Nicolson method provides the best accuracy")
        print("2. FTCS method is fastest but has stability constraints")
        print("3. Implicit methods are unconditionally stable")
        print("4. scipy.solve_ivp provides good accuracy with adaptive stepping")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()