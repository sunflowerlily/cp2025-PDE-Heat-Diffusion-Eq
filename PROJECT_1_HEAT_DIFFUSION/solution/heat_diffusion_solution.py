#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
铝棒热传导方程显式差分法数值解 - 参考答案

本模块使用显式差分法求解一维热传导方程，包含：
1. 显式差分格式的完整实现
2. 稳定性条件的验证和分析
3. 解析解的计算和比较
4. 误差分析和收敛性研究
5. 可视化和动画生成

Author: Reference Solution
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import warnings
warnings.filterwarnings('ignore')

# Set plotting parameters
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Physical parameters
K = 237.0      # Thermal conductivity W/(m·K)
C = 900.0      # Specific heat capacity J/(kg·K)
RHO = 2700.0   # Density kg/m³
ALPHA = K / (C * RHO)  # Thermal diffusivity m²/s

class HeatDiffusionSolver:
    """
    Heat diffusion equation solver using explicit finite difference method
    
    Focuses on explicit finite difference scheme implementation and stability analysis
    """
    
    def __init__(self, L=1.0, nx=50, nt=1000, total_time=0.1):
        """
        Initialize solver parameters
        
        Args:
            L (float): Length of aluminum rod (m)
            nx (int): Number of spatial grid points
            nt (int): Number of time steps
            total_time (float): Total simulation time (s)
        """
        self.L = L
        self.nx = nx
        self.nt = nt
        self.total_time = total_time
        
        # Grid parameters
        self.dx = L / (nx - 1)
        self.dt = total_time / nt
        self.eta = ALPHA * self.dt / (self.dx ** 2)
        
        # Spatial and temporal grids
        self.x = np.linspace(0, L, nx)
        self.t = np.linspace(0, total_time, nt + 1)
        
        print(f"Grid parameters: dx={self.dx:.6f}, dt={self.dt:.6f}, η={self.eta:.6f}")
        if self.eta > 0.25:
            print(f"Warning: η = {self.eta:.6f} > 0.25, may be unstable!")
        
    def initial_condition(self):
        """
        Set initial condition: T(x,0) = 100 K
        
        Returns:
            numpy.ndarray: Initial temperature distribution
        """
        T0 = np.ones(self.nx) * 100.0
        T0[0] = 0.0  # Boundary condition
        T0[-1] = 0.0  # Boundary condition
        return T0
    
    def explicit_finite_difference(self, dt=None, nt=None):
        """
        Solve heat diffusion equation using explicit finite difference method
        
        Forward Euler time discretization and central difference spatial discretization:
        T[i][j+1] = T[i][j] + η*(T[i+1][j] - 2*T[i][j] + T[i-1][j])
        
        Stability condition: η = α*Δt/(Δx)² ≤ 1/4
        
        Args:
            dt (float): Time step size, if provided recalculate grid parameters
            nt (int): Number of time steps, if provided recalculate grid parameters
        
        Returns:
            tuple: (time array, temperature matrix)
                - time array: shape (nt+1,)
                - temperature matrix: shape (nx, nt+1)
        """
        # Recalculate grid parameters if new time parameters provided
        if dt is not None:
            self.dt = dt
            self.nt = int(self.total_time / dt)
            self.eta = ALPHA * self.dt / (self.dx ** 2)
            self.t = np.linspace(0, self.total_time, self.nt + 1)
            print(f"Updated: dt={self.dt:.6f}, nt={self.nt}, η={self.eta:.6f}")
        
        if nt is not None:
            self.nt = nt
            self.dt = self.total_time / nt
            self.eta = ALPHA * self.dt / (self.dx ** 2)
            self.t = np.linspace(0, self.total_time, self.nt + 1)
            print(f"Updated: dt={self.dt:.6f}, nt={self.nt}, η={self.eta:.6f}")
        
        # Check stability condition
        if self.eta > 0.25:
            print(f"Warning: η = {self.eta:.6f} > 0.25, solution may be unstable!")
        
        # Initialize temperature matrix T_matrix[i,j] = T(x_i, t_j)
        T_matrix = np.zeros((self.nx, self.nt + 1))
        T_matrix[:, 0] = self.initial_condition()
        
        # Time stepping loop
        for j in range(self.nt):
            # Update interior nodes (i = 1 to nx-2)
            for i in range(1, self.nx - 1):
                T_matrix[i, j+1] = (T_matrix[i, j] + 
                                   self.eta * (T_matrix[i+1, j] - 2*T_matrix[i, j] + T_matrix[i-1, j]))
            
            # Apply boundary conditions T[0] = T[-1] = 0
            T_matrix[0, j+1] = 0.0
            T_matrix[-1, j+1] = 0.0
        
        return self.t, T_matrix
    
    def analytical_solution(self, n_terms=50):
        """
        Calculate analytical solution of heat diffusion equation
        
        Analytical solution as Fourier series:
        T(x,t) = Σ(n=1,3,5,...) (4*T₀)/(n*π) * sin(n*π*x/L) * exp(-n²*π²*α*t/L²)
        where T₀ = 100 K (initial temperature)
        
        Args:
            n_terms (int): Number of Fourier series terms
        
        Returns:
            tuple: (time array, temperature matrix)
        """
        # Initialize analytical solution matrix
        T_analytical = np.zeros((self.nx, self.nt + 1))
        
        # Create spatial and temporal grid matrices
        X, T_time = np.meshgrid(self.x, self.t, indexing='ij')
        
        # Calculate Fourier series (only odd terms n = 1,3,5,...)
        for n in range(1, 2*n_terms, 2):  # n = 1,3,5,...,2*n_terms-1
            coefficient = (4 * 100.0) / (n * np.pi)
            spatial_part = np.sin(n * np.pi * X / self.L)
            temporal_part = np.exp(-n**2 * np.pi**2 * ALPHA * T_time / self.L**2)
            T_analytical += coefficient * spatial_part * temporal_part
        
        # Apply boundary conditions
        T_analytical[0, :] = 0.0
        T_analytical[-1, :] = 0.0
        
        return self.t, T_analytical
    
    def plot_evolution(self, t_array, T_matrix, title="Temperature Evolution", save_fig=False):
        """
        Plot temperature evolution over time
        
        Args:
            t_array (np.ndarray): Time array
            T_matrix (np.ndarray): Temperature matrix
            title (str): Plot title
            save_fig (bool): Whether to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot temperature distribution at different times
        time_indices = [0, len(t_array)//4, len(t_array)//2, 3*len(t_array)//4, -1]
        colors = ['blue', 'green', 'orange', 'red', 'purple']
        
        for i, (idx, color) in enumerate(zip(time_indices, colors)):
            ax1.plot(self.x, T_matrix[:, idx], color=color, 
                    label=f't = {t_array[idx]:.4f} s', linewidth=2)
        
        ax1.set_xlabel('Position x (m)')
        ax1.set_ylabel('Temperature T (K)')
        ax1.set_title('Temperature Distribution at Different Times')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot temperature evolution at specific positions
        positions = [self.nx//4, self.nx//2, 3*self.nx//4]
        pos_colors = ['blue', 'red', 'green']
        
        for pos, color in zip(positions, pos_colors):
            ax2.plot(t_array, T_matrix[pos, :], color=color, 
                    label=f'x = {self.x[pos]:.3f} m', linewidth=2)
        
        ax2.set_xlabel('Time t (s)')
        ax2.set_ylabel('Temperature T (K)')
        ax2.set_title('Temperature Evolution at Specific Positions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_3d_surface(self, t_array, T_matrix, title="3D Temperature Distribution", save_fig=False):
        """
        Plot 3D temperature distribution
        
        Args:
            t_array (np.ndarray): Time array
            T_matrix (np.ndarray): Temperature matrix
            title (str): Plot title
            save_fig (bool): Whether to save figure
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create grid matrices
        X, T_time = np.meshgrid(self.x, t_array)
        
        # Plot surface
        surf = ax.plot_surface(X, T_time, T_matrix.T, cmap='hot', 
                              alpha=0.8, linewidth=0, antialiased=True)
        
        # Set labels and title
        ax.set_xlabel('Position x (m)')
        ax.set_ylabel('Time t (s)')
        ax.set_zlabel('Temperature T (K)')
        ax.set_title(title)
        
        # Add colorbar
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        if save_fig:
            plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_animation(self, t_array, T_matrix, filename="heat_diffusion.gif"):
        """
        Create temperature evolution animation
        
        Args:
            t_array (np.ndarray): Time array
            T_matrix (np.ndarray): Temperature matrix
            filename (str): Animation filename
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Set up the plot
        line, = ax.plot([], [], 'b-', linewidth=2)
        ax.set_xlim(0, self.L)
        ax.set_ylim(0, 100)
        ax.set_xlabel('Position x (m)')
        ax.set_ylabel('Temperature T (K)')
        ax.set_title('Heat Diffusion Animation')
        ax.grid(True, alpha=0.3)
        
        # Time text
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        
        def animate(frame):
            """Animation update function"""
            line.set_data(self.x, T_matrix[:, frame])
            time_text.set_text(f'Time = {t_array[frame]:.4f} s')
            return line, time_text
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(t_array), 
                                     interval=50, blit=True, repeat=True)
        
        # Save animation
        anim.save(filename, writer='pillow', fps=20)
        print(f"Animation saved as {filename}")
        
        plt.show()

def analyze_stability():
    """
    Analyze stability condition of explicit finite difference method
    
    Analysis includes:
    1. Test numerical stability under different eta values
    2. Demonstrate unstable phenomena
    3. Verify stability condition eta ≤ 1/4
    4. Visualize stability boundary
    """
    print("\n" + "="*50)
    print("STABILITY ANALYSIS")
    print("="*50)
    
    # Test different eta values
    eta_values = [0.2, 0.24, 0.25, 0.26, 0.3, 0.5]
    L = 1.0
    nx = 50
    total_time = 0.01  # Shorter time for stability test
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, eta_target in enumerate(eta_values):
        # Calculate required dt for target eta
        dx = L / (nx - 1)
        dt_required = eta_target * dx**2 / ALPHA
        nt = max(int(total_time / dt_required), 1)  # Ensure at least 1 time step
        
        print(f"\nTesting η = {eta_target:.3f}:")
        print(f"  Required dt = {dt_required:.2e} s")
        print(f"  Number of time steps = {nt}")
        
        # Create solver and run simulation
        solver = HeatDiffusionSolver(L=L, nx=nx, nt=nt, total_time=total_time)
        
        try:
            t_array, T_matrix = solver.explicit_finite_difference()
            
            # Check for instability
            max_temp = np.max(T_matrix)
            min_temp = np.min(T_matrix)
            is_stable = max_temp < 200 and min_temp > -50  # Reasonable bounds
            
            print(f"  Max temperature: {max_temp:.2f} K")
            print(f"  Min temperature: {min_temp:.2f} K")
            print(f"  Stable: {is_stable}")
            
            # Plot results
            ax = axes[i]
            
            # Plot final temperature distribution
            ax.plot(solver.x, T_matrix[:, -1], 'b-', linewidth=2, 
                   label=f'Final (t={total_time:.3f}s)')
            ax.plot(solver.x, T_matrix[:, 0], 'r--', linewidth=2, 
                   label='Initial')
            
            ax.set_title(f'η = {eta_target:.3f} ({"Stable" if is_stable else "Unstable"})')
            ax.set_xlabel('Position x (m)')
            ax.set_ylabel('Temperature T (K)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if not is_stable:
                ax.set_ylim(-100, 200)  # Extended range for unstable cases
                
        except Exception as e:
            print(f"  Error: {e}")
            axes[i].text(0.5, 0.5, f'η = {eta_target:.3f}\nFailed', 
                        transform=axes[i].transAxes, ha='center', va='center')
    
    plt.suptitle('Stability Analysis: Effect of η on Numerical Solution', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('stability_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def calculate_errors():
    """
    Calculate errors between numerical and analytical solutions
    
    Error analysis includes:
    1. L2 norm error
    2. Maximum error
    3. Grid convergence analysis
    4. Error evolution over time
    """
    print("\n" + "="*50)
    print("ERROR ANALYSIS")
    print("="*50)
    
    # Grid convergence study
    nx_values = [21, 41, 81, 161]
    L = 1.0
    total_time = 0.1
    
    errors_l2 = []
    errors_max = []
    dx_values = []
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for nx in nx_values:
        # Ensure stability: choose dt such that eta = 0.2
        dx = L / (nx - 1)
        eta_target = 0.2
        dt = eta_target * dx**2 / ALPHA
        nt = max(int(total_time / dt), 1)  # Ensure at least 1 time step
        
        print(f"\nGrid: nx = {nx}, dx = {dx:.6f}")
        print(f"Time: nt = {nt}, dt = {dt:.2e}")
        
        # Create solver
        solver = HeatDiffusionSolver(L=L, nx=nx, nt=nt, total_time=total_time)
        
        # Calculate numerical and analytical solutions
        t_num, T_num = solver.explicit_finite_difference()
        t_ana, T_ana = solver.analytical_solution()
        
        # Calculate errors
        error_l2 = np.linalg.norm(T_num - T_ana) / np.linalg.norm(T_ana)
        error_max = np.max(np.abs(T_num - T_ana))
        
        errors_l2.append(error_l2)
        errors_max.append(error_max)
        dx_values.append(dx)
        
        print(f"L2 relative error: {error_l2:.6e}")
        print(f"Max absolute error: {error_max:.6e} K")
    
    # Plot convergence
    ax1.loglog(dx_values, errors_l2, 'bo-', linewidth=2, markersize=8, label='L2 Error')
    ax1.loglog(dx_values, errors_max, 'rs-', linewidth=2, markersize=8, label='Max Error')
    
    # Add reference lines
    dx_ref = np.array(dx_values)
    ax1.loglog(dx_ref, 0.1 * dx_ref**2, 'k--', alpha=0.7, label='O(Δx²)')
    
    ax1.set_xlabel('Grid spacing Δx (m)')
    ax1.set_ylabel('Error')
    ax1.set_title('Grid Convergence Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Error evolution over time
    solver = HeatDiffusionSolver(L=L, nx=81, nt=1000, total_time=total_time)
    t_num, T_num = solver.explicit_finite_difference()
    t_ana, T_ana = solver.analytical_solution()
    
    # Calculate error at each time step
    time_errors_l2 = []
    time_errors_max = []
    
    for j in range(len(t_num)):
        error_l2_t = np.linalg.norm(T_num[:, j] - T_ana[:, j]) / np.linalg.norm(T_ana[:, j])
        error_max_t = np.max(np.abs(T_num[:, j] - T_ana[:, j]))
        time_errors_l2.append(error_l2_t)
        time_errors_max.append(error_max_t)
    
    ax2.semilogy(t_num, time_errors_l2, 'b-', linewidth=2, label='L2 Error')
    ax2.semilogy(t_num, time_errors_max, 'r-', linewidth=2, label='Max Error')
    
    ax2.set_xlabel('Time t (s)')
    ax2.set_ylabel('Error')
    ax2.set_title('Error Evolution Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('error_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate convergence order
    if len(errors_l2) >= 2:
        convergence_order = np.log(errors_l2[-2] / errors_l2[-1]) / np.log(dx_values[-2] / dx_values[-1])
        print(f"\nEstimated convergence order: {convergence_order:.2f}")

def compare_with_analytical():
    """
    Compare numerical solution with analytical solution
    
    Comparison includes:
    1. Temperature distribution comparison at different times
    2. Time evolution comparison at specific positions
    3. Error distribution visualization
    4. Convergence verification
    """
    print("\n" + "="*50)
    print("NUMERICAL vs ANALYTICAL COMPARISON")
    print("="*50)
    
    # Create solver with stable parameters
    solver = HeatDiffusionSolver(L=1.0, nx=81, nt=1000, total_time=0.1)
    
    # Calculate solutions
    print("Calculating numerical solution...")
    t_num, T_num = solver.explicit_finite_difference()
    
    print("Calculating analytical solution...")
    t_ana, T_ana = solver.analytical_solution(n_terms=100)
    
    # Create comparison plots
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Temperature distribution comparison
    ax1 = plt.subplot(2, 3, 1)
    time_indices = [0, len(t_num)//4, len(t_num)//2, 3*len(t_num)//4, -1]
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    for i, (idx, color) in enumerate(zip(time_indices, colors)):
        ax1.plot(solver.x, T_num[:, idx], color=color, linewidth=2, 
                label=f't = {t_num[idx]:.4f} s')
        ax1.plot(solver.x, T_ana[:, idx], color=color, linewidth=1, 
                linestyle='--', alpha=0.7)
    
    ax1.set_xlabel('Position x (m)')
    ax1.set_ylabel('Temperature T (K)')
    ax1.set_title('Temperature Distribution\n(Solid: Numerical, Dashed: Analytical)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Time evolution comparison
    ax2 = plt.subplot(2, 3, 2)
    positions = [solver.nx//4, solver.nx//2, 3*solver.nx//4]
    pos_colors = ['blue', 'red', 'green']
    
    for pos, color in zip(positions, pos_colors):
        ax2.plot(t_num, T_num[pos, :], color=color, linewidth=2, 
                label=f'Num: x = {solver.x[pos]:.3f} m')
        ax2.plot(t_ana, T_ana[pos, :], color=color, linewidth=1, 
                linestyle='--', alpha=0.7, label=f'Ana: x = {solver.x[pos]:.3f} m')
    
    ax2.set_xlabel('Time t (s)')
    ax2.set_ylabel('Temperature T (K)')
    ax2.set_title('Time Evolution\n(Solid: Numerical, Dashed: Analytical)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Error distribution
    ax3 = plt.subplot(2, 3, 3)
    error_matrix = np.abs(T_num - T_ana)
    
    for idx, color in zip(time_indices, colors):
        ax3.plot(solver.x, error_matrix[:, idx], color=color, linewidth=2, 
                label=f't = {t_num[idx]:.4f} s')
    
    ax3.set_xlabel('Position x (m)')
    ax3.set_ylabel('Absolute Error |T_num - T_ana| (K)')
    ax3.set_title('Absolute Error Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 3D error surface
    ax4 = plt.subplot(2, 3, 4, projection='3d')
    X, T_time = np.meshgrid(solver.x, t_num)
    surf = ax4.plot_surface(X, T_time, error_matrix.T, cmap='viridis', alpha=0.8)
    ax4.set_xlabel('Position x (m)')
    ax4.set_ylabel('Time t (s)')
    ax4.set_zlabel('Absolute Error (K)')
    ax4.set_title('3D Error Distribution')
    
    # 5. Error statistics over time
    ax5 = plt.subplot(2, 3, 5)
    error_l2_time = [np.linalg.norm(T_num[:, j] - T_ana[:, j]) for j in range(len(t_num))]
    error_max_time = [np.max(np.abs(T_num[:, j] - T_ana[:, j])) for j in range(len(t_num))]
    
    ax5.semilogy(t_num, error_l2_time, 'b-', linewidth=2, label='L2 Error')
    ax5.semilogy(t_num, error_max_time, 'r-', linewidth=2, label='Max Error')
    ax5.set_xlabel('Time t (s)')
    ax5.set_ylabel('Error')
    ax5.set_title('Error Evolution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Final error statistics
    ax6 = plt.subplot(2, 3, 6)
    final_error = T_num[:, -1] - T_ana[:, -1]
    ax6.plot(solver.x, final_error, 'b-', linewidth=2)
    ax6.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Position x (m)')
    ax6.set_ylabel('Error T_num - T_ana (K)')
    ax6.set_title(f'Final Error (t = {t_num[-1]:.4f} s)')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('numerical_analytical_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    final_l2_error = np.linalg.norm(T_num - T_ana) / np.linalg.norm(T_ana)
    final_max_error = np.max(np.abs(T_num - T_ana))
    
    print(f"\nFinal Error Statistics:")
    print(f"L2 relative error: {final_l2_error:.6e}")
    print(f"Max absolute error: {final_max_error:.6e} K")
    print(f"Mean absolute error: {np.mean(np.abs(T_num - T_ana)):.6e} K")

def demonstrate_instability():
    """
    Demonstrate numerical instability phenomena
    
    Demonstration includes:
    1. Using excessive time step size
    2. Observing solution divergence
    3. Analyzing instability causes
    4. Visualizing instability process
    """
    print("\n" + "="*50)
    print("INSTABILITY DEMONSTRATION")
    print("="*50)
    
    # Set up unstable parameters (eta > 0.25)
    L = 1.0
    nx = 50
    dx = L / (nx - 1)
    
    # Choose eta = 0.6 (well above stability limit)
    eta_unstable = 0.6
    dt_unstable = eta_unstable * dx**2 / ALPHA
    total_time = 0.01  # Longer time to observe instability
    nt = max(int(total_time / dt_unstable), 10)  # Ensure at least 10 time steps
    
    print(f"Unstable parameters:")
    print(f"  η = {eta_unstable:.3f} (> 0.25)")
    print(f"  dx = {dx:.6f} m")
    print(f"  dt = {dt_unstable:.2e} s")
    print(f"  nt = {nt}")
    
    # Create solver with unstable parameters
    solver = HeatDiffusionSolver(L=L, nx=nx, nt=nt, total_time=total_time)
    
    try:
        # Run unstable simulation
        print("\nRunning unstable simulation...")
        t_unstable, T_unstable = solver.explicit_finite_difference(dt=dt_unstable)
        
        # Monitor solution behavior
        max_temps = [np.max(T_unstable[:, j]) for j in range(len(t_unstable))]
        min_temps = [np.min(T_unstable[:, j]) for j in range(len(t_unstable))]
        
        print(f"Initial max temperature: {max_temps[0]:.2f} K")
        print(f"Final max temperature: {max_temps[-1]:.2e} K")
        print(f"Final min temperature: {min_temps[-1]:.2e} K")
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Temperature evolution at different times
        time_indices = [0, nt//4, nt//2, 3*nt//4, -1]
        colors = ['blue', 'green', 'orange', 'red', 'purple']
        
        for i, (idx, color) in enumerate(zip(time_indices, colors)):
            if idx < len(t_unstable):
                ax1.plot(solver.x, T_unstable[:, idx], color=color, linewidth=2, 
                        label=f't = {t_unstable[idx]:.2e} s')
        
        ax1.set_xlabel('Position x (m)')
        ax1.set_ylabel('Temperature T (K)')
        ax1.set_title('Unstable Solution Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Maximum and minimum temperature over time
        ax2.semilogy(t_unstable, np.abs(max_temps), 'r-', linewidth=2, label='Max |T|')
        ax2.semilogy(t_unstable, np.abs(min_temps), 'b-', linewidth=2, label='Min |T|')
        ax2.set_xlabel('Time t (s)')
        ax2.set_ylabel('|Temperature| (K)')
        ax2.set_title('Temperature Extrema Growth')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 3D surface of unstable solution
        if T_unstable.shape[1] > 1:
            X, T_time = np.meshgrid(solver.x, t_unstable)
            # Clip extreme values for visualization
            T_clipped = np.clip(T_unstable.T, -1000, 1000)
            surf = ax3.plot_surface(X, T_time, T_clipped, cmap='coolwarm', alpha=0.8)
            ax3.set_xlabel('Position x (m)')
            ax3.set_ylabel('Time t (s)')
            ax3.set_zlabel('Temperature T (K)')
            ax3.set_title('3D Unstable Solution')
        
        # 4. Comparison with stable solution
        # Run stable simulation for comparison
        eta_stable = 0.2
        dt_stable = eta_stable * dx**2 / ALPHA
        nt_stable = int(total_time / dt_stable)
        
        solver_stable = HeatDiffusionSolver(L=L, nx=nx, nt=nt_stable, total_time=total_time)
        t_stable, T_stable = solver_stable.explicit_finite_difference(dt=dt_stable)
        
        # Plot comparison at final time
        ax4.plot(solver.x, T_unstable[:, -1], 'r-', linewidth=2, 
                label=f'Unstable (η={eta_unstable:.1f})')
        ax4.plot(solver_stable.x, T_stable[:, -1], 'b-', linewidth=2, 
                label=f'Stable (η={eta_stable:.1f})')
        ax4.plot(solver.x, T_unstable[:, 0], 'k--', linewidth=1, 
                label='Initial condition')
        
        ax4.set_xlabel('Position x (m)')
        ax4.set_ylabel('Temperature T (K)')
        ax4.set_title(f'Final Solutions (t = {total_time:.2e} s)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('instability_demonstration.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"Simulation failed due to instability: {e}")
        print("This demonstrates the catastrophic failure of unstable schemes.")

if __name__ == "__main__":
    """
    Main program: Demonstrate explicit finite difference method for heat diffusion
    """
    print("=" * 60)
    print("ALUMINUM ROD HEAT DIFFUSION - EXPLICIT FINITE DIFFERENCE")
    print("=" * 60)
    
    # Create solver instance
    solver = HeatDiffusionSolver(L=1.0, nx=81, nt=1000, total_time=0.1)
    
    try:
        # 1. Explicit finite difference solution
        print("\n1. Running explicit finite difference method...")
        t_num, T_num = solver.explicit_finite_difference()
        print(f"Numerical solution completed, final max temperature: {np.max(T_num[:, -1]):.2f} K")
        
        # 2. Analytical solution
        print("\n2. Calculating analytical solution...")
        t_ana, T_ana = solver.analytical_solution()
        print(f"Analytical solution completed, final max temperature: {np.max(T_ana[:, -1]):.2f} K")
        
        # 3. Error analysis
        print("\n3. Error analysis...")
        error_l2 = np.linalg.norm(T_num - T_ana) / np.linalg.norm(T_ana)
        error_max = np.max(np.abs(T_num - T_ana))
        print(f"L2 relative error: {error_l2:.6f}")
        print(f"Max absolute error: {error_max:.6f} K")
        
        # 4. Visualization
        print("\n4. Generating visualization plots...")
        solver.plot_evolution(t_num, T_num, "Explicit Finite Difference Solution")
        solver.plot_3d_surface(t_num, T_num, "3D Temperature Distribution")
        
        # 5. Stability analysis
        print("\n5. Stability analysis...")
        analyze_stability()
        
        # 6. Comparison analysis
        print("\n6. Numerical vs analytical comparison...")
        compare_with_analytical()
        
        # 7. Error analysis
        print("\n7. Detailed error analysis...")
        calculate_errors()
        
        # 8. Instability demonstration
        print("\n8. Instability demonstration...")
        demonstrate_instability()
        
        # 9. Create animation
        print("\n9. Creating animation...")
        solver.create_animation(t_num, T_num, "heat_diffusion_explicit.gif")
        
        print("\n" + "=" * 60)
        print("ALL ANALYSES COMPLETED! Check generated plots and files.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        print("Please check the implementation.")