#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import warnings
warnings.filterwarnings('ignore')

# Default Physical parameters
DEFAULT_K = 237.0
DEFAULT_C = 900.0
DEFAULT_RHO = 2700.0
DEFAULT_L = 1.0
DEFAULT_NX = 101
DEFAULT_TOTAL_TIME = 1000.0
DEFAULT_DT = 0.5
DEFAULT_T0 = 100.0
ALPHA = DEFAULT_K / (DEFAULT_C * DEFAULT_RHO)

class HeatDiffusionSolver:

    
    def __init__(self, L=DEFAULT_L, K=DEFAULT_K, C=DEFAULT_C, rho=DEFAULT_RHO, 
                 nx=101, total_time=1000.0, dt=0.5, 
                 initial_condition_config=None, # Can be a float (uniform T0) or a dict for complex cases
                 boundary_conditions=((0,0), (0,0))): # ((type, val_left), (type, val_right)), 0 for Dirichlet

        self.L = L
        self.K = K
        self.C = C
        self.rho = rho
        self.D = K / (C * rho)  # Thermal diffusivity (alpha in some texts, D in problem statement)
        
        self.nx = nx
        self.dx = L / (nx - 1)
        self.x = np.linspace(0, L, nx)
        
        self.total_time = total_time
        self.dt = dt
        self.nt = int(total_time / dt)
        
        self.t = np.linspace(0, total_time, self.nt + 1)
        
        self.r = self.D * self.dt / (self.dx ** 2) # Stability parameter (eta in old code)
        
        self.initial_condition_config = initial_condition_config if initial_condition_config is not None else DEFAULT_T0
        self.boundary_conditions = boundary_conditions

        print(f"Physical parameters: L={L:.2f}m, K={K:.1f}, C={C:.1f}, rho={rho:.1f}, D={self.D:.2e} m^2/s")
        print(f"Grid parameters: nx={nx}, dx={self.dx:.4f}m, nt={self.nt}, dt={self.dt:.4f}s, total_time={total_time:.1f}s")
        print(f"Stability parameter r = D*dt/dx^2 = {self.r:.4f}")
        if self.r > 0.5:
            print(f"Warning: r = {self.r:.4f} > 0.5. The solution may be unstable for FTCS scheme without cooling.")

    def set_initial_condition(self):
        T_initial = np.zeros(self.nx)
        
        if isinstance(self.initial_condition_config, (int, float)):
            T_initial[:] = float(self.initial_condition_config)
        elif isinstance(self.initial_condition_config, dict):
            config_type = self.initial_condition_config.get('type')
            if config_type == 'two_rods':
                T1 = self.initial_condition_config.get('T1', DEFAULT_T0)
                T2 = self.initial_condition_config.get('T2', DEFAULT_T0 / 2)
                split_x_val = self.initial_condition_config.get('split_x', self.L / 2)
                for i in range(self.nx):
                    if self.x[i] < split_x_val:
                        T_initial[i] = T1
                    else:
                        T_initial[i] = T2
            else:
                # Default to uniform if unknown dict type
                T_initial[:] = DEFAULT_T0 
                print(f"Warning: Unknown initial_condition_config type: {config_type}. Using default T0.")
        else:
            T_initial[:] = DEFAULT_T0 # Default fallback
            print("Warning: Invalid initial_condition_config. Using default T0.")

        # Apply Dirichlet boundary conditions to the initial state if they are fixed
        if self.boundary_conditions[0][0] == 0: # Left boundary Dirichlet
            T_initial[0] = self.boundary_conditions[0][1]
        if self.boundary_conditions[1][0] == 0: # Right boundary Dirichlet
            T_initial[-1] = self.boundary_conditions[1][1]
            
        return T_initial
    
    def explicit_finite_difference(self, h_coeff=0.0, T_env=0.0):
        # Stability check (simplified)
        if self.r > 0.5 and h_coeff == 0:
            print(f"Warning: r = {self.r:.4f} > 0.5, solution may be unstable for pure FTCS.")
        stability_factor = 1 - 2*self.r - h_coeff*self.dt
        if stability_factor < 0:
             print(f"Warning: Stability factor (1-2r-h*dt) = {stability_factor:.4f} < 0. Solution may be unstable.")

        T_matrix = np.zeros((self.nx, self.nt + 1))
        T_matrix[:, 0] = self.set_initial_condition()

        for j in range(self.nt):
            for i in range(1, self.nx - 1):
                term_diffusion = self.r * (T_matrix[i+1, j] - 2*T_matrix[i, j] + T_matrix[i-1, j])
                term_cooling = 0.0
                if h_coeff > 0:
                    term_cooling = h_coeff * self.dt * (T_matrix[i, j] - T_env)
                
                T_matrix[i, j+1] = T_matrix[i, j] + term_diffusion - term_cooling
            
            # Apply boundary conditions at each time step
            if self.boundary_conditions[0][0] == 0: # Left boundary Dirichlet
                T_matrix[0, j+1] = self.boundary_conditions[0][1]
            # else: Neumann or Robin can be implemented here for future extension
                
            if self.boundary_conditions[1][0] == 0: # Right boundary Dirichlet
                T_matrix[-1, j+1] = self.boundary_conditions[1][1]
            # else: Neumann or Robin can be implemented here for future extension

        return self.t, T_matrix
    
    def analytical_solution(self, T0_analytical=None, n_terms=100):
        if T0_analytical is None:
            if isinstance(self.initial_condition_config, (int, float)):
                 T0_analytical = float(self.initial_condition_config)
            else: # Fallback if complex initial condition is used for the numerical part
                 T0_analytical = DEFAULT_T0
                 print(f"Warning: analytical_solution called with complex initial_condition_config. Using default T0={DEFAULT_T0} for analytical part.")

        T_an = np.zeros((self.nx, self.nt + 1))
        
        # Meshgrid for x and t. Note: self.t includes t=0 up to total_time.
        X_grid, Time_grid = np.meshgrid(self.x, self.t, indexing='ij')

        for n_odd in range(1, 2 * n_terms, 2): # n = 1, 3, 5, ... (sum over odd n)
            kn = n_odd * np.pi / self.L
            term_coeff = (4 * T0_analytical) / (n_odd * np.pi)
            term_spatial = np.sin(kn * X_grid)
            term_temporal = np.exp(-(kn**2) * self.D * Time_grid)
            T_an += term_coeff * term_spatial * term_temporal
            
        # Ensure boundary conditions are met (though series should converge to them for these specific BCs)
        # This is mainly for the case where T(0,t) or T(L,t) are non-zero in general, but here they are 0.
        if self.boundary_conditions[0][0] == 0:
            T_an[0, :] = self.boundary_conditions[0][1]
        if self.boundary_conditions[1][0] == 0:
            T_an[-1, :] = self.boundary_conditions[1][1]
            
        return self.t, T_an
    
    def plot_evolution(self, t_array, T_matrix, title="Temperature Evolution", save_fig=False):
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

    def plot_comparison_at_times(self, t_numerical, T_numerical, T_analytical, time_indices_to_plot, title="Numerical vs Analytical", save_fig=False, filename_suffix="comparison"):
        num_plots = len(time_indices_to_plot)
        if num_plots == 0:
            print("No time points specified for comparison plot.")
            return

        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots), sharex=True)
        if num_plots == 1:
            axes = [axes] # Make it iterable if only one subplot

        for i, time_idx in enumerate(time_indices_to_plot):
            ax = axes[i]
            current_time = t_numerical[time_idx]
            ax.plot(self.x, T_numerical[:, time_idx], 'b-', label=f'Numerical (t={current_time:.2f}s)', linewidth=2)
            ax.plot(self.x, T_analytical[:, time_idx], 'r--', label=f'Analytical (t={current_time:.2f}s)', linewidth=2)
            ax.set_ylabel('Temperature T (K)')
            ax.set_title(f'Comparison at t = {current_time:.2f}s')
            ax.legend()
            ax.grid(True, alpha=0.5)

        axes[-1].set_xlabel('Position x (m)')
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle

        if save_fig:
            filename = f"{filename_suffix.lower().replace(' ', '_')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved as {filename}")
        
        plt.show()

    # This is the existing plot_3d_surface, we will add another one based on markdown later or integrate.
    # For now, let's keep the existing one and add the new one as a standalone function for Problem 1.
    def plot_3d_surface(self, t_array, T_matrix, title="3D Temperature Distribution", save_fig=False):
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
    solver = HeatDiffusionSolver(L=1.0, nx=81, total_time=0.1, dt=0.1/1000) # dt = total_time / nt
    
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
    solver = HeatDiffusionSolver(L=1.0, nx=81, total_time=0.1, dt=0.1/1000) # dt = total_time / nt
    
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

def solve_heat_diffusion_problem_1():
    """
    Solves the heat diffusion problem as described in the first problem of the markdown file.
    Uses explicit finite difference method and plots the 3D temperature distribution.
    This function is based on the first Python code block in '铝棒的热传导.md'.
    """
    # Parameters from the markdown's first code block
    K_param = 237.0        # Thermal conductivity W/(m·K)
    C_param = 900.0      # Specific heat capacity J/(kg·K)
    rho_param = 2700.0   # Density kg/m³
    
    L_md = 1.0       # Length of aluminum rod (m)
    dx_md = 0.01   # Spatial step (m)
    dt_md = 0.5    # Time step (s)
    
    # Calculated diffusion coefficient and stability parameter eta
    D_md = K_param / (C_param * rho_param) # Diffusion coefficient
    r_md = D_md * dt_md / (dx_md**2) # Stability parameter (eta)
    print(f"Markdown Problem 1 Parameters: D = {D_md:.6e}, r (eta) = {r_md:.4f}")

    Nx_md = int(L_md / dx_md) + 1 # Number of spatial grid points
    # Nt_md from markdown was 2000, implying total_time = 2000 * 0.5 = 1000s
    total_time_md = 1000.0 # s
    Nt_md = int(total_time_md / dt_md) # Number of time steps

    print(f"Grid: Nx={Nx_md}, Nt={Nt_md}, dx={dx_md:.4f}, dt={dt_md:.4f}, Total Time={total_time_md:.2f}s")

    # Initialize temperature array u(x,t)
    u_md = np.zeros((Nx_md, Nt_md + 1))

    # Boundary conditions
    u_md[0, :] = 0.0   # T(x=0, t) = 0 K
    u_md[-1, :] = 0.0  # T(x=L, t) = 0 K

    # Initial condition
    u_md[:, 0] = 100.0 # T(x, t=0) = 100 K
    u_md[0, 0] = 0.0   # Ensure boundaries are 0 at t=0
    u_md[-1, 0] = 0.0

    # Explicit finite difference scheme
    for j in range(Nt_md): # Loop from t=0 to t=(Nt-1)*dt
        for i in range(1, Nx_md - 1):
            u_md[i, j+1] = (1 - 2*r_md) * u_md[i, j] + \
                           r_md * (u_md[i+1, j] + u_md[i-1, j])
    
    # Create grid for plotting (using mgrid as in markdown for direct compatibility)
    # x_coords_md = np.linspace(0, L_md, Nx_md)
    # t_coords_md = np.linspace(0, total_time_md, Nt_md + 1)
    
    # Plotting the 3D surface
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Using mgrid approach from markdown for consistency with its plotting
    x_mg, t_mg = np.mgrid[0:L_md:Nx_md*1j, 0:total_time_md:(Nt_md+1)*1j]
    ax.plot_surface(x_mg, t_mg, u_md, cmap='rainbow')

    ax.set_xlabel("L (m)")
    ax.set_ylabel("t (s)")
    ax.set_zlabel("T (K)")
    ax.set_title("Heat Diffusion in Aluminum Rod (Problem 1 from Markdown)")
    plt.show()

    return u_md

def solve_heat_diffusion_analytical_comparison(params, save_plot=False, plot_filename='analytical_comparison.png'):
    """
    Solves the heat diffusion problem and compares with the analytical solution.
    This function is designed for Task 2: Comparison with Analytical Solution.

    Args:
        params (dict): Dictionary of parameters including:
            L (float): Length of the rod.
            nx (int): Number of spatial points.
            total_time (float): Total simulation time.
            dt (float): Time step.
            K (float): Thermal conductivity.
            C (float): Specific heat capacity.
            rho (float): Density.
            T0 (float): Initial temperature (uniform).
            bc_left_val (float): Temperature at left boundary (x=0).
            bc_right_val (float): Temperature at right boundary (x=L).
            n_terms_analytical (int): Number of terms for analytical solution series.
            plot_times (list): List of time points (in seconds) to plot for comparison.
        save_plot (bool): Whether to save the plot.
        plot_filename (str): Filename for the saved plot.
    """
    print(f"\n--- Running Analytical Comparison (Task 2) ---")
    print(f"Parameters: {params}")

    solver = HeatDiffusionSolver(
        L=params.get('L', DEFAULT_L),
        nx=params.get('nx', DEFAULT_NX),
        total_time=params.get('total_time', DEFAULT_TOTAL_TIME),
        dt=params.get('dt', DEFAULT_DT),
        K=params.get('K', DEFAULT_K),
        C=params.get('C', DEFAULT_C),
        rho=params.get('rho', DEFAULT_RHO),
        initial_condition_config=params.get('T0', DEFAULT_T0),
        boundary_conditions=((0, params.get('bc_left_val', 0.0)), 
                             (0, params.get('bc_right_val', 0.0)))
    )

    print(f"Solver initialized with r = {solver.r:.4f}")
    if solver.r > 0.5:
        print(f"Warning: Stability condition r <= 0.5 not met (r = {solver.r:.4f}). Results may be unstable.")

    t_num, T_num = solver.explicit_finite_difference()
    
    # Ensure T0_analytical is correctly passed if initial_condition_config was a simple float/int
    T0_for_analytical = params.get('T0', DEFAULT_T0) 
    if not isinstance(solver.initial_condition_config, (int, float)):
        print(f"Warning: Complex initial condition used for numerical. Using T0={T0_for_analytical} for analytical.")

    t_an, T_an = solver.analytical_solution(
        T0_analytical=T0_for_analytical, 
        n_terms=params.get('n_terms_analytical', 100)
    )

    # Convert plot_times from seconds to time indices
    time_indices_to_plot = []
    for t_sec in params.get('plot_times', [0, solver.total_time*0.1, solver.total_time*0.5, solver.total_time]):
        idx = np.argmin(np.abs(t_num - t_sec))
        time_indices_to_plot.append(idx)
    time_indices_to_plot = sorted(list(set(time_indices_to_plot))) # Ensure unique and sorted

    solver.plot_comparison_at_times(
        t_num, T_num, T_an, 
        time_indices_to_plot,
        title=f"Task 2: Numerical vs Analytical (T0={T0_for_analytical:.1f}K)",
        save_fig=save_plot,
        filename_suffix=plot_filename.replace('.png','')
    )
    print(f"Analytical comparison plot {'saved as ' + plot_filename if save_plot else 'shown'}.")
    return t_num, T_num, t_an, T_an

if __name__ == "__main__":
    # Common parameters for demonstrations
    default_params = {
        'L': 1.0, 'nx': 51, 'total_time': 1000.0, 'dt': 1.0, # dt=1.0 for r approx 0.24 (stable)
        'K': DEFAULT_K, 'C': DEFAULT_C, 'rho': DEFAULT_RHO,
        'T0': 100.0, # Default initial temperature
        'bc_left_val': 0.0, 'bc_right_val': 0.0, # Default boundary conditions T=0
        'plot_times': [0, 100, 500, 1000] # s, for plotting evolution
    }

    # --- Task 1: Basic Simulation (Uniform T0, Fixed BCs T=0) ---
    print("\n--- Task 1: Basic Simulation (Uniform T0, Fixed BCs T=0) ---")
    params_task1 = default_params.copy()
    # solve_heat_diffusion_problem_1 uses its own internal plotting, 
    # but we can call the class directly for more control if needed or use the specific function.
    # For consistency with student tasks, let's use the specific function.
    solve_heat_diffusion_problem_1()

    # --- Task 2: Comparison with Analytical Solution (for T0=100, BCs=0) ---
    print("\n--- Task 2: Comparison with Analytical Solution ---")
    params_task2 = default_params.copy()
    params_task2['n_terms_analytical'] = 100
    params_task2['plot_times'] = [0, int(params_task2['total_time'] * 0.05), 
                                 int(params_task2['total_time'] * 0.2), 
                                 int(params_task2['total_time'] * 0.5),
                                 params_task2['total_time']]
    solve_heat_diffusion_analytical_comparison(params_task2, save_plot=True, plot_filename='task2_analytical_comp.png')

    # --- Task 3: Stability Analysis (Illustrative) ---
    # This is more of a conceptual task. The `analyze_stability` function can be called.
    # Or, demonstrate by creating a solver with unstable params.
    print("\n--- Task 3: Stability Analysis (Illustrative Demonstration) ---")
    params_stable = default_params.copy()
    solver_stable_demo = HeatDiffusionSolver(
        L=params_stable['L'], nx=params_stable['nx'], total_time=params_stable['total_time'], dt=params_stable['dt'],
        initial_condition_config=params_stable['T0'], boundary_conditions=((0,params_stable['bc_left_val']),(0,params_stable['bc_right_val']))
    )
    print(f"Stable case from Task 1/2 setup: r = {solver_stable_demo.r:.4f}")

    params_unstable = default_params.copy()
    # To make r > 0.5, e.g., r = 0.6. D = K/(C*rho) approx 9.753e-5. dx = L/(nx-1) = 1/50 = 0.02.
    # dt_unstable = r_target * dx^2 / D = 0.6 * (0.02)^2 / 9.753e-5 = 2.46s
    params_unstable['dt'] = 2.5 # s, to make it unstable
    params_unstable['total_time'] = 50.0 # Shorter time to see instability blow up quickly
    
    solver_unstable_demo = HeatDiffusionSolver(
        L=params_unstable['L'], nx=params_unstable['nx'], total_time=params_unstable['total_time'], dt=params_unstable['dt'],
        initial_condition_config=params_unstable['T0'], boundary_conditions=((0,params_unstable['bc_left_val']),(0,params_unstable['bc_right_val']))
    )
    print(f"Attempting unstable case: r = {solver_unstable_demo.r:.4f}")
    if solver_unstable_demo.r > 0.5:
        t_unstable, T_unstable = solver_unstable_demo.explicit_finite_difference()
        solver_unstable_demo.plot_evolution(t_unstable, T_unstable, title=f"Task 3: Unstable Demo (r={solver_unstable_demo.r:.2f})", save_fig=True)
    else:
        print(f"Could not achieve r > 0.5 for unstable demo. Current r = {solver_unstable_demo.r:.4f}. Try increasing dt further.")
    # For full analysis, one might call analyze_stability() if it's made part of the class or a utility.

    # --- Task 4: Different Initial Conditions (Two Rods Example) ---
    print("\n--- Task 4: Different Initial Conditions (Two Rods) ---")
    params_task4_raw = {
        'L': 1.0, 'nx': 51, 'total_time': 200.0, 'dt': 0.1,
        'T0': {'type': 'two_rods', 'T1': 100.0, 'T2': 20.0, 'split_x': 0.5},
        'boundary_conditions': ((0, 0), (0, 0)) # T(0,t)=0, T(L,t)=0
    }
    params_task4_processed = params_task4_raw.copy()
    if 'T0' in params_task4_processed:
        params_task4_processed['initial_condition_config'] = params_task4_processed.pop('T0')

    solver_task4 = HeatDiffusionSolver(**params_task4_processed) # Unpack dict for direct class init
    t_task4, T_task4 = solver_task4.explicit_finite_difference()
    solver_task4.plot_evolution(t_task4, T_task4, title="Task 4: Two Rods Initial Condition", 
                                positions_to_plot=[0.25, 0.5, 0.75], 
                                times_to_plot=[0, 0.1, 0.5, 1.0, 5.0, 20.0] # Adjusted times
                               )
    solver_task4.plot_3d_surface(t_task4, T_task4, title="Task 4: 3D Surface Plot - Two Rods")

    # --- Task 5: Newton's Cooling Law ---
    print("\n--- Task 5: Newton's Cooling Law ---")
    params_task5 = default_params.copy()
    params_task5['T0'] = 100.0
    params_task5['bc_left_val'] = 0.0 # Ends at 0K as per typical setup
    params_task5['bc_right_val'] = 0.0
    params_task5['h_coeff'] = 0.01 # s^-1, example value from markdown
    params_task5['T_env'] = 25.0 # K
    params_task5['total_time'] = 2000.0 # Longer time for cooling effects
    params_task5['plot_times'] = [0, 200, 1000, 2000]

    solve_heat_diffusion_newton_cooling(params_task5, save_plot=True, plot_filename='task5_newton_cooling.png')

    print("\nAll demonstrations complete. Check for saved figures in the current directory.")
    print("\nScript finished.")