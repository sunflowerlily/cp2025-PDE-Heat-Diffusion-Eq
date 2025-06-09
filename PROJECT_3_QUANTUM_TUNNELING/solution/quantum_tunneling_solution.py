#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: Quantum Tunneling Effect Numerical Simulation
File: quantum_tunneling_solution.py

This module implements the Crank-Nicolson method to solve the time-dependent
Schrödinger equation for a particle tunneling through a square barrier.

Physical Model:
- 1D time-dependent Schrödinger equation
- Square potential barrier
- Gaussian wave packet initial condition
- Quantum tunneling analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import warnings
warnings.filterwarnings('ignore')

# Physical constants (atomic units)
HBAR = 1.0  # Reduced Planck constant
M_E = 1.0   # Electron mass
EV_TO_AU = 1.0/27.211386245988  # eV to atomic units conversion

class QuantumTunnelingSolver:
    """
    Quantum tunneling effect solver using Crank-Nicolson method
    
    Solves the 1D time-dependent Schrödinger equation:
    i*hbar * ∂ψ/∂t = -hbar²/(2m) * ∂²ψ/∂x² + V(x)*ψ
    
    Attributes:
        L (float): Computational domain length
        nx (int): Number of spatial grid points
        nt (int): Number of time steps
        dx (float): Spatial step size
        dt (float): Time step size
        x (ndarray): Spatial coordinate array
        t (ndarray): Time coordinate array
        V (ndarray): Potential function
        psi (ndarray): Wavefunction array (nx, nt)
    """
    
    def __init__(self, L=20.0, nx=1000, t_final=50.0, nt=2000, 
                 barrier_height=0.18*EV_TO_AU, barrier_width=2.0, barrier_center=10.0):
        """
        Initialize quantum tunneling solver
        
        Args:
            L (float): Computational domain length (atomic units)
            nx (int): Number of spatial grid points
            t_final (float): Total simulation time (atomic units)
            nt (int): Number of time steps
            barrier_height (float): Barrier height (atomic units)
            barrier_width (float): Barrier width (atomic units)
            barrier_center (float): Barrier center position (atomic units)
        """
        self.L = L
        self.nx = nx
        self.nt = nt
        self.t_final = t_final
        
        # Initialize spatial and temporal grids
        self.dx = L / (nx - 1)
        self.dt = t_final / (nt - 1)
        self.x = np.linspace(0, L, nx)
        self.t = np.linspace(0, t_final, nt)
        
        # Set barrier parameters
        self.barrier_height = barrier_height
        self.barrier_width = barrier_width
        self.barrier_center = barrier_center
        
        # Initialize potential and wavefunction arrays
        self.V = np.zeros(nx)
        self.psi = np.zeros((nx, nt), dtype=complex)
        
        print(f"Solver initialized:")
        print(f"  Domain: [0, {L}] a.u., Grid: {nx} points, dx = {self.dx:.4f}")
        print(f"  Time: [0, {t_final}] a.u., Steps: {nt}, dt = {self.dt:.4f}")
        print(f"  Barrier: height = {barrier_height:.4f} a.u., width = {barrier_width} a.u.")
    
    def setup_potential(self):
        """
        Set up square barrier potential function
        
        Returns:
            ndarray: Potential function V(x)
        
        Barrier model:
        V(x) = 0           for x < x1 or x > x2
        V(x) = V0          for x1 ≤ x ≤ x2
        
        where x1, x2 are the left and right boundaries of the barrier
        """
        # Calculate barrier boundaries
        x1 = self.barrier_center - self.barrier_width / 2
        x2 = self.barrier_center + self.barrier_width / 2
        
        # Create potential array
        V = np.zeros(self.nx)
        
        # Set barrier region potential
        barrier_mask = (self.x >= x1) & (self.x <= x2)
        V[barrier_mask] = self.barrier_height
        
        print(f"Potential setup: barrier from {x1:.2f} to {x2:.2f} a.u.")
        return V
    
    def setup_initial_wavepacket(self, x0=5.0, sigma=1.0, k0=2.0):
        """
        Set up initial Gaussian wave packet
        
        Args:
            x0 (float): Wave packet center position
            sigma (float): Wave packet width parameter
            k0 (float): Initial wave number corresponding to momentum
        
        Returns:
            ndarray: Normalized initial wavefunction
        
        Gaussian wave packet formula:
        ψ(x,0) = exp[-(x-x0)²/(2σ²)] * exp(ik0*x)
        
        Physical meaning:
        - Gaussian envelope describes position uncertainty
        - Plane wave factor exp(ik0*x) describes initial momentum
        - k0 = p0/ħ, where p0 is initial momentum
        """
        # Calculate Gaussian envelope
        gaussian_envelope = np.exp(-(self.x - x0)**2 / (2 * sigma**2))
        
        # Calculate plane wave factor
        plane_wave = np.exp(1j * k0 * self.x)
        
        # Combine wavefunction
        psi_initial = gaussian_envelope * plane_wave
        
        # Normalize
        norm = np.sqrt(np.trapz(np.abs(psi_initial)**2, self.x))
        psi_initial = psi_initial / norm
        
        # Calculate initial energy
        initial_energy = HBAR**2 * k0**2 / (2 * M_E)
        print(f"Initial wave packet: center = {x0}, width = {sigma}, k0 = {k0}")
        print(f"Initial energy: {initial_energy:.4f} a.u. ({initial_energy/EV_TO_AU:.2f} eV)")
        
        return psi_initial
    
    def build_crank_nicolson_matrices(self):
        """
        Build coefficient matrices for Crank-Nicolson scheme
        
        Returns:
            tuple: (A_matrix, B_matrix) left and right coefficient matrices
        
        Crank-Nicolson scheme:
        (I + iΔt/(2ħ) * H) * ψ^(n+1) = (I - iΔt/(2ħ) * H) * ψ^n
        
        where H is the discretized Hamiltonian operator:
        H = -ħ²/(2m) * ∇² + V(x)
        
        Discretized kinetic operator (second-order central difference):
        ∇²ψ_j ≈ (ψ_{j+1} - 2ψ_j + ψ_{j-1}) / Δx²
        """
        # Calculate kinetic coefficient
        r = HBAR**2 / (2 * M_E * self.dx**2)
        
        # Build kinetic operator tridiagonal matrix
        # Main diagonal: 2r
        # Off-diagonals: -r
        kinetic_main = 2 * r * np.ones(self.nx)
        kinetic_off = -r * np.ones(self.nx - 1)
        
        # Add potential term
        hamiltonian_main = kinetic_main + self.V
        
        # Build A and B matrices
        # Time step factor
        alpha = 1j * self.dt / (2 * HBAR)
        
        # A matrix = I + α*H
        A_main = np.ones(self.nx) + alpha * hamiltonian_main
        A_off = alpha * kinetic_off
        
        # B matrix = I - α*H
        B_main = np.ones(self.nx) - alpha * hamiltonian_main
        B_off = -alpha * kinetic_off
        
        # Use scipy.sparse.diags to build sparse matrices
        A_matrix = diags([A_off, A_main, A_off], [-1, 0, 1], 
                        shape=(self.nx, self.nx), format='csc')
        B_matrix = diags([B_off, B_main, B_off], [-1, 0, 1], 
                        shape=(self.nx, self.nx), format='csc')
        
        # Handle boundary conditions (absorbing boundaries)
        # Set boundary points to have zero wavefunction
        A_matrix[0, :] = 0
        A_matrix[0, 0] = 1
        A_matrix[-1, :] = 0
        A_matrix[-1, -1] = 1
        
        B_matrix[0, :] = 0
        B_matrix[0, 0] = 1
        B_matrix[-1, :] = 0
        B_matrix[-1, -1] = 1
        
        print("Crank-Nicolson matrices constructed")
        return A_matrix, B_matrix
    
    def solve_time_evolution(self):
        """
        Solve wavefunction time evolution
        
        Returns:
            ndarray: Wavefunction evolution array psi(x,t)
        
        Algorithm flow:
        1. Set initial wavefunction
        2. Build Crank-Nicolson matrices
        3. Time loop solving linear system
        4. Store wavefunction at each time step
        
        Numerical stability:
        - Crank-Nicolson scheme is unconditionally stable
        - Preserves wavefunction normalization (within numerical error)
        - Conserves probability
        """
        # Set initial wavefunction
        psi_initial = self.setup_initial_wavepacket()
        self.psi[:, 0] = psi_initial
        
        # Build Crank-Nicolson matrices
        A_matrix, B_matrix = self.build_crank_nicolson_matrices()
        
        print("Starting time evolution...")
        
        # Time evolution loop
        for n in range(self.nt - 1):
            # Calculate right-hand side vector
            rhs = B_matrix @ self.psi[:, n]
            
            # Solve linear system
            self.psi[:, n+1] = spsolve(A_matrix, rhs)
            
            # Apply boundary conditions
            self.psi[0, n+1] = 0
            self.psi[-1, n+1] = 0
            
            # Check normalization periodically
            if n % 200 == 0:
                norm = np.trapz(np.abs(self.psi[:, n+1])**2, self.x)
                print(f"Time step {n}: t = {self.t[n+1]:.2f}, Norm = {norm:.6f}")
        
        print("Time evolution completed")
        return self.psi
    
    def calculate_probability_density(self):
        """
        Calculate probability density |ψ(x,t)|²
        
        Returns:
            ndarray: Probability density array
        
        Physical meaning:
        - |ψ(x,t)|² represents probability density of finding particle at position x, time t
        - Integral ∫|ψ(x,t)|²dx = 1 (normalization condition)
        """
        probability_density = np.abs(self.psi)**2
        return probability_density
    
    def calculate_probability_current(self):
        """
        Calculate probability current density j(x,t)
        
        Returns:
            ndarray: Probability current density array
        
        Probability current density formula:
        j(x,t) = (ħ/2mi) * [ψ* ∂ψ/∂x - ψ ∂ψ*/∂x]
        
        Physical meaning:
        - Describes probability flow in space
        - Satisfies continuity equation: ∂|ψ|²/∂t + ∂j/∂x = 0
        - Positive values indicate rightward flow, negative leftward
        
        Numerical implementation:
        - Use central difference for spatial derivatives
        - Handle boundary points specially
        """
        # Initialize current array
        current = np.zeros_like(self.psi, dtype=float)
        
        # Calculate spatial derivatives (central difference)
        for n in range(self.nt):
            psi_n = self.psi[:, n]
            
            # Interior points use central difference
            dpsi_dx = np.zeros_like(psi_n)
            dpsi_dx[1:-1] = (psi_n[2:] - psi_n[:-2]) / (2 * self.dx)
            
            # Boundary points use forward/backward difference
            dpsi_dx[0] = (psi_n[1] - psi_n[0]) / self.dx
            dpsi_dx[-1] = (psi_n[-1] - psi_n[-2]) / self.dx
            
            # Calculate probability current density
            current[:, n] = (HBAR / (2j * M_E)) * (np.conj(psi_n) * dpsi_dx - psi_n * np.conj(dpsi_dx))
            current[:, n] = np.real(current[:, n])  # Take real part
        
        return current
    
    def analyze_tunneling_effect(self):
        """
        Analyze tunneling effect
        
        Returns:
            dict: Dictionary containing transmission coefficient, reflection coefficient, etc.
        
        Analysis content:
        1. Transmission coefficient: probability of passing through barrier
        2. Reflection coefficient: probability of being reflected by barrier
        3. Tunneling time: characteristic time for wave packet to traverse barrier
        4. Energy analysis: comparison with classical expectation
        
        Calculation method:
        - Divide computational domain into three regions: incident, barrier, transmitted
        - Calculate probability in each region after sufficient time
        - Analyze wave packet shape changes
        """
        # Define region boundaries
        barrier_left = self.barrier_center - self.barrier_width / 2
        barrier_right = self.barrier_center + self.barrier_width / 2
        
        # Find corresponding grid indices
        idx_left = np.argmin(np.abs(self.x - barrier_left))
        idx_right = np.argmin(np.abs(self.x - barrier_right))
        
        # Calculate final time probability in each region
        prob_density_final = self.calculate_probability_density()[:, -1]
        
        incident_prob = np.trapz(prob_density_final[:idx_left], self.x[:idx_left])
        barrier_prob = np.trapz(prob_density_final[idx_left:idx_right], self.x[idx_left:idx_right])
        transmitted_prob = np.trapz(prob_density_final[idx_right:], self.x[idx_right:])
        
        # Calculate transmission and reflection coefficients
        transmission_coeff = transmitted_prob
        reflection_coeff = incident_prob
        
        # Calculate tunneling time (time when maximum probability reaches barrier center)
        prob_density = self.calculate_probability_density()
        center_idx = np.argmin(np.abs(self.x - self.barrier_center))
        center_prob_evolution = prob_density[center_idx, :]
        max_prob_time_idx = np.argmax(center_prob_evolution)
        tunneling_time = self.t[max_prob_time_idx]
        
        # Calculate classical velocity and expected arrival time
        k0 = 2.0  # Initial wave number (from setup_initial_wavepacket)
        classical_velocity = HBAR * k0 / M_E
        x0 = 5.0  # Initial position
        classical_time = (self.barrier_center - x0) / classical_velocity
        
        # Analysis results
        results = {
            'transmission_coefficient': transmission_coeff,
            'reflection_coefficient': reflection_coeff,
            'barrier_probability': barrier_prob,
            'total_probability': incident_prob + barrier_prob + transmitted_prob,
            'tunneling_time': tunneling_time,
            'classical_time': classical_time,
            'time_delay': tunneling_time - classical_time,
            'barrier_penetration_depth': self.barrier_width
        }
        
        print(f"\nTunneling Analysis Results:")
        print(f"  Transmission coefficient: {transmission_coeff:.4f}")
        print(f"  Reflection coefficient: {reflection_coeff:.4f}")
        print(f"  Barrier probability: {barrier_prob:.4f}")
        print(f"  Total probability: {results['total_probability']:.4f}")
        print(f"  Tunneling time: {tunneling_time:.2f} a.u.")
        print(f"  Classical time: {classical_time:.2f} a.u.")
        print(f"  Time delay: {results['time_delay']:.2f} a.u.")
        
        return results
    
    def plot_wavefunction_evolution(self, time_indices=None, save_fig=False):
        """
        Plot wavefunction evolution
        
        Args:
            time_indices (list): List of time indices to plot
            save_fig (bool): Whether to save figure
        
        Plot content:
        1. Probability density |ψ(x,t)|² time evolution
        2. Potential function overlay
        3. Multiple time snapshots comparison
        4. Clear legends and annotations
        """
        if time_indices is None:
            time_indices = [0, self.nt//4, self.nt//2, 3*self.nt//4, self.nt-1]
        
        # Calculate probability density
        prob_density = self.calculate_probability_density()
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot potential function
        plt.subplot(2, 1, 1)
        plt.plot(self.x, self.V/EV_TO_AU, 'k-', linewidth=2, label='Potential V(x)')
        plt.ylabel('Energy (eV)')
        plt.title('Quantum Tunneling: Wavefunction Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, self.L)
        
        # Plot probability density evolution
        plt.subplot(2, 1, 2)
        colors = plt.cm.viridis(np.linspace(0, 1, len(time_indices)))
        
        for i, t_idx in enumerate(time_indices):
            time_value = self.t[t_idx]
            plt.plot(self.x, prob_density[:, t_idx], 
                     color=colors[i], linewidth=2, 
                     label=f't = {time_value:.1f} a.u.')
        
        # Add barrier region shading
        barrier_left = self.barrier_center - self.barrier_width / 2
        barrier_right = self.barrier_center + self.barrier_width / 2
        plt.axvspan(barrier_left, barrier_right, alpha=0.2, color='red', label='Barrier')
        
        plt.xlabel('Position x (a.u.)')
        plt.ylabel('Probability Density |ψ|²')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, self.L)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig('wavefunction_evolution.png', dpi=300, bbox_inches='tight')
            print("Figure saved as wavefunction_evolution.png")
        
        plt.show()
    
    def create_animation(self, filename='tunneling_animation.gif', interval=50):
        """
        Create wave packet evolution animation
        
        Args:
            filename (str): Animation filename
            interval (int): Frame interval (milliseconds)
        
        Animation content:
        1. Real-time display of probability density changes
        2. Fixed display of potential barrier
        3. Time label updates
        4. Smooth animation effects
        """
        # Calculate probability density
        prob_density = self.calculate_probability_density()
        
        # Set up figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot potential function (static)
        ax1.plot(self.x, self.V/EV_TO_AU, 'k-', linewidth=2)
        ax1.set_ylabel('Potential V(x) (eV)')
        ax1.set_title('Quantum Tunneling Animation')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, self.L)
        
        # Add barrier region shading
        barrier_left = self.barrier_center - self.barrier_width / 2
        barrier_right = self.barrier_center + self.barrier_width / 2
        ax1.axvspan(barrier_left, barrier_right, alpha=0.2, color='red')
        
        # Initialize probability density plot line
        line, = ax2.plot([], [], 'b-', linewidth=2)
        ax2.set_xlim(0, self.L)
        ax2.set_ylim(0, np.max(prob_density) * 1.1)
        ax2.set_xlabel('Position x (a.u.)')
        ax2.set_ylabel('Probability Density |ψ|²')
        ax2.grid(True, alpha=0.3)
        
        # Add barrier region shading
        ax2.axvspan(barrier_left, barrier_right, alpha=0.2, color='red')
        
        # Add time text
        time_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, fontsize=12)
        
        # Define animation update function
        def animate(frame):
            line.set_data(self.x, prob_density[:, frame])
            time_text.set_text(f'Time = {self.t[frame]:.2f} a.u.')
            return line, time_text
        
        # Create animation
        ani = animation.FuncAnimation(fig, animate, frames=self.nt,
                                     interval=interval, blit=True, repeat=True)
        
        # Save animation
        try:
            ani.save(filename, writer='pillow', fps=1000//interval)
            print(f"Animation saved as {filename}")
        except Exception as e:
            print(f"Failed to save animation: {e}")
        
        plt.show()
        return ani
    
    def run_full_simulation(self):
        """
        Run complete quantum tunneling simulation
        
        Returns:
            dict: Dictionary containing all analysis results
        
        Simulation flow:
        1. Initialize system
        2. Solve time evolution
        3. Analyze tunneling effect
        4. Generate visualization results
        5. Return analysis report
        """
        print("Starting quantum tunneling simulation...")
        
        # Set up potential function
        self.V = self.setup_potential()
        print("Potential function setup completed")
        
        # Solve time evolution
        self.solve_time_evolution()
        print("Time evolution solved")
        
        # Analyze tunneling effect
        tunneling_results = self.analyze_tunneling_effect()
        print("Tunneling effect analysis completed")
        
        # Generate visualization
        self.plot_wavefunction_evolution(save_fig=True)
        print("Wavefunction evolution plot generated")
        
        # Create animation (optional)
        try:
            self.create_animation()
            print("Animation created")
        except Exception as e:
            print(f"Animation creation failed: {e}")
        
        # Return results
        results = {
            'solver_parameters': {
                'L': self.L,
                'nx': self.nx,
                'nt': self.nt,
                'dx': self.dx,
                'dt': self.dt
            },
            'physical_parameters': {
                'barrier_height': self.barrier_height,
                'barrier_width': self.barrier_width,
                'barrier_center': self.barrier_center
            },
            'tunneling_analysis': tunneling_results
        }
        
        print("\nSimulation completed successfully!")
        return results

def compare_different_energies(energies, barrier_height=0.18*EV_TO_AU):
    """
    Compare tunneling effects at different initial energies
    
    Args:
        energies (list): List of different initial kinetic energies
        barrier_height (float): Barrier height
    
    Returns:
        dict: Tunneling coefficients at different energies
    
    Physical background:
    - Classical particle: cannot pass through barrier when E < V0
    - Quantum particle: has tunneling probability even when E < V0
    - Tunneling probability increases with energy
    """
    # Initialize results dictionary
    results = {'energies': [], 'transmission_coefficients': []}
    
    print("\nComparing different energies:")
    print("=" * 40)
    
    # Loop through different energies
    for E in energies:
        # Calculate corresponding wave number k = sqrt(2mE)/ħ
        k0 = np.sqrt(2 * M_E * E) / HBAR
        
        # Create solver
        solver = QuantumTunnelingSolver(barrier_height=barrier_height, 
                                      L=20.0, nx=800, t_final=40.0, nt=1600)
        
        # Set up potential
        solver.V = solver.setup_potential()
        
        # Set up initial wave packet with corresponding energy
        psi_initial = solver.setup_initial_wavepacket(x0=5.0, sigma=1.0, k0=k0)
        solver.psi[:, 0] = psi_initial
        
        # Build matrices and solve
        A_matrix, B_matrix = solver.build_crank_nicolson_matrices()
        
        # Time evolution (simplified)
        for n in range(solver.nt - 1):
            rhs = B_matrix @ solver.psi[:, n]
            solver.psi[:, n+1] = spsolve(A_matrix, rhs)
            solver.psi[0, n+1] = 0
            solver.psi[-1, n+1] = 0
        
        # Analyze tunneling
        tunneling_analysis = solver.analyze_tunneling_effect()
        
        # Record results
        transmission_coeff = tunneling_analysis['transmission_coefficient']
        results['energies'].append(E)
        results['transmission_coefficients'].append(transmission_coeff)
        
        print(f"Energy: {E:.4f} a.u. ({E/EV_TO_AU:.2f} eV), Transmission: {transmission_coeff:.4f}")
    
    # Plot energy dependence
    plt.figure(figsize=(10, 6))
    plt.plot(np.array(results['energies'])/EV_TO_AU, results['transmission_coefficients'], 
             'bo-', linewidth=2, markersize=8)
    plt.axvline(x=barrier_height/EV_TO_AU, color='r', linestyle='--', 
               label=f'Barrier Height = {barrier_height/EV_TO_AU:.2f} eV')
    plt.xlabel('Initial Energy (eV)')
    plt.ylabel('Transmission Coefficient')
    plt.title('Quantum Tunneling: Energy Dependence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('energy_dependence.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

def verify_conservation_laws(solver):
    """
    Verify conservation laws
    
    Args:
        solver (QuantumTunnelingSolver): Solver with completed calculation
    
    Returns:
        dict: Conservation law verification results
    
    Verification content:
    1. Probability conservation: ∫|ψ|²dx = 1
    2. Continuity equation: ∂|ψ|²/∂t + ∂j/∂x = 0
    3. Energy conservation (in dissipationless system)
    """
    # Calculate probability density and current
    prob_density = solver.calculate_probability_density()
    prob_current = solver.calculate_probability_current()
    
    # Verify probability conservation
    total_prob = np.zeros(solver.nt)
    for n in range(solver.nt):
        total_prob[n] = np.trapz(prob_density[:, n], solver.x)
    
    # Verify continuity equation
    continuity_error = np.zeros((solver.nx-2, solver.nt-1))
    for n in range(solver.nt-1):
        # Time derivative
        dpdt = (prob_density[1:-1, n+1] - prob_density[1:-1, n]) / solver.dt
        
        # Space derivative
        djdx = (prob_current[2:, n] - prob_current[:-2, n]) / (2 * solver.dx)
        
        # Continuity equation error
        continuity_error[:, n] = dpdt + djdx
    
    # Generate verification results
    results = {
        'probability_conservation': {
            'mean_total_probability': np.mean(total_prob),
            'std_total_probability': np.std(total_prob),
            'max_deviation': np.max(np.abs(total_prob - 1.0))
        },
        'continuity_equation': {
            'max_error': np.max(np.abs(continuity_error)),
            'rms_error': np.sqrt(np.mean(continuity_error**2))
        }
    }
    
    print(f"\nConservation Laws Verification:")
    print(f"  Probability conservation:")
    print(f"    Mean total probability: {results['probability_conservation']['mean_total_probability']:.6f}")
    print(f"    Standard deviation: {results['probability_conservation']['std_total_probability']:.6f}")
    print(f"    Maximum deviation: {results['probability_conservation']['max_deviation']:.6f}")
    print(f"  Continuity equation:")
    print(f"    Maximum error: {results['continuity_equation']['max_error']:.6f}")
    print(f"    RMS error: {results['continuity_equation']['rms_error']:.6f}")
    
    return results

if __name__ == "__main__":
    # Example usage
    print("Quantum Tunneling Effect Numerical Simulation")
    print("=" * 50)
    
    # Create solver instance
    solver = QuantumTunnelingSolver(
        L=20.0,           # Computational domain length
        nx=1000,          # Number of spatial grid points
        t_final=50.0,     # Total time
        nt=2000,          # Number of time steps
        barrier_height=0.18*EV_TO_AU,  # Barrier height
        barrier_width=2.0,             # Barrier width
        barrier_center=10.0             # Barrier position
    )
    
    # Run complete simulation
    results = solver.run_full_simulation()
    
    # Print results
    print("\nSimulation Results:")
    print(f"Transmission coefficient: {results['tunneling_analysis']['transmission_coefficient']:.4f}")
    print(f"Reflection coefficient: {results['tunneling_analysis']['reflection_coefficient']:.4f}")
    
    # Verify conservation laws
    conservation_results = verify_conservation_laws(solver)
    print("\nConservation laws verified")
    
    # Compare different energies
    energies = np.linspace(0.05, 0.25, 8) * EV_TO_AU
    energy_results = compare_different_energies(energies)
    
    print("\nQuantum tunneling simulation completed successfully!")