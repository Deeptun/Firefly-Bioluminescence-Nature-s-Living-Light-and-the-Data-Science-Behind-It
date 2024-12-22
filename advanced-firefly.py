import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors

def arrhenius_equation(k, Ea, T, R=8.314):
    """
    Calculate temperature-dependent reaction rate
    
    Parameters:
    k: base reaction rate at reference temperature
    Ea: activation energy (J/mol)
    T: temperature (Kelvin)
    R: gas constant
    """
    T_ref = 298.15  # 25°C reference temperature
    return k * np.exp((Ea/R) * (1/T_ref - 1/T))

def hill_equation(S, Km, n):
    """
    Model ATP dependency using Hill equation
    
    Parameters:
    S: substrate concentration (ATP)
    Km: Michaelis constant
    n: Hill coefficient
    """
    return S**n / (Km**n + S**n)

def bioluminescence_reaction(state, t, params):
    """
    Enhanced model including temperature and ATP effects
    
    Parameters:
    state: concentrations [E, S, ES, P, ATP]
    params: dictionary of reaction parameters
    """
    E, S, ES, P, ATP = state
    
    # Get temperature-adjusted rate constants
    k1 = arrhenius_equation(params['k1'], params['Ea1'], params['T'])
    k2 = arrhenius_equation(params['k2'], params['Ea2'], params['T'])
    k3 = arrhenius_equation(params['k3'], params['Ea3'], params['T'])
    k4 = arrhenius_equation(params['k4'], params['Ea4'], params['T'])
    
    # ATP modulation
    atp_effect = hill_equation(ATP, params['Km_ATP'], params['n_ATP'])
    
    # Enhanced rate equations
    dE_dt = -k1*E*S*atp_effect + k2*ES + k3*ES
    dS_dt = -k1*E*S*atp_effect + k2*ES
    dES_dt = k1*E*S*atp_effect - k2*ES - k3*ES
    dP_dt = k3*ES - k4*P
    dATP_dt = -k3*ES*0.1  # ATP consumption
    
    return [dE_dt, dS_dt, dES_dt, dP_dt, dATP_dt]

def kuramoto_coupling(phases, natural_frequencies, K):
    """
    Implement Kuramoto model for firefly synchronization
    
    Parameters:
    phases: current phase of each firefly
    natural_frequencies: intrinsic flash frequencies
    K: coupling strength
    """
    N = len(phases)
    coupling = np.zeros(N)
    
    for i in range(N):
        coupling[i] = (K/N) * np.sum(np.sin(phases - phases[i]))
    
    return natural_frequencies + coupling

def simulate_firefly_group(num_fireflies, duration, num_points, temp_range=(293.15, 303.15)):
    """
    Simulate multiple interacting fireflies
    
    Parameters:
    num_fireflies: number of fireflies to simulate
    duration: simulation time in seconds
    num_points: number of time points
    temp_range: temperature range (K)
    """
    # Base parameters
    params = {
        'k1': 100.0, 'k2': 10.0, 'k3': 15.0, 'k4': 5.0,
        'Ea1': 50000, 'Ea2': 45000, 'Ea3': 55000, 'Ea4': 40000,
        'Km_ATP': 0.5, 'n_ATP': 2,
        'T': 298.15
    }
    
    # Initialize arrays
    t = np.linspace(0, duration, num_points)
    all_solutions = []
    positions = []
    phases = np.random.uniform(0, 2*np.pi, num_fireflies)
    natural_frequencies = np.random.normal(2*np.pi/5, 0.1, num_fireflies)
    
    # Generate random positions
    for _ in range(num_fireflies):
        positions.append((np.random.uniform(-10, 10), np.random.uniform(-10, 10)))
    
    # Simulate each firefly
    for i in range(num_fireflies):
        # Vary temperature and initial conditions slightly
        params['T'] = np.random.uniform(*temp_range)
        initial_state = [
            1.0 + np.random.normal(0, 0.1),  # E
            1.0 + np.random.normal(0, 0.1),  # S
            0,  # ES
            0,  # P
            1.0 + np.random.normal(0, 0.1)   # ATP
        ]
        
        solution = odeint(bioluminescence_reaction, initial_state, t, args=(params,))
        all_solutions.append(solution)
    
    return t, all_solutions, positions, phases, natural_frequencies

def visualize_firefly_group(t, all_solutions, positions, phases, natural_frequencies):
    """
    Create advanced visualization of firefly group
    
    Parameters:
    t: time points
    all_solutions: list of concentration profiles for each firefly
    positions: list of (x,y) positions for each firefly
    phases: initial phases of fireflies
    natural_frequencies: intrinsic frequencies of fireflies
    """
    fig = plt.figure(figsize=(15, 8))
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((2, 3), (1, 0), colspan=2)
    ax3 = plt.subplot2grid((2, 3), (0, 2), rowspan=2)
    
    # Plot average concentrations
    avg_solutions = np.mean(all_solutions, axis=0)
    species = ['Luciferase', 'Luciferin', 'ES Complex', 'Light', 'ATP']
    colors = ['b', 'g', 'r', 'y', 'm']
    
    for i, (name, color) in enumerate(zip(species, colors)):
        ax1.plot(t, avg_solutions[:, i], label=name, color=color)
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Average Concentration (M)')
    ax1.set_title('Average Species Concentrations')
    ax1.legend()
    ax1.grid(True)
    
    # Plot phase synchronization
    K = 1.0  # Coupling strength
    dt = t[1] - t[0]
    phase_evolution = np.zeros((len(t), len(phases)))
    phase_evolution[0] = phases
    
    for i in range(1, len(t)):
        dphases = kuramoto_coupling(phase_evolution[i-1], natural_frequencies, K)
        phase_evolution[i] = phase_evolution[i-1] + dphases * dt
    
    for i in range(len(phases)):
        ax2.plot(t, np.sin(phase_evolution[:, i]), alpha=0.3)
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Phase')
    ax2.set_title('Phase Synchronization')
    ax2.grid(True)
    
    def update(frame):
        ax3.clear()
        circles = []
        
        # for i, pos in enumerate(positions):
            # # Light intensity from reaction
            # intensity = all_solutions[i][frame, 3] / np.max(all_solutions[i][:, 3])
            # # Combine with phase
            # total_intensity = intensity * (0.5 + 0.5 * np.sin(phase_evolution[frame, i]))
            
            # circle = Circle(pos, 0.5, alpha=total_intensity)
            # circles.append(circle)
        
        for i, pos in enumerate(positions):
            # Light intensity from reaction
            # Ensure intensity is positive and normalized
            raw_intensity = all_solutions[i][frame, 3]
            max_intensity = np.max(all_solutions[i][:, 3])
            intensity = np.clip(raw_intensity / (max_intensity + 1e-10), 0, 1)
            
            # Combine with phase and ensure total_intensity is in [0,1]
            phase_factor = 0.5 + 0.5 * np.sin(phase_evolution[frame, i])
            total_intensity = np.clip(intensity * phase_factor, 0, 1)
            
            circle = Circle(pos, 0.5, alpha=total_intensity)
            circles.append(circle)
        
        collection = PatchCollection(circles, cmap='YlOrRd')
        collection.set_array(np.linspace(0, 1, len(circles)))
        ax3.add_collection(collection)
        
        ax3.set_xlim(-12, 12)
        ax3.set_ylim(-12, 12)
        ax3.set_title(f'Firefly Positions (t={t[frame]:.2f}s)')
        ax3.set_aspect('equal')
        ax3.grid(True)
        
        return ax3,
    
    ani = animation.FuncAnimation(fig, update, frames=len(t), 
                                interval=50, blit=True)
    
    plt.tight_layout()
    plt.show()
    # return fig

# Run simulation
num_fireflies = 10
duration = 20
num_points = 400

t, all_solutions, positions, phases, natural_frequencies = simulate_firefly_group(
    num_fireflies, duration, num_points
)

# Create visualization
visualize_firefly_group(t, all_solutions, positions, phases, natural_frequencies)
