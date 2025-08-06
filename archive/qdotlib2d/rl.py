# quantum_dot_sim/rl_agent_torch.py
# This is the main driver script. It uses an optimization algorithm (CMA-ES)
# to find the optimal control pulse for performing a specific quantum gate.

import numpy as np
import torch
import cma
import time

# --- Import our new, modular libraries ---
from qsolv import init_domain, run_simulation
from qdot_lib.gate_library import get_ground_state, get_gate_target, calculate_fidelity

# --- High-Level Experiment Configuration ---
GATE_TO_OPTIMIZE = "HADAMARD"  # Can be "NOT" or "HADAMARD"
POTENTIAL_CONFIG = {'name': 'ideal', 'params': {'omega': 1.0}}

# -----------------------------------------------------------------------------
# 1. THE OBJECTIVE FUNCTION (The "Environment")
# -----------------------------------------------------------------------------
def objective_function(params):
    """
    Runs one full simulation episode to test a set of pulse parameters.
    The goal is to maximize the fidelity of the desired quantum gate.

    Args:
        params (list): A list of parameters the agent is testing:
                       [amp, freq, phase, pulse_center, pulse_width]

    Returns:
        float: The score to be minimized (1.0 - fidelity).
    """
    control_amp, control_freq, control_phase, pulse_center_t, pulse_width = params
    
    print(f"--- Testing: A={control_amp:.1f}, F={control_freq:.2f}, P={control_phase:.2f}, "
          f"T={pulse_center_t:.2f}, W={abs(pulse_width):.2f} ---")

    # --- Setup the simulation environment ---
    dt = 0.005
    Nt = 1000
    X, Y, dx, dy, V, K2 = init_domain(potential_config=POTENTIAL_CONFIG)

    # --- Get the initial and target states from our gate library ---
    omega = POTENTIAL_CONFIG['params'].get('omega', 1.0)
    initial_psi = get_ground_state(X, Y, omega)
    target_psi = get_gate_target(GATE_TO_OPTIMIZE, X, Y, omega)

    # --- Create the control pulse using the agent's chosen parameters ---
    # Use abs(pulse_width) to ensure it's always positive
    dt_vec = np.arange(Nt) * dt
    envelope = np.exp(-((dt_vec - pulse_center_t) / abs(pulse_width))**2)
    drive_pulse = control_amp * envelope * np.sin(control_freq * dt_vec + control_phase)
    control_shape = X  # Simple dipole interaction in the x-direction

    # --- Run the simulation using our solver ---
    final_psi = run_simulation(initial_psi, V, K2, dt, Nt, drive_pulse, control_shape)

    # --- Calculate the final reward using fidelity ---
    final_fidelity = calculate_fidelity(final_psi, target_psi, dx, dy)
    print(f"  > Resulting Fidelity: {final_fidelity:.6f}\n")

    # The optimizer minimizes, so we return (1 - fidelity).
    # The minimum value is 0, which corresponds to a perfect fidelity of 1.0.
    return 1.0 - final_fidelity

# -----------------------------------------------------------------------------
# 2. THE MAIN SCRIPT (The "Agent's Brain")
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    print(f"--- Starting Optimization for {GATE_TO_OPTIMIZE} gate ---")
    start_time = time.time()

    # --- Define the search space for the agent ---
    # [amplitude, frequency, phase, pulse_center_time, pulse_width]
    initial_guess = [10.0, 1.0, 0.0, 1.5, 0.5]
    initial_std_dev = 5.0

    # Run the CMA-ES optimizer
    best_params, es = cma.fmin2(objective_function,
                               initial_guess,
                               initial_std_dev,
                               options={'maxiter': 50, 'popsize': 10})

    # --- Print the final results ---
    duration = time.time() - start_time
    print("\n--- Optimization Finished ---")
    print(f"Total duration: {duration:.2f} s")

    # The best score 'fbest' is the minimized (1 - fidelity)
    best_fidelity = 1.0 - es.result.fbest

    print(f"\nHighest fidelity achieved for {GATE_TO_OPTIMIZE} gate: {best_fidelity:.6f}")
    print("Discovered with the following optimal parameters:")
    print(f"  - Control Amplitude: {best_params[0]:.3f}")
    print(f"  - Control Frequency: {best_params[1]:.3f}")
    print(f"  - Control Phase:     {best_params[2]:.3f}")
    print(f"  - Pulse Center Time: {best_params[3]:.3f}")
    print(f"  - Pulse Width:       {abs(best_params[4]):.3f}")

