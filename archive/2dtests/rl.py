import numpy as np
import torch
import cma
import time

# Import the PyTorch-based simulation functions from your other script
from torchified import init_domain_torch, propagate_torch, reward_torch


# -----------------------------------------------------------------------------
# 1. THE OBJECTIVE FUNCTION (The "Environment")
# -----------------------------------------------------------------------------
# This is the core of the RL setup. The optimizer will call this function
# repeatedly with different parameters to find the best ones.
def objective_function(params):
    """
    Runs one full simulation "episode" with a given set of control parameters.

    Args:
        params (list): A list of parameters the agent is testing:
                       [amp, freq, phase, pulse_center, pulse_width]

    Returns:
        float: The score to be minimized. We return the NEGATIVE of the
               transmission because the optimizer's goal is to find the minimum.
    """
    # Unpack the parameters for this episode
    control_amp, control_freq, control_phase, pulse_center_t, pulse_width = params
    
    # --- Print feedback for the user ---
    print("--- New Episode ---")
    print(f"Testing Params: A={control_amp:.1f}, F={control_freq:.2f}, "
          f"P={control_phase:.2f}, T={pulse_center_t:.2f}, W={pulse_width:.2f}")

    # --- Setup the simulation environment for this episode ---
    dt = 0.005
    Nt = 1000
    X, Y, dx, dy, V, halfkprop, bmask = init_domain_torch(dt=dt)

    # --- Initial wavepacket (always starts the same) ---
    k0, sigma = 5.0, 0.5
    x0, y0 = -2.0, 0.0
    psi0 = (
        1 / (sigma * np.sqrt(np.pi))
        * torch.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
        * torch.exp(1j * k0 * X)
    ).to(torch.complex64)
    psi = psi0.clone()

    # --- Create the control pulse using the agent's chosen parameters ---
    dt_vec = np.arange(Nt) * dt
    envelope = np.exp(-((dt_vec - pulse_center_t) / pulse_width)**2)
    drive = control_amp * envelope * np.sin(control_freq * dt_vec + control_phase)
    control_shape = X * bmask

    # --- Run the simulation loop ---
    for n in range(Nt):
        psi = propagate_torch(psi, V, halfkprop, dt, drive[n], control_shape)

    # --- Get the final reward for the episode ---
    final_transmission = reward_torch(psi, X, dx, dy)
    print(f"  > Resulting Transmission: {final_transmission:.5f}\n")

    # The optimizer minimizes, so we return the inverse of our goal
    return -final_transmission

# -----------------------------------------------------------------------------
# 2. THE MAIN SCRIPT (The "Agent's Brain")
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    print("--- Starting Reinforcement Learning Agent ---")
    start_time = time.time()

    # --- Define the search space for the agent ---
    # [amplitude, frequency, phase, pulse_center_time, pulse_width]
    initial_guess = [100.0, 2 * np.pi * 2.0, 0.0, 1.0, 0.2]
    
    # This tells the agent how much to vary the parameters initially.
    # A larger value means a wider, more exploratory search.
    initial_std_dev = 30.0

    # This one line runs the entire optimization process!
    best_params, es = cma.fmin2(objective_function,
                               initial_guess,
                               initial_std_dev,
                               options={'maxiter': 20, 'popsize': 10})

    # --- Print the final results ---
    duration = time.time() - start_time
    print("\n--- Optimization Finished ---")
    print(f"Total duration: {duration:.2f} s")

    # The best score 'fbest' is the minimized negative transmission
    best_transmission = -es.result.fbest

    print(f"\nHighest transmission found: {best_transmission:.5f}")
    print("Discovered with the following optimal parameters:")
    print(f"  - Control Amplitude: {best_params[0]:.2f}")
    print(f"  - Control Frequency: {best_params[1]:.2f}")
    print(f"  - Control Phase:     {best_params[2]:.2f}")
    print(f"  - Pulse Center Time: {best_params[3]:.2f}")
    print(f"  - Pulse Width:       {best_params[4]:.2f}")

