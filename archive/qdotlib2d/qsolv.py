# split_step_solver.py
# This script is a dedicated solver for the 2D Time-Dependent Schrödinger Equation
# using the Split-Step Fourier Method. It acts as our "physics engine".

import torch
import numpy as np

# Import the potential generation library we created
# Assumes qdot_lib is in a path accessible by Python
from qdotconfig import get_potential

# --- Device Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Split-Step Solver using device: {device} ---")


def init_domain(Nx=256, Ny=256, Lx=10.0, Ly=10.0, potential_config={}):
    """
    Initializes the simulation grid and the static potential.

    Args:
        Nx, Ny, Lx, Ly (int, float): Grid size and dimensions.
        potential_config (dict): Dictionary specifying the potential to generate.
                                 e.g., {'name': 'ideal', 'params': {'omega': 1.0}}

    Returns:
        Tuple of Tensors: X, Y, dx, dy, V, K2
    """
    dx, dy = Lx / Nx, Ly / Ny
    x = np.linspace(-Lx / 2, Lx / 2 - dx, Nx)
    y = np.linspace(-Ly / 2, Ly / 2 - dy, Ny)
    X_np, Y_np = np.meshgrid(x, y)
    X = torch.from_numpy(X_np).to(device)
    Y = torch.from_numpy(Y_np).to(device)

    # Fourier space coordinates
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    KX_np, KY_np = np.meshgrid(kx, ky)
    K2 = torch.from_numpy(KX_np**2 + KY_np**2).to(device)

    # Generate the static potential V by calling our library
    config_name = potential_config.get('name', 'ideal')
    params = potential_config.get('params', {})
    V = get_potential(config_name, X, Y, params)

    return X, Y, dx, dy, V.to(torch.complex64), K2


def _propagate_one_step(psi, V, K2, dt, drive_val, control_shape):
    """
    (Internal function) Propagates the wavefunction for a single time step.
    """
    # Kinetic half-step
    halfkprop = torch.exp(-0.25j * K2 * dt)
    psi_hat = torch.fft.fft2(psi)
    psi_hat *= halfkprop
    psi = torch.fft.ifft2(psi_hat)

    # Potential full-step
    V_interaction = -control_shape * drive_val
    V_pulsed = V + V_interaction
    psi *= torch.exp(-1j * V_pulsed * dt)

    # Kinetic half-step again
    psi_hat = torch.fft.fft2(psi)
    psi_hat *= halfkprop
    psi = torch.fft.ifft2(psi_hat)
    return psi

def run_simulation(initial_psi, V, K2, dt, Nt, drive_pulse, control_shape):
    """
    Runs the full time-evolution simulation.

    Args:
        initial_psi (torch.Tensor): The starting wavefunction.
        V (torch.Tensor): The static potential.
        K2 (torch.Tensor): The squared kinetic energy operator in k-space.
        dt (float): Time step duration.
        Nt (int): Number of time steps.
        drive_pulse (np.array): Array of control pulse amplitudes for each time step.
        control_shape (torch.Tensor): The spatial shape of the control interaction.

    Returns:
        torch.Tensor: The final wavefunction after Nt steps.
    """
    psi = initial_psi.clone()
    for n in range(Nt):
        psi = _propagate_one_step(psi, V, K2, dt, drive_pulse[n], control_shape)
    return psi

# -----------------------------------------------------------------------------
# MAIN EXECUTION BLOCK FOR TESTING THE SOLVER
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import time
    from imageio import mimsave
    import matplotlib.pyplot as plt

    print("\n--- Testing the Split-Step Solver ---")
    start_time = time.time()

    # --- 1. Setup the Domain with a potential from our library ---
    potential_config = {'name': 'ideal', 'params': {'omega': 1.0}}
    X, Y, dx, dy, V, K2 = init_domain(potential_config=potential_config)

    # --- 2. Define a test initial state (ground state) ---
    # Note: In the final project, this will come from the gate_library
    sigma = np.sqrt(1.0 / 2.0)
    initial_psi = (
        1 / (sigma * np.sqrt(np.pi))
        * torch.exp(-((X**2 + Y**2) / (2 * sigma**2)))
    ).to(torch.complex64)

    # --- 3. Define a test pulse ---
    dt = 0.005
    Nt = 1000
    dt_vec = np.arange(Nt) * dt
    envelope = np.exp(-((dt_vec - 1.5) / 0.5)**2)
    drive_pulse = 10.0 * envelope * np.sin(1.0 * dt_vec)
    control_shape = X # Simple dipole interaction

    # --- 4. Run the simulation ---
    final_psi = run_simulation(initial_psi, V, K2, dt, Nt, drive_pulse, control_shape)

    duration = time.time() - start_time
    print(f"Solver test finished in {duration:.2f} seconds.")

    # --- Optional: Visualize the final state ---
    plt.imshow(torch.abs(final_psi).cpu().numpy()**2)
    plt.title("Final State Intensity")
    plt.show()

