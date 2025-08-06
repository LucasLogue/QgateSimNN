# qdot_lib/gate_library.py
# This script defines the quantum states (|0>, |1>) and target states
# for various quantum gates, as well as the fidelity calculation.

import torch
import numpy as np

# --- Qubit State Definitions ---

def get_ground_state(X, Y, omega=1.0):
    """
    Defines our logical |0> state.
    This is the ground state of a 2D harmonic oscillator.

    Args:
        X, Y (torch.Tensor): Meshgrid of coordinates.
        omega (float): The trapping frequency of the potential.

    Returns:
        torch.Tensor: The normalized ground state wavefunction.
    """
    # The width of the ground state depends on the trapping frequency
    sigma_sq = 1.0 / omega
    psi0 = torch.exp(-(X**2 + Y**2) / (2 * sigma_sq))
    
    # Normalize the wavefunction
    norm_const = torch.sqrt(torch.sum(torch.abs(psi0)**2))
    return (psi0 / norm_const).to(torch.complex64)

def get_first_excited_state(X, Y, omega=1.0):
    """
    Defines our logical |1> state.
    This is the first excited state (in the x-direction) of a 2D harmonic oscillator.

    Args:
        X, Y (torch.Tensor): Meshgrid of coordinates.
        omega (float): The trapping frequency of the potential.

    Returns:
        torch.Tensor: The normalized first excited state wavefunction.
    """
    ground_state = get_ground_state(X, Y, omega)
    # The first excited state has a shape proportional to X * ground_state
    psi1_unnormalized = X * ground_state
    
    # Normalize the wavefunction
    norm_const = torch.sqrt(torch.sum(torch.abs(psi1_unnormalized)**2))
    return (psi1_unnormalized / norm_const).to(torch.complex64)


# --- Fidelity Calculation ---

def calculate_fidelity(psi_final, psi_target, dx, dy):
    """
    Calculates the fidelity between the final simulated state and the ideal target state.

    Args:
        psi_final (torch.Tensor): The wavefunction at the end of the simulation.
        psi_target (torch.Tensor): The ideal target wavefunction for the gate.
        dx, dy (float): Grid spacing for numerical integration.

    Returns:
        float: The fidelity, a value between 0 and 1.
    """
    # The overlap is the integral of (psi_target_conjugate * psi_final)
    overlap = torch.sum(torch.conj(psi_target) * psi_final) * dx * dy
    # Fidelity is the squared magnitude of the overlap
    fidelity = torch.abs(overlap)**2
    return fidelity.item()


# --- Gate Target Factory ---

def get_gate_target(gate_name, X, Y, omega=1.0):
    """
    A factory function that returns the target wavefunction for a specific gate,
    assuming the initial state is the ground state |0>.

    Args:
        gate_name (str): The name of the gate ("NOT", "HADAMARD", etc.).
        X, Y (torch.Tensor): Meshgrid of coordinates.
        omega (float): The trapping frequency.

    Returns:
        torch.Tensor: The ideal target wavefunction.
    """
    psi_0 = get_ground_state(X, Y, omega)
    psi_1 = get_first_excited_state(X, Y, omega)

    if gate_name.upper() == "NOT":
        # A NOT gate flips |0> to |1>
        return psi_1
    elif gate_name.upper() == "HADAMARD":
        # A Hadamard gate transforms |0> to (1/sqrt(2)) * (|0> + |1>)
        return (1.0 / np.sqrt(2.0)) * (psi_0 + psi_1)
    else:
        raise ValueError(f"Unknown gate: {gate_name}")


# -----------------------------------------------------------------------------
# MAIN EXECUTION BLOCK FOR TESTING THE LIBRARY
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # --- Setup a sample grid for plotting ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Nx, Ny = 128, 128
    Lx, Ly = 10.0, 10.0
    dx, dy = Lx / Nx, Ly / Ny
    x = np.linspace(-Lx / 2, Lx / 2 - dx, Nx)
    y = np.linspace(-Ly / 2, Ly / 2 - dy, Ny)
    X_np, Y_np = np.meshgrid(x, y)
    X = torch.from_numpy(X_np).to(device)
    Y = torch.from_numpy(Y_np).to(device)

    # --- Generate the basis states ---
    psi_0 = get_ground_state(X, Y)
    psi_1 = get_first_excited_state(X, Y)
    hadamard_target = get_gate_target("HADAMARD", X, Y)

    # --- Sanity Checks ---
    print("--- Running Sanity Checks ---")
    fid_00 = calculate_fidelity(psi_0, psi_0, dx, dy)
    print(f"Fidelity(|0>, |0>): {fid_00:.6f}  (Should be 1.0)")
    
    fid_11 = calculate_fidelity(psi_1, psi_1, dx, dy)
    print(f"Fidelity(|1>, |1>): {fid_11:.6f}  (Should be 1.0)")

    fid_01 = calculate_fidelity(psi_0, psi_1, dx, dy)
    print(f"Fidelity(|0>, |1>): {fid_01:.6f}  (Should be 0.0, due to orthogonality)")

    # --- Plot the states ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Qubit Basis States and Hadamard Target", fontsize=16)

    ax1.imshow(torch.abs(psi_0).cpu().numpy()**2, cmap='magma', origin='lower')
    ax1.set_title("Probability |ψ₀|² (Ground State |0⟩)")
    
    ax2.imshow(torch.abs(psi_1).cpu().numpy()**2, cmap='magma', origin='lower')
    ax2.set_title("Probability |ψ₁|² (Excited State |1⟩)")

    ax3.imshow(torch.abs(hadamard_target).cpu().numpy()**2, cmap='magma', origin='lower')
    ax3.set_title("Probability |ψ_Hadamard|²")

    plt.tight_layout()
    plt.show()

