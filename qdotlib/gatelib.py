import torch
import numpy as np

def get_ground(X, Y, Z, omega=1.0, dtype=torch.complex64):
    """
    Returns ground state ideal value
    """
    sigsq = 1.0 / omega
    psi0 = torch.exp(-(X**2 + Y**2 + Z**2) / (2 * sigsq))
    norm_const = torch.sqrt(torch.sum(torch.abs(psi0)**2))
    return (psi0 / norm_const).to(dtype)

def get_maxexcited(X, Y, Z, omega=1.0, dtype=torch.complex64):
    """
    Returns max excited state ideal value
    """
    ground = get_ground(X, Y, Z, omega, dtype=dtype)
    psi1_unnormalized = X * ground
    norm_const = torch.sqrt(torch.sum(torch.abs(psi1_unnormalized)**2))
    return (psi1_unnormalized / norm_const).to(dtype)

def calc_fidelity(psi_final, psi_target, dx, dy, dz):
    """
    Calculates fidelity between our produced and goal wave
    """
    overlap = torch.sum(torch.conj(psi_target) * psi_final) * dx * dy * dz
    fidelity = torch.abs(overlap)**2
    return fidelity.item()

def target_gate_function(gate, X, Y, Z, omega=1.0, dtype=torch.complex64):
    """
    Returns the ideal value for selected gate
    """
    psi_0 = get_ground(X, Y, Z, omega, dtype)
    psi_1 = get_maxexcited(X, Y, Z, omega, dtype)

    if gate.upper() == "NOT":
        return psi_1
    elif gate.upper() == "HADAMARD":
        return (1.0 / np.sqrt(2.0)) * (psi_0 + psi_1)
    else:
        raise ValueError(f"Unknown gate: {gate}")