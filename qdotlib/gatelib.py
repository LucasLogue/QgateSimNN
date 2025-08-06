import torch
import numpy as np

def get_ground(X, Y, Z, omega=1.0, vol=1.0):
    """
    Returns ground state ideal value
    """
    sigsq = 1.0 / omega
    psi0 = torch.exp(-(X**2 + Y**2 + Z**2) / (2 * sigsq))
    norm_const = torch.sqrt(torch.sum(torch.abs(psi0)**2) * vol)
    return (psi0 / norm_const).to(torch.complex64)

def get_maxexcited(X, Y, Z, omega=1.0, vol=1.0):
    """
    Returns max excited state ideal value
    """
    ground = get_ground(X, Y, Z, omega)
    psi1_unnormalized = X * ground
    norm_const = torch.sqrt(torch.sum(torch.abs(psi1_unnormalized)**2) * vol)
    return (psi1_unnormalized / norm_const).to(torch.complex64)

def calc_fidelity(psi_final, psi_target, vol=1.0):
    """
    Calculates fidelity between our produced and goal wave
    """
    overlap = torch.sum(torch.conj(psi_target) * psi_final) * vol
    fidelity = torch.abs(overlap)**2
    return fidelity.item()

def dw_qbit_wave(X, Y, Z, bit, omega=1.0, delta=2.0, vol=1.0) -> torch.Tensor:
    center = +delta if bit == 0 else -delta
    sigma2 = 1.0 / omega
    psi    = torch.exp(-((X - center) ** 2 + Y ** 2 + Z ** 2) / (2 * sigma2))
    norm   = torch.sqrt(torch.sum(torch.abs(psi) ** 2) * vol)
    return psi / norm

def dw_twoqbit_basis(X, Y, Z, bit, omega=1.0, delta=2.0, vol=1.0):
    psi0 = get_qubit_wave(X, Y, Z, bit=0, omega=omega, delta=delta, vol=vol)
    psi1 = get_qubit_wave(X, Y, Z, bit=1, omega=omega, delta=delta, vol=vol)
    basis = {
        "00": psi0 * psi0,
        "01": psi0 * psi1,
        "10": psi1 * psi0,
        "11": psi1 * psi1,
    }
    return basis

def target_gate_function(gate, X, Y, Z, omega=1.0, vol=1.0):
    """
    Returns the ideal value for selected gate
    """
    psi_0 = get_ground(X, Y, Z, omega, vol)
    psi_1 = get_maxexcited(X, Y, Z, omega, vol)

    if gate.upper() == "NOT":
        return psi_1
    elif gate.upper() == "HADAMARD":
        return (1.0 / np.sqrt(2.0)) * (psi_0 + psi_1)
    elif gate.upper() == "DOUBLEWELL":
        #
    else:
        raise ValueError(f"Unknown gate: {gate}")