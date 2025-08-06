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

def dw_twoqbit_basis(X, Y, Z, *, omega=1.0, delta=2.0, vol=1.0):
    psi0 = dw_qbit_wave(X, Y, Z, bit=0, omega=omega, delta=delta, vol=vol) #left
    psi1 = dw_qbit_wave(X, Y, Z, bit=1, omega=omega, delta=delta, vol=vol) #right
    def normem(psi):
        return psi / torch.sqrt(torch.sum(torch.abs(psi)**2) * vol)
    basis = {
        "00": psi0 * psi0,
        "01": psi0 * psi1,
        "10": psi1 * psi0,
        "11": psi1 * psi1,
    }
    return {
        "00": normem(psi0 * psi0),
        "01": normem(psi0 * psi1),
        "10": normem(psi1 * psi0),
        "11": normem(psi1 * psi1),
    }
    #return basis

def target_gate_function(gate, X, Y, Z, *, params:dict | None = None):
    """
    Returns the ideal value for selected gate
    """
    cfg   = {'omega': 1.0, 'delta': 2.0, 'vol': 1.0}
    if params is not None:
        cfg.update(params)
    omega = cfg['omega']
    delta = cfg['delta']
    vol   = cfg['vol']


    # psi_0 = get_ground(X, Y, Z, omega, vol)
    # psi_1 = get_maxexcited(X, Y, Z, omega, vol)

    if gate.upper() == "NOT":
        return get_maxexcited(X, Y, Z, omega, vol)
    elif gate.upper() == "HADAMARD":
        psi_0 = get_ground(X, Y, Z, omega, vol)
        psi_1 = get_maxexcited(X, Y, Z, omega, vol)
        # return (1.0 / np.sqrt(2.0)) * (psi_0 + psi_1)
        return (psi_0 + psi_1) / torch.sqrt(torch.tensor(2.0, device=X.device))

    elif gate.upper() in ("BELL", "PHIPLUS"):
        basis = dw_twoqbit_basis(X,Y,Z, omega=omega, delta=delta, vol=vol)
        return (basis["00"] + basis["11"]) / torch.sqrt(torch.tensor(2.0, device=X.device))
    elif gate.upper() == "CNOT":
        ctrl  = params.get("input_state", "10")
        basis = dw_twoqbit_basis(X, Y, Z, omega=omega, delta=delta, vol=vol)
        mapping = {"00": "00",
                   "01": "01",
                   "10": "11",
                   "11": "10"}
        if ctrl not in mapping:
            raise ValueError("CNOT: input_state must be one of '00','01','10','11'")

        return basis[mapping[ctrl]]
    else:
        raise ValueError(f"Unknown gate: {gate}")