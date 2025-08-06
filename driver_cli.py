import numpy as np
import torch
import cma
import qdotlib
from pathlib import Path
import matplotlib.pyplot as plt

#doesnt look bad
def build_domain(grid_size, dt, trap_name, trap_params):
    domain = {}
    time_steps = 500
    domain["t"] = torch.arange(time_steps, device="cuda") * dt

    # Setup the trap potential
    pot_cfg = {"name": trap_name, "params": trap_params.copy()}
    X, Y, Z, dx, dy, dz, V, halfkprop, K2 = qdotlib.init_domain(
        Nx=grid_size, Ny=grid_size, Nz=grid_size, dt=dt, potential_cfg=trap_params
    )

    domain.update({
        "volume": dx * dy * dz,
        "X": X, "Y": Y, "Z": Z, "V": V, "halfk": halfkprop,
        "K2": K2, "dx": dx, "dy": dy
    })

    return domain

def make_target(gate, domain, pot_params):
    pot_params = pot_params.copy()
    pot_params['vol'] = domain['volume']
    return qdotlib.target_gate_function(gate, domain["X"], domain["Y"], domain["Z"], params=pot_params)

def objective_function(params, *, domain, dt, time_steps):
    control_amp, control_freq, control_phase, pulse_center_t, pulse_width = params
    pulse_width = abs(pulse_width)

    dt_vec = domain["t"]
    envelope = torch.exp(-((dt_vec - pulse_center_t) / pulse_width) ** 2)
    drive_pulse = control_amp * envelope * torch.sin(control_freq * dt_vec + control_phase)
    control_shape = domain["X"]

    final_psi = qdotlib.RUN_SIM(domain["initial_psi"], domain["V"], domain["halfk"],
                                domain["K2"], dt, time_steps, drive_pulse, control_shape)

    fidelity = qdotlib.calc_fidelity(final_psi, domain["target_psi"], domain["volume"])
    infidelity = 1.0 - fidelity + 1e-12
    return np.log(infidelity)

def optimise(domain, t_vec, maxiter, popsize, init_guess, bounds, cma_stds):
    time_steps = len(t_vec)
    dt = t_vec[1].item() - t_vec[0].item()

    def obj_fn(params): return objective_function(params, domain, dt, time_steps)

    res, es = cma.fmin2(obj_fn, init_guess, 5.0, {
        "bounds": bounds,
        "maxiter": maxiter,
        "popsize": popsize,
        "CMA_stds": cma_stds,
    })

    best_fidelity = 1.0 - np.exp(es.result.fbest)
    return best_fidelity, res