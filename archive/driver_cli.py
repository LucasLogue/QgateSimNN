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

    def obj_fn(params): return obj_fn(params, domain, dt, time_steps)

    res, es = cma.fmin2(obj_fn, init_guess, 5.0, {
        "bounds": bounds,
        "maxiter": maxiter,
        "popsize": popsize,
        "CMA_stds": cma_stds,
    })

    best_fidelity = 1.0 - np.exp(es.result.fbest)
    return best_fidelity, res

def obj_fn(params):
    return objective_function(params, domain=domain, dt=dt, time_steps=time_steps)
def run_cli_job(trap, gate, omega, delta=None, maxiter=20, popsize=10):
    # --- Step 1: Set potential config
    np.random.seed(42)
    pot_params = {"omega": omega}
    if trap == "doublewell" and delta is not None:
        pot_params["delta"] = delta
        pot_params["input_state"] = "10"

    GRID = 64
    DT = 0.005
    STEPS = 500
    total_time = STEPS * DT
    pot_params["time_steps"] = total_time
    # --- Step 2: Build domain
    print()
    print()
    print(GRID, DT, trap, pot_params)
    print()
    print()
    dom = build_domain(GRID, DT, trap, pot_params)
    dom["initial_psi"] = qdotlib.get_ground(dom["X"], dom["Y"], dom["Z"], omega, dom["volume"])
    dom["target_psi"] = make_target(gate, dom, pot_params)

    # --- Step 3: Setup CMA‑ES params
    t_vec = torch.arange(STEPS, device="cuda") * DT
    init_guess = [10.0, omega, 0.0, total_time/2, total_time/4]
    bounds = [[-50, 0.1, -np.pi, 0, 0.1],
              [50, omega*2, np.pi, total_time, total_time]]
    cma_stds = [10.0, 0.5, np.pi/4, total_time/4, total_time/4]

    # --- Step 4: Run optimiser
    best_fid, best_params = optimise(dom, t_vec, maxiter, popsize, init_guess, bounds, cma_stds)

    return {
        "fidelity": best_fid,
        "params": {
            "amplitude": best_params[0],
            "frequency": best_params[1],
            "phase": best_params[2],
            "pulse_center": best_params[3],
            "pulse_width": abs(best_params[4]),
        }
    }