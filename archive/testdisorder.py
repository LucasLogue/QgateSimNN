import numpy as np
import torch
import cma
import time
from pathlib import Path
import matplotlib.pyplot as plt
import qdotlib # Your library of quantum dot functions

# ======================================================================
# === HARDCODED CONFIGURATION FOR DISORDERED NOT GATE TEST ===
# ======================================================================
# --- System & Gate ---
GATE_TO_OPTIMIZE = "NOT"
ELECCONFIG = "disordered"
OMEGA = 1.0

# --- Disordered Potential Settings ---
NOISE_AMP = 1.0
CORR_LEN = 5.0

# --- Simulation Grid & Time ---
GRID_SIZE = 64      # High resolution for accuracy
TIME_STEPS = 500
DT = 0.005

# --- Optimizer Budget ---
# Phase 1: Broad search to find a good general area
MAX_ITER_PHASE_1 = 40
POP_SIZE_PHASE_1 = 15

# Phase 2: Focused search to refine the result
MAX_ITER_PHASE_2 = 30
POP_SIZE_PHASE_2 = 15
REFINEMENT_FACTOR = 0.25 # Shrink search space to 25% of original

# --- Reproducibility ---
RANDOM_SEED = 42 # Use a fixed seed to get the same potential every time
# ======================================================================

def run_isolated_test():
    """
    An isolated, hardcoded test to maximize fidelity for the NOT gate
    on a disordered potential landscape.
    """
    # 1. SETUP AND DOMAIN INITIALIZATION
    start_time = time.time()
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    print(f"--- Running Isolated Test: {GATE_TO_OPTIMIZE} on {ELECCONFIG} potential ---")
    print(f"    Random Seed: {RANDOM_SEED}, Grid Size: {GRID_SIZE}x{GRID_SIZE}x{GRID_SIZE}")

    potential_params = {
        'omega': OMEGA,
        'noise_amp': NOISE_AMP,
        'corr_len': CORR_LEN
    }
    POTENTIAL_CONFIG = {'name': ELECCONFIG, 'params': potential_params}

    domain = {}
    (X, Y, Z, dx, dy, dz, V, halfkprop, K2) = qdotlib.init_domain(
        Nx=GRID_SIZE, Ny=GRID_SIZE, Nz=GRID_SIZE, dt=DT, potential_cfg=POTENTIAL_CONFIG)
    
    domain["volume"] = dx * dy * dz
    potential_params['vol'] = domain['volume']
    
    domain["initial_psi"] = qdotlib.get_ground(X, Y, Z, OMEGA, domain["volume"])
    domain["target_psi"] = qdotlib.target_gate_function(GATE_TO_OPTIMIZE, X, Y, Z, params=potential_params)
    
    domain.update({"X": X, "V": V, "K2": K2, "halfk": halfkprop})
    domain["t"] = torch.arange(TIME_STEPS, device="cuda") * DT

    # 2. OBJECTIVE FUNCTION
    # (This is the same function used before)
    def objective_function(params):
        control_amp, control_freq, control_phase, pulse_center_t, pulse_width = params
        pulse_width = abs(pulse_width)
        drive_pulse = control_amp * torch.exp(-((domain["t"] - pulse_center_t) / pulse_width)**2) * torch.sin(control_freq * domain["t"] + control_phase)
        final_psi = qdotlib.RUN_SIM(domain["initial_psi"], domain["V"], domain["halfk"], domain["K2"], DT, TIME_STEPS, drive_pulse, domain["X"])
        fidelity = qdotlib.calc_fidelity(final_psi, domain["target_psi"], domain["volume"])
        return np.log(1.0 - fidelity + 1e-12)

    # 3. PHASE 1: EXPLORATORY SEARCH
    total_time = TIME_STEPS * DT
    initial_guess = [10.0, OMEGA, 0.0, total_time / 2, total_time / 4]
    std_devs_param = np.array([10.0, 0.5, np.pi / 4, total_time / 4, total_time / 4])
    lower_bounds = [-50, 0.1, -np.pi, 0, 0.1]
    upper_bounds = [50, OMEGA * 2, np.pi, total_time, total_time]
    
    options_1 = {
        'bounds': [lower_bounds, upper_bounds],
        'maxiter': MAX_ITER_PHASE_1,
        'popsize': POP_SIZE_PHASE_1,
        'CMA_stds': std_devs_param
    }

    print("\n--- Starting Phase 1: Exploratory Search ---")
    best_params_1, es_1 = cma.fmin2(objective_function, initial_guess, 5.0, options=options_1)
    fidelity_1 = 1.0 - np.exp(es_1.result.fbest)
    print(f"--- Fidelity after Phase 1: {fidelity_1:.6f} ---")

    # 4. PHASE 2: REFINEMENT SEARCH
    print("\n--- Starting Phase 2: Refinement Search ---")
    search_range = np.array(upper_bounds) - np.array(lower_bounds)
    refined_range = search_range * REFINEMENT_FACTOR
    
    lower_bounds_2 = best_params_1 - refined_range / 2
    upper_bounds_2 = best_params_1 + refined_range / 2
    lower_bounds_2 = np.maximum(lower_bounds_2, lower_bounds)
    upper_bounds_2 = np.minimum(upper_bounds_2, upper_bounds)
    
    options_2 = {
        'bounds': [lower_bounds_2.tolist(), upper_bounds_2.tolist()],
        'maxiter': MAX_ITER_PHASE_2,
        'popsize': POP_SIZE_PHASE_2,
        'CMA_stds': std_devs_param * REFINEMENT_FACTOR
    }
    
    best_params_2, es_2 = cma.fmin2(objective_function, best_params_1, 2.0, options=options_2)

    # 5. FINAL RESULTS
    duration = time.time() - start_time
    final_fidelity = 1.0 - np.exp(es_2.result.fbest)
    
    print("\n==============================================")
    print("--- ISOLATED TEST FINISHED ---")
    print(f"    Total duration: {duration:.2f} s")
    print(f"    HIGHEST FIDELITY ACHIEVED: {final_fidelity:.6f}")
    print("    Optimal Parameters:")
    print(f"      - Control Amplitude: {best_params_2[0]:.3f}")
    print(f"      - Control Frequency: {best_params_2[1]:.3f}")
    print(f"      - Control Phase:     {best_params_2[2]:.3f}")
    print(f"      - Pulse Center Time: {best_params_2[3]:.3f}")
    print(f"      - Pulse Width:       {abs(best_params_2[4]):.3f}")
    print("==============================================")

if __name__ == '__main__':
    run_isolated_test()