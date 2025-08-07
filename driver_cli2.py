import numpy as np
import torch
import cma
import time
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='cma.*')
# Import our installed quantum dot library
import qdotlib

def run_cli_job(
    trap: str,
    gate: str,
    omega: float,
    maxiter: int,
    popsize: int,
    grid_size: int,
    time_steps: int,
    dt: float,
    delta: float | None = None,
    noise_amp: float | None = None,
    corr_len: float | None = None,
    # --- ✨ NEW: Parameters to control the refinement phase ---
    run_refinement_phase: bool = False,
    refine_maxiter: int = 30,
    refinement_factor: float = 0.25
):
    """
    Runs a complete quantum gate optimization job, with an optional
    second refinement phase.
    """
    start_time = time.time()
    print(f"--- Received Job: Optimise {gate} gate for {trap} trap ---")
    print(f"    Parameters: ω={omega}, grid={grid_size}, steps={time_steps}, dt={dt}")
    
    # --- (No changes in this section) ---
    # 1. Construct Potential Configuration
    potential_params = {'omega': omega}
    if trap == "disordered":
        if noise_amp is None or corr_len is None:
            raise ValueError("For a disordered trap, 'noise_amp' and 'corr_len' must be provided.")
        potential_params.update({'noise_amp': noise_amp, 'corr_len': corr_len})
    if trap == "doublewell":
        if delta is None:
            raise ValueError("For a double-well trap, 'delta' must be provided.")
        potential_params['delta'] = delta
        if gate in ("CNOT", "BELL"):
            potential_params['input_state'] = '10' 
    POTENTIAL_CONFIG = {'name': trap, 'params': potential_params}

    # 2. Initialize Simulation Domain
    domain = {}
    _snapshot_done = False
    (X, Y, Z, dx, dy, dz, V, halfkprop, K2) = qdotlib.init_domain(
        Nx=grid_size, Ny=grid_size, Nz=grid_size, dt=dt, potential_cfg=POTENTIAL_CONFIG)
    domain["volume"] = dx * dy * dz
    potential_params['vol'] = domain['volume']
    domain["initial_psi"] = qdotlib.get_ground(X, Y, Z, omega, domain["volume"])
    domain["target_psi"] = qdotlib.target_gate_function(gate, X, Y, Z, params=potential_params)
    domain.update({"X": X, "Y": Y, "Z": Z, "V": V, "K2": K2, "halfk": halfkprop, "dx": dx, "dy": dy})
    domain["t"] = torch.arange(time_steps, device="cuda") * dt
    print("\n--- PyTorch Sanity Check ---")
    print(f"    PyTorch version: {torch.__version__}")
    # The .device attribute tells you if the tensor is on 'cpu' or 'cuda:0'
    print(f"    Simulation tensors are on device: '{domain['t'].device}'")
    print("----------------------------\n")
    # 3. Define the Objective Function (reused for both phases)
    def objective_function(params):
        nonlocal _snapshot_done
        control_amp, control_freq, control_phase, pulse_center_t, pulse_width = params
        pulse_width = abs(pulse_width)
        dt_vec = domain["t"]
        envelope = torch.exp(-((dt_vec - pulse_center_t) / pulse_width)**2)
        drive_pulse = control_amp * envelope * torch.sin(control_freq * dt_vec + control_phase)
        control_shape = domain["X"] * domain["Y"]
        if not _snapshot_done:
            # (Plotting logic remains the same)
            project_root = Path(__file__).resolve().parent; out_dir = project_root / "outputinfo"; out_dir.mkdir(exist_ok=True)
            zidx = domain["Z"].shape[2] // 2; V_np = domain["V"].cpu().numpy().real; fig, ax = plt.subplots(figsize=(6, 5))
            width_x = domain["dx"] * grid_size; width_y = domain["dy"] * grid_size
            im = ax.imshow(V_np[:, :, zidx], extent=[-width_x/2, width_x/2, -width_y/2, width_y/2], origin="lower", cmap="viridis")
            ax.set_title(f"{trap.capitalize()} Potential (z-slice)"); ax.set_xlabel("x"); ax.set_ylabel("y"); fig.colorbar(im, ax=ax, label="V(x,y)")
            plt.tight_layout(); filename = out_dir / f"{trap}_potential.png"; fig.savefig(filename, dpi=150); plt.close(fig)
            print(f"    Saved potential snapshot to {filename}"); _snapshot_done = True
        final_psi = qdotlib.RUN_SIM(domain["initial_psi"], domain["V"], domain["halfk"], domain["K2"], dt, time_steps, drive_pulse, control_shape)
        final_fidelity = qdotlib.calc_fidelity(final_psi, domain["target_psi"], domain["volume"])
        return np.log(1.0 - final_fidelity + 1e-12)

    # --- 4. Setup and Run Phase 1: Exploratory Search ---
    total_time = time_steps * dt
    initial_guess = [10.0, omega, 0.0, total_time / 2, total_time / 4]
    std_devs_param = np.array([10.0, 0.5, np.pi / 4, total_time / 4, total_time / 4])
    lower_bounds = [-50, 0.1, -np.pi, 0, 0.1]
    upper_bounds = [50, omega * 2, np.pi, total_time, total_time]
    
    options_1 = {
        'bounds': [lower_bounds, upper_bounds],
        'maxiter': maxiter,
        'popsize': popsize,
        'CMA_stds': std_devs_param
    }

    print("\n--- Starting Phase 1: Exploratory Search ---")
    best_params, es = cma.fmin2(objective_function, initial_guess, 5.0, options=options_1)

    # --- ✨ NEW: 5. Conditionally Run Phase 2: Refinement Search ---
    if run_refinement_phase:
        print("\n--- Starting Phase 2: Refinement Search ---")
        
        # New initial guess is the best result from Phase 1
        initial_guess_2 = best_params
        
        # Calculate new, tighter bounds around the Phase 1 result
        search_range = np.array(upper_bounds) - np.array(lower_bounds)
        refined_range = search_range * refinement_factor
        
        lower_bounds_2 = initial_guess_2 - refined_range / 2
        upper_bounds_2 = initial_guess_2 + refined_range / 2
        
        # Ensure refined bounds don't exceed original absolute bounds
        lower_bounds_2 = np.maximum(lower_bounds_2, lower_bounds)
        upper_bounds_2 = np.minimum(upper_bounds_2, upper_bounds)
        
        # Use smaller standard deviations for finer searching
        std_devs_2 = std_devs_param * refinement_factor

        options_2 = {
            'bounds': [lower_bounds_2.tolist(), upper_bounds_2.tolist()],
            'maxiter': refine_maxiter, # Use new iteration count
            'popsize': popsize, # Can reuse or add a new parameter for this
            'CMA_stds': std_devs_2
        }
        
        # Run the optimizer again with the refined settings
        best_params, es = cma.fmin2(objective_function, initial_guess_2, 2.0, options=options_2)

    # --- 6. Process and Return Final Results ---
    duration = time.time() - start_time
    best_infidelity = np.exp(es.result.fbest)
    best_fidelity = 1.0 - best_infidelity
    
    print("\n--- Optimization Finished ---")
    print(f"    Total duration: {duration:.2f} s")
    print(f"    Highest fidelity achieved: {best_fidelity:.6f}")

    result_params = {
        "Control Amplitude": f"{best_params[0]:.3f}",
        "Control Frequency": f"{best_params[1]:.3f}",
        "Control Phase": f"{best_params[2]:.3f}",
        "Pulse Center Time": f"{best_params[3]:.3f}",
        "Pulse Width": f"{abs(best_params[4]):.3f}"
    }

    return {"fidelity": best_fidelity, "params": result_params}