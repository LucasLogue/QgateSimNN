import numpy as np
import torch
import cma
import time
from pathlib import Path
import matplotlib.pyplot as plt

# Import our installed quantum dot library
import qdotlib

#TORCH CUDA SANITY CHECK
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda devices:", torch.cuda.device_count(),
          torch.cuda.get_device_name(0))
#EXPERIMENT CONFIG PARAMS
ISBREV = False
ELECCONFIG = "DISORDERED"

GATE_TO_OPTIMIZE = "HADAMARD"  # Can be "NOT" or "HADAMARD"
if ELECCONFIG == "ideal":
    POTENTIAL_CONFIG = {'name': 'ideal', 'params': {'omega': 1.0}}
elif ELECCONFIG == "DISORDERED":
    POTENTIAL_CONFIG = {
        'name':   'disordered',
        'params': {
            'omega':      1.0,
            'noise_amp':  1.0,
            'corr_len':   5
        }
    }

if ISBREV:
    print("--- RUNNING IN HIGH-PERFORMANCE (BREV) MODE ---")
    GRID_SIZE = 96  # Larger grid for more spatial accuracy
    MAX_ITER = 80  # jesus christ 200 x 30 was going to take 80 mins fuck my asshole
    POP_SIZE = 12   # 
else:
    print("--- RUNNING IN FAST-PREVIEW (LOCAL) MODE ---")
    GRID_SIZE = 64
    MAX_ITER = 10 #suprisingly chill for local system
    POP_SIZE = 5

#Variables and Project Initialization
_snapshot_done = False

TIME_STEPS = 500 # Number of time steps in the simulation
DT = 0.005 # Time step duration

#Chosen Domain
domain = {}
(X, Y, Z, dx, dy, dz, V, halfkprop, K2) = qdotlib.init_domain(
        Nx=GRID_SIZE, Ny=GRID_SIZE, Nz=GRID_SIZE, dt=DT,
        potential_cfg=POTENTIAL_CONFIG 
    )

domain["volume"] = dx * dy * dz
omega = POTENTIAL_CONFIG['params'].get('omega', 1.0)


domain["initial_psi"] = qdotlib.get_ground(X, Y, Z, omega, domain["volume"])
domain["target_psi"] = qdotlib.target_gate_function(GATE_TO_OPTIMIZE, X, Y, Z, omega, domain["volume"])

domain["X"] = X
domain["V"] = V
domain["Y"] = Y
domain["Z"] = Z
domain["K2"] = K2
domain["halfk"] = halfkprop
domain["dx"] = dx
domain["dy"] = dy

def objective_function(params, *, st=domain):
    global _snapshot_done
    control_amp, control_freq, control_phase, pulse_center_t, pulse_width = params

    # Ensure pulse width is positive
    pulse_width = abs(pulse_width)

    print(f"--- Testing: A={control_amp:.1f}, F={control_freq:.2f}, "
          f"T={pulse_center_t:.2f}, W={pulse_width:.2f} ---")
    
    # #in theory could be done once
    # dt = 0.005
    Nt = TIME_STEPS
    # X, Y, Z, dx, dy, dz, V, halfkprop, K2 = qdotlib.init_domain(
    #     Nx=GRID_SIZE, Ny=GRID_SIZE, Nz=GRID_SIZE, dt=DT,
    #     potential_cfg=POTENTIAL_CONFIG 
    # )

    #Save the electron configuration potential graph to output folder, only once per loop
    if not _snapshot_done:
        _snapshot_done = True
        project_root = Path(__file__).resolve().parent
        print(project_root)
        out_dir      = project_root / "outputinfo"
        out_dir.mkdir(exist_ok=True)
        zidx = st["Z"].shape[2] // 2
        V_np = st["V"].cpu().numpy().real
        #plotting
        fig, ax = plt.subplots(figsize=(6,5))
        width_x = st["dx"] * GRID_SIZE
        width_y = st["dy"] * GRID_SIZE
        im = ax.imshow(
            V_np[:, :, zidx],
            extent=[-width_x/2, width_x/2, -width_y/2, width_y/2],
            origin="lower",
            cmap="viridis"
        )
        ax.set_title(f"{POTENTIAL_CONFIG['name'].capitalize()} Potential (z={zidx})")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, label="V(x,y)")

        plt.tight_layout()
        # save
        filename = out_dir / f"{POTENTIAL_CONFIG['name']}_potential.png"
        fig.savefig(filename, dpi=300)
        plt.close(fig)



    # #!!!!!!!!!could be optimized
    # volume = dy*dx*dz
    # omega = POTENTIAL_CONFIG['params'].get('omega', 1.0)
    # initial_psi = qdotlib.get_ground(X, Y, Z, omega, volume)
    # target_psi = qdotlib.target_gate_function(GATE_TO_OPTIMIZE, X, Y, Z, omega, volume)

    #Commented out this debug statement for efficiency
    # fid_self   = qdotlib.calc_fidelity(initial_psi, initial_psi, volume)
    # assert abs(fid_self - 1.0) < 1e-6
    # fid_target = qdotlib.calc_fidelity(initial_psi, target_psi, volume)
    # print(f"[DEBUG] ⟨ψ₀|ψ₀⟩ = {fid_self:.6f}, ⟨ψ₀|ψ_target⟩ = {fid_target:.6f}")

    #Control Pulse, Needs to be Remade every loop, so fine
    dt_vec = np.arange(Nt) * DT
    envelope = np.exp(-((dt_vec - pulse_center_t) / pulse_width)**2)
    drive_pulse = control_amp * envelope * np.sin(control_freq * dt_vec + control_phase)
    control_shape = st["X"]  # Simple dipole interaction in the x-direction

    #Simulation call, needs to be done in every loop
    final_psi = qdotlib.run_sim(st["initial_psi"], st["V"], st["halfk"], st["K2"], DT, Nt, drive_pulse, control_shape)

    #final score, needs to be done every loop
    final_fidelity = qdotlib.calc_fidelity(final_psi, st["target_psi"], st["volume"])
    print(f"  > Resulting Fidelity: {final_fidelity:.6f}\n")

    infidelity = 1.0 - final_fidelity + 1e-12 #add 1e-12 to prevent log(0)
    #SWITCHING to log(1-fidelity) for better accuracy in such a low range
    return np.log(infidelity)

if __name__ == '__main__':
    print(f"--- Starting 3D Optimization for {GATE_TO_OPTIMIZE} gate ---")
    start_time = time.time()

    #collect inputs to create first guess
    omega = POTENTIAL_CONFIG['params'].get('omega', 1.0)
    total_time = TIME_STEPS * DT

    # --- Define the search space for the agent ---
    # [amplitude, frequency, phase, pulse_center_time, pulse_width]
    initial_guess = [10.0, omega, 0.0, total_time/2, total_time/4]

    initial_std_dev = 5.0
    std_devs_param = [10.0, 0.5, np.pi / 4, total_time / 4, total_time / 4]

    lower_bounds = [-50, 0.1, -np.pi, 0, 0.1]
    upper_bounds = [50, omega * 2, np.pi, total_time, total_time]
    bounds = [lower_bounds, upper_bounds]
    options = {
        'bounds': bounds, 
        'maxiter': MAX_ITER, 
        'popsize': POP_SIZE,
        'CMA_stds': std_devs_param # Correct way to specify per-parameter stds
    }
    # Run the CMA-ES optimizer
    best_params, es = cma.fmin2(objective_function,
                               initial_guess,
                               initial_std_dev,
                               options=options
                               ) # Fewer iterations for 3D

    # --- Print the final results ---
    duration = time.time() - start_time
    print("\n--- Optimization Finished ---")
    print(f"Total duration: {duration:.2f} s")

    best_infidelity = np.exp(es.result.fbest)
    best_fidelity = 1.0 - best_infidelity

    print(f"\nHighest fidelity achieved for {GATE_TO_OPTIMIZE} gate: {best_fidelity:.6f}")
    print("Discovered with the following optimal parameters:")
    print(f"  - Control Amplitude: {best_params[0]:.3f}")
    print(f"  - Control Frequency: {best_params[1]:.3f}")
    print(f"  - Control Phase:     {best_params[2]:.3f}")
    print(f"  - Pulse Center Time: {best_params[3]:.3f}")
    print(f"  - Pulse Width:       {abs(best_params[4]):.3f}")