import numpy as np
import torch
import cma
import time
from pathlib import Path
import matplotlib.pyplot as plt

# Import our installed quantum dot library
import qdotlib

# TORCH CUDA SANITY CHECK
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda devices:", torch.cuda.device_count(), torch.cuda.get_device_name(0))

# EXPERIMENT CONFIG PARAMS
ISBREV = False
ELECCONFIG = "disordered"  # can be ideal, disordered, doublewell

GATE_TO_OPTIMIZE = "NOT"  # Can be "NOT" or "HADAMARD", "CNOT", "BELL"
if ELECCONFIG == "ideal":
    POTENTIAL_CONFIG = {'name': 'ideal', 'params': {'omega': 1.0}}
elif ELECCONFIG == "disordered":
    POTENTIAL_CONFIG = {
        'name': 'disordered',
        'params': {
            'omega': 1.0,
            'noise_amp': 0.1,
            'corr_len': 5
        }
    }
elif ELECCONFIG == "doublewell":
    if GATE_TO_OPTIMIZE in ("CNOT", "BELL"):
        POTENTIAL_CONFIG = {
            'name': 'doublewell',
            'params': {'omega': 1.0, 'delta': 2.0, 'input_state': '10'}
        }
    else:
        POTENTIAL_CONFIG = {'name': 'doublewell', 'params': {'omega': 1.0, 'delta': 2.0}}

if ISBREV:
    print("--- RUNNING IN HIGH-PERFORMANCE (BREV) MODE ---")
    GRID_SIZE = 96
    MAX_ITER = 40
    POP_SIZE = 15
else:
    print("--- RUNNING IN FAST-PREVIEW (LOCAL) MODE ---")
    GRID_SIZE = 64
    MAX_ITER = 20
    POP_SIZE = 10

# Variables and Project Initialization
_snapshot_done = False

domain = {}
params_target = POTENTIAL_CONFIG['params'].copy()
TIME_STEPS = 500  # Number of time steps in the simulation
DT = 0.005        # Time step duration
domain["t"] = torch.arange(TIME_STEPS, device="cuda") * DT

# Initialize domain with full config
(X, Y, Z, dx, dy, dz, V, halfkprop, K2) = qdotlib.init_domain(
    Nx=GRID_SIZE, Ny=GRID_SIZE, Nz=GRID_SIZE, dt=DT,
    potential_cfg=POTENTIAL_CONFIG
)
# Ensure consistent dtypes
X = X.to(torch.float32)
Y = Y.to(torch.float32)
Z = Z.to(torch.float32)
halfkprop = halfkprop.to(torch.complex64)
K2 = K2.to(torch.float32)

# Print potential stats
V_real = V.real
print(
    f"Potential stats:  min={V_real.min().item():.3f}, max={V_real.max().item():.3f}, mean={V_real.mean().item():.3f}"
)

domain["volume"] = dx * dy * dz
params_target['vol'] = domain['volume']
omega = POTENTIAL_CONFIG['params'].get('omega', 1.0)

domain["initial_psi"] = qdotlib.get_ground(X, Y, Z, omega, domain['volume'])
domain["target_psi"] = qdotlib.target_gate_function(
    GATE_TO_OPTIMIZE, X, Y, Z, params=params_target
)

domain.update({
    "X": X,
    "V": V,
    "Y": Y,
    "Z": Z,
    "K2": K2,
    "halfk": halfkprop,
    "dx": dx,
    "dy": dy
})

def objective_function(params, *, st=domain):
    global _snapshot_done
    control_amp, control_freq, control_phase, pulse_center_t, pulse_width = params
    pulse_width = abs(pulse_width)

    print(f"--- Testing: A={control_amp:.3f}, F={control_freq:.3f}, T={pulse_center_t:.3f}, W={pulse_width:.3f} ---")

    # One-time potential snapshot
    if not _snapshot_done:
        _snapshot_done = True
        project_root = Path(__file__).resolve().parent
        out_dir = project_root / "outputinfo"
        out_dir.mkdir(exist_ok=True)
        zidx = st['Z'].shape[2] // 2
        V_np = st['V'].cpu().numpy().real
        fig, ax = plt.subplots(figsize=(6, 5))
        width_x = st['dx'] * GRID_SIZE
        width_y = st['dy'] * GRID_SIZE
        im = ax.imshow(
            V_np[:, :, zidx],
            extent=[-width_x/2, width_x/2, -width_y/2, width_y/2],
            origin='lower',
            cmap='viridis'
        )
        ax.set_title(f"{POTENTIAL_CONFIG['name'].capitalize()} Potential (z={zidx})")
        ax.set_xlabel('x'); ax.set_ylabel('y')
        fig.colorbar(im, ax=ax, label='V(x,y)')
        plt.tight_layout()
        filename = out_dir / f"{POTENTIAL_CONFIG['name']}_potential.png"
        fig.savefig(filename, dpi=300)
        plt.close(fig)

    # Control Pulse computation
    dt_vec = st['t']
    envelope = torch.exp(-((dt_vec - pulse_center_t) / pulse_width)**2)
    drive_pulse = control_amp * envelope * torch.sin(control_freq * dt_vec + control_phase)
    control_shape = st['X'].to(device=drive_pulse.device, dtype=drive_pulse.dtype)

    # Debug prints
    print(">> drive_pulse[:5] =", drive_pulse[:5].detach().cpu().numpy())
    print(f">> drive_pulse mean={drive_pulse.mean().item():.3e}, std={drive_pulse.std().item():.3e}")
    nz = int((control_shape != 0).sum().item())
    tot = control_shape.numel()
    print(f">> control_shape nonzero entries: {nz}/{tot}")
    print(
        ">> dtypes/devices:",
        f"psi0={st['initial_psi'].device},{st['initial_psi'].dtype};",
        f"V={st['V'].device},{st['V'].dtype};",
        f"pulse={drive_pulse.device},{drive_pulse.dtype};",
        f"mask={control_shape.device},{control_shape.dtype}"
    )

    # Propagate
    psi0 = st['initial_psi']
    psi_test = qdotlib.RUN_SIM(
        psi0, st['V'], st['halfk'], st['K2'],
        DT, TIME_STEPS, drive_pulse, control_shape
    )
    delta = torch.norm(psi_test - psi0).item()
    print(f">> ‖ψ_test – ψ0‖ = {delta:.3e}")

    # Fidelity
    final_fidelity = qdotlib.calc_fidelity(psi_test, st['target_psi'], st['volume'])
    print(f"  > Resulting Fidelity: {final_fidelity:.6f}\n")

    # Return log-infidelity
    infidelity = 1.0 - final_fidelity + 1e-12
    return np.log(infidelity)


if __name__ == '__main__':
    print(f"--- Starting 3D Optimization for {GATE_TO_OPTIMIZE} gate ---")
    start_time = time.time()

    # Initial guesses and bounds
    omega = POTENTIAL_CONFIG['params'].get('omega', 1.0)
    total_time = TIME_STEPS * DT
    initial_guess = [10.0, omega, 0.0, total_time/2, total_time/4]
    initial_std_dev = 5.0
    std_devs_param = [10.0, 0.5, np.pi/4, total_time/4, total_time/4]
    bounds = [[-50, 0.1, -np.pi, 0.0, 0.1], [50, omega*2, np.pi, total_time, total_time]]
    options = {'bounds': bounds, 'maxiter': MAX_ITER, 'popsize': POP_SIZE, 'CMA_stds': std_devs_param}

    # Self-consistency checks
    print("⟨ψ₀|ψ₀⟩ =", qdotlib.calc_fidelity(domain['initial_psi'], domain['initial_psi'], domain['volume']),
          "  ⟨ψ_tgt|ψ_tgt⟩ =", qdotlib.calc_fidelity(domain['target_psi'], domain['target_psi'], domain['volume']),
          "  ⟨ψ₀|ψ_tgt⟩² =", qdotlib.calc_fidelity(domain['initial_psi'], domain['target_psi'], domain['volume'])
    )

    best_params, es = cma.fmin2(
        objective_function,
        initial_guess,
        sigma0=initial_std_dev,
        options=options
    )

    # Results
    duration = time.time() - start_time
    print(f"\n--- Optimization Finished in {duration:.2f}s ---")
    best_fid = 1.0 - np.exp(es.result.fbest)
    print(f"Highest fidelity: {best_fid:.6f}")
    print("Optimal params:")
    for name, val in zip(["Amp","Freq","Phase","Center","Width"], best_params):
        print(f" {name}: {val:.3f}")
