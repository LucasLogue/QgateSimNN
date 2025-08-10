# -*- coding: utf-8 -*-
"""
Physics-Informed Neural Network (PINN) for the 3D Time-Dependent Schrödinger Equation.

This script solves the following equation:
  i * ∂ψ/∂t = -1/2 * (∂²ψ/∂x² + ∂²ψ/∂y² + ∂²ψ/∂z²) + [1/2 * (x² + y² + z²) - E(t) * x] * ψ
on the domain (x, y, z) ∈ [-5, 5]³ and t ∈ [0, 3].

The external electric field is given by:
  E(t) = -6.3 * exp(-((t - 0) / 1.7)²) * sin(1.13 * t + 0.64)

The initial condition at t=0 is the ground state of the 3D Quantum Harmonic Oscillator:
  ψ(x, y, z, 0) = (1/π)^(3/4) * exp(-1/2 * (x² + y² + z²))

Update: Implemented a co-moving frame of reference, as suggested by the user.
1.  Classical Trajectory: The classical path Xc(t) and velocity Xc_dot(t) of the
    wavepacket center are pre-computed.
2.  Coordinate Transformation: The network learns in a transformed coordinate system
    x' = x - Xc(t). This simplifies the problem by removing the large-scale motion.
3.  Transformed PDE: The time derivative in the PDE loss is adjusted to account for
    the moving frame via the chain rule: ∂ψ/∂t -> ∂ψ/∂t - Xc_dot * ∂ψ/∂x.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import time
import os
CONFIG = "6-FAST"

# ---------- DEBUG HELPER ----------
def dbg(tag, **vals):
    """Pretty-print a set of named tensors / scalars."""
    msg = [f"[{tag}]"]
    for k, v in vals.items():
        if torch.is_tensor(v):
            v = v.detach().cpu().numpy()
        msg.append(f"{k}={v}")
    print("  ".join(msg))
# ----------------------------------

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- Configuration ---
if CONFIG == "6-FAST":
    NUM_ITERATIONS = 20000
    LEARNING_RATE = 2e-5
    N_COLLOCATION = 4096
    N_INITIAL = 2048
    N_NORM = 4096
    LAYERS = [4] + [128] * 6 + [2]

# Domain boundaries
#X_MIN, X_MAX = -5.0, 5.0
X_MIN, X_MAX = -10.0, 10.0 
Y_MIN, Y_MAX = -5.0, 5.0
Z_MIN, Z_MAX = -5.0, 5.0
T_MIN, T_MAX = 0.0, 3.0

# Spatial domain length for cutoff function
L = 5.0

# Hyperparameters
# # FUTURE !! HYPERPARAMETERS FOR A HIGH-FIDELITY RUN
# # 1. More Power: A deeper and wider network
# LAYERS = [4] + [256] * 8 + [2]
# # 2. More Data: 4x the training points
# N_COLLOCATION = 16384
# N_INITIAL = 4096
# N_NORM = 16384
# # 3. More Time: 5x the training iterations
# NUM_ITERATIONS = 50000
# #!!!!!!!!!!!!!!!!!!!!!!!!!!




# LEARNING_RATE = 1e-5
#NUM_ITERATIONS = 10000 # Let's give this promising architecture a good run
# NUM_ITERATIONS = 50000
#N_COLLOCATION = 4096
#N_COLLOCATION = 8192
# N_COLLOCATION = 16384
#N_INITIAL = 2048
#N_INITIAL = 4096
# N_INITIAL = 8192
#N_NORM = 4096
#N_NORM = 16384
# N_NORM = 32768
# Loss weights
W_PDE = 60.0
W_IC = 15.0
W_NORM = 80.0

# Network architecture
# LAYERS = [4] + [128] * 6 + [2]
# LAYERS = [4] + [256] * 8 + [2]
# --- Pre-compute Classical Trajectory ---
def classical_ode(t, state):
    x, p = state
    dxdt = p
    E_t_np = lambda t_val: -6.3 * np.exp(-((t_val - 0)/1.7)**2) * np.sin(1.13*t_val + 0.64)
    dpdt = -x - E_t_np(t)
    return [dxdt, dpdt]

print("Pre-computing classical trajectory...")
t_eval = np.linspace(T_MIN, T_MAX, 500)
sol = solve_ivp(classical_ode, [T_MIN, T_MAX], [0, 0], t_eval=t_eval, dense_output=True)

# Create fast interpolation functions for position and velocity
Xc_func = interp1d(sol.t, sol.y[0], kind='cubic', fill_value="extrapolate")
Xc_dot_func = interp1d(sol.t, sol.y[1], kind='cubic', fill_value="extrapolate")

# Convert to PyTorch tensors on the correct device
t_for_interp = torch.linspace(T_MIN, T_MAX, 500, device=device)
Xc_t = torch.tensor(Xc_func(t_for_interp.cpu().numpy()), dtype=torch.float32, device=device)
Xc_dot_t = torch.tensor(Xc_dot_func(t_for_interp.cpu().numpy()), dtype=torch.float32, device=device)

def get_Xc(t):
    """Gets classical position Xc at time t via interpolation."""
    indices = torch.searchsorted(t_for_interp, t.squeeze(-1).contiguous()).clamp(max=len(t_for_interp)-1)
    out = Xc_t[indices].unsqueeze(-1)
    if not hasattr(get_Xc, "_flag"):
        dbg("Xc_lookup", sample_t=t[:3].flatten(), Xc_sample=out[:3].flatten())
        get_Xc._flag = False
    return out

def get_Xc_dot(t):
    """Gets classical velocity Xc_dot at time t via interpolation."""
    indices = torch.searchsorted(t_for_interp, t.squeeze(-1).contiguous()).clamp(max=len(t_for_interp)-1)
    out = Xc_dot_t[indices].unsqueeze(-1)
    if not hasattr(get_Xc, "_flag"):
        dbg("Xc_lookup", sample_t=t[:3].flatten(), Xc_sample=out[:3].flatten())
        get_Xc._flag = False
    return out


# --- Physics Definitions ---
def E_t(t):
    """External electric field E(t)."""
    return -6.3 * torch.exp(-((t - 0) / 1.7)**2) * torch.sin(1.13 * t + 0.64)

def V_potential(x, y, z, t):
    """Time-dependent potential V(x, y, z, t)."""
    return 0.5 * (x**2 + y**2 + z**2) + E_t(t) * x

def initial_condition_A_S(x, y, z):
    """Analytic initial condition for Amplitude (A) and Phase (S)."""
    norm_factor = (1.0 / np.pi)**(0.75)
    exponent = -0.5 * (x**2 + y**2 + z**2)
    A0 = norm_factor * torch.exp(exponent)
    S0 = torch.zeros_like(A0)
    return A0, S0

# --- Neural Network Model ---
# --- FINAL, OPTIMIZED PINN CLASS ---
# --- The Final, Correct, and Consistent PINN Class ---
# --- FINAL, CORRECTED PINN CLASS ---
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.net = self.build_net(layers)

    def build_net(self, layers):
        net_layers = []
        for i in range(len(layers) - 1):
            net_layers.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                net_layers.append(nn.Tanh())
        return nn.Sequential(*net_layers)

    def _forward_base(self, x, y, z, t):
        x_prime = x - get_Xc(t)
        x_prime_norm = x_prime / L
        y_norm = y / L
        z_norm = z / L
        t_norm = 2.0 * (t - T_MIN) / (T_MAX - T_MIN) - 1.0
        
        inputs = torch.cat([x_prime_norm, y_norm, z_norm, t_norm], dim=1)
        nn_output = self.net(inputs)
        
        A_raw = nn_output[:, 0:1]
        S_raw = nn_output[:, 1:2] # The NN now learns the full phase S
        
        return A_raw, S_raw, x_prime, y_norm, z_norm

    def forward(self, x, y, z, t):
        A_raw, S_raw, x_prime, y_norm, z_norm = self._forward_base(x, y, z, t)
        
        cutoff = (1 - (x_prime / L)**2) * (1 - y_norm**2) * (1 - z_norm**2)
        A = cutoff * A_raw
        S = S_raw
        
        u = A * torch.cos(S)
        v = A * torch.sin(S)
        
        return u, v
    
    def get_A_S(self, x, y, z, t, apply_cutoff=True):
        A_raw, S_raw, x_prime, y_norm, z_norm = self._forward_base(x, y, z, t)
        
        if apply_cutoff:
            cutoff = (1 - (x_prime / L)**2) * (1 - y_norm**2) * (1 - z_norm**2)
            A = cutoff * A_raw
        else:
            A = A_raw
        
        S = S_raw
        return A, S


def pde_residual(model, x, y, z, t):
    """Calculate the residual of the Schrödinger PDE (Optimized)."""
    x.requires_grad_(True); y.requires_grad_(True); z.requires_grad_(True); t.requires_grad_(True)
    
    u, v = model(x, y, z, t)

    # --- Bundle derivative calculations for u ---
    # Calculate all first-order derivatives of u in one go
    u_grads = torch.autograd.grad(u, [x, y, z, t], grad_outputs=torch.ones_like(u), create_graph=True)
    u_x, u_y, u_z, u_t_incomplete = u_grads[0], u_grads[1], u_grads[2], u_grads[3]

    # Calculate second-order derivatives from the first-order ones
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    u_zz = torch.autograd.grad(u_z, z, grad_outputs=torch.ones_like(u_z), create_graph=True)[0]
    lap_u = u_xx + u_yy + u_zz

    # --- Bundle derivative calculations for v ---
    v_grads = torch.autograd.grad(v, [x, y, z, t], grad_outputs=torch.ones_like(v), create_graph=True)
    v_x, v_y, v_z, v_t_incomplete = v_grads[0], v_grads[1], v_grads[2], v_grads[3]

    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
    v_zz = torch.autograd.grad(v_z, z, grad_outputs=torch.ones_like(v_z), create_graph=True)[0]
    lap_v = v_xx + v_yy + v_zz

    # --- Apply chain rule and compute residuals (no changes here) ---
    xc_dot = get_Xc_dot(t)
    u_t = u_t_incomplete - xc_dot * u_x
    v_t = v_t_incomplete - xc_dot * v_x
    
    V = V_potential(x, y, z, t)
    f_u = u_t + 0.5 * lap_v - V * v
    f_v = v_t - 0.5 * lap_u + V * u
    
    return f_u, f_v


# --- Data Sampling ---
def sample_points():
    """
    Draw collocation / IC / normalisation points *uniformly in the co-moving frame*.
    Every point then satisfies |x'| ≤ L, so the cutoff never switches sign.
    """
    # --- collocation set -------------------------------------------------
    t_col = torch.rand(N_COLLOCATION, 1, device=device)*(T_MAX-T_MIN) + T_MIN          # U[0,T]
    x_prime = torch.rand_like(t_col)*(2*L) - L                                         # U[-L,L]
    x_col   = x_prime + get_Xc(t_col)                                                  # back-shift to lab
    y_col   = torch.rand_like(x_col)*(Y_MAX-Y_MIN) + Y_MIN
    z_col   = torch.rand_like(x_col)*(Z_MAX-Z_MIN) + Z_MIN

    # --- initial-condition set (t = 0; Xc(0)=0 so x'=x) -------------------
    x_ic = torch.rand(N_INITIAL, 1, device=device)*(2*L) - L
    y_ic = torch.rand_like(x_ic)*(Y_MAX-Y_MIN) + Y_MIN
    z_ic = torch.rand_like(x_ic)*(Z_MAX-Z_MIN) + Z_MIN
    t_ic = torch.zeros_like(x_ic)

    # --- normalisation batch (reuse the same trick) ----------------------
    t_norm = torch.rand(N_NORM, 1, device=device)*(T_MAX-T_MIN) + T_MIN
    x_prime_n = torch.rand_like(t_norm)*(2*L) - L
    x_norm = x_prime_n + get_Xc(t_norm)
    y_norm = torch.rand_like(x_norm)*(Y_MAX-Y_MIN) + Y_MIN
    z_norm = torch.rand_like(x_norm)*(Z_MAX-Z_MIN) + Z_MIN

    return x_col, y_col, z_col, t_col, x_ic, y_ic, z_ic, t_ic, x_norm, y_norm, z_norm, t_norm

# --- Main Training Loop ---
if __name__ == "__main__":
    model = PINN(layers=LAYERS).to(device)
    #model = torch.compile(model)

    # STAGE 1: ADAM FOR EXPLORATION 
    # ----------------------------------------------------
    warmupfrac = 0.3
    warmupsteps = int(NUM_ITERATIONS*warmupfrac)
    print("--- Starting Stage 1: Adam Optimizer ---")
    def get_w_pde(iteration, warmup_steps=warmupsteps, max_weight=60.0):
        """Linearly ramps up the PDE weight from 0 to max_weight."""
        if iteration < 1000: # Start with a tiny, non-zero weight to avoid issues
            return 0.01
        progress = min(1.0, (iteration - 1000) / warmup_steps)
        return max_weight * progress
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer_adam, step_size=10000, gamma=0.5)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_adam, T_max=NUM_ITERATIONS, eta_min=1e-7)
    # Keep LR high for 80% of the run, then drop it sharply for the final fine-tuning.
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_adam, step_size=int(NUM_ITERATIONS * 0.8), gamma=0.1)
    domain_volume = (X_MAX - X_MIN) * (Y_MAX - Y_MIN) * (Z_MAX - Z_MIN)

    print("--- Starting Training (Co-Moving Frame Model) ---")
    start_time = time.time()
    for i in range(1, NUM_ITERATIONS + 1):
        model.train()
        optimizer_adam.zero_grad()

        (x_col, y_col, z_col, t_col,
        x_ic,  y_ic,  z_ic,  t_ic,
        x_norm, y_norm, z_norm, t_norm) = sample_points()

        f_u, f_v = pde_residual(model, x_col, y_col, z_col, t_col)
        loss_pde = torch.mean(f_u**2) + torch.mean(f_v**2)

        A_ic_pred, S_ic_pred = model.get_A_S(x_ic, y_ic, z_ic, t_ic, apply_cutoff=False)
        A_ic_true, S_ic_true = initial_condition_A_S(x_ic, y_ic, z_ic)

        #FOR SIMPLIFIED IC LOSS
        # loss_ic = torch.mean((A_ic_pred - A_ic_true)**2) + torch.mean((S_ic_pred - S_ic_true)**2)


        # # Create a weight function (the Gaussian itself) to focus the loss on the center
        ic_weight = A_ic_true**2
        
        loss_ic_A = torch.mean(ic_weight * (A_ic_pred - A_ic_true)**2)
        loss_ic_S = torch.mean((S_ic_pred - S_ic_true)**2) # Phase loss doesn't need weighting
        loss_ic = loss_ic_A + loss_ic_S


        # loss_ic = torch.mean((A_ic_pred - A_ic_true)**2) + torch.mean((S_ic_pred - S_ic_true)**2)
        u_norm, v_norm = model(x_norm, y_norm, z_norm, t_norm)
        psi_sq_norm = u_norm**2 + v_norm**2
        norm_pred = domain_volume * torch.mean(psi_sq_norm)
        loss_norm = (norm_pred - 1.0)**2

        # Get A and S at the collocation points to calculate the phase gradients
        # We use apply_cutoff=True because we're regularizing the main solution for t > 0
        A_col, S_col = model.get_A_S(x_col, y_col, z_col, t_col, apply_cutoff=True)
        
        # Calculate the spatial gradients of the phase
        S_grads = torch.autograd.grad(S_col, [x_col, y_col, z_col], grad_outputs=torch.ones_like(S_col), create_graph=True)
        S_x, S_y, S_z = S_grads[0], S_grads[1], S_grads[2]
        
        # The new loss term penalizes the magnitude of the phase gradients to enforce smoothness
        loss_reg = torch.mean(S_x**2) + torch.mean(S_y**2) + torch.mean(S_z**2)
        
        # Add a small weight for this regularization term
        W_REG = 0.01
        # --- End of New Block ---
        # -----------------------------------------------------------------


        W_PDE = get_w_pde(i, warmup_steps=warmupsteps, max_weight=60.0)

        total_loss = W_PDE * loss_pde + W_IC * loss_ic + W_NORM * loss_norm #+ W_REG * loss_reg

        #very working for loss being updated to line above for new fidelity test
        #total_loss = W_PDE * loss_pde + W_IC * loss_ic + W_NORM * loss_norm
        
        total_loss.backward()
        optimizer_adam.step()
        scheduler.step()
        if i % 500 == 0:
            # centre-of-mass in the *lab* frame for this batch
            x_lab_mean = (u_norm**2 + v_norm**2).squeeze()  # weighted later, fine for trend
            #print((x_norm.squeeze()).min(), x_norm.squeeze().max())
            ballslol = x_norm.squeeze()
            x_cm_batch = (ballslol * x_lab_mean).sum() / x_lab_mean.sum()
            #psisq = (u_norm**2 )
            dbg("iter", step=i, loss=total_loss.item(), PDE=loss_pde.item(),
                x_cm_batch=x_cm_batch,
                norm=float(norm_pred))
            

    #Stage 2 Optimizer disabled rn cause it sucks dick
    print("\n--- Starting Stage 2: L-BFGS Optimizer ---")
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=0.1,  # L-BFGS often works well with a higher learning rate
        max_iter=1000,
        max_eval=1250,
        history_size=100,
        line_search_fn="strong_wolfe" # A robust line search algorithm
    )
    # L-BFGS requires a 'closure' function that it can call multiple times
    (x_col_lbfgs, y_col_lbfgs, z_col_lbfgs, t_col_lbfgs,
    x_ic_lbfgs,  y_ic_lbfgs,  z_ic_l_lbfgs,  t_ic_lbfgs,
    x_norm_lbfgs, y_norm_lbfgs, z_norm_lbfgs, t_norm_lbfgs) = sample_points()
    def closure():
        optimizer_lbfgs.zero_grad()


        f_u, f_v = pde_residual(model, x_col_lbfgs, y_col_lbfgs, z_col_lbfgs, t_col_lbfgs)
        loss_pde = torch.mean(f_u**2) + torch.mean(f_v**2)

        A_ic_pred, S_ic_pred = model.get_A_S(x_ic_lbfgs, y_ic_lbfgs, z_ic_l_lbfgs, t_ic_lbfgs)
        A_ic_true, S_ic_true = initial_condition_A_S(x_ic_lbfgs, y_ic_lbfgs, z_ic_l_lbfgs)
        loss_ic = torch.mean((A_ic_pred - A_ic_true)**2) + torch.mean((S_ic_pred - S_ic_true)**2)
        u_norm, v_norm = model(x_norm_lbfgs, y_norm_lbfgs, z_norm_lbfgs, t_norm_lbfgs)
        psi_sq_norm = u_norm**2 + v_norm**2
        norm_pred = domain_volume * torch.mean(psi_sq_norm)
        loss_norm = (norm_pred - 1.0)**2

        total_loss = W_PDE * loss_pde + W_IC * loss_ic + W_NORM * loss_norm
        
        total_loss.backward()
        print(f"L-BFGS Loss: {total_loss.item():.3e}")
        return total_loss
    # Run the L-BFGS optimizer
    #optimizer_lbfgs.step(closure)

    print("Training finished.")

    MODEL_PATH = "pinn_schrodinger_3d.pth"
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    print("Starting post-processing and visualization...")

    # For analytic plot, we need the original |ψ|²
    def initial_condition_psi_np(x, y, z):
        norm_factor = (1.0 / np.pi)**(0.75)
        exponent = -0.5 * (x**2 + y**2 + z**2)
        u0 = norm_factor * np.exp(exponent)
        v0 = np.zeros_like(u0)
        return u0, v0

    # --- CHANGES START HERE ---

    # Calculate the final classical position FIRST.
    x_shift_final = Xc_func(T_MAX)
    print(f"Corrected classical x-shift at t={T_MAX}: {x_shift_final:.4f}")

    # Define a new evaluation domain centered on the final classical position
    EVAL_X_MIN = x_shift_final - L
    EVAL_X_MAX = x_shift_final + L
    print(f"Setting evaluation domain to [{EVAL_X_MIN:.2f}, {EVAL_X_MAX:.2f}] to match the co-moving frame.")

    # Create the grid using the CORRECTED evaluation domain
    N_grid = 100
    # x_coords = np.linspace(X_MIN, X_MAX, N_grid) # OLD LINE
    x_coords = np.linspace(EVAL_X_MIN, EVAL_X_MAX, N_grid) # NEW LINE
    y_coords = np.linspace(Y_MIN, Y_MAX, N_grid)
    
    # --- CHANGES END HERE ---
    X_np, Y_np = np.meshgrid(x_coords, y_coords, indexing='ij')
    Z_np_slice = np.zeros_like(X_np)
    
    x_flat = torch.tensor(X_np.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    y_flat = torch.tensor(Y_np.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    z_flat = torch.tensor(Z_np_slice.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    t_final_flat = torch.full_like(x_flat, T_MAX)

    model.eval()
    with torch.no_grad():
        u_pred, v_pred = model(x_flat, y_flat, z_flat, t_final_flat)
        psi_pred_sq = (u_pred**2 + v_pred**2).reshape(N_grid, N_grid).cpu().numpy()

    x_shift_final = Xc_func(T_MAX)
    print(f"Corrected classical x-shift at t={T_MAX}: {x_shift_final:.4f}")

    print("\n--- Quantitative Sanity Check ---")
    with torch.no_grad():
        # --- CHANGES START HERE ---

        # Create a high-resolution line of points along the CORRECTED domain
        # x_line = torch.linspace(X_MIN, X_MAX, 4096, device=device).unsqueeze(1) # OLD LINE
        x_line = torch.linspace(EVAL_X_MIN, EVAL_X_MAX, 4096, device=device).unsqueeze(1) # NEW LINE
        
        # --- CHANGES END HERE ---

        y0_line = torch.zeros_like(x_line)
        z0_line = torch.zeros_like(x_line)
        t_final_line = torch.full_like(x_line, T_MAX)

        # Get the model's prediction along this line
        u_pred_line, v_pred_line = model(x_line, y0_line, z0_line, t_final_line)
        psi_sq_line = (u_pred_line**2 + v_pred_line**2).squeeze()

        # Numerically integrate to find <x>
        # --- CHANGES START HERE ---
        
        # dx = (X_MAX - X_MIN) / (4096 - 1) # OLD LINE
        dx = (EVAL_X_MAX - EVAL_X_MIN) / (4096 - 1) # NEW LINE

        # --- CHANGES END HERE ---
        
        # Numerator: ∫|ψ|² * x dx
        numerator = (psi_sq_line * x_line.squeeze()).sum() * dx
        
        # Denominator: ∫|ψ|² dx
        denominator = psi_sq_line.sum() * dx

        # Expectation value <x>
        x_expectation = numerator / denominator

    print(f"PINN <x>(t=3.0) ≈ {x_expectation.item():.4f}")
    print(f"Classical Xc(t=3.0) = {float(Xc_func(T_MAX)):.4f}")

    # --- NEW: Numerical Comparison Metrics ---
    print("\n--- Fidelity and Error Analysis ---")

    # Helper function to get the analytical solution on torch tensors
    def get_analytic_solution_torch(x, y, z, x_shift):
        # --- FIX #1: Convert the CPU float x_shift to a tensor ---
        x_shift_tensor = torch.tensor(x_shift, device=x.device, dtype=x.dtype)
        x_shifted = x - x_shift_tensor
        
        # --- FIX #2: Convert the CPU float norm_factor to a tensor ---
        norm_factor_val = (1.0 / np.pi)**(0.75)
        norm_factor = torch.tensor(norm_factor_val, device=x.device, dtype=x.dtype)

        exponent = -0.5 * (x_shifted**2 + y**2 + z**2)
        u_analytic = norm_factor * torch.exp(exponent)
        v_analytic = torch.zeros_like(u_analytic)
        return u_analytic, v_analytic

    with torch.no_grad():
        # Get the analytical solution on the same high-resolution line
        u_analytic_line, v_analytic_line = get_analytic_solution_torch(
            x_line, y0_line, z0_line, x_shift_final
        )

        # 1. L2 Error: A measure of the overall difference between the wavefunctions
        error_real_part = (u_pred_line - u_analytic_line)**2
        error_imag_part = (v_pred_line - v_analytic_line)**2
        l2_error = torch.sqrt(torch.mean(error_real_part + error_imag_part))
        print(f"L2 Error between PINN and Analytic solution: {l2_error.item():.4f}")

        # 2. Fidelity: Measures the overlap between the two quantum states. Target is 1.0.
        psi_analytic_torch = u_analytic_line
        psi_pinn_torch = u_pred_line + 1j * v_pred_line

        # Calculate the inner product
        integrand = psi_analytic_torch * psi_pinn_torch
        inner_product = torch.sum(integrand) * dx

        # Normalize and calculate fidelity
        norm_pinn = torch.sum(psi_pinn_torch.abs()**2) * dx
        fidelity = (inner_product.abs()**2) / norm_pinn
        print(f"Fidelity of PINN solution: {fidelity.item():.4f}")

    # --- End of New Block ---
    
    x_shifted = X_np - x_shift_final
    u_analytic, v_analytic = initial_condition_psi_np(x_shifted, Y_np, Z_np_slice)
    psi_analytic_sq = u_analytic**2 + v_analytic**2

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=120)
    fig.suptitle(f'|ψ(x, y, z=0, t={T_MAX})|² Comparison', fontsize=16)
    
    vmax = max(psi_pred_sq.max(), psi_analytic_sq.max())
    if vmax < 1e-6: vmax = 0.1
    vmin = 1e-9 
    
    cp1 = axes[0].contourf(X_np, Y_np, psi_pred_sq, 100, cmap='viridis', norm=colors.LogNorm(vmin=vmin, vmax=vmax))
    fig.colorbar(cp1, ax=axes[0]); axes[0].set_title('PINN Prediction')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('y'); axes[0].set_aspect('equal', 'box')

    cp2 = axes[1].contourf(X_np, Y_np, psi_analytic_sq, 100, cmap='viridis', vmin=0, vmax=vmax)
    fig.colorbar(cp2, ax=axes[1]); axes[1].set_title('Analytic (Shifted Ground State)')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('y'); axes[1].set_aspect('equal', 'box')
    
    error = np.abs(psi_pred_sq - psi_analytic_sq)
    cp3 = axes[2].contourf(X_np, Y_np, error, 100, cmap='inferno')
    fig.colorbar(cp3, ax=axes[2]); axes[2].set_title('Absolute Error')
    axes[2].set_xlabel('x'); axes[2].set_ylabel('y'); axes[2].set_aspect('equal', 'box')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    PLOT_PATH = "pinn_schrodinger_3d_results.png"
    plt.savefig(PLOT_PATH)
    print(f"Plot saved to {PLOT_PATH}")
    plt.show()
