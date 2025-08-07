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

Boundary conditions enforce ψ -> 0 at the boundaries of our truncated domain.

Update: Implemented loss weight annealing during Phase 2. The weight of the IC loss
is gradually decreased while the PDE loss weight is increased. This prevents the
network from getting stuck in a local minimum where the wavefunction is static.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- Configuration ---
# Domain boundaries
X_MIN, X_MAX = -5.0, 5.0
Y_MIN, Y_MAX = -5.0, 5.0
Z_MIN, Z_MAX = -5.0, 5.0
T_MIN, T_MAX = 0.0, 3.0

# Hyperparameters
LEARNING_RATE = 1e-4
# Curriculum Learning & Annealing Iterations (reduced for debugging)
ITERATIONS_PHASE1 = 3000  # Iterations for learning the IC only
ITERATIONS_PHASE2 = 12000 # Iterations for learning the full physics with annealing

N_COLLOCATION = 4096
N_INITIAL = 2048
N_BOUNDARY = 2048
N_NORM = 4096

# Base Loss weights
W_PDE_FINAL = 1.0
W_IC_FINAL = 1.0
W_BC = 10.0
W_NORM = 100.0

# Network architecture
LAYERS = [4] + [128] * 6 + [2]

# Fourier Feature Mapping
USE_FOURIER_FEATURES = True
FOURIER_SCALE = 1.0
M_FOURIER = 256

# Early Stopping Configuration
CHECK_NORM_INTERVAL = 2000
N_NORM_SAMPLES_CHECK = 16384
NORM_LOWER_THRESHOLD = 0.9
NORM_UPPER_THRESHOLD = 1.1

# --- Physics Definitions ---
def E_t(t):
    """External electric field E(t)."""
    return -6.3 * torch.exp(-((t - 0) / 1.7)**2) * torch.sin(1.13 * t + 0.64)

def V_potential(x, y, z, t):
    """Time-dependent potential V(x, y, z, t)."""
    return 0.5 * (x**2 + y**2 + z**2) - E_t(t) * x

def initial_condition_psi(x, y, z):
    """Analytic initial condition ψ(x, y, z, 0). Returns (u, v)."""
    norm_factor = (1.0 / np.pi)**(0.75)
    exponent = -0.5 * (x**2 + y**2 + z**2)
    u0 = norm_factor * torch.exp(exponent)
    v0 = torch.zeros_like(u0)
    return u0, v0

# --- Neural Network Model ---
class FourierFeatureMapping(nn.Module):
    """Fourier feature mapping layer."""
    def __init__(self, input_dims, mapping_size, scale=1.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn((input_dims, mapping_size)) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class PINN(nn.Module):
    """Physics-Informed Neural Network."""
    def __init__(self, layers, use_fourier=False, fourier_dims=0, fourier_scale=1.0):
        super(PINN, self).__init__()
        self.use_fourier = use_fourier
        if self.use_fourier:
            self.fourier_map = FourierFeatureMapping(layers[0], fourier_dims, fourier_scale)
            layers[0] = 2 * fourier_dims

        self.net = self.build_net(layers)

    def build_net(self, layers):
        net_layers = []
        for i in range(len(layers) - 1):
            net_layers.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                net_layers.append(nn.SiLU())
        return nn.Sequential(*net_layers)

    def forward(self, x, y, z, t):
        inputs = torch.cat([x, y, z, t], dim=1)
        if self.use_fourier:
            inputs = self.fourier_map(inputs)
        uv = self.net(inputs)
        u = uv[:, 0:1]
        v = uv[:, 1:2]
        return u, v

# --- Loss Calculation ---
def pde_residual(model, x, y, z, t):
    """Calculate the residual of the Schrödinger PDE."""
    x.requires_grad_(True); y.requires_grad_(True); z.requires_grad_(True); t.requires_grad_(True)
    u, v = model(x, y, z, t)
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_z = torch.autograd.grad(v, z, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    u_zz = torch.autograd.grad(u_z, z, grad_outputs=torch.ones_like(u_z), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
    v_zz = torch.autograd.grad(v_z, z, grad_outputs=torch.ones_like(v_z), create_graph=True)[0]
    lap_u = u_xx + u_yy + u_zz
    lap_v = v_xx + v_yy + v_zz
    V = V_potential(x, y, z, t)
    f_u = u_t + 0.5 * lap_v - V * v
    f_v = v_t - 0.5 * lap_u + V * u
    return f_u, f_v

def check_norm_monte_carlo(model, t_value, n_samples):
    """(Non-differentiable) Calculates ∫|ψ|² dV for early stopping checks."""
    domain_volume = (X_MAX - X_MIN) * (Y_MAX - Y_MIN) * (Z_MAX - Z_MIN)
    x = torch.rand(n_samples, 1, device=device) * (X_MAX - X_MIN) + X_MIN
    y = torch.rand(n_samples, 1, device=device) * (Y_MAX - Y_MIN) + Y_MIN
    z = torch.rand(n_samples, 1, device=device) * (Z_MAX - Z_MIN) + Z_MIN
    t = torch.full((n_samples, 1), t_value, device=device)
    
    model.eval()
    with torch.no_grad():
        u, v = model(x, y, z, t)
        psi_sq = u**2 + v**2
        norm = domain_volume * torch.mean(psi_sq)
    model.train()
    return norm.item()

# --- Data Sampling ---
def sample_points():
    """Sample points for all loss components."""
    x_col = torch.rand(N_COLLOCATION, 1, device=device) * (X_MAX - X_MIN) + X_MIN
    y_col = torch.rand(N_COLLOCATION, 1, device=device) * (Y_MAX - Y_MIN) + Y_MIN
    z_col = torch.rand(N_COLLOCATION, 1, device=device) * (Z_MAX - Z_MIN) + Z_MIN
    t_col = torch.rand(N_COLLOCATION, 1, device=device) * (T_MAX - T_MIN) + T_MIN

    x_ic = torch.rand(N_INITIAL, 1, device=device) * (X_MAX - X_MIN) + X_MIN
    y_ic = torch.rand(N_INITIAL, 1, device=device) * (Y_MAX - Y_MIN) + Y_MIN
    z_ic = torch.rand(N_INITIAL, 1, device=device) * (Z_MAX - Z_MIN) + Z_MIN
    t_ic = torch.zeros(N_INITIAL, 1, device=device)

    n_face = N_BOUNDARY // 6
    y_bc1 = torch.rand(n_face, 1, device=device)*(Y_MAX-Y_MIN)+Y_MIN
    z_bc1 = torch.rand(n_face, 1, device=device)*(Z_MAX-Z_MIN)+Z_MIN
    t_bc1 = torch.rand(n_face, 1, device=device)*(T_MAX-T_MIN)+T_MIN
    x_bc_min = torch.full((n_face, 1), X_MIN, device=device); x_bc_max = torch.full((n_face, 1), X_MAX, device=device)
    x_bc2 = torch.rand(n_face, 1, device=device)*(X_MAX-X_MIN)+X_MIN
    z_bc2 = torch.rand(n_face, 1, device=device)*(Z_MAX-Z_MIN)+Z_MIN
    t_bc2 = torch.rand(n_face, 1, device=device)*(T_MAX-T_MIN)+T_MIN
    y_bc_min = torch.full((n_face, 1), Y_MIN, device=device); y_bc_max = torch.full((n_face, 1), Y_MAX, device=device)
    x_bc3 = torch.rand(n_face, 1, device=device)*(X_MAX-X_MIN)+X_MIN
    y_bc3 = torch.rand(n_face, 1, device=device)*(Y_MAX-Y_MIN)+Y_MIN
    t_bc3 = torch.rand(n_face, 1, device=device)*(T_MAX-T_MIN)+T_MIN
    z_bc_min = torch.full((n_face, 1), Z_MIN, device=device); z_bc_max = torch.full((n_face, 1), Z_MAX, device=device)
    x_bc = torch.cat([x_bc_min, x_bc_max, x_bc2, x_bc2, x_bc3, x_bc3], dim=0)
    y_bc = torch.cat([y_bc1, y_bc1, y_bc_min, y_bc_max, y_bc3, y_bc3], dim=0)
    z_bc = torch.cat([z_bc1, z_bc1, z_bc2, z_bc2, z_bc_min, z_bc_max], dim=0)
    t_bc = torch.cat([t_bc1, t_bc1, t_bc2, t_bc2, t_bc3, t_bc3], dim=0)
    if x_bc.shape[0] < N_BOUNDARY:
        n_extra = N_BOUNDARY - x_bc.shape[0]
        indices = torch.randint(0, x_bc.shape[0], (n_extra,), device=device)
        x_bc = torch.cat([x_bc, x_bc[indices]], dim=0); y_bc = torch.cat([y_bc, y_bc[indices]], dim=0)
        z_bc = torch.cat([z_bc, z_bc[indices]], dim=0); t_bc = torch.cat([t_bc, t_bc[indices]], dim=0)

    return x_col, y_col, z_col, t_col, x_ic, y_ic, z_ic, t_ic, x_bc, y_bc, z_bc, t_bc

# --- Main Training Loop ---
if __name__ == "__main__":
    model = PINN(layers=LAYERS, use_fourier=USE_FOURIER_FEATURES, fourier_dims=M_FOURIER//2, fourier_scale=FOURIER_SCALE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8000, gamma=0.5)

    domain_volume = (X_MAX - X_MIN) * (Y_MAX - Y_MIN) * (Z_MAX - Z_MIN)

    # --- Phase 1: Learn Initial Condition ---
    W_IC_PHASE1 = 200.0
    print("--- Starting Training Phase 1: Learning Initial Condition ---")
    start_time_phase1 = time.time()
    for i in range(1, ITERATIONS_PHASE1 + 1):
        model.train()
        optimizer.zero_grad()
        _, _, _, _, x_ic, y_ic, z_ic, t_ic, _, _, _, _ = sample_points()
        u_ic_pred, v_ic_pred = model(x_ic, y_ic, z_ic, t_ic)
        u_ic_true, v_ic_true = initial_condition_psi(x_ic, y_ic, z_ic)
        loss_ic = torch.mean((u_ic_pred - u_ic_true)**2) + torch.mean((v_ic_pred - v_ic_true)**2)
        total_loss = W_IC_PHASE1 * loss_ic
        total_loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"Phase 1 - Iter: {i:6d}, IC Loss: {loss_ic.item():.4e}")
    print(f"--- Phase 1 Finished. Time: {time.time() - start_time_phase1:.2f}s ---")

    # --- Phase 2: Learn Full Physics with Annealing ---
    print("\n--- Starting Training Phase 2: Learning Full Physics with Annealing ---")
    start_time_phase2 = time.time()
    for i in range(1, ITERATIONS_PHASE2 + 1):
        model.train()
        optimizer.zero_grad()

        # Annealing factor: goes from 0 to 1 over the course of Phase 2
        annealing_factor = i / ITERATIONS_PHASE2
        
        # Anneal IC weight down and PDE weight up
        w_ic_annealed = W_IC_PHASE1 * (1 - annealing_factor) + W_IC_FINAL * annealing_factor
        w_pde_annealed = 0.1 * (1 - annealing_factor) + W_PDE_FINAL * annealing_factor


        x_col, y_col, z_col, t_col, x_ic, y_ic, z_ic, t_ic, x_bc, y_bc, z_bc, t_bc = sample_points()

        f_u, f_v = pde_residual(model, x_col, y_col, z_col, t_col)
        loss_pde = torch.mean(f_u**2) + torch.mean(f_v**2)

        u_ic_pred, v_ic_pred = model(x_ic, y_ic, z_ic, t_ic)
        u_ic_true, v_ic_true = initial_condition_psi(x_ic, y_ic, z_ic)
        loss_ic = torch.mean((u_ic_pred - u_ic_true)**2) + torch.mean((v_ic_pred - v_ic_true)**2)

        u_bc_pred, v_bc_pred = model(x_bc, y_bc, z_bc, t_bc)
        loss_bc = torch.mean(u_bc_pred**2) + torch.mean(v_bc_pred**2)
        
        t_norm_slice = torch.rand(1, 1, device=device) * T_MAX
        x_norm = torch.rand(N_NORM, 1, device=device) * (X_MAX - X_MIN) + X_MIN
        y_norm = torch.rand(N_NORM, 1, device=device) * (Y_MAX - Y_MIN) + Y_MIN
        z_norm = torch.rand(N_NORM, 1, device=device) * (Z_MAX - Z_MIN) + Z_MIN
        t_norm = t_norm_slice.expand(N_NORM, 1)
        u_norm, v_norm = model(x_norm, y_norm, z_norm, t_norm)
        psi_sq_norm = u_norm**2 + v_norm**2
        norm_pred = domain_volume * torch.mean(psi_sq_norm)
        loss_norm = (norm_pred - 1.0)**2

        total_loss = w_pde_annealed * loss_pde + w_ic_annealed * loss_ic + W_BC * loss_bc + W_NORM * loss_norm
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if i % 500 == 0: # Print less frequently to see the trend
            print(f"P2-Iter: {i:6d}, Loss: {total_loss.item():.3e}, "
                  f"PDE: {loss_pde.item():.3e} (w={w_pde_annealed:.2f}), "
                  f"IC: {loss_ic.item():.3e} (w={w_ic_annealed:.1f}), "
                  f"BC: {loss_bc.item():.3e}, Norm: {loss_norm.item():.3e}")

        if i % CHECK_NORM_INTERVAL == 0:
            check_norm_val = check_norm_monte_carlo(model, t_value=T_MAX / 2.0, n_samples=N_NORM_SAMPLES_CHECK)
            print(f"--- Norm check at iter {i}: ∫|ψ(t=T_MAX/2)|² dV ≈ {check_norm_val:.4f} ---")
            if not (NORM_LOWER_THRESHOLD < check_norm_val < NORM_UPPER_THRESHOLD):
                print(f"\n!!! EARLY STOPPING TRIGGERED !!! Norm ({check_norm_val:.4f}) is outside stable range.")
                break

    print("Training finished.")

    MODEL_PATH = "pinn_schrodinger_3d.pth"
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    print("Starting post-processing and visualization...")

    def classical_ode(t, state):
        x, p = state
        dxdt = p
        E_t_np = lambda t_val: -6.3 * np.exp(-((t_val - 0)/1.7)**2) * np.sin(1.13*t_val + 0.64)
        dpdt = -x + E_t_np(t)
        return [dxdt, dpdt]

    sol = solve_ivp(classical_ode, [T_MIN, T_MAX], [0, 0], dense_output=True, t_eval=[T_MAX])
    x_shift_final = sol.y[0, -1]
    print(f"Corrected classical x-shift at t={T_MAX}: {x_shift_final:.4f}")

    N_grid = 100
    x_plot = torch.linspace(X_MIN, X_MAX, N_grid, device=device)
    y_plot = torch.linspace(Y_MIN, Y_MAX, N_grid, device=device)
    X, Y = torch.meshgrid(x_plot, y_plot, indexing='ij')
    Z_slice = torch.zeros_like(X, device=device)
    T_final = torch.full_like(X, T_MAX, device=device)
    x_flat = X.flatten().unsqueeze(1); y_flat = Y.flatten().unsqueeze(1)
    z_flat = Z_slice.flatten().unsqueeze(1); t_flat = T_final.flatten().unsqueeze(1)

    model.eval()
    with torch.no_grad():
        u_pred, v_pred = model(x_flat, y_flat, z_flat, t_flat)
        psi_pred_sq = (u_pred**2 + v_pred**2).reshape(N_grid, N_grid).cpu().numpy()

    x_shifted = X.cpu() - x_shift_final
    u_analytic, v_analytic = initial_condition_psi(x_shifted, Y.cpu(), Z_slice.cpu())
    psi_analytic_sq = (u_analytic**2 + v_analytic**2).cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=120)
    fig.suptitle(f'|ψ(x, y, z=0, t={T_MAX})|² Comparison', fontsize=16)
    
    vmax = max(psi_pred_sq.max(), psi_analytic_sq.max())
    if vmax < 1e-6: vmax = 0.1
    
    cp1 = axes[0].contourf(X.cpu(), Y.cpu(), psi_pred_sq, 100, cmap='viridis', vmin=0, vmax=vmax)
    fig.colorbar(cp1, ax=axes[0]); axes[0].set_title('PINN Prediction')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('y'); axes[0].set_aspect('equal', 'box')

    cp2 = axes[1].contourf(X.cpu(), Y.cpu(), psi_analytic_sq, 100, cmap='viridis', vmin=0, vmax=vmax)
    fig.colorbar(cp2, ax=axes[1]); axes[1].set_title('Analytic (Shifted Ground State)')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('y'); axes[1].set_aspect('equal', 'box')
    
    error = np.abs(psi_pred_sq - psi_analytic_sq)
    cp3 = axes[2].contourf(X.cpu(), Y.cpu(), error, 100, cmap='inferno')
    fig.colorbar(cp3, ax=axes[2]); axes[2].set_title('Absolute Error')
    axes[2].set_xlabel('x'); axes[2].set_ylabel('y'); axes[2].set_aspect('equal', 'box')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    PLOT_PATH = "pinn_schrodinger_3d_results.png"
    plt.savefig(PLOT_PATH)
    print(f"Plot saved to {PLOT_PATH}")
    plt.show()
