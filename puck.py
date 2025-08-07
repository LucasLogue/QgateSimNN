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

Update: Major architectural change based on feedback.
1.  New Ansatz: The model now has the form ψ = cutoff(x,y,z) * NN(x,y,z,t), which
    enforces boundary conditions by construction, eliminating loss_bc.
2.  Simplified Training: Removed curriculum learning. Back to a single training phase.
3.  Input Normalization: All inputs (x,y,z,t) are scaled to [-1, 1].
4.  No Fourier Features: Disabled for this debugging run for clarity.
5.  Activation Function: Reverted to tanh for smoother second derivatives.
6.  Loss Weights: Significantly increased the PDE loss weight.
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

# Spatial domain length for cutoff function
L = 5.0

# Hyperparameters
LEARNING_RATE = 1e-4
NUM_ITERATIONS = 15000 # Kept low for debugging

N_COLLOCATION = 4096
N_INITIAL = 2048
N_NORM = 4096

# Loss weights - PDE loss is now dominant
W_PDE = 10.0
W_IC = 5.0
W_NORM = 1.0

# Network architecture
LAYERS = [4] + [128] * 6 + [2] # 4 inputs (x,y,z,t)

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
class PINN(nn.Module):
    """
    Physics-Informed Neural Network.
    The forward pass is constructed to automatically satisfy the boundary conditions.
    ψ(x,y,z,t) = cutoff(x,y,z) * NN(x_norm, y_norm, z_norm, t_norm)
    """
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.net = self.build_net(layers)

    def build_net(self, layers):
        net_layers = []
        for i in range(len(layers) - 1):
            net_layers.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                net_layers.append(nn.Tanh()) # Using tanh for smoother derivatives
        return nn.Sequential(*net_layers)

    def forward(self, x, y, z, t):
        # Normalize inputs to [-1, 1] for stability
        x_norm = x / L
        y_norm = y / L
        z_norm = z / L
        t_norm = 2.0 * (t - T_MIN) / (T_MAX - T_MIN) - 1.0
        
        inputs = torch.cat([x_norm, y_norm, z_norm, t_norm], dim=1)
        
        # Get raw NN output
        nn_output = self.net(inputs)
        
        # Define a cutoff function that is 0 at the boundaries
        cutoff = (1 - x_norm**2) * (1 - y_norm**2) * (1 - z_norm**2)
        
        # Apply the cutoff to enforce boundary conditions
        # The network now learns the behavior *inside* the box
        u = cutoff * nn_output[:, 0:1]
        v = cutoff * nn_output[:, 1:2]
        
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

    return x_col, y_col, z_col, t_col, x_ic, y_ic, z_ic, t_ic

# --- Main Training Loop ---
if __name__ == "__main__":
    model = PINN(layers=LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.5)

    domain_volume = (X_MAX - X_MIN) * (Y_MAX - Y_MIN) * (Z_MAX - Z_MIN)

    print("--- Starting Training ---")
    start_time = time.time()
    for i in range(1, NUM_ITERATIONS + 1):
        model.train()
        optimizer.zero_grad()

        x_col, y_col, z_col, t_col, x_ic, y_ic, z_ic, t_ic = sample_points()

        # PDE Loss
        f_u, f_v = pde_residual(model, x_col, y_col, z_col, t_col)
        loss_pde = torch.mean(f_u**2) + torch.mean(f_v**2)

        # Initial Condition Loss
        u_ic_pred, v_ic_pred = model(x_ic, y_ic, z_ic, t_ic)
        u_ic_true, v_ic_true = initial_condition_psi(x_ic, y_ic, z_ic)
        loss_ic = torch.mean((u_ic_pred - u_ic_true)**2) + torch.mean((v_ic_pred - v_ic_true)**2)
        
        # Norm Conservation Loss
        t_norm_slice = torch.rand(1, 1, device=device) * T_MAX
        x_norm = torch.rand(N_NORM, 1, device=device) * (X_MAX - X_MIN) + X_MIN
        y_norm = torch.rand(N_NORM, 1, device=device) * (Y_MAX - Y_MIN) + Y_MIN
        z_norm = torch.rand(N_NORM, 1, device=device) * (Z_MAX - Z_MIN) + Z_MIN
        t_norm = t_norm_slice.expand(N_NORM, 1)
        u_norm, v_norm = model(x_norm, y_norm, z_norm, t_norm)
        psi_sq_norm = u_norm**2 + v_norm**2
        norm_pred = domain_volume * torch.mean(psi_sq_norm)
        loss_norm = (norm_pred - 1.0)**2

        # Total Loss (No BC loss needed)
        total_loss = W_PDE * loss_pde + W_IC * loss_ic + W_NORM * loss_norm
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if i % 500 == 0:
            print(f"Iter: {i:6d}, Loss: {total_loss.item():.3e}, "
                  f"PDE: {loss_pde.item():.3e}, IC: {loss_ic.item():.3e}, Norm: {loss_norm.item():.3e}")

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
