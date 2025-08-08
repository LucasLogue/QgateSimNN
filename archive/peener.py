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
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
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
NUM_ITERATIONS = 15000 # Let's give this promising architecture a good run

N_COLLOCATION = 4096
N_INITIAL = 2048
N_NORM = 4096

# Loss weights
W_PDE = 10.0
W_IC = 15.0
W_NORM = 1.0

# Network architecture
LAYERS = [4] + [128] * 6 + [2]

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
    return Xc_t[indices].unsqueeze(-1)

def get_Xc_dot(t):
    """Gets classical velocity Xc_dot at time t via interpolation."""
    indices = torch.searchsorted(t_for_interp, t.squeeze(-1).contiguous()).clamp(max=len(t_for_interp)-1)
    return Xc_dot_t[indices].unsqueeze(-1)


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
class PINN(nn.Module):
    """
    PINN in a co-moving frame of reference.
    The network learns the wavefunction's shape relative to its classical center.
    ψ(x,y,z,t) = cutoff(x',y,z) * A(x',y,z,t) * exp(i*S(x',y,z,t))
    where x' = x - Xc(t)
    """
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

    def forward(self, x, y, z, t):
        # Transform to the co-moving frame
        x_prime = x + get_Xc(t)
        
        # Normalize inputs for stability
        x_prime_norm = x_prime / L
        y_norm = y / L
        z_norm = z / L
        t_norm = 2.0 * (t - T_MIN) / (T_MAX - T_MIN) - 1.0
        
        inputs = torch.cat([x_prime_norm, y_norm, z_norm, t_norm], dim=1)
        
        nn_output = self.net(inputs)
        A_raw = nn_output[:, 0:1]
        S_raw = nn_output[:, 1:2]
        
        # Cutoff function is also in the moving frame
        cutoff = (1 - (x_prime / L)**2) * (1 - y_norm**2) * (1 - z_norm**2)
        
        A = cutoff * A_raw
        S = S_raw
        
        u = A * torch.cos(S)
        v = A * torch.sin(S)
        
        return u, v
    
    def get_A_S(self, x, y, z, t):
        """Helper function to get A and S directly for the IC loss."""
        x_prime = x + get_Xc(t)
        x_prime_norm = x_prime / L
        y_norm = y / L
        z_norm = z / L
        t_norm = 2.0 * (t - T_MIN) / (T_MAX - T_MIN) - 1.0
        inputs = torch.cat([x_prime_norm, y_norm, z_norm, t_norm], dim=1)
        nn_output = self.net(inputs)
        A_raw = nn_output[:, 0:1]
        S_raw = nn_output[:, 1:2]
        cutoff = (1 - (x_prime / L)**2) * (1 - y_norm**2) * (1 - z_norm**2)
        A = cutoff * A_raw
        S = S_raw
        return A, S

# --- Loss Calculation ---
def pde_residual(model, x, y, z, t):
    """Calculate the residual of the Schrödinger PDE."""
    x.requires_grad_(True); y.requires_grad_(True); z.requires_grad_(True); t.requires_grad_(True)
    u, v = model(x, y, z, t)

    # u_t and v_t from autograd are INCOMPLETE because the gradient path through get_Xc is broken.
    # They represent the change in psi within the moving frame (∂ψ/∂t_moving).
    u_t_incomplete = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_t_incomplete = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]

    # Manually add the missing part of the chain rule to get the full lab-frame time derivative.
    # ∂ψ/∂t_lab = ∂ψ/∂t_moving + Xc_dot * ∂ψ/∂x
    xc_dot = get_Xc_dot(t)
    u_t = u_t_incomplete + xc_dot * u_x
    v_t = v_t_incomplete + xc_dot * v_x

    # The rest of the calculation proceeds as before
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    # ... (and so on for all other derivatives) ...
    u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]
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

    # The residual calculation now uses the full, correct lab-frame time derivative
    f_u = u_t + 0.5 * lap_v - V * v
    f_v = v_t - 0.5 * lap_u + V * u
    
    return f_u, f_v

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

    # STAGE 1: ADAM FOR EXPLORATION (14,000 iterations)
    # ----------------------------------------------------
    print("--- Starting Stage 1: Adam Optimizer ---")
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_adam, step_size=10000, gamma=0.5)

    domain_volume = (X_MAX - X_MIN) * (Y_MAX - Y_MIN) * (Z_MAX - Z_MIN)

    print("--- Starting Training (Co-Moving Frame Model) ---")
    start_time = time.time()
    for i in range(1, NUM_ITERATIONS + 1):
        model.train()
        optimizer_adam.zero_grad()

        x_col, y_col, z_col, t_col, x_ic, y_ic, z_ic, t_ic = sample_points()

        f_u, f_v = pde_residual(model, x_col, y_col, z_col, t_col)
        loss_pde = torch.mean(f_u**2) + torch.mean(f_v**2)

        A_ic_pred, S_ic_pred = model.get_A_S(x_ic, y_ic, z_ic, t_ic)
        A_ic_true, S_ic_true = initial_condition_A_S(x_ic, y_ic, z_ic)
        loss_ic = torch.mean((A_ic_pred - A_ic_true)**2) + torch.mean((S_ic_pred - S_ic_true)**2)
        
        t_norm_slice = torch.rand(1, 1, device=device) * T_MAX
        x_norm = torch.rand(N_NORM, 1, device=device) * (X_MAX - X_MIN) + X_MIN
        y_norm = torch.rand(N_NORM, 1, device=device) * (Y_MAX - Y_MIN) + Y_MIN
        z_norm = torch.rand(N_NORM, 1, device=device) * (Z_MAX - Z_MIN) + Z_MIN
        t_norm = t_norm_slice.expand(N_NORM, 1)
        u_norm, v_norm = model(x_norm, y_norm, z_norm, t_norm)
        psi_sq_norm = u_norm**2 + v_norm**2
        norm_pred = domain_volume * torch.mean(psi_sq_norm)
        loss_norm = (norm_pred - 1.0)**2

        total_loss = W_PDE * loss_pde + W_IC * loss_ic + W_NORM * loss_norm
        
        total_loss.backward()
        optimizer_adam.step()
        scheduler.step()

        if i % 500 == 0:
            print(f"Adam Iter: {i:6d}, Loss: {total_loss.item():.3e}, "
                  f"PDE: {loss_pde.item():.3e}, IC: {loss_ic.item():.3e}, Norm: {loss_norm.item():.3e}")

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
    def closure():
        optimizer_lbfgs.zero_grad()
        
        # We use the same collocation points for this step for stability
        x_col, y_col, z_col, t_col, x_ic, y_ic, z_ic, t_ic = sample_points()

        f_u, f_v = pde_residual(model, x_col, y_col, z_col, t_col)
        loss_pde = torch.mean(f_u**2) + torch.mean(f_v**2)

        A_ic_pred, S_ic_pred = model.get_A_S(x_ic, y_ic, z_ic, t_ic)
        A_ic_true, S_ic_true = initial_condition_A_S(x_ic, y_ic, z_ic)
        loss_ic = torch.mean((A_ic_pred - A_ic_true)**2) + torch.mean((S_ic_pred - S_ic_true)**2)
        
        t_norm_slice = torch.rand(1, 1, device=device) * T_MAX
        x_norm = torch.rand(N_NORM, 1, device=device) * (X_MAX - X_MIN) + X_MIN
        y_norm = torch.rand(N_NORM, 1, device=device) * (Y_MAX - Y_MIN) + Y_MIN
        z_norm = torch.rand(N_NORM, 1, device=device) * (Z_MAX - Z_MIN) + Z_MIN
        t_norm = t_norm_slice.expand(N_NORM, 1)
        u_norm, v_norm = model(x_norm, y_norm, z_norm, t_norm)
        psi_sq_norm = u_norm**2 + v_norm**2
        norm_pred = domain_volume * torch.mean(psi_sq_norm)
        loss_norm = (norm_pred - 1.0)**2

        total_loss = W_PDE * loss_pde + W_IC * loss_ic + W_NORM * loss_norm
        
        total_loss.backward()
        print(f"L-BFGS Loss: {total_loss.item():.3e}")
        return total_loss

    # Run the L-BFGS optimizer
    optimizer_lbfgs.step(closure)

    print("Training finished.")

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

    N_grid = 100
    x_coords = np.linspace(X_MIN, X_MAX, N_grid)
    y_coords = np.linspace(Y_MIN, Y_MAX, N_grid)
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
        # Create a high-resolution line of points along x-axis at y=0, z=0
        x_line = torch.linspace(X_MIN, X_MAX, 4096, device=device).unsqueeze(1)
        y0_line = torch.zeros_like(x_line)
        z0_line = torch.zeros_like(x_line)
        t_final_line = torch.full_like(x_line, T_MAX)

        # Get the model's prediction along this line
        u_pred_line, v_pred_line = model(x_line, y0_line, z0_line, t_final_line)
        psi_sq_line = (u_pred_line**2 + v_pred_line**2).squeeze()

        # Numerically integrate to find <x>
        dx = (X_MAX - X_MIN) / (4096 - 1)
        
        # Numerator: ∫|ψ|² * x dx
        numerator = (psi_sq_line * x_line.squeeze()).sum() * dx
        
        # Denominator: ∫|ψ|² dx
        denominator = psi_sq_line.sum() * dx

        # Expectation value <x>
        x_expectation = numerator / denominator

    print(f"PINN <x>(t=3.0) ≈ {x_expectation.item():.4f}")
    print(f"Classical Xc(t=3.0) = {float(Xc_func(T_MAX)):.4f}")
    
    x_shifted = X_np - x_shift_final
    u_analytic, v_analytic = initial_condition_psi_np(x_shifted, Y_np, Z_np_slice)
    psi_analytic_sq = u_analytic**2 + v_analytic**2

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=120)
    fig.suptitle(f'|ψ(x, y, z=0, t={T_MAX})|² Comparison', fontsize=16)
    
    vmax = max(psi_pred_sq.max(), psi_analytic_sq.max())
    if vmax < 1e-6: vmax = 0.1
    
    cp1 = axes[0].contourf(X_np, Y_np, psi_pred_sq, 100, cmap='viridis', vmin=0, vmax=vmax)
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
