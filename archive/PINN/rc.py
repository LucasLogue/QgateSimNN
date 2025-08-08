# test_pinn.py (FIXED version)

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration: Must match your training script ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {DEVICE}")

OMEGA = 1.0
X_MIN, X_MAX = -5.0, 5.0
T_MAX = 3.0

# --- PINN Architecture: Must match your training script ---
class AdaptiveTanh(nn.Module):
    def __init__(self, initial_a=1.0):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(initial_a))

    def forward(self, x):
        return self.a * torch.tanh(x)

class SchrodingerPINN(nn.Module):
    def __init__(self, layers=8, units=256):
        super().__init__()
        net, in_dim = [], 2
        for _ in range(layers):
            net += [nn.Linear(in_dim, units), AdaptiveTanh()]
            in_dim = units
        net.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*net)
        self.timefactor = 5.0

    def initial_state(self, x):
        coeff = (OMEGA / torch.pi) ** 0.25
        real_part = coeff * torch.exp(-0.5 * OMEGA * x ** 2)
        return torch.complex(real_part, torch.zeros_like(real_part))

    def forward(self, x, t):
        psi_nn_raw = self.net(torch.cat([x, t], dim=-1))
        psi_nn = torch.complex(psi_nn_raw[..., 0], psi_nn_raw[..., 1])
        tenvelope = (1.0 - torch.exp(-self.timefactor * t))
        boundary_term = (x - X_MIN) * (X_MAX - x)
        
        # --- FIX 1: The unsqueeze is necessary to keep arrays as [N, 1] not [N, N] ---
        # This matches the broadcasting logic in your original training script.
        return tenvelope * boundary_term * psi_nn.unsqueeze(1) + self.initial_state(x)

# --- Main Test Execution ---
if __name__ == "__main__":
    # 1. Load data
    ref_data = np.load("fft_ref_snapshots.npz")
    x_ref = ref_data['x']
    psi0_ref_density = ref_data['psi0']
    psiT_ref_density = ref_data['psiT']
    print("✅ Reference data loaded.")

    # 2. Load model
    pinn_model = SchrodingerPINN().to(DEVICE)
    pinn_model.load_state_dict(torch.load("pinn1d_tdse_pulse.pt", map_location=DEVICE))
    pinn_model.eval()
    print("✅ Trained PINN model loaded.")

    # 3. Prepare input tensors
    x_tensor = torch.tensor(x_ref, dtype=torch.float32).view(-1, 1).to(DEVICE)
    t0_tensor = torch.zeros_like(x_tensor)
    tT_tensor = torch.full_like(x_tensor, T_MAX)

    # 4. Generate predictions
    print("⏳ Generating PINN predictions...")
    with torch.no_grad():
        psi0_pinn = pinn_model(x_tensor, t0_tensor)
        psiT_pinn = pinn_model(x_tensor, tT_tensor)

    # --- FIX 2: Squeeze the output to be a 1D array for plotting ---
    # The model output is [1024, 1], .squeeze() makes it [1024], which is a 1D line.
    psi0_pinn_density = (psi0_pinn.abs()**2).squeeze().cpu().numpy()
    psiT_pinn_density = (psiT_pinn.abs()**2).squeeze().cpu().numpy()

    # 5. Quantify error
    mse_error = np.mean((psiT_pinn_density - psiT_ref_density)**2)
    print(f"📊 Mean Squared Error (MSE) at T_MAX: {mse_error:.2e}")

    # 6. Plot results
    print("🎨 Generating plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(x_ref, psi0_ref_density, 'k-', label=r"Reference $|\psi(x, 0)|^2$", linewidth=2)
    ax.plot(x_ref, psi0_pinn_density, 'r--', label=r"PINN $|\psi(x, 0)|^2$", linewidth=2)
    ax.plot(x_ref, psiT_ref_density, 'b-', label=f"Reference $|\psi(x, T={T_MAX})|^2$", linewidth=2.5)
    ax.plot(x_ref, psiT_pinn_density, 'c--', label=f"PINN $|\psi(x, T={T_MAX})|^2$", linewidth=2.5)
    
    
    ax.set_title("PINN vs. Split-Step Fourier Reference", fontsize=16)
    ax.set_xlabel("Position (x)", fontsize=12)
    ax.set_ylabel("Probability Density $|\psi|^2$", fontsize=12)
    ax.legend(fontsize=11)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(bottom=-0.05)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- FIX 3: Use a more robust method for saving the figure ---
    fig.tight_layout()
    output_filename = "pinn_vs_reference_comparison.png"
    fig.savefig(output_filename, dpi=300)
    plt.close(fig) # Close the figure to free up memory
    
    print(f"✅ Plot saved successfully as '{output_filename}'")