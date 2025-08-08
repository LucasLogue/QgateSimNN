import torch, math
import torch.nn as nn  # Use the standard alias
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Point this at your saved model and reference data ---
currdir = os.getcwd()
model_path = os.path.join(currdir, "pinn1d_tdse_pulse.pt")
fft_ref_path = os.path.join(currdir, "fft_ref_snapshots.npz")

# --- Define simulation parameters to ensure consistency ---
DT = 0.005  # Time step
NT = 600    # Number of time steps


# >>>>>>>>>>>>>>>>>>>>>>>>>> HIGHLIGHT START: RE-CREATE THE EXACT NETWORK ARCHITECTURE <<<<<<<<<<<<<<<<<<<<<<<<
# This section must be an exact copy of the network from the training script.

# 1) Re-create the Adaptive Tanh activation function
class AdaptiveTanh(nn.Module):
    def __init__(self, initial_a=1.0):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(initial_a))

    def forward(self, x):
        return self.a * torch.tanh(x)

# 2) Re-create the main PINN class with the hard-constraint forward pass
class SchrodingerPINN(nn.Module):
    def __init__(self, layers=8, units=256): # Must match training parameters
        super().__init__()
        net, in_dim = [], 2
        for _ in range(layers):
            net += [nn.Linear(in_dim, units), AdaptiveTanh()] # Use the adaptive activation
            in_dim = units
        net.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*net)
        
        # Define constants needed for the forward pass
        self.time_factor = 5.0
        self.X_MIN, self.X_MAX = -5.0, 5.0
        self.OMEGA = 1.0

    def initial_state(self, x):
        # This function takes x with shape (N, 1) and must return a tensor of shape (N, 1)
        coeff = (self.OMEGA / torch.pi) ** 0.25
        real_part = coeff * torch.exp(-0.5 * self.OMEGA * x ** 2)
        return torch.complex(real_part, torch.zeros_like(real_part))

    def forward(self, x, t):
        # x and t have shape (N, 1)
        y = self.net(torch.cat([x, t], dim=-1)) # y has shape (N, 2)
        
        # --- CORRECTED SHAPE ---
        # Ensure psi_nn has shape (N, 1) to match the other terms for element-wise multiplication
        ψ_nn = torch.complex(y[..., 0], y[..., 1]).unsqueeze(1)

        time_envelope = (1.0 - torch.exp(-self.time_factor * t))
        boundary_term = (x - self.X_MIN) * (self.X_MAX - x)
        
        # All terms now have shape (N, 1), so multiplication is element-wise
        return time_envelope * boundary_term * ψ_nn + self.initial_state(x)

# >>>>>>>>>>>>>>>>>>>>>>>>>> HIGHLIGHT END <<<<<<<<<<<<<<<<<<<<<<<<


# --- Load the trained model ---
net = SchrodingerPINN()
net.load_state_dict(torch.load(model_path, map_location="cpu"))
net.eval()
print("✅  Model loaded successfully")


# --- Load the FFT reference data ---
ref = np.load(fft_ref_path)
x_ref = torch.tensor(ref["x"], dtype=torch.float32).unsqueeze(1)
# Ensure reference wavefunctions are loaded as complex numbers
psi_ref0 = torch.tensor(ref["psi0"], dtype=torch.complex64)
psi_refT = torch.tensor(ref["psiT"], dtype=torch.complex64)
t_final = DT * NT


# --- Compare PINN vs FFT at t=0 and t=T_MAX ---
for t_val, psi_ref_complex in [(0.0, psi_ref0), (t_final, psi_refT)]:
    t = torch.full_like(x_ref, t_val)
    with torch.no_grad():
        # The network output will have shape (N, 1), so we squeeze it to (N,) for plotting
        psi_pred_complex = net(x_ref, t).squeeze()
    
    # Calculate probabilities for plotting
    psi_pred_prob = psi_pred_complex.abs()**2
    psi_ref_prob = psi_ref_complex.abs()**2

    plt.figure(figsize=(6,4))
    # --- CORRECTED PLOTTING ---
    # The .flatten() ensures we are plotting 1D arrays, preventing plotting errors.
    plt.plot(x_ref.numpy().flatten(), psi_ref_prob.numpy().flatten(), label="FFT ref", linewidth=2)
    plt.plot(x_ref.numpy().flatten(), psi_pred_prob.numpy().flatten(), "--", label="PINN", linewidth=2)
    plt.xlabel("x"); plt.ylabel("|ψ|²"); plt.legend(); plt.tight_layout()
    
    output_filename = f"pinn_vs_fft_t_{t_val:.1f}.png"
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"✅  Plot saved to {output_filename}")

    # Calculate relative error on the complex wavefunctions
    rel_err = torch.linalg.norm(psi_pred_complex - psi_ref_complex) / torch.linalg.norm(psi_ref_complex)
    print(f"  > Relative L2 error = {rel_err.item():.3e}")
