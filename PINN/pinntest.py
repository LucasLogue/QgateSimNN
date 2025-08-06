import torch, math
import matplotlib.pyplot as plt
import numpy as np
import os

# 1)  <<< point this at your .pt file >>>
currdir = os.getcwd()
model_path = os.path.join(currdir, "pinn1d_tdse_pulse.pt")
output = os.path.join("PINN", "balls.png")

DT         = 0.005          # time step
NT         = 600 
#include the pinntest script to add pulse to the prediction
from proto1 import control_pulse as ctrl_orig
def control_pulse(t):
    return 0.0 * t
# 2)  re-create the exact network architecture
class AdaptiveTanh(torch.nn.Module):
    def __init__(self, initial_a=1.0):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(initial_a))

    def forward(self, x):
        return self.a * torch.tanh(x)

# 2) Re-create the main PINN class with the hard-constraint forward pass
class SchrodingerPINN(torch.nn.Module):
    def __init__(self, layers=8, units=384): # Must match training parameters
        super().__init__()
        net, in_dim = [], 2
        for _ in range(layers):
            net += [torch.nn.Linear(in_dim, units), AdaptiveTanh()] # Use the adaptive activation
            in_dim = units
        net.append(torch.nn.Linear(in_dim, 2))
        self.net = torch.nn.Sequential(*net)
        
        # Define constants needed for the forward pass
        self.time_factor = 5.0
        self.X_MIN, self.X_MAX = -5.0, 5.0
        self.OMEGA = 1.0

    def initial_state(self, x):
        coeff = (self.OMEGA / torch.pi) ** 0.25
        real_part = coeff * torch.exp(-0.5 * self.OMEGA * x ** 2)
        return torch.complex(real_part, torch.zeros_like(real_part))

    def forward(self, x, t):
        ψ_nn = self.net(torch.cat([x, t], dim=-1))
        ψ_nn = torch.complex(ψ_nn[..., 0], ψ_nn[..., 1])

        time_envelope = (1.0 - torch.exp(-self.time_factor * t))
        boundary_term = (x - self.X_MIN) * (self.X_MAX - x)
        
        return time_envelope * boundary_term * ψ_nn + self.initial_state(x)
net = SchrodingerPINN()
net.load_state_dict(torch.load(model_path, map_location="cpu"))
net.eval()
print("✅  model loaded")

# 3)  analytic ground-state for comparison  (ω = 1)
def psi_true(x, t):
    coeff = (1 / math.pi) ** 0.25
    return coeff * torch.exp(-0.5 * x**2 - 0.5j * t)

# 4)  sample two time slices and compare
#x = torch.linspace(-5, 5, 400).unsqueeze(1)
ref = np.load("fft_ref_snapshots.npz")
x_ref = torch.tensor(ref["x"]).unsqueeze(1)
psi_ref0 = ref["psi0"]
psi_refT = ref["psiT"]

# load trained PINN
net = SchrodingerPINN(); net.load_state_dict(torch.load("pinn1d_tdse_pulse.pt", map_location="cpu")); net.eval()

for t_val, psi_ref in [(0.0, psi_ref0), (DT*NT, psi_refT)]:
    t = torch.full_like(x_ref, t_val)
    with torch.no_grad():
        psi_pred = net(x_ref, t).abs()**2

    plt.figure(figsize=(4.2,3))
    plt.plot(x_ref, psi_ref, label="FFT ref")
    plt.plot(x_ref, psi_pred, "--", label="PINN")
    plt.xlabel("x"); plt.ylabel("|ψ|²"); plt.legend(); plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()

    rel_err = torch.linalg.norm(torch.tensor(psi_pred) - torch.tensor(psi_ref)) / torch.linalg.norm(torch.tensor(psi_ref))
    print(f"t = {t_val:.2f}  relative L2 error = {rel_err.item():.2e}")

# 5)  quick normalization check at random t
t_rand = torch.rand(32,1) * 1.5                 # 32 random times up to t=1.5
x_rand = (torch.rand(32,1) * 10) - 5            # x ∈ [-5,5]
psi     = net(x_rand, t_rand)
norm_mc = 10 * (psi.abs()**2).mean()           # Monte-Carlo integral estimate
print(f"MC normalization ⟨|ψ|²⟩≈ {norm_mc.item():.3f}  (target 1.00)")
