import torch, math
import matplotlib.pyplot as plt
import numpy as np
import os

# 1)  <<< point this at your .pt file >>>
currdir = os.getcwd()
model_path = os.path.join(currdir, "pinn1d_tdse.pt")

# 2)  re-create the exact network architecture
class SchrodingerPINN(torch.nn.Module):
    def __init__(self, hidden_layers=8, hidden_units=256):
        super().__init__()
        layers, in_dim = [], 2                         # (x, t)
        for _ in range(hidden_layers):
            layers += [torch.nn.Linear(in_dim, hidden_units), torch.nn.Tanh()]
            in_dim = hidden_units
        layers.append(torch.nn.Linear(in_dim, 2))      # Re, Im
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x, t):
        y = self.net(torch.cat([x, t], dim=-1))
        return torch.complex(y[..., 0], y[..., 1])

net = SchrodingerPINN()
net.load_state_dict(torch.load(model_path, map_location="cpu"))
net.eval()
print("✅  model loaded")

# 3)  analytic ground-state for comparison  (ω = 1)
def psi_true(x, t):
    coeff = (1 / math.pi) ** 0.25
    return coeff * torch.exp(-0.5 * x**2 - 0.5j * t)

# 4)  sample two time slices and compare
x = torch.linspace(-5, 5, 400).unsqueeze(1)
for t_val in (0.0, 1.0):
    t = torch.full_like(x, t_val)
    with torch.no_grad():
        psi_pred = net(x, t)
    psi_ref  = psi_true(x.squeeze(), t_val)

    rel_err = torch.linalg.norm(psi_pred - psi_ref) / torch.linalg.norm(psi_ref)
    print(f"t = {t_val:4.1f}   relative L2 error = {rel_err:.3e}")

    plt.figure(figsize=(4.2,3))
    plt.plot(x, psi_ref.abs()**2,  label="analytic |ψ|²")
    plt.plot(x, psi_pred.abs()**2, "--", label="PINN |ψ|²")
    plt.title(f"Probability density at t={t_val}")
    plt.xlabel("x"); plt.ylabel("|ψ|²"); plt.legend(); plt.tight_layout()
    plt.show()

# 5)  quick normalization check at random t
t_rand = torch.rand(32,1) * 1.5                 # 32 random times up to t=1.5
x_rand = (torch.rand(32,1) * 10) - 5            # x ∈ [-5,5]
psi     = net(x_rand, t_rand)
norm_mc = 10 * (psi.abs()**2).mean()           # Monte-Carlo integral estimate
print(f"MC normalization ⟨|ψ|²⟩≈ {norm_mc.item():.3f}  (target 1.00)")
