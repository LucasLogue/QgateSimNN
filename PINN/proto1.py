import torch
import torch.nn as nn
from torch.autograd import grad
from torch.amp import autocast, GradScaler
from contextlib import nullcontext
from torch.optim.lr_scheduler import ExponentialLR
import time
"""
1‑D PINN for the TDSE **with fixed control pulse**
=================================================
* Harmonic potential `V(x)=½ ω² x²` plus drive term `‑E(t)x`.
* Mixed‑precision **forward** pass for speed; derivatives stay FP32 for stability.
* Works on CUDA 12.+ GPUs without `torch.compile` quirks.
* Saves to `pinn1d_tdse_pulse.pt`.
"""

# ---------------- Runtime constants ----------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on", device)
if device.type != "cuda":
    raise SystemExit("⚠️  CUDA GPU required")

amp_ctx = autocast("cuda")        # fp16 forward only
scaler  = GradScaler("cuda")

OMEGA = 1.0
X_MIN, X_MAX = -5.0, 5.0
T_MAX = 3.0

PULSE_AMP, PULSE_FREQ, PULSE_PHASE = -6.3, 1.13, 0.64
PULSE_T0,  PULSE_WIDTH             = 0.0, 1.7

N_IC, N_BC, N_COL, N_ENERGY = 1024, 1024, 2048, 1024
EPOCHS, LR        = 12_000, 2e-3
PRINT_EVERY       = 400


w_pde, w_ic, w_bc, w_norm = 10.0, 5.0, 5.0, 5.0

# ---------------- Helper functions -----------------

def harmonic_potential(x):
    return 0.5 * (OMEGA ** 2) * x ** 2

def control_pulse(t):
    env = torch.exp(-((t - PULSE_T0) / PULSE_WIDTH) ** 2)
    return PULSE_AMP * env * torch.sin(PULSE_FREQ * t + PULSE_PHASE)

def initial_state(x):
    coeff = (OMEGA / torch.pi) ** 0.25
    return coeff * torch.exp(-0.5 * OMEGA * x ** 2)

# ---------------- Network --------------------------

class AdaptiveTanh(nn.Module):
    def __init__(self, initial_a=1.0):
        super().__init__()
        # Create a learnable parameter 'a' for the slope of the activation
        self.a = nn.Parameter(torch.tensor(initial_a))


    def forward(self, x):
        # The activation is now a * tanh(x)
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
        #self.boundary_factor = 0.1
        self.timefactor = 5.0

    #INITIALSTATE!
    def initial_state(self, x):
        coeff = (OMEGA / torch.pi) ** 0.25
        real_part = coeff * torch.exp(-0.5 * OMEGA * x ** 2)
        return torch.complex(real_part, torch.zeros_like(real_part))
    #Updated forward with boundaries
    def forward(self, x, t):
        psi_nn_raw = self.net(torch.cat([x, t], dim=-1))
        psi_nn = torch.complex(psi_nn_raw[..., 0], psi_nn_raw[..., 1])

        #now using a neural network switch and aVOids vanishing gradient
        tenvelope = (1.0 - torch.exp(-self.timefactor * t))
        boundary_term = (x - X_MIN) * (X_MAX - x)
        return tenvelope * boundary_term * psi_nn.unsqueeze(1) + self.initial_state(x)

       # boundary_term = self.boundary_factor * (x - X_MIN) * (X_MAX - x)

        #return boundary_term * t * psi_nn + initial_state(x)
        # y = self.net(torch.cat([x, t], dim=-1))
        # return torch.complex(y[..., 0], y[..., 1])

# ---------------- Loss helpers ---------------------
def hamiltonian(net, x, t):
    #compute Hpsi for network and coordinates
    x.requires_grad_(True)
    psi = net(x, t)
    psi_x = grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
    psi_xx = grad(psi_x, x, grad_outputs=torch.ones_like(psi_x), create_graph=True)[0]

    V = harmonic_potential(x)
    E_t = control_pulse(t)
    Hpsi = -0.5 * psi_xx + (V - E_t * x) * psi
    return Hpsi, psi

def residual(net, x, t):
    x.requires_grad_(True); t.requires_grad_(True)
    psi = net(x, t)
    psi_grads = grad(psi, (x, t), grad_outputs=torch.ones_like(psi), create_graph=True)
    psi_x, psi_t = psi_grads[0], psi_grads[1]
    psi_xx = grad(psi_x, x, grad_outputs=torch.ones_like(psi_x), create_graph=True)[0]
    # psi_t  = grad(psi, t, torch.ones_like(psi), create_graph=True)[0]
    # psi_x  = grad(psi, x, torch.ones_like(psi), create_graph=True)[0]
    # psi_xx = grad(psi_x, x, torch.ones_like(psi_x), create_graph=True)[0]

    V   = harmonic_potential(x.squeeze(-1))
    E_t = control_pulse(t.squeeze(-1))
    Hpsi  = -0.5 * psi_xx + (V - E_t * x.squeeze(-1)) * psi
    r   = 1j*psi_t.squeeze(-1) - Hpsi
    return (r.real**2 + r.imag**2).mean()

def norm_loss(psi):
    return ((psi.abs()**2).mean() * (X_MAX - X_MIN) - 1).abs()


def energy_conservation_loss(net, n):
    x = torch.empty(n, 1, device=device).uniform_(X_MIN, X_MAX)
    t1 = torch.empty(n, 1, device=device).uniform_(0.0, T_MAX)
    t2 = torch.empty(n, 1, device=device).uniform_(0.0, T_MAX)

    # Calculate Hamiltonian and wavefunction at both times
    Hpsi1, psi1 = hamiltonian(net, x, t1)
    Hpsi2, psi2 = hamiltonian(net, x, t2)

    # Estimate energy expectation value via Monte Carlo integration
    # <E> = ∫ psi* Hpsi dx
    energy1 = (torch.conj(psi1) * Hpsi1).mean() * (X_MAX - X_MIN)
    energy2 = (torch.conj(psi2) * Hpsi2).mean() * (X_MAX - X_MIN)

    # The loss is the squared difference between the two energies
    return (energy1.real - energy2.real)**2 + (energy1.imag - energy2.imag)**2
# ---------------- Samplers -------------------------

#ADDING IMPORTANCE SAMPLING
def collocation(n):
    #x = torch.empty(n,1, device=device).uniform_(X_MIN, X_MAX)
    t = torch.empty(n,1, device=device).uniform_(0.0, T_MAX)

    x = torch.randn(n, 1, device=device) * 2.0 #std = 2.0
    x = torch.clamp(x, X_MIN, X_MAX)
    return x, t

def ic_batch(n):
    x = torch.empty(n,1, device=device).uniform_(X_MIN, X_MAX)
    t = torch.zeros_like(x)
    return x, t, initial_state(x.squeeze(-1))

def bc_batch(n):
    x = torch.full((n,1), X_MIN, device=device); x[n//2:] = X_MAX
    t = torch.empty_like(x).uniform_(0.0, T_MAX)
    return x, t

# ---------------- Training -------------------------

if __name__ == "__main__":
    net = SchrodingerPINN().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    scheduler = ExponentialLR(opt, gamma=0.99)
    
    start = time.time()
    for ep in range(1, EPOCHS+1):
        opt.zero_grad(set_to_none=True)

        # >>>>>>>>>>>>>>>>>>>>>>>>>> HIGHLIGHT START: DYNAMIC LOSS WEIGHTS <<<<<<<<<<<<<<<<<<<<<<<<
        # Define the annealing schedule.
        # 'progress' goes from 0.0 at the start to 1.0 at the end.
        progress = ep / EPOCHS

        # w_pde ramps UP from 0.1 to 10.0
        w_pde = 0.1 + (10.0 - 0.1) * progress
        
        # Constraint weights ramp DOWN from 20.0 to 5.0
        w_constraint = 20.0 + (5.0 - 20.0) * progress
        w_ic = w_constraint
        w_bc = w_constraint
        w_norm = w_constraint
        w_energy = 1.0 + (5.0- 1.0)*progress
        # >>>>>>>>>>>>>>>>>>>>>>>>>> HIGHLIGHT END <<<<<<<<<<<<<<<<<<<<<<<<

        xc, tc = collocation(N_COL)
        L_pde = residual(net, xc, tc)

        # xi, ti, psi0 = ic_batch(N_IC)
        # psi_ic = net(xi, ti)
        # L_ic = ((psi_ic - psi0).abs()**2).mean()

        # xb, tb = bc_batch(N_BC)
        # psi_bc = net(xb, tb)
        # L_bc = (psi_bc.abs()**2).mean()

        xn, tn = collocation(N_IC)
        L_norm = norm_loss(net(xn, tn))

        #L_energy = energy_conservation_loss(net, N_ENERGY)

        total = w_pde*L_pde + w_norm*L_norm #+ w_energy*L_energy

        total.backward()
        opt.step()
        
        if ep % 100 == 0:
            scheduler.step()

        if ep % PRINT_EVERY == 0:
            print(f"Ep {ep:>5} | L {total.item():.2e} | "
                #   f"PDE {L_pde.item():.1e} IC {L_ic.item():.1e} "
                #   f"BC {L_bc.item():.1e} N {L_norm.item():.1e} | "
                  f"LR {scheduler.get_last_lr()[0]:.1e}")
    end = time.time()
    dur = end - start
    print(f"Duration: {dur:.2f} s")
    torch.save(net.state_dict(), "pinn1d_tdse_pulse.pt")
    print("✅  Training finished – pinn1d_tdse_pulse.pt saved")

