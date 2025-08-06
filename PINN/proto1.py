import torch
import torch.nn as nn
from torch.autograd import grad
#1-D PINN test

# HYPER PARAMS!
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32 #real dtype


OMEGA        = 1.0                      # harmonic ω
X_MIN, X_MAX = -5.0, 5.0               # spatial box
T_MAX        = 3.0                     # total evolution time

N_IC  = 1024                            # points for initial condition loss
N_BC  = 1024                            # points for boundary loss (x = ±L)
N_COL = 8192                            # collocation points per batch

EPOCHS       = 20000
LR           = 1e-3
PRINT_EVERY  = 500

#loss weights, will need to change a lot
w_pde, w_ic, w_bc, w_norm = 1.0, 10.0, 10.0, 1.0


#Helpers
def harmonic_potential(x: torch.Tensor) -> torch.Tensor:
    """½ ω² x²"""
    return 0.5 * (OMEGA ** 2) * x ** 2


def initial_state(x: torch.Tensor) -> torch.Tensor:
    """Ground state of harmonic oscillator (analytic)"""
    # ψ₀(x) = (ω/π)^{1/4} exp(-½ ω x²)
    coeff = (OMEGA / torch.pi) ** 0.25
    psi0 = coeff * torch.exp(-0.5 * OMEGA * x ** 2)
    return psi0


def complex_split_to_tensor(re: torch.Tensor, im: torch.Tensor) -> torch.Tensor:
    """Combine real/imag parts (…,) → complex tensor."""
    return torch.complex(re, im)


class SchrodingerPINN(nn.module):
    def __init__(self, hlayers: int = 8, hunits: int = 256):
        super().__init__()
        layers=[]
        in_dim = 2 #x, t

        #NN layering
        for i in range(hlayers):
            out_dim = hunits
            layers.append(nn.Linear(in_dim), out_dim)
            layers.append(nn.Tanh())
            in_dim = out_dim #set dimensions for next loop
        layers.append(nn.Linear(in_dim,2)) #set final output to 2, RE, IM
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        inp = torch.cat([x, t], dim=-1)
        out = self.net(inp)
        psi_re, psi_im = out[..., 0], out[..., 1]
        return psi_re, psi_im
    

#LOSS COMPONENETS!!!!!!!!!!!!!!!!!!!!!!!
def pde_res(net: SchrodingerPINN, x, t):
    x.requires_grad_(True)
    t.requires_grad_(True)

    psi_re, psi_im = net(x, t)
    psi = complex_split_to_tensor(psi_re, psi_im)

    # First derivatives
    psi_t = grad(psi, t, torch.ones_like(psi), create_graph=True)[0]
    psi_x = grad(psi, x, torch.ones_like(psi), create_graph=True)[0]
    # Second spatial derivative
    psi_xx = grad(psi_x, x, torch.ones_like(psi_x), create_graph=True)[0]

    V = harmonic_potential(x.squeeze(-1))

    lhs = 1j * psi_t.squeeze(-1)
    rhs = (-0.5 * psi_xx.squeeze(-1) + V * psi)

    resid = lhs - rhs
    return resid.real ** 2 + resid.imag ** 2  # |residual|^2


def normalization_loss(psi_re, psi_im, x):
    """Monte Carlo estimate of |ψ|^2 integral – 1."""
    prob = psi_re ** 2 + psi_im ** 2
    vol  = (X_MAX - X_MIN)
    integral = vol * prob.mean()
    return (integral - 1.0).abs()


#TRAINING UTILITY
###############################################################################
# 4.  Training utilities
###############################################################################

def sample_collocation(n):
    x = torch.empty(n, 1, device=device).uniform_(X_MIN, X_MAX)
    t = torch.empty(n, 1, device=device).uniform_(0.0, T_MAX)
    return x.requires_grad_(True), t.requires_grad_(True)


def sample_initial(n):
    x = torch.empty(n, 1, device=device).uniform_(X_MIN, X_MAX)
    t0 = torch.zeros_like(x)
    psi0 = initial_state(x.squeeze(-1))
    return x, t0, psi0


def sample_boundary(n):
    # x = ±L, t ∈ [0,T]
    xb = torch.empty(n, 1, device=device).uniform_(0.0, 1.0)
    xb = torch.where(xb < 0.5, torch.full_like(xb, X_MIN), torch.full_like(xb, X_MAX))
    tb = torch.empty_like(xb).uniform_(0.0, T_MAX)
    return xb, tb


if __name__ == "__main__":
    net = SchrodingerPINN().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()

        # Collocation residual
        xc, tc = sample_collocation(N_COL)
        loss_pde = pde_res(net, xc, tc).mean()

        # Initial condition loss
        xi, ti, psi0 = sample_initial(N_IC)
        psi_re_i, psi_im_i = net(xi, ti)
        loss_ic = ((psi_re_i - psi0).pow(2) + psi_im_i.pow(2)).mean()

        # Boundary condition loss (Dirichlet ~0)
        xb, tb = sample_boundary(N_BC)
        psi_re_b, psi_im_b = net(xb, tb)
        loss_bc = (psi_re_b.pow(2) + psi_im_b.pow(2)).mean()

        # Normalization loss (estimate at random slice t=uniform)
        xn = torch.empty(N_IC, 1, device=device).uniform_(X_MIN, X_MAX)
        tn = torch.empty_like(xn).uniform_(0.0, T_MAX)
        psi_re_n, psi_im_n = net(xn, tn)
        loss_norm = normalization_loss(psi_re_n, psi_im_n, xn)

        total_loss = (
            w_pde * loss_pde + w_ic * loss_ic + w_bc * loss_bc + w_norm * loss_norm
        )

        total_loss.backward()
        optimizer.step()

        if epoch % PRINT_EVERY == 0:
            print(f"Epoch {epoch:>6} | L_tot {total_loss.item():.4e} | "
                  f"L_pde {loss_pde.item():.3e} | L_ic {loss_ic.item():.3e} | "
                  f"L_bc {loss_bc.item():.3e} | L_norm {loss_norm.item():.3e}")

    # Save trained model
    torch.save(net.state_dict(), "pinn1d_tdse.pt")
    print("Training complete – model saved to pinn1d_tdse.pt")