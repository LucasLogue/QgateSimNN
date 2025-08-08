import torch
import torch.nn as nn
from torch.autograd import grad
from torch.optim.lr_scheduler import ExponentialLR
import time
import numpy as np
import matplotlib.pyplot as plt
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on", device)

OMEGA = 1.0
X_MIN, X_MAX = -5.0, 5.0
T_MIN, T_MAX = 0.0, 3.0

PULSE_AMP, PULSE_FREQ, PULSE_PHASE = -6.3, 1.13, 0.64
PULSE_T0,  PULSE_WIDTH             = 0.0, 1.7

N_IC, N_BC, N_COL = 1024, 1024, 2048 # Increased collocation points for stability
EPOCHS, LR        = 12000, 2e-4 # Adjusted epochs and LR
PRINT_EVERY       = 500

def harmonic_potential(x):
    return 0.5 * (OMEGA ** 2) * x ** 2

def control_pulse(t):
    env = torch.exp(-((t - PULSE_T0) / PULSE_WIDTH) ** 2)
    return PULSE_AMP * env * torch.sin(PULSE_FREQ * t + PULSE_PHASE)

def initial_state(x):
    coeff = (OMEGA / torch.pi) ** 0.25
    real_part = coeff * torch.exp(-0.5 * OMEGA * x ** 2)
    return torch.complex(real_part, torch.zeros_like(real_part))

class FourierEncode(nn.Module):
    def __init__(self, in_dim=2, M=128, scale=10.0):
        super().__init__()
        # fixed random projection, shape=(in_dim, M)
        B = torch.randn(in_dim, M) * scale
        self.register_buffer('B', B)

    def forward(self, x):
        # x: (..., in_dim)
        x_proj = (2*torch.pi * x) @ self.B    # (..., M)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)  # (..., 2M)

#Swapped for fourier
# class AdaptiveTanh(nn.Module):
#     def __init__(self, initial_a=1.0):
#         super().__init__()
#         # Create a learnable parameter 'a' for the slope of the activation
#         self.a = nn.Parameter(torch.tensor(initial_a))

#     def forward(self, x):
#         # The activation is now a * tanh(x)
#         return self.a * torch.tanh(x)


class SchrodingerPINN(nn.Module):
    def __init__(self, layers=8, units=256):
        super().__init__()
        
        #bounds for normalization so it actually works
        # self.lower_bound = torch.tensor([X_MIN, T_MIN], device=device)
        # self.upper_bound = torch.tensor([X_MAX, T_MAX], device=device)
        
        self.encoder = FourierEncode(in_dim=2, M=128, scale=10.0)
        in_dim = 128*2
        net_layers= []
        #for old super simple version
        # net_layers = []
        # in_dim = 2

        for _ in range(layers):
            linear_layer = nn.Linear(in_dim, units)
            # ADAPTED: Apply Xavier Normal Initialization like the Wave PINN
            nn.init.xavier_normal_(linear_layer.weight)
            net_layers.append(linear_layer)

            #Currently unused as adding fourier
            #net_layers.append(AdaptiveTanh()) #custom adaptive tanh

            net_layers.append(nn.Tanh())
            in_dim = units

        final_layer = nn.Linear(in_dim, 2)
        nn.init.xavier_normal_(final_layer.weight)
        net_layers.append(final_layer)

        self.net = nn.Sequential(*net_layers)
        self.E0 = 0.5

    def forward(self, x, t):
        # --- 1) compute classical shift x_cl(t) for each sample ---
        # classical_shift returns a Python float, so we map it over the batch
        t_flat = t.detach().cpu().squeeze(-1).tolist()      # list of times
        x_cl_list = [classical_shift(ti) for ti in t_flat]  # one float per sample
        x_cl = torch.tensor(x_cl_list, device=x.device).unsqueeze(-1)  # (batch,1)

        # shift coordinates
        x_shifted = x - x_cl

        # --- 2) build shifted‐ground‐state + network residual ---
        psi0_s = initial_state(x_shifted)  # exact ground‐state at the shifted center
        alpha  = 1.0 - t / T_MAX
        beta   =       t / T_MAX

        # encode on the shifted coordinates
        inp    = torch.cat([x_shifted, t], dim=-1)
        feats  = self.encoder(inp)
        raw    = self.net(feats)
        phi    = torch.complex(raw[...,0], raw[...,1])

        # interpolate: at t=0 → psi0_s, at t=T_MAX → network correction
        psi_unphased = alpha * psi0_s + beta * phi

        # --- 3) factor out the known oscillatory phase ---
        real_phase = torch.cos(self.E0 * t)
        imag_phase = -torch.sin(self.E0 * t)
        phase      = torch.complex(real_phase, imag_phase)

        return phase * psi_unphased

    # def forward(self, x, t):
    #     input_tensor = torch.cat([x, t], dim=-1)
    #     feats = self.encoder(input_tensor)
    #     #norm = 2*(feats - feats.min()) / (feats.max - feats.min()) -1 hollup I already normalzie them with FFT
    #     raw = self.net(feats)
    #     bigO = torch.complex(raw[..., 0], raw[..., 1])
    #     #AdaptiveTanh() version
    #     # input_tensor = torch.cat([x, t], dim=-1)
    #     # normalized = 2 * (input_tensor - self.lower_bound) / (self.upper_bound - self.lower_bound) - 1
    #     # raw = self.net(normalized)
    #     # bigO = torch.complex(raw[..., 0], raw[..., 1])

    #     #ansat peices
    #     alpha = 1.0 - t/T_MAX
    #     beta = t/T_MAX
    #     env = (x - X_MIN) * (X_MAX - x)
    #     psi0 = initial_state(x)

    #     env_bc = 1 - (x / X_MAX)**2 #MURDER BOUNDARIES ONLY
    #     #unphase dat boi
    #     psi_unphase = alpha * psi0 + beta*env_bc*bigO

    #     #factor out the oscilatory boy
    #     real_phase = torch.cos(self.E0 * t)
    #     imag_phase = -torch.sin(self.E0 * t)
    #     phase = torch.complex(real_phase, imag_phase)

    #     psi_raw = phase * psi_unphase
    #     # p = torch.abs(psi_raw)**2
    #     # norm_factor = torch.sqrt(torch.mean(p)*(X_MAX-X_MIN))
    #     # psi = psi_raw / norm_factor
    #     return psi_raw #no normalization because we'll do that for the weights
        #broken but produces something
        # normalized_input = 2.0 * (input_tensor - self.lower_bound) / (self.upper_bound - self.lower_bound) - 1.0
        # raw_output = self.net(normalized_input)
        # betterout = torch.complex(raw_output[..., 0], raw_output[..., 1])
        # time_factor = (1.0 - torch.exp(-0.1*t))
        # psi = initial_state(x) + time_factor * betterout
        # return psi #added ANSATZ

def residual(net, x, t):
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)
    psi = net(x, t)      # <-- includes envelope + phase + network
    u, v = psi.real, psi.imag

    # u_t  = torch.autograd.grad(u, t,   grad_outputs=torch.ones_like(u),
    #                         create_graph=True)[0]
    # v_t  = torch.autograd.grad(v, t,   grad_outputs=torch.ones_like(v),
    #                         create_graph=True)[0]
    # u_x  = torch.autograd.grad(u, x,   grad_outputs=torch.ones_like(u),
    #                         create_graph=True)[0]
    # v_x  = torch.autograd.grad(v, x,   grad_outputs=torch.ones_like(v),
    #                         create_graph=True)[0]
    # u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
    #                         create_graph=True)[0]
    # v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x),
    #                         create_graph=True)[0]

    u_t = grad(u.sum(),   t,   create_graph=True)[0]
    v_t = grad(v.sum(),   t,   create_graph=True)[0]
    u_x = grad(u.sum(),   x,   create_graph=True)[0]
    v_x = grad(v.sum(),   x,   create_graph=True)[0]
    u_xx= grad(u_x.sum(), x,   create_graph=True)[0]
    v_xx= grad(v_x.sum(), x,   create_graph=True)[0]

    psi_t  = torch.complex(u_t,  v_t)
    psi_xx = torch.complex(u_xx, v_xx)

    # now build the true Schrödinger residual
    i    = torch.complex(torch.tensor(0., device=x.device),
                        torch.tensor(1., device=x.device))
    V = harmonic_potential(x)
    E_t = control_pulse(t)
    Hpsi = -0.5*psi_xx + (V - control_pulse(t)*x)*psi
    res  = i * psi_t - Hpsi

    return torch.mean(torch.abs(res)**2)


    #Old logic for AdaptiveTanh()
    # x.requires_grad_(True)
    # t.requires_grad_(True)
    # psi = net(x, t)

    # # --- FIX: Calculate gradients for real and imaginary parts separately ---
    # u = psi.real
    # v = psi.imag

    # # First derivatives of the real part
    # u_grads = grad(u.sum(), (x, t), create_graph=True)
    # u_x = u_grads[0]
    # u_t = u_grads[1]

    # # First derivatives of the imaginary part
    # v_grads = grad(v.sum(), (x, t), create_graph=True)
    # v_x = v_grads[0]
    # v_t = v_grads[1]

    # # Second derivative of the real part
    # u_xx = grad(u_x.sum(), x, create_graph=True)[0]

    # # Second derivative of the imaginary part
    # v_xx = grad(v_x.sum(), x, create_graph=True)[0]

    # # Reconstruct the complex derivatives
    # psi_t = torch.complex(u_t,  v_t)
    # psi_xx = torch.complex(u_xx, v_xx)

    # # --- END FIX ---

    # # Now, calculate the PDE residual as before
    # i = torch.complex(torch.tensor(0., device=x.device),
    #               torch.tensor(1., device=x.device))
    # V = harmonic_potential(x)
    # E_t = control_pulse(t)
    # H_psi = -0.5 * psi_xx + (V - E_t * x) * psi
    
    # pde_residual = i * psi_t - H_psi
    # return torch.mean(pde_residual.real**2 + pde_residual.imag**2)

# ---------------- Samplers -------------------------
def collocation(n):
    x = torch.empty(n, 1, device=device).uniform_(X_MIN, X_MAX)
    t = torch.empty(n, 1, device=device).uniform_(T_MIN, T_MAX)
    return x, t

def ic_batch(n):
    x = torch.empty(n, 1, device=device).uniform_(X_MIN, X_MAX)
    t = torch.full_like(x, T_MIN)
    psi0 = initial_state(x)
    return x, t, psi0

def bc_batch(n):
    t = torch.empty(n, 1, device=device).uniform_(T_MIN, T_MAX)
    # Points on the left and right spatial boundaries
    x_left = torch.full_like(t, X_MIN)
    x_right = torch.full_like(t, X_MAX)
    return x_left, x_right, t


def classical_shift(t_final, n_steps=1000):
    """
    Compute x_cl(t_final) = (1/Ω) ∫₀ᵗ sin[Ω (t_final - τ)] E(τ) dτ
    using a simple trapezoidal rule.
    """
    # sample τ between 0 and t_final
    tau = torch.linspace(0.0, t_final, n_steps, device=device)
    E_tau = control_pulse(tau)                    # shape (n_steps,)
    kernel = torch.sin(OMEGA * (t_final - tau)) / OMEGA
    # trapz over τ
    xcl = torch.trapz(E_tau * kernel, tau)
    return xcl.item()  # return as Python float

def save_comparison_plot(net, epoch_number):
    """Generates and saves a plot comparing the PINN to the analytical solution."""
    
    # Set the network to evaluation mode
    net.eval()

    def psi_true(x, t):
        coeff = (1 / math.pi)**0.25
        return coeff * torch.exp(-0.5 * x**2 - 0.5j * t)

    with torch.no_grad():
        x_plot = torch.linspace(X_MIN, X_MAX, 500).view(-1, 1).to(device)
        t_plot = torch.full_like(x_plot, T_MAX) # Always plot the state at the final time T_MAX

        psi_pinn = net(x_plot, t_plot)
        psi_analytical = psi_true(x_plot, t_plot)
        shift_val = classical_shift(T_MAX)
        psi_analytical = psi_true(x_plot - shift_val, t_plot)
        pinn_density = (psi_pinn.abs()**2).cpu().numpy()
        analytical_density = (psi_analytical.abs()**2).cpu().numpy()
        x_coords = x_plot.cpu().numpy()

    plt.figure(figsize=(10, 6))
    #plt.plot(x_coords, analytical_density, 'k-', label=f'Analytical Ground State', linewidth=2)
    plt.plot(x_coords, analytical_density, 'k-', label=f'Analytic (shifted by {shift_val:.2f})', linewidth=2)
    plt.plot(x_coords, pinn_density, 'c--', label=f'PINN Final State (t={T_MAX:.1f})', linewidth=2)

    plt.title(f"Comparison at Epoch {epoch_number}")
    plt.xlabel("Position (x)")
    plt.ylabel("Probability Density |ψ|²")
    #plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(X_MIN, X_MAX)
    plt.ylim(bottom=-0.05, top=max(0.6, np.max(pinn_density)*1.1)) # Adjust ylim dynamically
    
    plot_filename = f"comparison_epoch_{epoch_number}.png"
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print(f"✅ Diagnostic plot saved as '{plot_filename}'")
    
    # Set the network back to training mode
    net.train()

# ---------------- Training -------------------------
if __name__ == "__main__":
    # Use a modest learning rate and more epochs to allow for stable convergence.

    net = SchrodingerPINN().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min',
        factor=0.5,    # cut LR in half
        patience=500,
        verbose=True
    )

    start = time.time()
    for ep in range(1, EPOCHS+1):
        opt.zero_grad()

        #static weights for hard constraints
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!DISABLED BECAUSE DONE IN ansatz
        w_bc= 0
        w_norm = 100.0

        #linearly ramp PDE weights from 0 to 1 over first quarter
        # Old ramp
        # ramp_epochs = EPOCHS // 4
        # w_pde = min(ep / ramp_epochs, 1.0)

        # new ramp: full weight by half the run, floor at 0.1
        ramp_epochs = EPOCHS // 2
        w_pde = 0.1 + 0.9 * min(ep / ramp_epochs, 1.0)
        
        # --- Loss calculations are the same ---
        xc, tc = collocation(N_COL)
        L_pde = residual(net, xc, tc)

        xb_left, xb_right, tb = bc_batch(N_BC)
        psi_bc_left = net(xb_left, tb)
        psi_bc_right = net(xb_right, tb)
        L_bc = torch.mean(psi_bc_left.abs()**2) + torch.mean(psi_bc_right.abs()**2)

        #================================fix the norm
        xn = torch.linspace(X_MIN, X_MAX, N_IC, device=device).unsqueeze(-1)
        tn = torch.full_like(xn, T_MIN)   # or whatever times you use for normalization check
        psi_norm = net(xn, tn) 

        prob2    = psi_norm.abs()**2         # shape (N_IC, 1)
        prob2    = prob2.squeeze(-1)         # shape (N_IC,)
        integral = torch.trapz(prob2, xn.squeeze(-1))
        L_norm_vec = (integral - 1.0)**2
        L_norm     = L_norm_vec.mean()
        # xn, tn = collocation(N_IC)
        # psi_norm = net(xn, tn)
        # prob_density_integral = torch.mean(psi_norm.abs()**2) * (X_MAX - X_MIN)
        # L_norm = (prob_density_integral - 1.0)**2
        #================================fix the norm

        #================================manual setting of loss to 0 to prevent double dip
        #L_norm = 0.0
        L_bc = 0.0
        total_loss = w_pde * L_pde + w_norm * L_norm #+ w_bc * L_bc + w_norm * L_norm
        total_loss = (w_pde * L_pde + w_norm * L_norm).sum() #failsafe
        #print("total_loss shape:", total_loss.shape)
        total_loss.backward()
        opt.step()
        scheduler.step(L_pde)
        
        # --- Simplified Diagnostics ---
        if ep % PRINT_EVERY == 0 or ep == 1:
            print("-" * 60)
            print(f"Epoch {ep:>5} | Total Loss: {total_loss.item():.2e} | LR: {opt.param_groups[0]['lr']:.1e}")
            print(f"  L_pde: {L_pde.item():.2e} | L_norm: {L_norm.item():.2e}")# | L_bc: {L_bc.item():.2e} | L_norm: {L_norm.item():.2e}")

        # Save a plot midway to check progress
        if ep == EPOCHS // 2:
            save_comparison_plot(net, ep)

    end = time.time()
    dur = end - start
    print("-" * 60)
    print(f"\nDuration: {dur:.2f} s")
    torch.save(net.state_dict(), "pinn1d_tdse_pulse.pt")
    print("✅  Training finished – pinn1d_tdse_pulse.pt saved")

    # Save the final plot
    save_comparison_plot(net, EPOCHS)

    # #=====================GRAPH COMPARISON
    # def psi_true(x, t):
    #     # This is the ground state solution for the simple harmonic oscillator (E=0.5)
    #     # It's a good reference but won't match perfectly due to the control pulse.
    #     coeff = (1 / math.pi)**0.25
    #     return coeff * torch.exp(-0.5 * x**2 - 0.5j * t)

    # # Prepare tensors for plotting
    # with torch.no_grad(): # Disable gradient calculations for inference
    #     # Create a high-resolution grid of x-points for a smooth plot
    #     x_plot = torch.linspace(X_MIN, X_MAX, 500).view(-1, 1).to(device)
    #     # We will compare the state at the final time, T_MAX
    #     t_plot = torch.full_like(x_plot, T_MAX)

    #     # Get the PINN's prediction at T_MAX
    #     psi_pinn = net(x_plot, t_plot)
    #     # Calculate the analytical solution at T_MAX
    #     psi_analytical = psi_true(x_plot, t_plot)

    #     # Calculate probability densities (|ψ|²) for plotting
    #     pinn_density = (psi_pinn.abs()**2).cpu().numpy()
    #     analytical_density = (psi_analytical.abs()**2).cpu().numpy()
    #     x_coords = x_plot.cpu().numpy()

    # plt.figure(figsize=(10, 6))
    # plt.plot(x_coords, analytical_density, 'k-', label=f'Analytical Ground State at t={T_MAX:.1f}', linewidth=2)
    # plt.plot(x_coords, pinn_density, 'r--', label=f'PINN Final State at t={T_MAX:.1f}', linewidth=2)
    # plt.title("PINN Final State vs. Analytical Solution")
    # plt.xlabel("Position (x)")
    # plt.ylabel("Probability Density |ψ|²")
    # plt.legend()
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.xlim(X_MIN, X_MAX)
    # plt.ylim(bottom=0)

    # plot_filename = "final_comparison.png"
    # plt.savefig(plot_filename, dpi=300)
    # plt.close()
