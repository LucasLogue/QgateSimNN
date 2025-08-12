#Lucas Logue 8/11/2025
#Refactored PINN For Simple Example
#The external electric field is given by:
#   E(t) = -6.3 * exp(-((t - 0) / 1.7)²) * sin(1.13 * t + 0.64)
# The initial condition at t=0 is the ground state of the 3D Quantum Harmonic Oscillator:
#   ψ(x, y, z, 0) = (1/π)^(3/4) * exp(-1/2 * (x² + y² + z²))
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import time
import os
#SELECT CONFIGURATION================================================
#CONFIGURATION SELECTOR
# CONFIG = "6-FAST" #10000epoch, small grid
# CONFIG = "6-SLOW" #25000 epoch, bigger grid
CONFIG = "6-MID" #20000 epoch, small grid
#mid gets better fidelity, but slow puts us flat on the center confidently. yknow type stuff

if CONFIG == "6-FAST":
    NUM_ITERATIONS = 10000
    LEARNING_RATE = 2e-5
    N_COLLOCATION = 4096
    N_INITIAL = 2048
    N_NORM = 4096
    LAYERS = [4] + [128] * 6 + [2]
elif CONFIG == "6-MID":
    NUM_ITERATIONS = 20000
    LEARNING_RATE = 2e-5
    N_COLLOCATION = 4096
    N_INITIAL = 2048
    N_NORM = 4096
    LAYERS = [4] + [128] * 6 + [2]
elif CONFIG == "6-SLOW":
    NUM_ITERATIONS = 25000
    LEARNING_RATE = 1e-5
    N_COLLOCATION = 8192
    N_INITIAL = 4096
    N_NORM = 8192
    LAYERS = [4] + [128] * 6 + [2]
elif CONFIG == "8-FAST":
    #the poor CRC A10 can't handle dis shit cuh 💔 🕊️
    NUM_ITERATIONS = 15000
    LEARNING_RATE = 1e-5
    N_COLLOCATION = 4096
    N_INITIAL = 2048
    N_NORM = 4096
    #exploding amnt
    # N_COLLOCATION = 8192
    # N_INITIAL = 4096
    # N_NORM = 8192
    LAYERS = [4] + [256] * 8 + [2]


#DOMAIN BOUNDARIES
X_MIN, X_MAX = -10.0, 10.0 
Y_MIN, Y_MAX = -5.0, 5.0
Z_MIN, Z_MAX = -5.0, 5.0
T_MIN, T_MAX = 0.0, 3.0
L = 5.0# Spatial domain length for cutoff function

DOMAIN_VOLUME = (X_MAX - X_MIN) * (Y_MAX - Y_MIN) * (Z_MAX - Z_MIN)


#=======DEVICE SETUP(CUDA)==========================
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)

#=======Initial Weights and Weight Functions
W_PDE = 60.0 #kind of seems like the same for how it traces against the statistical solution
W_IC = 25.0 #How well it traces against the statistical solution.
W_NORM = 80.0 #for how well it adheres to P(infinity volume) = 1
W_REG = 0.5 # Add a small weight for this regularization term
W_FIDEL = 65.0 #weight for fidelity at final time

#Functions for weight scaling if we want to be doing that
def get_w_pde(iteration, warmup_steps=(0.3 * NUM_ITERATIONS), max_weight=W_PDE):
    """Linearly ramps up the PDE weight from 0 to max_weight."""
    if iteration < 1000: # Start with a tiny, non-zero weight to avoid issues
        return 0.01
    progress = min(1.0, (iteration - 1000) / warmup_steps)
    return max_weight * progress

def get_w_ic(iteration, num_iterations, initial_weight=150.0, final_weight=W_IC):
    """Gradually decreases the IC weight from an initial high value."""
    # Define the point at which the decay finishes
    decay_end_iter = int(num_iterations * 0.75)
    
    if iteration >= decay_end_iter:
        return final_weight
        
    progress = iteration / decay_end_iter
    # Linear interpolation from initial_weight down to final_weight
    return initial_weight - (initial_weight - final_weight) * progress

#========Physics Definitions
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

#Classical Trajectory===============================================================
def classical_ode(t, state):
    x, p = state
    dxdt = p
    E_t_np = lambda t_val: -6.3 * np.exp(-((t_val - 0)/1.7)**2) * np.sin(1.13*t_val + 0.64)
    dpdt = -x - E_t_np(t)
    return [dxdt, dpdt]
t_eval = np.linspace(T_MIN, T_MAX, 500)
sol = solve_ivp(classical_ode, [T_MIN, T_MAX], [0, 0], t_eval=t_eval, dense_output=True)

# Create fast interpolation functions for position and velocity
Xc_func = interp1d(sol.t, sol.y[0], kind='cubic', fill_value="extrapolate")
Xc_dot_func = interp1d(sol.t, sol.y[1], kind='cubic', fill_value="extrapolate")

# Convert to PyTorch tensors on the correct device
t_for_interp = torch.linspace(T_MIN, T_MAX, 500, device=device)
Xc_t = torch.tensor(Xc_func(t_for_interp.cpu().numpy()), dtype=torch.float32, device=device)
Xc_dot_t = torch.tensor(Xc_dot_func(t_for_interp.cpu().numpy()), dtype=torch.float32, device=device)



#Xc functions=================================================================
def get_Xc(t):
    """Gets classical position Xc at time t via interpolation."""
    indices = torch.searchsorted(t_for_interp, t.squeeze(-1).contiguous()).clamp(max=len(t_for_interp)-1)
    out = Xc_t[indices].unsqueeze(-1)
    # if not hasattr(get_Xc, "_flag"):
    #     dbg("Xc_lookup", sample_t=t[:3].flatten(), Xc_sample=out[:3].flatten())
    #     get_Xc._flag = False
    return out

def get_Xc_dot(t):
    """Gets classical velocity Xc_dot at time t via interpolation."""
    indices = torch.searchsorted(t_for_interp, t.squeeze(-1).contiguous()).clamp(max=len(t_for_interp)-1)
    out = Xc_dot_t[indices].unsqueeze(-1)
    # if not hasattr(get_Xc, "_flag"):
    #     dbg("Xc_lookup", sample_t=t[:3].flatten(), Xc_sample=out[:3].flatten())
    #     get_Xc._flag = False
    return out

#=====================PINNNNN MODELL!!!==============================
class PINN(nn.Module):
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

    def _forward_base(self, x, y, z, t):
        x_prime = x - get_Xc(t)
        x_prime_norm = x_prime / L
        y_norm = y / L
        z_norm = z / L
        t_norm = 2.0 * (t - T_MIN) / (T_MAX - T_MIN) - 1.0
        
        inputs = torch.cat([x_prime_norm, y_norm, z_norm, t_norm], dim=1)
        nn_output = self.net(inputs)
        
        A_raw = nn_output[:, 0:1]
        S_raw = nn_output[:, 1:2] # The NN now learns the full phase S
        
        return A_raw, S_raw, x_prime, y_norm, z_norm

    def forward(self, x, y, z, t):
        A_raw, S_raw, x_prime, y_norm, z_norm = self._forward_base(x, y, z, t)
        
        cutoff = (1 - (x_prime / L)**2) * (1 - y_norm**2) * (1 - z_norm**2)
        A = cutoff * A_raw
        S = S_raw
        
        u = A * torch.cos(S)
        v = A * torch.sin(S)
        
        return u, v
    
    def get_A_S(self, x, y, z, t, apply_cutoff=True):
        A_raw, S_raw, x_prime, y_norm, z_norm = self._forward_base(x, y, z, t)
        
        if apply_cutoff:
            cutoff = (1 - (x_prime / L)**2) * (1 - y_norm**2) * (1 - z_norm**2)
            A = cutoff * A_raw
        else:
            A = A_raw
        
        S = S_raw
        return A, S
#=================END OF PINN MODEL=========================


#==========Training Functions================================
def pde_residual(model, x, y, z, t):
    """attempt at schrodingies residual calculation"""
    x.requires_grad_(True); y.requires_grad_(True); z.requires_grad_(True); t.requires_grad_(True)
    
    u, v = model(x, y, z, t)

    # --- Bundle derivative calculations for u ---
    # Calculate all first-order derivatives of u in one go
    u_grads = torch.autograd.grad(u, [x, y, z, t], grad_outputs=torch.ones_like(u), create_graph=True)
    u_x, u_y, u_z, u_t_incomplete = u_grads[0], u_grads[1], u_grads[2], u_grads[3]

    # Calculate second-order derivatives from the first-order ones
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    u_zz = torch.autograd.grad(u_z, z, grad_outputs=torch.ones_like(u_z), create_graph=True)[0]
    lap_u = u_xx + u_yy + u_zz

    # --- Bundle derivative calculations for v ---
    v_grads = torch.autograd.grad(v, [x, y, z, t], grad_outputs=torch.ones_like(v), create_graph=True)
    v_x, v_y, v_z, v_t_incomplete = v_grads[0], v_grads[1], v_grads[2], v_grads[3]

    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
    v_zz = torch.autograd.grad(v_z, z, grad_outputs=torch.ones_like(v_z), create_graph=True)[0]
    lap_v = v_xx + v_yy + v_zz

    # --- Apply chain rule and compute residuals (no changes here) ---
    xc_dot = get_Xc_dot(t)
    u_t = u_t_incomplete - xc_dot * u_x
    v_t = v_t_incomplete - xc_dot * v_x
    
    V = V_potential(x, y, z, t)
    f_u = u_t + 0.5 * lap_v - V * v #why are we doing it like this?
    f_v = v_t - 0.5 * lap_u + V * u
    
    return f_u, f_v


def fidelity_loss(model):
    PTSFIDELITY = 1024 #number of sampled poitns
    x_final = (torch.rand(PTSFIDELITY, 1, device=device) * (X_MAX - X_MIN)) + X_MIN
    y_final = (torch.rand(PTSFIDELITY, 1, device=device) * (Y_MAX - Y_MIN)) + Y_MIN
    z_final = (torch.rand(PTSFIDELITY, 1, device=device) * (Z_MAX - Z_MIN)) + Z_MIN
    t_final = torch.full_like(x_final, T_MAX) 

    #Get prediction wavefunc
    u_pred, v_pred = model(x_final, y_final, z_final, t_final)
    psi_pinn_final = u_pred + 1j * v_pred

    #Define target state (later will we use the values for the actual target states)
    classical_center = Xc_func(T_MAX).item()
    x_shifted = x_final - classical_center
    psi_target_final = torch.tensor((1.0 / np.pi)**(0.75), device=device) * torch.exp(-0.5 * (x_shifted**2 + y_final**2 + z_final**2))

    # who tf is monte carlo? but it is monte carlo or so they say
    inner_product = (DOMAIN_VOLUME / PTSFIDELITY) * torch.sum(torch.conj(psi_target_final) * psi_pinn_final)
    norm_pinn_sq = (DOMAIN_VOLUME / PTSFIDELITY) * torch.sum(torch.abs(psi_pinn_final)**2)
    norm_target_sq = (DOMAIN_VOLUME / PTSFIDELITY) * torch.sum(torch.abs(psi_target_final)**2)


    epsilon = 1e-8
    fidelity = (torch.abs(inner_product)**2) / (norm_pinn_sq * norm_target_sq + epsilon)
    return (1.0 - fidelity)

#===================DATASAMPLING_=======================================
def sample_points():
    """
    Draw collocation / IC / normalisation points *uniformly in the co-moving frame*.
    Every point then satisfies |x'| ≤ L, so the cutoff never switches sign.
    """
    # --- collocation set -------------------------------------------------
    t_col = torch.rand(N_COLLOCATION, 1, device=device)*(T_MAX-T_MIN) + T_MIN          # U[0,T]
    x_prime = torch.rand_like(t_col)*(2*L) - L                                         # U[-L,L]
    x_col   = x_prime + get_Xc(t_col)                                                  # back-shift to lab
    y_col   = torch.rand_like(x_col)*(Y_MAX-Y_MIN) + Y_MIN
    z_col   = torch.rand_like(x_col)*(Z_MAX-Z_MIN) + Z_MIN

    # --- initial-condition set (t = 0; Xc(0)=0 so x'=x) -------------------
    x_ic = torch.rand(N_INITIAL, 1, device=device)*(2*L) - L
    y_ic = torch.rand_like(x_ic)*(Y_MAX-Y_MIN) + Y_MIN
    z_ic = torch.rand_like(x_ic)*(Z_MAX-Z_MIN) + Z_MIN
    t_ic = torch.zeros_like(x_ic)

    # --- normalisation batch (reuse the same trick) ----------------------
    t_norm = torch.rand(N_NORM, 1, device=device)*(T_MAX-T_MIN) + T_MIN
    x_prime_n = torch.rand_like(t_norm)*(2*L) - L
    x_norm = x_prime_n + get_Xc(t_norm)
    y_norm = torch.rand_like(x_norm)*(Y_MAX-Y_MIN) + Y_MIN
    z_norm = torch.rand_like(x_norm)*(Z_MAX-Z_MIN) + Z_MIN

    return x_col, y_col, z_col, t_col, x_ic, y_ic, z_ic, t_ic, x_norm, y_norm, z_norm, t_norm

#Final anaylsis check
def run_final_analysis(model):
    """
    pretty big n accurate analysis for t=3 / t_MAX prolly fix it in future
    """
    print("\n--- FINAL MODEL ANALYSIS --- 🤯🤯🤯🤯")
    model.eval()
    with torch.no_grad():
        # Create analysis grid
        N_FINAL_GRID = 4096
        classical_center = Xc_func(T_MAX).item()
        
        #DEFINE THE DOMAIN
        XMINEVAL = classical_center - L
        XMAXEVAL = classical_center + L
        
        x_line = torch.linspace(XMINEVAL, XMAXEVAL, N_FINAL_GRID, device=device).unsqueeze(1)
        y0_line = torch.zeros_like(x_line)
        z0_line = torch.zeros_like(x_line)
        t_final_line = torch.full_like(x_line, T_MAX)

        #PINN's predicted wavefunction
        u_pred, v_pred = model(x_line, y0_line, z0_line, t_final_line)
        psi_pinn = u_pred + 1j * v_pred

        #true analytical wavefunction at the final position
        norm_factor = (1.0 / np.pi)**(0.75)
        x_shifted = x_line - classical_center
        psi_analytic = norm_factor * torch.exp(-0.5 * (x_shifted**2))

        #dx for fidelity
        dx = (XMAXEVAL - XMINEVAL) / (N_FINAL_GRID - 1)
        
        # Fidelity
        inner_product = torch.sum(torch.conj(psi_analytic) * psi_pinn) * dx
        norm_pinn_sq = torch.sum(torch.abs(psi_pinn)**2) * dx
        norm_analytic_sq = torch.sum(torch.abs(psi_analytic)**2) * dx
        fidelity = (torch.abs(inner_product)**2) / (norm_pinn_sq * norm_analytic_sq)
        
        # Calculate X Center
        numerator = torch.sum(x_line * torch.abs(psi_pinn)**2) * dx
        pinn_center = numerator / norm_pinn_sq

        #Debug Printing
        print(f"  - Final Fidelity      : {fidelity.item():.4f}")
        print(f"  - Final PINN Center <x>: {pinn_center.item():.4f}")
        print(f"  - Target Center        : {classical_center:.4f}")
        print("--------------------------")

if __name__ == "__main__":
    model = PINN(layers=LAYERS).to(device)

    #Optimizer and Scheduler
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_adam, step_size=int(NUM_ITERATIONS * 0.25), gamma=0.1)

    print("🤓🤓🤓 Starting Training (Co-Moving Frame Model) 🤓🤓🤓")
    start_time = time.time()
    for i in range(1, NUM_ITERATIONS+1):
        model.train()
        optimizer_adam.zero_grad() #why?

        #Get Sample for Epoch
        (x_col, y_col, z_col, t_col,
        x_ic,  y_ic,  z_ic,  t_ic,
        x_norm, y_norm, z_norm, t_norm) = sample_points()

        f_u, f_v = pde_residual(model, x_col, y_col, z_col, t_col)
        loss_pde = torch.mean(f_u**2) + torch.mean(f_v**2)

        #Compare the Amplitude and Phase from model against Initial Condition 
        A_ic_pred, S_ic_pred = model.get_A_S(x_ic, y_ic, z_ic, t_ic, apply_cutoff=False)
        A_ic_true, S_ic_true = initial_condition_A_S(x_ic, y_ic, z_ic)

        #Loss_ic Choices--------------------------

        #Simplified Version, Working Decently Under Certain params mean((Prediction Amplitude & Phase - True)**2)
        loss_ic = torch.mean((A_ic_pred - A_ic_true)**2) + torch.mean((S_ic_pred - S_ic_true)**2)

        #Weighted on the Gaussian (goal to focus on center)
        # ic_weight = A_ic_true**2
        # loss_ic_A = torch.mean(ic_weight * (A_ic_pred - A_ic_true)**2)
        # loss_ic_S = torch.mean((S_ic_pred - S_ic_true)**2) # Phase loss doesn't need weighting
        # loss_ic = loss_ic_A + loss_ic_S
        #--------------End of Loss Ic Choices

        #Loss norm
        u_norm, v_norm = model(x_norm, y_norm, z_norm, t_norm)
        psi_sq_norm = u_norm**2 + v_norm**2
        norm_pred = DOMAIN_VOLUME * torch.mean(psi_sq_norm)
        loss_norm = (norm_pred - 1.0)**2

        #Loss for Regularizing the Phase
        A_col, S_col = model.get_A_S(x_col, y_col, z_col, t_col, apply_cutoff=True)
        # Calculate the spatial gradients of the phase
        S_grads = torch.autograd.grad(S_col, [x_col, y_col, z_col], grad_outputs=torch.ones_like(S_col), create_graph=True)
        S_x, S_y, S_z = S_grads[0], S_grads[1], S_grads[2]
        # The new loss term penalizes the magnitude of the phase gradients to enforce smoothness
        loss_reg = torch.mean(S_x**2) + torch.mean(S_y**2) + torch.mean(S_z**2)

        #Loss for Fidelity
        loss_fidel = fidelity_loss(model)


        #Set up the ramping / scaling weights
        W_PDE = get_w_pde(i, #warmup_steps=warmupsteps, 
                    max_weight=60.0)
        w_ic = get_w_ic(i, num_iterations=NUM_ITERATIONS, initial_weight=150.0, final_weight=W_IC)


        #Final loss estimate
        total_loss = W_PDE * loss_pde + w_ic * loss_ic + W_NORM * loss_norm + W_REG * loss_reg + W_FIDEL*loss_fidel

        #Propogate and Step with optim and scheduler
        total_loss.backward()
        optimizer_adam.step()
        scheduler.step()

        if i % 1000 == 0: #debug print every 1000
            print(f"\n--- DEBUG REPORT @ ITERATION {i} ---")
            print("Loss Contributions:")
            print(f"  - PDE Loss  : {W_PDE * loss_pde.item():.4e}")
            print(f"  - IC Loss   : {w_ic * loss_ic.item():.4e}")
            print(f"  - Norm Loss : {W_NORM * loss_norm.item():.4e}")
            print(f"  - Reg Loss  : {W_REG * loss_reg.item():.4e}")
            print(f"  - Fidelity Loss  : {W_FIDEL * loss_fidel.item():.4e}")


            model.eval() #set to evaluation for debug
            with torch.no_grad():
                N_DBG = 128
                x_shift_final = Xc_func(3.0).item()
                XMINEVAL = x_shift_final - L
                XMAXEVAL = x_shift_final + L

                x_line = torch.linspace(XMINEVAL, XMAXEVAL, N_DBG, device=device).unsqueeze(1)
                y0_line = torch.zeros_like(x_line)
                z0_line = torch.zeros_like(x_line)
                #set time = 3 for all points on the meshgrid
                t_final_line = torch.full_like(x_line, T_MAX)

                #get prediction at time =3
                u_pred_line, v_pred_line = model(x_line, y0_line, z0_line, t_final_line)
                psi_pinn_torch = u_pred_line + 1j * v_pred_line

                norm_factor = (1.0 / np.pi)**(0.75)
                x_shifted = x_line - x_shift_final
                psi_analytic_torch = norm_factor * torch.exp(-0.5 * (x_shifted**2 + y0_line**2 + z0_line**2))

                # Calculate Fidelity
                dx = (XMAXEVAL - XMINEVAL) / (N_DBG - 1)
                inner_product = torch.sum(torch.conj(psi_analytic_torch) * psi_pinn_torch) * dx
                norm_pinn_sq = torch.sum(torch.abs(psi_pinn_torch)**2) * dx
                norm_analytic_sq = torch.sum(torch.abs(psi_analytic_torch)**2) * dx
                fidelity = (torch.abs(inner_product)**2) / (norm_pinn_sq * norm_analytic_sq)
                
                # Calculate PINN Center <x>
                numerator = torch.sum(x_line * torch.abs(psi_pinn_torch)**2) * dx
                x_expectation = numerator / norm_pinn_sq

                print("Fidelity and Centering at t=3.0:")
                print(f"  - Fidelity      : {fidelity.item():.4f}")
                print(f"  - PINN Center <x>: {x_expectation.item():.4f}")
                print(f"  - Target Center  : {x_shift_final:.4f}")
                print("---------------------------------")
            model.train() #just incase
    endtime = time.time()
    print(f"Duration: {endtime - start_time}")
    #Save our better version
    MODEL_PATH = "pinn_schrodinger_3d.pth"
    torch.save(model.state_dict(), MODEL_PATH)

    run_final_analysis(model)

