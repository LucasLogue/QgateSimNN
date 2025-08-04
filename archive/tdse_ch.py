import torch
import torch.nn as nn
import torch.autograd as autograd
import matplotlib.pyplot as plt
import numpy as np
import imageio
import time

### NEW ARCHITECTURE: FOURIER FEATURE MAPPING ###
# This layer transforms the input coordinates into a set of high-frequency features,
# which makes it much easier for the network to learn wave-like functions.
class FourierFeatureMapping(nn.Module):
    def __init__(self, input_dims, mapping_size, scale=10.0):
        super().__init__()
        self.input_dims = input_dims
        self.mapping_size = mapping_size
        # B is a random but fixed matrix used for the mapping
        self.B = nn.Parameter(torch.randn(input_dims, mapping_size) * scale, requires_grad=False)

    def forward(self, x):
        # x is shape (batch_size, input_dims)
        x_proj = x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# The new PINN incorporates the Fourier Feature mapping
class PINN(nn.Module):
    def __init__(self, layers, mapping_size=128):
        super().__init__()
        self.fourier_mapping = FourierFeatureMapping(layers[0], mapping_size)
        
        # The main network now takes the mapped features as input
        network_layers = [mapping_size * 2] + layers[1:]
        
        self.layers = nn.ModuleList()
        for i in range(len(network_layers)-2):
            self.layers.append(nn.Linear(network_layers[i], network_layers[i+1]))
        self.out = nn.Linear(network_layers[-2], network_layers[-1])

    def forward(self, xyt):
        # First, apply the Fourier mapping
        h = self.fourier_mapping(xyt)
        # Then, pass the features through the main network
        for layer in self.layers:
            h = torch.tanh(layer(h))
        return self.out(h)

# The residual calculator remains the same
def schrodinger_residual(model, xyt, V_func):
    xyt = xyt.clone().detach().requires_grad_(True)
    psi = model(xyt)
    psi_r, psi_i = psi[:, 0:1], psi[:, 1:2]

    grad_r1 = autograd.grad(psi_r.sum(), xyt, create_graph=True)[0]
    psi_r_x, psi_r_y, psi_r_t = grad_r1[:, 0:1], grad_r1[:, 1:2], grad_r1[:, 2:3]
    psi_r_xx = autograd.grad(psi_r_x.sum(), xyt, create_graph=True, retain_graph=True)[0][:, 0:1]
    psi_r_yy = autograd.grad(psi_r_y.sum(), xyt, create_graph=True)[0][:, 1:2]
    lap_r = psi_r_xx + psi_r_yy

    grad_i1 = autograd.grad(psi_i.sum(), xyt, create_graph=True)[0]
    psi_i_x, psi_i_y, psi_i_t = grad_i1[:, 0:1], grad_i1[:, 1:2], grad_i1[:, 2:3]
    psi_i_xx = autograd.grad(psi_i_x.sum(), xyt, create_graph=True, retain_graph=True)[0][:, 0:1]
    psi_i_yy = autograd.grad(psi_i_y.sum(), xyt, create_graph=True)[0][:, 1:2]
    lap_i = psi_i_xx + psi_i_yy

    V = V_func(xyt[:, 0:1], xyt[:, 1:2])
    f_r = lap_r - V * psi_r - psi_i_t
    f_i = psi_r_t + lap_i - V * psi_i

    return f_r, f_i

def V_free_space(x, y):
    return torch.zeros_like(x)

# ── Training loop ─────────────────────────────────────────────────────
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # "Laptop-Safe" but powerful settings
    N_colloc = 10000
    N_ic = 4000
    N_bc = 4000
    # The network layers. Input is 3 (x,y,t), output is 2 (Re, Im)
    layers  = [3, 128, 128, 128, 128, 2]
    lr      = 1e-4 # Lower LR is often better for Fourier Feature nets
    epochs  = 20000 
    
    w_pde = 1.0
    w_ic = 2000.0 # Keep a very high IC weight
    w_bc = 1.0 

    # Collocation points
    x = (torch.rand(N_colloc,1)*10 - 5)
    y = (torch.rand(N_colloc,1)*6  - 3)
    t = torch.rand(N_colloc,1)
    xyt = torch.cat([x,y,t], dim=1).to(device)

    # Initial condition points
    x0_pts = (torch.rand(N_ic,1)*10 - 5)
    y0_pts = (torch.rand(N_ic,1)*6  - 3)
    t0_pts = torch.zeros_like(x0_pts)
    xyt0 = torch.cat([x0_pts,y0_pts,t0_pts], dim=1).to(device)
    
    momentum = 10.0
    psi0_wave = torch.exp(-((x0_pts + 2.5)**2 + y0_pts**2) / 0.8) * torch.exp(1j * momentum * x0_pts)
    psi0_val = torch.view_as_real(psi0_wave).to(device)

    # Boundary conditions
    N_side = N_bc // 4
    x_left  = torch.full((N_side,1), -5.0); y_left  = torch.rand((N_side,1))*6.0 - 3.0
    x_right = torch.full((N_side,1), +5.0); y_right = torch.rand((N_side,1))*6.0 - 3.0
    y_bot   = torch.full((N_side,1), -3.0); x_bot   = torch.rand((N_side,1))*10.0 - 5.0
    y_top   = torch.full((N_side,1), +3.0); x_top   = torch.rand((N_side,1))*10.0 - 5.0
    x_bc = torch.cat([x_left, x_right, x_bot, x_top], dim=0)
    y_bc = torch.cat([y_left, y_right, y_bot, y_top], dim=0)
    t_bc = torch.rand((4*N_side,1))
    xyt_bc = torch.cat([x_bc, y_bc, t_bc], dim=1).to(device)

    model = PINN(layers).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=2000, gamma=0.9)
    scaler = torch.amp.GradScaler('cuda')

    print("🔥 Starting training for the 'free particle' case with Fourier Features...")
    start_time = time.time()
    for ep in range(epochs+1):
        opt.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda'):
            fr, fi = schrodinger_residual(model, xyt, V_free_space)
            loss_pde = (fr**2 + fi**2).mean()
            
            psi_pred0 = model(xyt0)
            loss_ic = ((psi_pred0 - psi0_val)**2).mean()
            
            psi_bc = model(xyt_bc)
            loss_bc = (psi_bc**2).mean()
            
            loss = w_pde * loss_pde + w_ic * loss_ic + w_bc * loss_bc
        
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        if ep % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {ep:5d} | PDE={loss_pde:.2e} IC={loss_ic:.2e} BC={loss_bc:.2e} | Total={loss:.2e} | Time: {elapsed:.1f}s")

    # --- Visualization ---
    model.eval()
    X, Y = np.meshgrid(np.linspace(-5, 5, 250), np.linspace(-3, 3, 150)) 
    ptvect = np.stack((X,Y), axis=-1).reshape(-1, 2)
    timevec = np.linspace(0, 1, 150)

    def analytical_solution(x, y, t, k=10.0, sigma_sq=0.4):
        # Correct group velocity: v_g = hbar*k/m = 1*k/(1/2) = 2k
        v = 2 * k
        
        denominator = sigma_sq + 1j * t
        exponent = -((x - v*t + 2.5)**2 + y**2) / (2 * denominator)
        
        prefactor = np.sqrt(sigma_sq / denominator) / (2 * np.pi * sigma_sq)**0.5
        
        psi = prefactor * np.exp(exponent) * np.exp(1j * k * (x - v*t/2))
        return np.abs(psi)**2

    print("✨ Generating frames for PINN prediction and Ground Truth...")
    pinn_frames = []
    analytical_frames = []

    vis_batch_size = 4096 
    for tk in timevec:
        # --- PINN Frame ---
        outputs = []
        for batch_start in range(0, ptvect.shape[0], vis_batch_size):
            batch_end = batch_start + vis_batch_size
            pt_batch = ptvect[batch_start:batch_end]
            tket = np.full((pt_batch.shape[0],1), tk, dtype=np.float32)
            inp = torch.from_numpy(np.hstack([pt_batch, tket]).astype(np.float32)).to(device)
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=False):
                    out_batch = model(inp).cpu().numpy()
                    outputs.append(out_batch)
        
        out = np.vstack(outputs)
        psi_pinn = out[:,0] + 1j*out[:,1]
        prob_pinn = (np.abs(psi_pinn)**2).reshape(Y.shape)
        
        frame_max_pinn = prob_pinn.max()
        norm_prob_pinn = prob_pinn / (frame_max_pinn + 1e-8) 
        pinn_img = (plt.cm.inferno(norm_prob_pinn)[:,:,:3] * 255).astype(np.uint8)
        pinn_frames.append(pinn_img)

        # --- Analytical Frame ---
        prob_analytical = analytical_solution(X, Y, tk, k=momentum).reshape(Y.shape)
        frame_max_analytical = prob_analytical.max()
        norm_prob_analytical = prob_analytical / (frame_max_analytical + 1e-8)
        analytical_img = (plt.cm.inferno(norm_prob_analytical)[:,:,:3] * 255).astype(np.uint8)
        analytical_frames.append(analytical_img)

    duration = 5.0
    fps = len(timevec)/duration
    print(f"⏱ Generating GIFs...")

    imageio.mimsave("pinn_prediction.gif", pinn_frames, fps=fps)
    print(f"✅ Saved PINN prediction GIF.")
    
    imageio.mimsave("ground_truth.gif", analytical_frames, fps=fps)
    print(f"✅ Saved Ground Truth GIF.")

if __name__=='__main__':
    train()
