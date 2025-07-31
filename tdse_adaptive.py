import torch
import torch.nn as nn
import torch.autograd as autograd
import matplotlib.pyplot as plt
import numpy as np
import imageio
import time

# ── PINN definition ────────────────────────────────────────────────────
class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers)-2):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        self.out = nn.Linear(layers[-2], layers[-1])

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = torch.tanh(layer(h))
        return self.out(h)

# The stable, non-JIT version of the residual calculator
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

# ── Double‑slit potential with smooth edges ───────────────────────────
def V_double_slit(x, y):
    V0 = 1e4; barrier_width = 0.1; slit_pos = 0.5; slit_width = 0.2; smoothness = 50.0
    barrier_shape = 0.5 * (torch.tanh(smoothness * (x + barrier_width)) - torch.tanh(smoothness * (x - barrier_width)))
    slit1 = 0.5 * (torch.tanh(smoothness * (y - (slit_pos - slit_width))) - torch.tanh(smoothness * (y - (slit_pos + slit_width))))
    slit2 = 0.5 * (torch.tanh(smoothness * (y - (-slit_pos - slit_width))) - torch.tanh(smoothness * (y - (-slit_pos + slit_width))))
    slits_shape = slit1 + slit2
    potential = V0 * barrier_shape * (1 - slits_shape)
    return potential

# ── Training loop ─────────────────────────────────────────────────────
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Using the "Balanced" settings as a base
    N_colloc = 15000
    N_ic    = 4000
    N_bc    = 4000
    layers  = [3, 96, 96, 96, 96, 2]
    lr      = 5e-4
    epochs  = 15000
    
    w_pde_max = 1.0
    ### CHANGE 1: MASSIVELY INCREASED IC WEIGHT ###
    w_ic = 1000.0 # Was 100.0. This is the main fix.
    w_bc = 10.0

    # Collocation points
    x = (torch.rand(N_colloc,1)*10 - 5)
    y = (torch.rand(N_colloc,1)*6  - 3)
    t = torch.rand(N_colloc,1)
    xyt = torch.cat([x,y,t], dim=1).to(device)

    # Initial condition
    x0 = (torch.rand(N_ic,1)*10 - 5)
    y0 = (torch.rand(N_ic,1)*6  - 3)
    t0 = torch.zeros_like(x0)
    xyt0 = torch.cat([x0,y0,t0], dim=1).to(device)
    
    # Slower velocity for easier learning
    momentum = 6.0
    psi0_wave = torch.exp(-((x0+2.5)**2 + y0**2)/0.8) * torch.exp(1j*momentum*x0)
    
    ### CHANGE 2: STRICTER INITIAL CONDITION ###
    # Create a mask that is 1 on the left side and 0 on the right.
    # This forces the initial wavepacket to be exactly zero on the right.
    mask = (x0 < -1.0).float()
    psi0 = psi0_wave * mask
    
    psi0_val = torch.view_as_real(psi0).to(device)

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

    start_time = time.time()
    for ep in range(epochs+1):
        opt.zero_grad(set_to_none=True)
        w_pde = w_pde_max * min(1.0, ep / (epochs/2.0))

        with torch.amp.autocast('cuda'):
            fr, fi = schrodinger_residual(model, xyt, V_double_slit)
            loss_pde = (fr**2 + fi**2).mean()
            psi_pred0 = model(xyt0)
            loss_ic  = ((psi_pred0 - psi0_val)**2).mean()
            psi_bc = model(xyt_bc)
            loss_bc = (psi_bc**2).mean()
            loss = w_pde * loss_pde + w_ic * loss_ic + w_bc * loss_bc
        
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        if ep % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {ep:5d} | PDE={loss_pde:.2e} IC={loss_ic:.2e} BC={loss_bc:.2e} | Total={loss:.2e} | w_pde={w_pde:.2f} | Time: {elapsed:.1f}s")

    # --- Visualization ---
    model.eval()
    X, Y = np.meshgrid(np.linspace(-5, 5, 250), np.linspace(-3, 3, 150)) 
    ptvect = np.stack((X,Y), axis=-1).reshape(-1, 2)
    timevec = np.linspace(0, 1, 150)

    def V_double_slit_np(x, y):
        V0 = 1.0; barrier_width = 0.1; slit_pos = 0.5; slit_width = 0.2; smoothness = 50.0
        barrier_shape = 0.5 * (np.tanh(smoothness * (x + barrier_width)) - np.tanh(smoothness * (x - barrier_width)))
        slit1 = 0.5 * (np.tanh(smoothness * (y - (slit_pos - slit_width))) - np.tanh(smoothness * (y - (slit_pos + slit_width))))
        slit2 = 0.5 * (np.tanh(smoothness * (y - (-slit_pos - slit_width))) - np.tanh(smoothness * (y - (-slit_pos + slit_width))))
        slits_shape = slit1 + slit2
        return V0 * barrier_shape * (1 - slits_shape)

    V_np = V_double_slit_np(X, Y)
    cyan_color = np.array([0.0, 1.0, 1.0])
    barrier_img = V_np[:, :, np.newaxis] * cyan_color 

    print("✨ Generating frames with dynamic normalization and cyan barrier...")
    frames = []
    for tk in timevec:
        tket = np.full((ptvect.shape[0],1), tk, dtype=np.float32)
        inp = torch.from_numpy(np.hstack([ptvect, tket]).astype(np.float32)).to(device)
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=False):
                out = model(inp).cpu().numpy()
        
        psi = out[:,0] + 1j*out[:,1]
        prob = (np.abs(psi)**2).reshape(Y.shape)
        
        frame_max = prob.max()
        norm_prob = prob / (frame_max + 1e-8) 
        
        wave_color = plt.cm.inferno(norm_prob)[:,:,:3]
        wave_alpha = norm_prob[:,:,np.newaxis] 
        blended_img = barrier_img + (1 - V_np[:, :, np.newaxis]) * wave_alpha * wave_color
        
        img = (np.clip(blended_img, 0, 1) * 255).astype(np.uint8)
        frames.append(img)

    duration = 5.0
    fps = len(frames)/duration
    print(f"⏱ Generating GIF with {len(frames)} frames...")

    plt.figure(figsize=(8, 8*Y.shape[0]/Y.shape[1]))
    plt.imshow(frames[-1], origin='lower', extent=[-5,5,-3,3])
    plt.title(f'|ψ|² at t=1.0 (Stricter IC)')
    plt.xlabel('x'); plt.ylabel('y')
    plt.tight_layout()
    plt.show()

    imageio.mimsave("schrodinger_pinn_strict_ic.gif", frames, fps=fps)
    print(f"✅ Saved schrodinger_pinn_strict_ic.gif ({len(frames)} frames, {duration}s, {fps:.1f} FPS)")

if __name__=='__main__':
    train()
