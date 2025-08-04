import torch
import torch.nn as nn
import torch.autograd as autograd
import matplotlib.pyplot as plt
import numpy as np
import imageio
import time

### FOURIER FEATURE MAPPING ###
class FourierFeatureMapping(nn.Module):
    def __init__(self, input_dims, mapping_size, scale=1.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(input_dims, mapping_size) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class PINN(nn.Module):
    def __init__(self, layers, mapping_size=128):
        super().__init__()
        self.fourier_mapping = FourierFeatureMapping(layers[0], mapping_size)
        net_dims = [mapping_size*2] + layers[1:]
        self.layers = nn.ModuleList([nn.Linear(net_dims[i], net_dims[i+1]) for i in range(len(net_dims)-1)])
    def forward(self, xyt):
        h = self.fourier_mapping(xyt)
        for layer in self.layers[:-1]:
            h = torch.tanh(layer(h))
        return self.layers[-1](h)

# Double-slit barrier potential
def V_double_slit(x, y):
    V0 = 1e5
    barrier_width = 0.2
    slit_half_h = 0.5
    slit_centers = [1.0, -1.0]
    mask_barrier = (torch.abs(x) < barrier_width/2)
    mask_slits = torch.zeros_like(x, dtype=torch.bool)
    for yc in slit_centers:
        mask_slits |= (torch.abs(y - yc) < slit_half_h)
    V = torch.zeros_like(x)
    V[mask_barrier & ~mask_slits] = V0
    return V

# PDE residual for TDSE
def schrodinger_residual(model, xyt, V_func):
    xyt = xyt.clone().detach().requires_grad_(True)
    psi = model(xyt)
    psi_r, psi_i = psi[:,0:1], psi[:,1:2]
    # derivatives real
    grad_r = autograd.grad(psi_r.sum(), xyt, create_graph=True)[0]
    psi_r_x, psi_r_y, psi_r_t = grad_r[:,0:1], grad_r[:,1:2], grad_r[:,2:3]
    psi_r_xx = autograd.grad(psi_r_x.sum(), xyt, create_graph=True, retain_graph=True)[0][:,0:1]
    psi_r_yy = autograd.grad(psi_r_y.sum(), xyt, create_graph=True)[0][:,1:2]
    lap_r = psi_r_xx + psi_r_yy
    # derivatives imag
    grad_i = autograd.grad(psi_i.sum(), xyt, create_graph=True)[0]
    psi_i_x, psi_i_y, psi_i_t = grad_i[:,0:1], grad_i[:,1:2], grad_i[:,2:3]
    psi_i_xx = autograd.grad(psi_i_x.sum(), xyt, create_graph=True, retain_graph=True)[0][:,0:1]
    psi_i_yy = autograd.grad(psi_i_y.sum(), xyt, create_graph=True)[0][:,1:2]
    lap_i = psi_i_xx + psi_i_yy
    V = V_func(xyt[:,0:1], xyt[:,1:2])
    f_r = lap_r - V*psi_r - psi_i_t
    f_i = psi_r_t + lap_i - V*psi_i
    return f_r, f_i

# Training
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # sample sizes
    N_colloc = 10000
    N_ic = 4000
    N_bc = 4000
    # Smaller domain
    x_min, x_max = -5.0, 5.0
    y_min, y_max = -3.0, 3.0
    layers = [3,128,128,128,2]
    lr = 1e-4
    epochs = 10000  # Reduced for quicker feedback cycles
    # weights
    w_pde = 1.0
    w_ic  = 50.0
    w_bc  = 1.0
    w_bar = 100.0   # weight for barrier interior BC (enforce psi≈0 on barrier edges)

    # collocation points
    x = torch.rand(N_colloc,1)*(x_max-x_min)+x_min
    y = torch.rand(N_colloc,1)*(y_max-y_min)+y_min
    t = torch.rand(N_colloc,1)
    xyt = torch.cat([x,y,t],dim=1).to(device)
    # IC points
    x0 = torch.rand(N_ic,1)*(x_max-x_min)+x_min
    y0 = torch.rand(N_ic,1)*(y_max-y_min)+y_min
    t0 = torch.zeros_like(x0)
    xyt0 = torch.cat([x0,y0,t0],dim=1).to(device)
    # initial wave
    momentum = 5.0
    psi0 = torch.exp(-((x0+3.0)**2 + y0**2)/0.4) * torch.exp(1j * momentum * x0)
    psi0_val = torch.view_as_real(psi0).to(device)
    # BC points on boundaries
    N_side = N_bc//4
    xl = torch.full((N_side,1),x_min); yl = torch.rand(N_side,1)*(y_max-y_min)+y_min
    xr = torch.full((N_side,1),x_max); yr = torch.rand(N_side,1)*(y_max-y_min)+y_min
    yb = torch.full((N_side,1),y_min); xb = torch.rand(N_side,1)*(x_max-x_min)+x_min
    yt = torch.full((N_side,1),y_max); xt = torch.rand(N_side,1)*(x_max-x_min)+x_min
    xbcs = torch.cat([xl,xr,xb,xt],dim=0)
    ybcs = torch.cat([yl,yr,yb,yt],dim=0)
    tbcs = torch.rand((4*N_side,1))
    xyt_bc = torch.cat([xbcs,ybcs,tbcs],dim=1).to(device)

    # --- Barrier interior BC: enforce psi=0 at barrier edges ---
    barrier_width = 0.2
    slit_half_h = 0.5
    N_bar = 2000
    yb_int_full = torch.rand(N_bar*5,1)*(y_max-y_min)+y_min
    mask_block = (torch.abs(yb_int_full - 1.0) >= slit_half_h) & (torch.abs(yb_int_full + 1.0) >= slit_half_h)
    yb_int = yb_int_full[mask_block][:N_bar].view(-1,1)
    xb_pos = torch.full((N_bar,1), barrier_width/2)
    xb_neg = torch.full((N_bar,1), -barrier_width/2)
    tb_int = torch.rand((2*N_bar,1))
    x_int = torch.cat([xb_pos, xb_neg], dim=0)
    y_int = torch.cat([yb_int, yb_int], dim=0)
    xyt_barrier = torch.cat([x_int, y_int, tb_int], dim=1).to(device)

    model = PINN(layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2000, gamma=0.9)
    scaler = torch.amp.GradScaler()

    print("Training double-slit PINN...")
    start=time.time()
    for ep in range(epochs+1):
        opt.zero_grad()
        with torch.amp.autocast(device_type=device.type):
            fr, fi = schrodinger_residual(model, xyt, V_double_slit)
            loss_pde = (fr**2 + fi**2).mean()
            pred0 = model(xyt0)
            loss_ic = ((pred0-psi0_val)**2).mean()
            pred_bc = model(xyt_bc)
            loss_bc = (pred_bc**2).mean()
            loss_bar = (model(xyt_barrier)**2).mean()
            loss = w_pde*loss_pde + w_ic*loss_ic + w_bc*loss_bc + w_bar*loss_bar
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if ep % 1000 == 0:
            print(f"Ep {ep:5d} | PDE={loss_pde:.2e} IC={loss_ic:.2e} BC={loss_bc:.2e} BAR={loss_bar:.2e} | Tot={loss:.2e} | {time.time()-start:.1f}s")

    # --- Visualization & GIF Generation ---
    print("✨ Generating PINN wavefunction GIF for double‐slit.")
    model.eval()
    nx, ny = 200, 100
    X, Y = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
    pts = np.stack((X, Y), axis=-1).reshape(-1, 2)
    tvec = np.linspace(0, 1, 150)
    frames = []
    for tk in tvec:
        outs = []
        for i in range(0, pts.shape[0], 4096):
            batch = pts[i:i+4096]
            t_in = np.full((batch.shape[0], 1), tk, dtype=np.float32)
            inp = torch.from_numpy(np.hstack([batch, t_in]).astype(np.float32)).to(device)
            with torch.no_grad():
                out = model(inp).cpu().numpy()
            outs.append(out)
        out = np.vstack(outs)
        psi_complex = out[:,0] + 1j * out[:,1]
        prob = np.abs(psi_complex)**2
        prob = prob.reshape(ny, nx)
        normed = prob / (prob.max() + 1e-8)
        img = (plt.cm.inferno(normed)[:,:,:3] * 255).astype(np.uint8)
        frames.append(img)
    fps = len(tvec) / 5.0
    imageio.mimsave("double_slit_pinn.gif", frames, fps=fps)
    print("✅ Saved GIF: double_slit_pinn.gif")

# --- Analytical double-slit simulation for baseline comparison ---
print("✨ Generating analytical double-slit pattern.")
def generate_analytic():
    # Domain and time vector for analytical pattern
    x_min, x_max = -5.0, 5.0
    y_min, y_max = -3.0, 3.0
    nx, ny = 200, 100
    tvec = np.linspace(0, 1, 150)
    # Grid setup
    X, Y = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
    # Wave parameters
    k = 5.0  # same momentum as PINN
    slit_y = [1.0, -1.0]
    # Sum point-source contributions from slits at x=0
    Z = np.zeros_like(X, dtype=np.complex64)
    for y0 in slit_y:
        r = np.sqrt((X - 0.0)**2 + (Y - y0)**2)
        # avoid division by zero
        r[r < 1e-6] = 1e-6
        Z += np.exp(1j * k * r) / r
    I = np.abs(Z)**2
    I = I / I.max()
    # Create a static GIF for direct visual comparison
    analytic_frames = []
    for _ in range(len(tvec)):
        img = (plt.cm.inferno(I)[:,:,:3] * 255).astype(np.uint8)
        analytic_frames.append(img)
    fps = len(tvec) / 5.0
    imageio.mimsave("analytic_double_slit.gif", analytic_frames, fps=fps)
    print("✅ Saved analytical pattern GIF: analytic_double_slit.gif")

generate_analytic()

if __name__=='__main__':
    train()
