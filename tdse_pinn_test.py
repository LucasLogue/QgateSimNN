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

# ── Residual for the 2D time‑dependent Schrödinger equation ──────────
def schrodinger_residual(model, xyt, Vfunc):
    # xyt: [N,3] tensor with (x,y,t)
    xyt = xyt.clone().detach().requires_grad_(True)
    psi = model(xyt)                       # [N,2] real & imag parts
    psi_r, psi_i = psi[:,0:1], psi[:,1:2]

    # first deriv
    grads = autograd.grad(psi.sum(), xyt, create_graph=True)[0]
    psi_rx, psi_ry, psi_rt = grads[:,0:1], grads[:,1:2], grads[:,2:3]

    grads_i = autograd.grad(psi_i.sum(), xyt, create_graph=True)[0]
    psi_it = grads_i[:,2:3]

    # ∇²ψ (laplacian)
    def lap(f):
        f_x = autograd.grad(f.sum(), xyt, create_graph=True)[0][:,0:1]
        f_y = autograd.grad(f.sum(), xyt, create_graph=True)[0][:,1:2]
        f_xx = autograd.grad(f_x.sum(), xyt, create_graph=True)[0][:,0:1]
        f_yy = autograd.grad(f_y.sum(), xyt, create_graph=True)[0][:,1:2]
        return f_xx + f_yy

    lap_r = lap(psi_r)
    lap_i = lap(psi_i)

    V = Vfunc(xyt[:,0:1], xyt[:,1:2])
    # i*h*d(psi)/dt = -h^2/2m * lap(psi) + V*psi
    # With h=1, m=1/2, this becomes: i*d(psi)/dt = -lap(psi) + V*psi
    # Let psi = u + iv.  i*(u_t + i*v_t) = -(u_xx+u_yy) - i*(v_xx+v_yy) + V*(u+iv)
    # -v_t + i*u_t = (-lap_u + V*u) + i*(-lap_v + V*v)
    # Real part: -v_t = -lap_u + V*u  =>  f_r = lap_u - V*u - v_t = 0
    # Imag part:  u_t = -lap_v + V*v  =>  f_i = u_t + lap_v - V*v = 0
    f_r =  lap_r - V * psi_r - psi_it
    f_i =  psi_it + lap_i - V * psi_i

    return f_r, f_i

# ── Double‑slit potential ───────────────────────────────────────────────
def V_double_slit(x, y):
    # barrier at |x|<0.05 except two slits at y=±0.5±0.1
    barrier = (torch.abs(x) < 0.1).float() * 1e4
    slit  = ((torch.abs(y-0.5)<0.2) | (torch.abs(y+0.5)<0.2)).float()
    return barrier * (1 - slit)

# ── Training loop ─────────────────────────────────────────────────────
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    # hyperparams GO TIME FOR CUDA
    N_colloc = 30000
    N_ic    = 7500
    N_bc = 7500
    layers  = [3,128,128,128,128,128,2]
    lr      = 1e-4
    epochs  = 30000

    #proper weight values for the losses' so it doesnt fucking break my lazy ass model
    w_pde_max = 1.0
    w_ic = 100.0
    w_bc = 10.0

    # collocation points (x∈[-5,5], y∈[-3,3], t∈[0,1])
    x = (torch.rand(N_colloc,1)*10 - 5)
    y = (torch.rand(N_colloc,1)*6  - 3)
    t = torch.rand(N_colloc,1)
    xyt = torch.cat([x,y,t], dim=1).to(device)

    # initial condition at t=0: Gaussian wavepacket centered at x=-2
    x0 = (torch.rand(N_ic,1)*10 - 5)
    y0 = (torch.rand(N_ic,1)*6  - 3)
    t0 = torch.zeros_like(x0)
    xyt0 = torch.cat([x0,y0,t0], dim=1).to(device)

    #initial wavepacket goes to the right
    psi0 = torch.exp(-((x0+2.5)**2 + y0**2)/0.8) * torch.exp(1j*12*x0)
    psi0_val = torch.view_as_real(psi0).to(device)  # [N_ic,2]

    #Boundary conditions
    N_side = N_bc // 4
    x_left  = torch.full((N_side,1), -5.0); y_left  = torch.rand((N_side,1))*6.0 - 3.0
    x_right = torch.full((N_side,1), +5.0); y_right = torch.rand((N_side,1))*6.0 - 3.0
    y_bot   = torch.full((N_side,1), -3.0); x_bot   = torch.rand((N_side,1))*10.0 - 5.0
    y_top   = torch.full((N_side,1), +3.0); x_top   = torch.rand((N_side,1))*10.0 - 5.0
    x_bc = torch.cat([x_left, x_right, x_bot, x_top], dim=0)
    y_bc = torch.cat([y_left, y_right, y_bot, y_top], dim=0)
    t_bc = torch.rand((4*N_side,1))
    xyt_bc = torch.cat([x_bc, y_bc, t_bc], dim=1).to(device)

    # model + optimizer
    model = PINN(layers).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=2500, gamma=0.9)

    start_time = time.time()
    for ep in range(epochs+1):
        opt.zero_grad()
        w_pde = w_pde_max * min(1.0, ep / (epochs/2.0))
        fr, fi = schrodinger_residual(model, xyt, V_double_slit)
        loss_pde = (fr**2 + fi**2).mean()
        psi_pred0 = model(xyt0)
        loss_ic  = ((psi_pred0 - psi0_val)**2).mean()
        #compute BC loss: force psi = 0 on edges
        psi_bc = model(xyt_bc)
        loss_bc = (psi_bc**2).mean()
        #Final loss
        loss = w_pde*loss_pde + w_ic*loss_ic + w_bc*loss_bc
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        if ep % 1000 == 0:
            print(f"Epoch {ep:4d} PDE={loss_pde:.2e} IC={loss_ic:.2e} BC={loss_bc:.2e} Total={loss:.2e}")


    #---------------MOVING PART!!
    #print model eval
    model.eval()

    #spatial grid, 20 samples per unit [-5, 5] for x, [-3, 3] for Y
    X, Y = np.meshgrid(np.linspace(-5, 5, 250), np.linspace(-3, 3, 150))
    ptvect = np.stack((X,Y), axis=-1).reshape(-1, 2)
    timevec = np.linspace(0, 1, 150)   #we only trained for one second so far so no more than 1 :(
     # We use a numpy version of the potential function to create an image
    def V_double_slit_np(x, y):
        V0 = 1.0 # Just need the shape, not the full potential height
        barrier_width = 0.1; slit_pos = 0.5; slit_width = 0.2; smoothness = 50.0
        barrier_shape = 0.5 * (np.tanh(smoothness * (x + barrier_width)) - np.tanh(smoothness * (x - barrier_width)))
        slit1 = 0.5 * (np.tanh(smoothness * (y - (slit_pos - slit_width))) - np.tanh(smoothness * (y - (slit_pos + slit_width))))
        slit2 = 0.5 * (np.tanh(smoothness * (y - (-slit_pos - slit_width))) - np.tanh(smoothness * (y - (-slit_pos + slit_width))))
        slits_shape = slit1 + slit2
        return V0 * barrier_shape * (1 - slits_shape)

    V_np = V_double_slit_np(X, Y)
    # Create a background image for the barrier. We'll make it light gray.
    #barrier_img = plt.cm.gray(V_np)[:,:,:3] * 0.5 
    cyan_color = np.array([0.0, 1.0, 1.0])
    #barrier_img = (barrier_img * 255).astype(np.uint8)
    barrier_img = V_np[:, :, np.newaxis] * cyan_color
    #precomputing the max works, but i'm not sure how accurate, switching to normalized frame by frame
    # g_max_prob = 0.0
    # for tk in timevec:
    #     tcol = np.full((ptvect.shape[0],1), tk, dtype=np.float32)
    #     inp  = torch.from_numpy(np.hstack([ptvect, tcol])).float().to(device)
    #     with torch.no_grad():
    #         out = model(inp).cpu().numpy()
    #     psi  = out[:,0] + 1j*out[:,1]
    #     prob = np.abs(psi)**2
    #     if prob.max() > g_max_prob:
    #         g_max_prob = prob.max()
    #     #g_max_prob = max(g_max_prob, prob.max())

    # if g_max_prob == 0:
    #     print("globamax is fucked!")
    #     g_max_prob = 1.0 # Avoid division by zero
    # print(f"✨ Global max probability for normalization: {g_max_prob:.4f}")

    #properly draw the barrier for our eyes============================
    ### NEW BLOCK ### - Create a visual representation of the barrier potential
   

    frames = []
    for tk in timevec:
        tket = np.full((ptvect.shape[0],1), tk, dtype=np.float32)
        inp = torch.from_numpy(np.hstack([ptvect, tket]).astype(np.float32)).to(device)
        with torch.no_grad():
            out = model(inp).cpu().numpy()
        psi = out[:,0] + 1j*out[:,1]
        prob=(np.abs(psi)**2).reshape(Y.shape) #Now we got PSI for each frame

        #Normalize the prob might fix the shit idk
        # norm_prob = np.clip(prob / g_max_prob, 0, 1)
        frame_max = prob.max()
        norm_prob = prob / (frame_max + 1e-8)
        #img = (255*(prob/g_max_prob())).astype(np.uint8) BROKEN ASS SHIT
        #CYAN BARRIER TIME===========================================================================
        cyan_color = np.array([0.0, 1.0, 1.0])
        barrier_img = V_np[:, :, np.newaxis] * cyan_color 
        wave_color = plt.cm.inferno(norm_prob)[:,:,:3]

        wave_alpha = norm_prob[:,:,np.newaxis] 

        blended_img = barrier_img + (1 - V_np[:, :, np.newaxis]) * wave_alpha * wave_color

        img = (np.clip(blended_img, 0, 1) * 255).astype(np.uint8)
        frames.append(img)
    #Set duration and FPS
    
    duration = 5.0 #in seconds baby
    fps = len(frames)/duration
    print("⏱ Generating", len(frames), "frames, expected", len(timevec))
    #DEBUG DISPLAY
    plt.figure(figsize=(8, 8*Y.shape[0]/Y.shape[1]))
    plt.imshow(frames[-1], origin='lower', extent=[-5,5,-3,3])
    plt.title(f'|ψ|² at t=1.0 (GPU TRAINED BABY)')
    plt.xlabel('x'); plt.ylabel('y')
    plt.tight_layout()
    plt.show()

    #TRY TO SAVE PLASEAW OWRK
    imageio.mimsave("BIGPEENER.gif", frames, fps=fps)
    print(f"🔹 Saved {len(frames)} frames over {duration}s → fps={fps:.1f}")




if __name__=='__main__':
    train()
