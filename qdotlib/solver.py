#needs to be adjusted for PINN
import torch
import numpy as np
from .qdotconfig import get_potential

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- 3D Split-Step Solver using device: {device} ---")

def init_domain(Nx=64, Ny=64, Nz=64, Lx=10.0, Ly=10.0, Lz=10.0, dt=0.005, potential_cfg={}):
    """
    Initializes the simulation grid and potential V
    """
    #Step/Grid spacing for 3 dimensions
    dx, dy, dz = Lx/Nx, Ly/Ny, Lz/Nz
    
    #create grid vectors
    x = np.linspace(-Lx/2, Lx/2, Nx)
    y = np.linspace(-Ly/2, Ly/2, Ny)
    z = np.linspace(-Lz/2, Lz/2, Nz)
    
    #numpy meshgrid -> torch grid
    X_np, Y_np, Z_np = np.meshgrid(x, y, z, indexing='ij')
    X = torch.from_numpy(X_np).to(device)
    Y = torch.from_numpy(Y_np).to(device)
    Z = torch.from_numpy(Z_np).to(device)

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #will be updated for PINN
    # 3D Fourier space coordinates
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    kz = 2 * np.pi * np.fft.fftfreq(Nz, d=dz)
    KX_np, KY_np, KZ_np = np.meshgrid(kx, ky, kz, indexing='ij')
    K2 = torch.from_numpy(KX_np**2 + KY_np**2 + KZ_np**2).to(device)


    halfkprop = torch.exp(-0.5j * K2 * dt)

    cfgname = potential_cfg.get('name', 'ideal')
    params = potential_cfg.get('params', {})
    V = get_potential(cfgname=cfgname, X=X, Y=Y, Z=Z, params=params)

    return X, Y, Z, dx, dy, dz, V.to(torch.complex64), halfkprop, K2

#Normal run_sim definition
#def run_sim(initial_psi, V, halfkprop, K2, dt, Nt, drive_pulse, control_shape):

#jit run_sim def
def run_sim(
    initial_psi: torch.Tensor,
    V: torch.Tensor,
    halfkprop: torch.Tensor,
    K2: torch.Tensor,
    dt: float,
    Nt: int,
    drive_pulse: torch.Tensor,
    control_shape: torch.Tensor
) -> torch.Tensor:
    """
    Runs the 3D time-evolution simulation
    """
    psi = initial_psi.clone()
    for n in range(Nt):
        # Kinetic half-step using 3D FFT
        psi_hat = torch.fft.fftn(psi) # 3D FFT
        psi_hat *= halfkprop
        psi = torch.fft.ifftn(psi_hat) # Inverse 3D FFT
        # Potential full-step
        V_interaction = -control_shape * drive_pulse[n]
        V_pulsed = V + V_interaction
        psi *= torch.exp(-1j * V_pulsed * dt)

        # Kinetic half-step again
        psi_hat = torch.fft.fftn(psi)
        psi_hat *= halfkprop
        psi = torch.fft.ifftn(psi_hat)
        
    return psi

#This should be fine
try:
    run_sim_ts = torch.jit.script(run_sim)
    RUN_SIM = run_sim_ts
    print("TorchScripted the Simulation")
except Exception as e:
    print("TorchScript failed, falling back → eager mode:", e)
    RUN_SIM = run_sim