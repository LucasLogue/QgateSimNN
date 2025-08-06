# fft_ref_1d.py  ------------------------------------------------------
import numpy as np
import torch

# ---------------- physics / pulse params ----------------------------
OMEGA      = 1.0
DT         = 0.005          # time step
NT         = 600            # number of steps  (T = NT·DT = 3.0)
NX, LBOX   = 1024, 5.0      # grid points, half-box size  (domain [-L,+L])

P_AMP      = -6.3
P_FREQ     = 1.13
P_PHASE    = 0.64
P_T0       = 0.0
P_WIDTH    = 1.7

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- helper: drive pulse E(t) --------------------------
def make_drive(dt, nt, amp, freq, phase, t0, width, device):
    t = torch.arange(nt, device=device) * dt
    env = torch.exp(-((t - t0) / width) ** 2)
    return amp * env * torch.sin(freq * t + phase)

drive = make_drive(DT, NT, P_AMP, P_FREQ, P_PHASE, P_T0, P_WIDTH, DEVICE)

# ---------------- build 1-D grid ------------------------------------
x  = torch.linspace(-LBOX, LBOX, NX, device=DEVICE)
dx = x[1] - x[0]
k  = 2 * np.pi * torch.fft.fftfreq(NX, float(dx)).to(DEVICE)   # momentum grid

V  = 0.5 * (OMEGA ** 2) * x**2                                 # harmonic
kprop = torch.exp(-0.5j * (k**2) * DT)                         # kinetic half-step

# ground state (analytic)  ψ₀(x) = (ω/π)^{1/4} e^{-½ ω x²}
psi = ((OMEGA / np.pi) ** 0.25 * torch.exp(-0.5 * OMEGA * x**2)).to(torch.complex64)

psi_snap0 = psi.clone()                    # store t = 0 probability density

# ---------------- split-step FFT propagation ------------------------
for n in range(NT):
    # kinetic half-step
    psi = torch.fft.ifft(torch.fft.fft(psi) * kprop)
    # potential + drive   (dipole operator is just x)
    psi *= torch.exp(-1j * (V - drive[n] * x) * DT)
    # kinetic half-step
    psi = torch.fft.ifft(torch.fft.fft(psi) * kprop)

psi_snapT = psi.clone()                    # t = NT·DT snapshot

# ---------------- save snapshots to disk ----------------------------
np.savez("fft_ref_snapshots.npz",
         x=x.cpu().numpy(),
         psi0=(psi_snap0.abs()**2).cpu().numpy(),
         psiT=(psi_snapT.abs()**2).cpu().numpy())
print("✅  FFT reference saved → fft_ref_snapshots.npz")
