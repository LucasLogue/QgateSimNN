import torch
import numpy as np
#To act as a library for torchified functions.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- TDSE Simulation using device: {device} ---")

def init_domain_torch(Nx=256, Ny=256, Lx=10.0, Ly=10.0, dt=0.005):
    """
    Initializes the simulation domain using PyTorch tensors.
    """
    dx, dy = Lx / Nx, Ly / Ny

    # Create grid coordinates as NumPy arrays first
    x = np.linspace(-Lx / 2, Lx / 2 - dx, Nx)
    y = np.linspace(-Ly / 2, Ly / 2 - dy, Ny)
    X_np, Y_np = np.meshgrid(x, y)

    # Convert NumPy arrays to PyTorch tensors and move them to the GPU
    X = torch.from_numpy(X_np).to(device)
    Y = torch.from_numpy(Y_np).to(device)

    # Fourier space coordinates (wave numbers)
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    KX_np, KY_np = np.meshgrid(kx, ky)
    K2_np = KX_np**2 + KY_np**2

    # Convert K-space grid to a tensor and move to GPU
    K2 = torch.from_numpy(K2_np).to(device)

    # Kinetic propagator (note the dtype=torch.complex64 for complex numbers)
    halfkprop = torch.exp(-0.25j * K2 * dt).to(torch.complex64)

    # Potential barrier
    V = torch.zeros_like(X)
    V0, bwidth, sheight = 1e3, 0.2, 0.25
    barmask = (torch.abs(X) < bwidth / 2)
    slitmask = (torch.abs(Y - 1.0) < sheight) | (torch.abs(Y + 1.0) < sheight)
    bmask = barmask & ~slitmask
    V[bmask] = V0

    return X, Y, dx, dy, V.to(torch.complex64), halfkprop, bmask.to(torch.complex64)


def propagate_torch(psi, V, halfkprop, dt, drive_val, control_shape):
    """
    Propagates the wavefunction for one time step on the GPU.
    """
    # Kinetic half-step (using torch.fft)
    psi_hat = torch.fft.fft2(psi)
    psi_hat *= halfkprop
    psi = torch.fft.ifft2(psi_hat)

    # Potential full-step
    V_interaction = -control_shape * drive_val
    V_pulsed = V + V_interaction
    psi *= torch.exp(-1j * V_pulsed * dt)

    # Kinetic half-step again
    psi_hat = torch.fft.fft2(psi)
    psi_hat *= halfkprop
    psi = torch.fft.ifft2(psi_hat)
    return psi


def reward_torch(psi, X, dx, dy):
    """
    Calculates the reward (transmission) on the GPU.
    """
    # torch.abs() for complex tensors gives the magnitude
    intensity = torch.abs(psi)**2
    # Calculate the sum of intensity where X > 0
    transmission = torch.sum(intensity[X > 0]) * dx * dy
    # .item() moves the final scalar value from the GPU back to the CPU
    return transmission.item()

if __name__ == '__main__':
    import time
    from imageio import mimsave
    import matplotlib.pyplot as plt
    
    print("\n--- Running Single Test Simulation ---")
    start_time = time.time()

    # --- Setup the simulation ---
    dt = 0.005
    Nt = 1000
    X, Y, dx, dy, V, halfkprop, bmask = init_domain_torch(dt=dt)

    # --- Initial wavepacket ---
    k0, sigma = 5.0, 0.5
    x0, y0 = -2.0, 0.0
    psi0 = (
        1 / (sigma * np.sqrt(np.pi))
        * torch.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
        * torch.exp(1j * k0 * X)
    ).to(torch.complex64)
    psi = psi0.clone()

    # --- Define a fixed control pulse for the test ---
    dt_vec = np.arange(Nt) * dt
    control_amp = 100.0
    control_freq = 2 * np.pi * 2.0
    control_phase = 0.0

    # Define the shape of the pulse
    pulse_center_t = 1.0 # Time (in simulation units) when the pulse peaks
    pulse_width = 0.2    # Duration of the pulse
    
    # Create the Gaussian envelope
    envelope = np.exp(-((dt_vec - pulse_center_t) / pulse_width)**2)

    drive = control_amp *envelope*np.sin(control_freq * dt_vec + control_phase)
    control_shape = X * bmask

    # --- Run the simulation and save frames for a GIF ---
    frames = []
    for n in range(Nt):
        psi = propagate_torch(psi, V, halfkprop, dt, drive[n], control_shape)

        # Capture a frame every 20 steps
        if n % (Nt//50) == 0:
            # Move data to CPU and convert to NumPy for plotting
            intensity_np = torch.abs(psi).cpu().numpy()**2
            norm = intensity_np / intensity_np.max()
            # Use a colormap to create an RGB image
            frame_rgb = (plt.cm.inferno(norm)[:, :, :3] * 255).astype(np.uint8)
            frames.append(frame_rgb)

    # --- Calculate final results ---
    final_transmission = reward_torch(psi, X, dx, dy)
    duration = time.time() - start_time

    print(f"\nSimulation finished in {duration:.2f} seconds.")
    print(f"Final Transmission: {final_transmission:.5f}")

    # --- Save the GIF ---
    mimsave("test_simulation.gif", frames, fps=25)
    print("Saved simulation as test_simulation.gif")