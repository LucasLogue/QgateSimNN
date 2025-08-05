# qdot_config.py
# This script is a library for generating different potential energy landscapes
# for a single-electron quantum dot.

import torch
import numpy as np

# --- Potential Generation Functions ---

def get_harmonic_potential(X, Y, omega=1.0):
    """
    Generates a perfect, symmetrical, harmonic potential (a "bowl").
    This represents an ideal, single quantum dot.

    Args:
        X (torch.Tensor): Meshgrid of X coordinates.
        Y (torch.Tensor): Meshgrid of Y coordinates.
        omega (float): The trapping frequency, controlling the steepness of the trap.

    Returns:
        torch.Tensor: The potential energy V(x, y).
    """
    print("Generating ideal harmonic potential.")
    V = 0.5 * omega**2 * (X**2 + Y**2)
    return V

def get_disordered_potential(X, Y, omega=1.0, disorder_strength=0.1, correlation_length=20):
    """
    Generates a realistic potential by adding smooth, random noise to an
    ideal harmonic potential. This mimics imperfections in a real device.

    Args:
        X (torch.Tensor): Meshgrid of X coordinates.
        Y (torch.Tensor): Meshgrid of Y coordinates.
        omega (float): The base trapping frequency.
        disorder_strength (float): The amplitude of the random potential fluctuations.
        correlation_length (int): Controls the "smoothness" or "bumpiness" of the disorder.
                                  Larger values mean smoother disorder.

    Returns:
        torch.Tensor: The potential energy V(x, y) with disorder.
    """
    print("Generating disordered potential.")
    # 1. Start with the ideal harmonic potential
    V_ideal = get_harmonic_potential(X, Y, omega)
    device = X.device
    Nx, Ny = X.shape

    # 2. Create a smooth random noise field
    # Create random noise in Fourier space
    noise_k = torch.randn(Nx, Ny, dtype=torch.complex64, device=device)

    # Create a low-pass filter to make the noise smooth
    kx = torch.fft.fftfreq(Nx, d=1).to(device)
    ky = torch.fft.fftfreq(Ny, d=1).to(device)
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')
    K_dist = torch.sqrt(KX**2 + KY**2)
    
    # The filter removes high-frequency (sharp) components
    low_pass_filter = torch.exp(-(K_dist * correlation_length)**2)
    
    # Apply the filter and transform back to real space
    filtered_noise_k = noise_k * low_pass_filter
    V_disorder_unnormalized = torch.fft.ifft2(filtered_noise_k).real

    # 3. Normalize and scale the disorder
    # Add a small epsilon to prevent division by zero if the noise is flat
    std_dev = torch.std(V_disorder_unnormalized)
    V_disorder = disorder_strength * (V_disorder_unnormalized / (std_dev + 1e-10))
    
    # 4. Add the disorder to the ideal potential and return the result
    return V_ideal + V_disorder

# --- Main "Factory" Function ---

def get_potential(config_name, X, Y, params={}):
    """
    A factory function that returns a potential based on a configuration name.
    This is the function your main solver will call.

    Args:
        config_name (str): The name of the desired configuration ("ideal" or "disordered").
        X (torch.Tensor): Meshgrid of X coordinates.
        Y (torch.Tensor): Meshgrid of Y coordinates.
        params (dict): A dictionary of parameters for the potential function.

    Returns:
        torch.Tensor: The requested potential energy V(x, y).
    """
    if config_name == "ideal":
        return get_harmonic_potential(X, Y, **params)
    elif config_name == "disordered":
        return get_disordered_potential(X, Y, **params)
    else:
        raise ValueError(f"Unknown potential configuration: {config_name}")

# -----------------------------------------------------------------------------
# MAIN EXECUTION BLOCK FOR TESTING AND VISUALIZATION
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # --- Setup a sample grid for plotting ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Nx, Ny = 256, 256
    Lx, Ly = 10.0, 10.0
    dx, dy = Lx / Nx, Ly / Ny
    x = np.linspace(-Lx / 2, Lx / 2 - dx, Nx)
    y = np.linspace(-Ly / 2, Ly / 2 - dy, Ny)
    X_np, Y_np = np.meshgrid(x, y)
    X = torch.from_numpy(X_np).to(device)
    Y = torch.from_numpy(Y_np).to(device)

    # --- Generate the two potentials ---
    V_ideal = get_potential("ideal", X, Y, params={'omega': 1.0})
    V_disordered = get_potential("disordered", X, Y, params={'omega': 1.0, 'disorder_strength': 0.2})

    # --- Plot them side-by-side for comparison ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Quantum Dot Potential Configurations", fontsize=16)

    # Plot Ideal Potential
    im1 = ax1.imshow(V_ideal.cpu().numpy(), extent=[-Lx/2, Lx/2, -Ly/2, Ly/2], cmap='viridis', origin='lower')
    ax1.set_title("1. Ideal Harmonic Trap")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    fig.colorbar(im1, ax=ax1, label="Potential Energy")

    # Plot Disordered Potential
    im2 = ax2.imshow(V_disordered.cpu().numpy(), extent=[-Lx/2, Lx/2, -Ly/2, Ly/2], cmap='viridis', origin='lower')
    ax2.set_title("3. Disordered Potential")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    fig.colorbar(im2, ax=ax2, label="Potential Energy")

    plt.tight_layout()
    plt.show()
