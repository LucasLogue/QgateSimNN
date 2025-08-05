import torch
import numpy as np
from scipy.ndimage import gaussian_filter

def harmonic_potential(X, Y, Z, omega=1.0):
    """
    Potential field for ideal harmonic
    """
    V = 0.5 * omega**2 * (X**2 + Y**2 + Z**2)
    return V

def disordered_harmonic_potential(X, Y, Z, omega: float = 1.0, noise_amp: float = 0.1, corr_len: float = 10.0):
    V0 = 0.5 * omega**2 * (X**2 + Y**2 + Z**2) #base harmonic
    noise = np.random.randn(*X.cpu().shape) #create noise
    dx = float((X[1,0,0] - X[0,0,0]))  
    sigma_pts = corr_len / dx
    noise_smooth = gaussian_filter(noise, sigma=sigma_pts)
    noise_t = torch.from_numpy(noise_smooth).to(X.device).to(X.dtype)

    return V0+noise_amp*noise_t

def get_potential(cfgname, X, Y, Z, params={}):
    """
    Returns the potential for selected electron configuration
    """
    if cfgname == "ideal":
        return harmonic_potential(X, Y, Z, **params)
    elif cfgname == "disordered":
        return disordered_harmonic_potential(X, Y, Z, **params)
    else:
        raise ValueError("bruh we dont have that")
    