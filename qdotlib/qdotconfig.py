import torch
import numpy as np
from scipy.ndimage import gaussian_filter
import inspect

def _filtered_kwargs(func, kw):
    """Return a copy of dict `kw` that only keeps keys accepted by `func`."""
    sig = inspect.signature(func)
    allowed = sig.parameters.keys()
    return {k: v for k, v in kw.items() if k in allowed}

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

def doublewell_potential(X, Y, Z, omega=1.0, delta=2.0):
    """
    Potential field for double well with two electrons 
    """
    left = 0.5 * omega**2 * ((X - delta) ** 2 + Y ** 2 + Z ** 2)
    right = 0.5 * omega**2 * ((X + delta) ** 2 + Y ** 2 + Z ** 2)
    return torch.minimum(left, right)

def get_potential(cfgname, X, Y, Z, params=None):
    """
    Returns the potential for selected electron configuration
    """
    if params is None:
        params = {}

    if cfgname == "ideal":
        f = harmonic_potential
    elif cfgname == "disordered":
        f = disordered_harmonic_potential
    elif cfgname == "doublewell":
        f = doublewell_potential
    else:
        raise ValueError(f"Unknown cfgname '{cfgname}'")

    
    return f(X, Y, Z, **_filtered_kwargs(f, params))
    # if cfgname == "ideal":
    #     return harmonic_potential(X, Y, Z, **params)
    # elif cfgname == "disordered":
    #     #Debugging logic, commented out for efficiency
    #     # omega= params.get("omega", 1.0)
    #     # V_ideal = harmonic_potential(X, Y, Z, omega=omega)
    #     # V_disord = disordered_harmonic_potential(X, Y, Z, **params)
    #     # D = (V_disord- V_ideal).flatten().abs().mean().item()
    #     # print("SANITY CHECK! ", D)
    #     return disordered_harmonic_potential(X, Y, Z, **params)
    # elif cfgname == "doublewell":
    #     return doublewell_potential(X, Y, Z, **params)
    # else:
    #     raise ValueError("bruh we dont have that")
    