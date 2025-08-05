import torch
import numpy as np

def harmonic_potential(X, Y, Z, omega=1.0):
    """
    Potential field for ideal harmonic
    """
    V = 0.5 * omega**2 * (X**2 + Y**2 + Z**2)
    return V

def get_potential(cfgname, X, Y, Z, params={}):
    """
    Returns the potential for selected electron configuration
    """
    if cfgname == "ideal":
        return harmonic_potential(X, Y, Z, **params)
    else:
        raise ValueError("bruh we dont have that")
    