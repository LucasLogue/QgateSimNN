import numpy as np
import torch
import cma
import qdotlib
from pathlib import Path
import matplotlib.pyplot as plt

def build_domain(grid_size, dt, trap_name, trap_params):
    domain = {}
    time_steps = 500
    domain["t"] = torch.arange(time_steps, device="cuda") * dt

    # Setup the trap potential
    pot_cfg = {"name": trap_name, "params": trap_params.copy()}
    X, Y, Z, dx, dy, dz, V, halfkprop, K2 = qdotlib.init_domain(
        Nx=grid_size, Ny=grid_size, Nz=grid_size, dt=dt, potential_cfg=trap_params
    )

    domain.update({
        "volume": dx * dy * dz,
        "X": X, "Y": Y, "Z": Z, "V": V, "halfk": halfkprop,
        "K2": K2, "dx": dx, "dy": dy
    })

    return domain