"""
====================================================================
PINN Pipeline — *external* wrapper for the ultra‑fragile TDSE‑PINN
--------------------------------------------------------------------
Keeps **all physics code 100 % untouched**.  This module only:
    1.  Collects the user‑chosen *trap* / *gate* / grid params.
    2.  Writes a tiny JSON containing **global‑variable overrides**.
    3.  Spawns a fresh Python process that reads those overrides and
        then executes the original `tdse_pinn.py` via `runpy.run_path()`.

Nothing inside the PINN script itself needs to be edited.  If you later
*want* to let the script read more params natively, just add a single
line at its top:  `from pinn_pipeline import apply_overrides; apply_overrides(globals())`
—but that’s optional.  The default launch path below does **zero** edits.
====================================================================
"""

import json
import subprocess
import tempfile
import sys
import runpy
import os
from dataclasses import dataclass, asdict
from typing import Literal, Dict, Any

# ------------------------------------------------------------------
# 1) Lightweight config objects so RL code is pleasant to read
# ------------------------------------------------------------------
@dataclass
class TrapCfg:
    """Physics trap configuration (harmonic, disordered, double‑well …)."""
    name: Literal["ideal", "disordered", "doublewell"]
    params: Dict[str, float]

@dataclass
class GateCfg:
    """Logical gate target state (NOT, HADAMARD, BELL, CNOT)."""
    name: Literal["NOT", "HADAMARD", "BELL", "CNOT"]
    params: Dict[str, float] | None = None

# ------------------------------------------------------------------
# 2) Core helper to *spawn* the fragile script in a sandboxed process
# ------------------------------------------------------------------

def _write_cfg_and_run(run_cfg: dict):
    """Dump JSON -> spawn new Python proc -> run core PINN with overrides."""
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
        json.dump(run_cfg, f)
        cfg_path = f.name

    # Re‑invoke *this* file as a CLI shim; see the __main__ block below.
    cmd = [sys.executable, os.path.abspath(__file__), cfg_path]
    subprocess.run(cmd, check=False)

# ------------------------------------------------------------------
# 3) Public API:  train_pinn_for_cfg()
# ------------------------------------------------------------------

def train_pinn_for_cfg(
    trap_cfg: TrapCfg,
    gate_cfg: GateCfg,
    *,
    grid_size: int = 64,
    n_collocation: int | None = None,
    n_initial: int | None = None,
    n_norm: int | None = None,
    L: float | None = None,
    core_script: str = "uberpwnage/pinn_core/tdse_pinn.py",
    extra_overrides: Dict[str, Any] | None = None,
):
    """Spawn one training run for the given (trap, gate) combo.

    Parameters
    ----------
    trap_cfg / gate_cfg :   Domain‑specific dataclasses.
    grid_size           :   Not used by the core script yet, but future‑proof.
    n_collocation …     :   Optional overrides for the global constants at the
                             *top* of the fragile script (safe to tweak).
    L                   :   Override for the cutoff length in the script.
    core_script         :   Path to the unchanged PINN file.
    extra_overrides     :   Any other `{global_var: new_value}` you want.
    """
    overrides: Dict[str, Any] = {}
    if L is not None:
        overrides["L"] = L
    if n_collocation is not None:
        overrides["N_COLLOCATION"] = n_collocation
    if n_initial is not None:
        overrides["N_INITIAL"] = n_initial
    if n_norm is not None:
        overrides["N_NORM"] = n_norm
    if extra_overrides:
        overrides.update(extra_overrides)

    run_cfg = {
        "script_path": core_script,          # where the fragile beast lives
        "globals": overrides,               # things we mutate up‑front
        "trap_cfg": asdict(trap_cfg),       # kept for provenance / logs
        "gate_cfg": asdict(gate_cfg),
        "grid_size": grid_size,
    }
    _write_cfg_and_run(run_cfg)

# ------------------------------------------------------------------
# 4) __main__ acts as the *launcher* when we get called by _write_cfg_and_run
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Case 1:  We were invoked with a JSON path -> behave as a shim runner.
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        cfg_path = sys.argv[1]
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        overrides = cfg.get("globals", {})

        # Prepare the initial globals for the fragile script
        exec_globals: Dict[str, Any] = {"__name__": "__main__"}
        exec_globals.update(overrides)

        # Finally run the untouched TDSE‑PINN *in this interpreter*.
        runpy.run_path(cfg["script_path"], run_name="__main__", init_globals=exec_globals)
        sys.exit(0)

    # ------------------------------------------------------------------
    # Case 2:  User ran `python pinn_pipeline.py --trap ideal --gate NOT …`
    # ------------------------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description="Launch the fragile TDSE‑PINN with runtime overrides.")
    parser.add_argument("--trap", required=True, choices=["ideal", "disordered", "doublewell"],
                        help="Trap configuration name.")
    parser.add_argument("--gate", required=True, choices=["NOT", "HADAMARD", "BELL", "CNOT"],
                        help="Gate target state.")
    parser.add_argument("--grid_size", type=int, default=64,
                        help="(Optional) Grid resolution hint for future use.")
    parser.add_argument("--L", type=float, default=None,
                        help="Override the cutoff length L in the PINN script.")
    args = parser.parse_args()

    # Fire off a training run with *zero* other overrides
    train_pinn_for_cfg(
        TrapCfg(args.trap, {}),
        GateCfg(args.gate, {}),
        grid_size=args.grid_size,
        L=args.L,
    )
