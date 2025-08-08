"""
staydetermined.py, launcher for youarefilledwithdetermination.py
============================================================================
This file **never** edits physics logic.  Instead it:
1. Accepts a single CLI arg: path to a JSON file created by `pinn_pipeline.py`.
2. Reads the JSON → extracts `script_path` (defaults to ballzach.py) and a
   `globals` dict containing any top‑level constant tweaks (e.g. L, N_COLLOCATION…).
3. Spawns the fragile script inside *this* interpreter via `runpy.run_path()`,
   pre‑seeding its global namespace with those overrides.

Usage from pipeline (auto):
    python determination.py /tmp/overrides_abc.json
Manual poke:
    python determination.py               # runs ballzach with stock globals
    python determination.py --L 12 --N_COLLOCATION 8192   # quick test
"""

from __future__ import annotations
import sys, json, os, runpy, argparse
from pathlib import Path
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Helper: load overrides & launch the fragile script
# ---------------------------------------------------------------------------

def _launch(script_path: str | os.PathLike, overrides: Dict[str, Any]):
    # Prepare initial globals dict for runpy
    init_globals: Dict[str, Any] = {"__name__": "__main__"}
    init_globals.update(overrides)

    # Absolute path resolution
    script_path = str(Path(script_path).expanduser().resolve())

    print(f"[determination] ➜ Launching '{script_path}' with overrides: {list(overrides.keys())}")
    runpy.run_path(script_path, run_name="__main__", init_globals=init_globals)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # ---------------------------------------------------------------------
    # Case 1: invoked by pinn_pipeline with a JSON config path
    # ---------------------------------------------------------------------
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json") and Path(sys.argv[1]).is_file():
        cfg_path = Path(sys.argv[1])
        with cfg_path.open("r") as f:
            cfg = json.load(f)
        script_path = cfg.get("script_path") or Path(__file__).with_name("ballzach.py")
        overrides = cfg.get("globals", {})
        _launch(script_path, overrides)
        sys.exit(0)

    # ---------------------------------------------------------------------
    # Case 2: manual dev / debug usage via CLI flags
    # ---------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Run the ballzach PINN with optional global overrides.")
    parser.add_argument("--script_path", default=Path(__file__).with_name("ballzach.py"),
                        help="Path to the unmodified PINN script (default: ballzach.py in same folder).")
    parser.add_argument("--override", nargs=2, action="append", metavar=("NAME", "VALUE"), default=[],
                        help="Set a global constant before execution, e.g. --override L 12.0")
    args = parser.parse_args()

    # Build overrides dict (attempt to cast numbers)
    overrides: Dict[str, Any] = {}
    for name, val in args.override:
        try:
            overrides[name] = int(val)
        except ValueError:
            try:
                overrides[name] = float(val)
            except ValueError:
                overrides[name] = val  # leave as string

    _launch(args.script_path, overrides)
