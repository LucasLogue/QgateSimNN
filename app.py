import streamlit as st
import numpy as np
import torch
import time
import driver_cli  # reuse domain builder and optimiser
import qdotlib

# ---------------- Sidebar: trap first -------------------------------
st.sidebar.title("Optimiser Controls")
trap = st.sidebar.selectbox("Trap (potential landscape)",
                            ["ideal", "disordered", "doublewell"],
                            index=0)
omega = st.sidebar.number_input("ω (harmonic freq)", 0.1, 5.0, 1.0, 0.1)

delta = None
if trap == "doublewell":
    delta = st.sidebar.slider("Δ separation (a.u.)", 1.0, 5.0, 2.0, 0.1)

# ---------------- Gate selection (filtered) -------------------------
if trap == "doublewell":
    gate_options = ["NOT", "HADAMARD", "BELL", "CNOT"]
else:
    gate_options = ["NOT", "HADAMARD"]

gate = st.sidebar.selectbox("Gate / Target", gate_options)

# ---------------- Optimiser hyper‑params ---------------------------
maxiter = st.sidebar.number_input("CMA‑ES max iterations",
                                 min_value=5, max_value=200, value=20,
                                 help="How many generations CMA‑ES runs. More ⇒ better fidelity, slower run.")
popsize = st.sidebar.number_input("Population size",
                                 min_value=4, max_value=64, value=10,
                                 help="Number of candidate pulses per generation. Larger pop explores more but costs time.")

run_btn = st.sidebar.button("Optimise!")

# ---------------- Main panel ---------------------------------------
st.title("Quantum‑Dot Gate Optimiser")
st.write("Pick a trap, choose a gate, tune hyper‑parameters, then click **Optimise!**.  All heavy lifting happens on the GPU.")

if run_btn:
    st.info("Building domain & target state …")

    pot_params = {"omega": omega}
    if trap == "doublewell":
        pot_params["delta"] = delta

    GRID, DT, STEPS = 64, 0.005, 500
    dom = driver_cli.build_domain(GRID, DT, trap, pot_params)
    dom["initial_psi"] = qdotlib.get_ground(dom["X"], dom["Y"], dom["Z"],
                                            omega, dom["volume"])
    dom["target_psi"]  = driver_cli.make_target(gate, dom, pot_params)

    # baseline overlaps
    fid0 = qdotlib.calc_fidelity(dom["initial_psi"], dom["target_psi"], dom["volume"])
    st.write(f"Baseline overlap (no pulse): {fid0:.3f}")

    t_vec = torch.arange(STEPS, device="cuda") * DT
    init_guess = [10.0, omega, 0.0, (STEPS*DT)/2, (STEPS*DT)/4]
    bounds = [[-50,0.1,-np.pi,0,0.1], [50,omega*2,np.pi,STEPS*DT,STEPS*DT]]
    cma_stds = [10.0,0.5,np.pi/4,(STEPS*DT)/4,(STEPS*DT)/4]

    st.info("Running CMA‑ES … this may take a minute.")
    start = time.time()
    best_fid, best_params = driver_cli.optimise(dom, t_vec, maxiter, popsize,
                                                init_guess, bounds, cma_stds)
    dur = time.time() - start

    st.success(f"Finished in {dur:.1f} s — Best fidelity {best_fid:.4f}")
    labels = ["Amplitude", "Frequency", "Phase", "Pulse center t", "Pulse width"]
    st.json({k: float(v) for k, v in zip(labels, best_params)})
