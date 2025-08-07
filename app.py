import streamlit as st
import os
import numpy as np
import torch
import time
import driver_cli2 as driver_cli
import qdotlib

st.set_page_config(page_title="Quantum Dot Gate Optimiser", layout="wide")

# ---------------- Header ----------------
st.title("Quantum-Dot Gate Optimiser")
st.caption("Pick a trap, choose a gate, tune hyper-parameters, then click **Optimise!**. All heavy lifting happens on the GPU.")

# ---------------- Horizontal Layout ----------------
# ---------------- Trap & Gate Selection ----------------
col_left, col_right = st.columns([2, 1])

with col_left:
    col_trap, col_gate = st.columns(2)

    with col_trap:
        trap = st.selectbox("Trap landscape", ["ideal", "disordered", "doublewell"], index=0)

    with col_gate:
        gate_options = ["NOT", "HADAMARD"]
        if trap == "doublewell":
            gate_options += ["BELL", "CNOT"]
        gate = st.selectbox("Target Gate", gate_options)

    omega = st.slider("ω (harmonic frequency)", 0.1, 5.0, 1.0, 0.1)
    
    delta = None
    if trap == "doublewell":
        delta = st.slider("Δ (Delta) separation", 1.0, 5.0, 2.0, 0.1)

    # --- ✨ NEW: Conditional sliders for the disordered trap ---
    noise_amp = None
    corr_len = None
    if trap == "disordered":
        st.markdown("##### Disordered Trap Settings")
        noise_amp = st.slider("Noise Amplitude", 0.0, 2.0, 1.0, 0.1, help="Controls the strength of the random potential noise.")
        corr_len = st.slider("Noise Correlation Length", 1.0, 20.0, 5.0, 0.5, help="Defines the spatial smoothness of the noise.")

    trap_descriptions = {
        "ideal": "*Ideal:* smooth harmonic potential for clean dynamics.",
        "disordered": "🌀 *Disordered:* noisy trap to simulate imperfections.",
        "doublewell": "🪙 *Double well:* required for two-qubit gates like BELL and CNOT."
    }
    st.markdown(trap_descriptions.get(trap, ""))

with col_right:
    # This will now depend on the full set of potential parameters.
    # We assume the CLI job will generate the appropriate potential image.
    trap_img_path = f"outputinfo/{trap}_potential.png"
    if os.path.exists(trap_img_path):
        st.image(trap_img_path, caption=f"{trap.capitalize()} potential", width=260)

st.markdown("### Optimiser Settings")

c1, c2 = st.columns([1, 1])
with c1:
    maxiter = st.number_input("Max iterations", min_value=5, max_value=200, value=20)
with c2:
    popsize = st.number_input("Population size", min_value=2, max_value=64, value=10)

# --- ✨ NEW: Inputs for simulation grid and time settings ---
st.markdown("##### Simulation Grid & Time Settings")
c3, c4, c5 = st.columns(3)
with c3:
    grid_size = st.number_input("Grid Size", min_value=16, max_value=128, value=64, step=16, help="Resolution of the simulation space (e.g., 64x64x64 grid).")
with c4:
    time_steps = st.number_input("Time Steps", min_value=50, max_value=1000, value=500, step=50, help="Total number of discrete time steps in the simulation.")
with c5:
    dt = st.number_input("Time Step (dt)", min_value=0.001, max_value=0.1, value=0.005, format="%.3f", help="Duration of each individual time step.")

st.caption("CMA-ES = Covariance Matrix Adaptation Evolution Strategy")

if maxiter > 20 or popsize > 10 or grid_size > 64:
    st.warning("⚠️ We recommend using a GPU (e.g. Brev) for high iteration, population, or grid sizes.")

st.markdown("### Refinement Phase")
st.caption("Optionally run a second, more focused optimization to improve the result.")

run_refinement = st.checkbox("✅ Run second refinement phase", value=True)

# These controls will only be used if the checkbox is ticked
if run_refinement:
    c6, c7 = st.columns(2)
    with c6:
        refine_maxiter = st.number_input("Refinement Iterations", min_value=10, max_value=200, value=30, help="Number of iterations for the focused search.")
    with c7:
        # The slider value is divided by 100 to pass a float (e.g., 25 -> 0.25)
        refinement_factor = st.slider("Refinement Search Size (%)", 1, 50, 25, help="How much to shrink the search area. 25% is a good starting point.") / 100.0
else:
    # Set default values if the box is unchecked
    refine_maxiter = 0
    refinement_factor = 0.0

run_btn = st.button("🚀 Optimise!")

if run_btn:
    st.info("Running optimisation... this may take a moment ⏳")
    
    # --- ✨ UPDATED: The run_cli_job call now includes all new parameters ---
    result = driver_cli.run_cli_job(
        trap=trap,
        gate=gate,
        omega=omega,
        delta=delta,
        maxiter=maxiter,
        popsize=popsize,
        grid_size=grid_size,
        time_steps=time_steps,
        dt=dt,
        noise_amp=noise_amp,
        corr_len=corr_len,
        run_refinement_phase=run_refinement,
        refine_maxiter=refine_maxiter if run_refinement else 0,
        refinement_factor=refinement_factor if run_refinement else 0.0
    )

    st.success(f"Optimisation complete! Best fidelity: {result['fidelity']:.4f}")
    st.json(result["params"])



# import streamlit as st
# import os
# import numpy as np
# import torch
# import time
# import driver_cli
# import qdotlib

# st.set_page_config(page_title="Quantum Dot Gate Optimiser", layout="wide")

# # ---------------- Header ----------------
# st.title("Quantum‑Dot Gate Optimiser")
# st.caption("Pick a trap, choose a gate, tune hyper‑parameters, then click **Optimise!**. All heavy lifting happens on the GPU.")

# # ---------------- Horizontal Layout ----------------
# # ---------------- Trap & Gate Selection ----------------
# col_left, col_right = st.columns([2, 1])

# with col_left:
#     col_trap, col_gate = st.columns(2)

#     with col_trap:
#         trap = st.selectbox("Trap landscape", ["ideal", "disordered", "doublewell"], index=0)

#     with col_gate:
#         gate_options = ["NOT", "HADAMARD"]
#         if trap == "doublewell":
#             gate_options += ["BELL", "CNOT"]
#         gate = st.selectbox("Target Gate", gate_options)

#     omega = st.slider("ω (harmonic frequency)", 0.1, 5.0, 1.0, 0.1)
    
#     delta = None
#     if trap == "doublewell":
#         delta = st.slider("Δ (Delta) separation", 1.0, 5.0, 2.0, 0.1)

#     trap_descriptions = {
#         "ideal": "*Ideal:* smooth harmonic potential for clean dynamics.",
#         "disordered": "🌀 *Disordered:* noisy trap to simulate imperfections.",
#         "doublewell": "🪙 *Double well:* required for two‑qubit gates like BELL and CNOT."
#     }
#     st.markdown(trap_descriptions.get(trap, ""))

# with col_right:
#     trap_img_path = f"outputinfo/{trap}_potential.png"
#     if os.path.exists(trap_img_path):
#         st.image(trap_img_path, caption=f"{trap.capitalize()} potential", width=260)

# st.markdown("### Optimiser Settings")

# c1, c2 = st.columns([1, 1])
# with c1:
#     maxiter = st.number_input("Max iterations", min_value=5, max_value=200, value=20)
# with c2:
#     popsize = st.number_input("Population size", min_value=2, max_value=64, value=10)

# st.caption("CMA‑ES = Covariance Matrix Adaptation Evolution Strategy")

# if maxiter > 20 or popsize > 5:
#     st.warning("⚠️ We recommend using a GPU (e.g. Brev) for high iteration or population sizes.")

# run_btn = st.button("🚀 Optimise!")

# if run_btn:
#     st.info("Running optimisation... this may take a moment ⏳")
#     result = driver_cli.run_cli_job(
#         trap=trap,
#         gate=gate,
#         omega=omega,
#         delta=delta,
#         maxiter=maxiter,
#         popsize=popsize
#     )

#     st.success(f"Optimisation complete! Best fidelity: {result['fidelity']:.4f}")
#     st.json(result["params"])
