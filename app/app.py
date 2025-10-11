import json
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from model import load_params, apply_preset, simulate

from pathlib import Path
BASE_DIR = Path(__file__).parent

st.set_page_config(page_title="GI Motility & Absorption Simulator", layout="wide")

st.title("GI Motility & Fluid Absorption Simulator (Educational)")
st.caption(
    "This tool simulates simplified gastrointestinal transit and water absorption. "
    "For education only — not medical advice or a diagnostic tool."
)

# Loads base parameters and presets
with open(BASE_DIR / "params.json", "r") as f:
    base_params = json.load(f)
with open(BASE_DIR / "presets.json", "r") as f:
    preset_dict = json.load(f)

# Sidebar for configuration
st.sidebar.header("Configuration")
preset_name = st.sidebar.selectbox("Preset scenario", list(preset_dict.keys()), index=0)

st.sidebar.subheader("Manual parameter overrides (optional)")
mot_si = st.sidebar.slider("Small Intestine motility", 0.1, 2.0, float(base_params["compartments"][0]["motility"]), 0.05)
mot_co = st.sidebar.slider("Colon motility", 0.1, 2.0, float(base_params["compartments"][1]["motility"]), 0.05)
abs_si = st.sidebar.slider("Small Intestine absorption rate", 0.0, 0.05, float(base_params["compartments"][0]["absorption_rate"]), 0.001)
abs_co = st.sidebar.slider("Colon absorption rate", 0.0, 0.05, float(base_params["compartments"][1]["absorption_rate"]), 0.001)
res_si = st.sidebar.slider("SI outflow resistance", 0.5, 8.0, float(base_params["compartments"][0]["resistance_out"]), 0.1)
res_co = st.sidebar.slider("Colon outflow resistance", 0.5, 8.0, float(base_params["compartments"][1]["resistance_out"]), 0.1)

damage_thresh = st.sidebar.slider("Damage threshold (pressure units)", 0.5, 5.0, float(base_params["damage"]["threshold"]), 0.1)
kP = st.sidebar.slider("Pressure gain kP", 0.005, 0.03, float(base_params["pressure"]["kP"]), 0.001)
t_end = st.sidebar.slider("Simulation duration (minutes)", 120, 1440, int(base_params["time"]["t_end"]), 60)

# Build GIParams with preset + overrides
gi = load_params(base_params)
gi.time["t_end"] = float(t_end)

preset = preset_dict[preset_name]
gi = apply_preset(gi, preset)

gi.compartments[0].motility = float(mot_si)
gi.compartments[1].motility = float(mot_co)
gi.compartments[0].absorption_rate = float(abs_si)
gi.compartments[1].absorption_rate = float(abs_co)
gi.compartments[0].resistance_out = float(res_si)
gi.compartments[1].resistance_out = float(res_co)
gi.damage.threshold = float(damage_thresh)
gi.pressure.kP = float(kP)

run = st.button("Run simulation")
if run:
    t, Y = simulate(gi)
    V_SI, W_SI, V_CO, W_CO, D = Y

    WaterVol_SI = V_SI * np.clip(W_SI, 0, 1)
    WaterVol_CO = V_CO * np.clip(W_CO, 0, 1)
    P_SI = gi.pressure.kP * np.maximum(V_SI - gi.pressure.V_ref, 0.0) * (1.0 + gi.compartments[0].resistance_out)
    P_CO = gi.pressure.kP * np.maximum(V_CO - gi.pressure.V_ref, 0.0) * (1.0 + gi.compartments[1].resistance_out)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("Compartment Volumes")
        fig1 = plt.figure()
        plt.plot(t, V_SI, label="Small Intestine Vol (mL)")
        plt.plot(t, V_CO, label="Colon Vol (mL)")
        plt.xlabel("Time (min)")
        plt.ylabel("Volume (mL)")
        plt.legend()
        st.pyplot(fig1)

        st.subheader("Water Fractions")
        fig2 = plt.figure()
        plt.plot(t, np.clip(W_SI, 0, 1), label="SI water fraction")
        plt.plot(t, np.clip(W_CO, 0, 1), label="Colon water fraction")
        plt.xlabel("Time (min)")
        plt.ylabel("Water fraction (0-1)")
        plt.legend()
        st.pyplot(fig2)

    with col2:
        st.subheader("Pressure Proxies")
        fig3 = plt.figure()
        plt.plot(t, P_SI, label="SI pressure proxy")
        plt.plot(t, P_CO, label="Colon pressure proxy")
        plt.axhline(gi.damage.threshold, linestyle="--", label="Damage threshold")
        plt.xlabel("Time (min)")
        plt.ylabel("Pressure (arb units)")
        plt.legend()
        st.pyplot(fig3)

        st.subheader("Cumulative Damage Index")
        fig4 = plt.figure()
        plt.plot(t, D, label="Damage index")
        plt.xlabel("Time (min)")
        plt.ylabel("Damage (arb units)")
        plt.legend()
        st.pyplot(fig4)
    st.markdown("### Summary")
    stool_liquidity = float(np.clip(W_CO[-1], 0, 1))
    consistency = "liquid" if stool_liquidity>0.8 else ("soft" if stool_liquidity>0.6 else ("formed" if stool_liquidity>0.4 else "hard"))
    st.write(f"- Final colon volume: **{V_CO[-1]:.1f} mL**")
    st.write(f"- Final colon water fraction: **{W_CO[-1]:.2f}** (≈ **{consistency}** stool)")
    st.write(f"- Max pressure proxy: **{max(P_SI.max(), P_CO.max()):.2f}**, damage threshold: **{gi.damage.threshold:.2f}**")
    st.write(f"- Final damage index: **{D[-1]:.3f}** (educational proxy, not clinical)")

st.markdown("Disclaimer: Educational simulator only, not a medical device. Do not use for diagnosis or treatment.")