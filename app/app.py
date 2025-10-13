import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
from model import load_params, apply_preset, simulate

BASE_DIR = Path(__file__).parent
st.set_page_config(page_title="Gastrointestinal Motility and Fluid Absorption Simulator", layout="wide")
st.title("Gastrointestinal Motility and Fluid Absorption Simulator")

with open(BASE_DIR / "params.json", "r") as f:
    base_params = json.load(f)
with open(BASE_DIR / "presets.json", "r") as f:
    preset_dict = json.load(f)

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

st.markdown("## 3D Peristalsis Animation (Gently Curved Tube)")
st.caption("Drag to rotate  •  Scroll to zoom •  Use Play to animate a traveling wave. The left behaves like the small intestine, and the right behaves like the colon.")

mot_si_val = float(gi.compartments[0].motility)
mot_co_val = float(gi.compartments[1].motility)
abs_si_val = float(gi.compartments[0].absorption_rate)
abs_co_val = float(gi.compartments[1].absorption_rate)

def contrast_from_abs(abs_rate: float) -> float:
    return float(np.clip(0.8 + abs_rate * 25.0, 0.6, 1.8))

contrast_si = contrast_from_abs(abs_si_val)
contrast_co = contrast_from_abs(abs_co_val)

def phase_step_from_mot(mot: float) -> float:
    return float(np.clip(0.20 * mot, 0.05, 0.45))

phase_step_si = phase_step_from_mot(mot_si_val)
phase_step_co = phase_step_from_mot(mot_co_val)

n_long = 180
Xc = np.linspace(-5.0, 5.0, n_long)
Yc = 0.8 * np.sin(0.6 * Xc)
Zc = 0.3 * np.sin(0.3 * Xc + 1.0)
dxyz = np.sqrt(np.diff(Xc)**2 + np.diff(Yc)**2 + np.diff(Zc)**2)
s = np.concatenate([[0], np.cumsum(dxyz)])
s_total = s[-1]
s_split = 0.60 * s_total

def orthonormal_frames(X, Y, Z):
    n = len(X)
    T = np.zeros((n, 3))
    N = np.zeros((n, 3))
    B = np.zeros((n, 3))
    V = np.column_stack([np.gradient(X), np.gradient(Y), np.gradient(Z)])
    eps = 1e-9
    for i in range(n):
        t = V[i]
        t = t / (np.linalg.norm(t) + eps)
        ref = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(ref, t)) > 0.95:
            ref = np.array([0.0, 1.0, 0.0])
        nvec = np.cross(ref, t)
        nvec = nvec / (np.linalg.norm(nvec) + eps)
        bvec = np.cross(t, nvec)
        T[i], N[i], B[i] = t, nvec, bvec
    return T, N, B

T, N, B = orthonormal_frames(Xc, Yc, Zc)
n_circ = 40
theta = np.linspace(0, 2*np.pi, n_circ, endpoint=False)
base_R = 0.20
amp_R = 0.08
lam_si = 0.25 * s_total
lam_co = 0.35 * s_total

def build_frame(phase_si: float, phase_co: float):
    wave_si = np.sin(2 * np.pi * s / lam_si - phase_si)
    wave_co = np.sin(2 * np.pi * s / lam_co - phase_co)
    wave = np.where(s <= s_split, wave_si, wave_co)
    R = base_R + amp_R * wave
    C = 0.5 + 0.5 * wave
    C = np.where(s <= s_split, np.clip(C * contrast_si, 0, 1), np.clip(C * contrast_co, 0, 1))
    Xs, Ys, Zs, Cs = [], [], [], []
    for i in range(len(Xc)):
        ring = (N[i][None, :] * np.cos(theta)[:, None] + B[i][None, :] * np.sin(theta)[:, None])
        pts = np.array([Xc[i], Yc[i], Zc[i]])[None, :] + R[i] * ring
        Xs.append(pts[:, 0]); Ys.append(pts[:, 1]); Zs.append(pts[:, 2]); Cs.append(np.full(n_circ, C[i]))
    Xs = np.array(Xs); Ys = np.array(Ys); Zs = np.array(Zs); Cs = np.array(Cs)
    return Xs.T, Ys.T, Zs.T, Cs.T

num_frames = 60
frames = []
for k in range(num_frames):
    phase_si = k * phase_step_si
    phase_co = k * phase_step_co
    Xs, Ys, Zs, Cs = build_frame(phase_si, phase_co)
    frames.append(go.Frame(
        data=[go.Surface(
            x=Xs, y=Ys, z=Zs,
            surfacecolor=Cs,
            cmin=0, cmax=1,
            colorscale="Viridis",
            showscale=False,
            opacity=0.95,
        )],
        name=str(k)
    ))

Xs0, Ys0, Zs0, Cs0 = build_frame(0.0, 0.0)
fig = go.Figure(
    data=[go.Surface(
        x=Xs0, y=Ys0, z=Zs0,
        surfacecolor=Cs0,
        cmin=0, cmax=1,
        colorscale="Viridis",
        showscale=False,
        opacity=0.95,
    )],
    frames=frames
)
fig.update_layout(
    scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode="data"),
    margin=dict(l=0, r=0, t=0, b=0),
    updatemenus=[{
        "type": "buttons", "showactive": False, "x": 0.05, "y": 1.08, "xanchor": "left", "yanchor": "top",
        "buttons": [
            {"label": "▶ Play", "method": "animate",
             "args": [None, {"frame": {"duration": 45, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}]},
            {"label": "⏸ Pause", "method": "animate",
             "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]}
        ]
    }],
    sliders=[{
        "active": 0, "x": 0.05, "y": 1.02, "xanchor": "left", "yanchor": "top", "len": 0.9,
        "currentvalue": {"prefix": "Frame: ", "visible": True},
        "steps": [{"label": str(k), "method": "animate",
                   "args": [[str(k)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True},
                                       "transition": {"duration": 0}}]} for k in range(num_frames)]
    }]
)
fig.update_layout(scene_camera=dict(eye=dict(x=2.2, y=0.6, z=0.35), up=dict(x=0, y=0, z=1)))
st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

run = st.button("Run simulation")
if run:
    t, Y = simulate(gi)
    V_SI, W_SI, V_CO, W_CO, D = Y
    P_SI = gi.pressure.kP * np.maximum(V_SI - gi.pressure.V_ref, 0.0) * (1.0 + gi.compartments[0].resistance_out)
    P_CO = gi.pressure.kP * np.maximum(V_CO - gi.pressure.V_ref, 0.0) * (1.0 + gi.compartments[1].resistance_out)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("Compartment Volumes")
        fig1 = plt.figure()
        plt.plot(t, V_SI, label="Small Intestine Volume (in mL)")
        plt.plot(t, V_CO, label="Colon Volume (in mL)")
        plt.xlabel("Time (in mins)")
        plt.ylabel("Volume (in mL)")
        plt.legend()
        st.pyplot(fig1)
        st.subheader("Water Fractions")
        fig2 = plt.figure()
        plt.plot(t, np.clip(W_SI, 0, 1), label="Small intestine water fraction")
        plt.plot(t, np.clip(W_CO, 0, 1), label="Colon water fraction")
        plt.xlabel("Time (in mins)")
        plt.ylabel("Water fraction (0–1)")
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
    consistency = "liquid" if stool_liquidity > 0.8 else "soft" if stool_liquidity > 0.6 else "formed" if stool_liquidity > 0.4 else "hard"
    st.write(f"- Final colon volume: **{V_CO[-1]:.1f} mL**")
    st.write(f"- Final colon water fraction: **{W_CO[-1]:.2f}** (≈ **{consistency}** stool)")
    st.write(f"- Max pressure proxy: **{max(P_SI.max(), P_CO.max()):.2f}**, damage threshold: **{gi.damage.threshold:.2f}**")
    st.write(f"- Final damage index: **{D[-1]:.3f}** (educational proxy, not clinical)")

st.markdown("**Disclaimer:** This is an educational simulator only, not a medical device. Do not use this for diagnosis or treatment.")