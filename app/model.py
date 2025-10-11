import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Tuple
from scipy.integrate import solve_ivp

@dataclass
class CompartmentParams:
    name: str
    V0: float
    W0: float
    motility: float
    absorption_rate: float  # per minute water absorption fraction
    resistance_out: float   # flow resistance to next compartment

@dataclass
class FlowParams:
    k_flow: float

@dataclass
class PressureParams:
    kP: float
    V_ref: float

@dataclass
class DamageParams:
    threshold: float
    rate: float

@dataclass
class GIParams:
    time: Dict[str, float]
    compartments: Tuple[CompartmentParams, CompartmentParams]
    flow: FlowParams
    pressure: PressureParams
    damage: DamageParams

def load_params(params_json: Dict[str, Any]) -> GIParams:
    c0 = params_json["compartments"][0]
    c1 = params_json["compartments"][1]
    return GIParams(
        time=params_json["time"],
        compartments=(
            CompartmentParams(**c0),
            CompartmentParams(**c1),
        ),
        flow=FlowParams(**params_json["flow"]),
        pressure=PressureParams(**params_json["pressure"]),
        damage=DamageParams(**params_json["damage"]),
    )

def apply_preset(gi: GIParams, preset: Dict[str, Any]) -> GIParams:
    import copy
    new = copy.deepcopy(gi)

    def set_attr_path(path, value):
        parts = path.split(".")
        if parts[0] in ("SmallIntestine", "Colon"):
            comp_idx = 0 if parts[0] == "SmallIntestine" else 1
            setattr(new.compartments[comp_idx], parts[1], value)
        elif parts[0] == "pressure":
            setattr(new.pressure, parts[1], value)
        elif parts[0] == "damage":
            setattr(new.damage, parts[1], value)
        elif parts[0] == "flow":
            setattr(new.flow, parts[1], value)

    for k, v in preset.items():
        set_attr_path(k, v)
    return new

def gi_rhs(t, y, gi: GIParams):
    """
    State y = [V_SI, W_SI, V_CO, W_CO, D]
      volumes in mL, water fractions 0-1, cumulative damage D (arb units)
    """
    V_SI, W_SI, V_CO, W_CO, D = y
    cp_SI, cp_CO = gi.compartments
    k_flow = gi.flow.k_flow

    W_SI = np.clip(W_SI, 0.0, 1.0)
    W_CO = np.clip(W_CO, 0.0, 1.0)

    flow_SI_to_CO = k_flow * cp_SI.motility * V_SI / (1.0 + max(cp_SI.resistance_out, 1e-6))
    flow_CO_out   = k_flow * cp_CO.motility * V_CO / (1.0 + max(cp_CO.resistance_out, 1e-6))

    water_abs_SI = cp_SI.absorption_rate * (V_SI * W_SI)
    water_abs_CO = cp_CO.absorption_rate * (V_CO * W_CO)

    dV_SI = -flow_SI_to_CO
    dV_CO =  flow_SI_to_CO - flow_CO_out

    water_SI = V_SI * W_SI
    water_CO = V_CO * W_CO

    dWater_SI = -(flow_SI_to_CO * (water_SI / (V_SI + 1e-9))) - water_abs_SI
    dWater_CO =  (flow_SI_to_CO * (water_SI / (V_SI + 1e-9))) \
               - (flow_CO_out * (water_CO / (V_CO + 1e-9))) - water_abs_CO

    dW_SI = 0.0
    dW_CO = 0.0
    if V_SI > 1e-9:
        dW_SI = (dWater_SI - W_SI * dV_SI) / V_SI
    if V_CO > 1e-9:
        dW_CO = (dWater_CO - W_CO * dV_CO) / V_CO

    P_SI = gi.pressure.kP * max(V_SI - gi.pressure.V_ref, 0.0) * (1.0 + cp_SI.resistance_out)
    P_CO = gi.pressure.kP * max(V_CO - gi.pressure.V_ref, 0.0) * (1.0 + cp_CO.resistance_out)
    P_max = max(P_SI, P_CO)
    dD = gi.damage.rate * max(P_max - gi.damage.threshold, 0.0)

    return [dV_SI, dW_SI, dV_CO, dW_CO, dD]

def simulate(gi: GIParams):
    cp_SI, cp_CO = gi.compartments
    y0 = [cp_SI.V0, cp_SI.W0, cp_CO.V0, cp_CO.W0, 0.0]

    t_end = gi.time["t_end"]
    dt = gi.time.get("dt", 0.5)
    t_eval = np.arange(0.0, t_end + dt, dt)

    sol = solve_ivp(
        fun=lambda t, y: gi_rhs(t, y, gi),
        t_span=(0.0, t_end),
        y0=y0,
        t_eval=t_eval,
        method="RK45",
        max_step=1.0,
    )
    return sol.t, sol.y  # shape: [5, len(t)]