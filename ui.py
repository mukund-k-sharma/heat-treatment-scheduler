"""
Heat Treatment Digital Twin - Streamlit Frontend.

Interactive dashboard for manually controlling and visualizing the Heat Treatment
Scheduler environment. Runs the ODE physics engine locally (no network required)
and provides real-time Plotly charts for:
- Thermal inertia visualization (furnace air vs. material core temperature)
- Nanoprecipitate growth trajectory vs. target radius
- Step-by-step metrics (elapsed time, temperature, radius, reward)

The dashboard exposes SMDP controls (action selection + duration slider) for
manual experimentation with the physics engine.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from server.heat_treatment_scheduler_environment import HeatTreatmentSchedulerEnvironment, AgentGrade
from models import HeatTreatmentSchedulerAction

# --- CONFIGURATION ---
st.set_page_config(page_title="Heat Treatment Digital Twin", layout="wide")

# --- SESSION STATE ---
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=[
        "step", "time_sec", "furnace_temp", "material_temp", "radius", "target_radius", "reward"
    ])
if "env" not in st.session_state:
    # Initialize the local ODE environment directly! No network required.
    st.session_state.env = HeatTreatmentSchedulerEnvironment(
        difficulty=AgentGrade.HARD,
        alloy_key="Ti_6Al_4V",
        hardware_key="massive_casting"
    )
if "done" not in st.session_state:
    st.session_state.done = False
if "score" not in st.session_state:
    st.session_state.score = 0.0

# --- HELPERS ---
def reset_env():
    """Reset the environment and initialize the history DataFrame with the initial state."""
    result = st.session_state.env.reset()
    state = st.session_state.env.state # Accessed as a property, not a method!
    
    st.session_state.history = pd.DataFrame([{
        "step": 0,
        "time_sec": state.time,
        "furnace_temp": state.temperature, 
        "material_temp": state.temperature,
        "radius": state.radius,
        "target_radius": state.target_radius,
        "reward": 0.0
    }])
    st.session_state.done = False
    st.session_state.score = 0.0

def step_env(action_num: int, duration: float):
    """Execute one SMDP action and append the resulting state to the session history."""
    action = HeatTreatmentSchedulerAction(action_num=action_num, duration_minutes=duration)
    
    current_furnace = st.session_state.history.iloc[-1]["furnace_temp"]
    dT = HeatTreatmentSchedulerAction.ACTION_MAP.get(action_num)
    new_furnace = current_furnace if dT is None else current_furnace + dT
    
    result = st.session_state.env.step(action)
    state = st.session_state.env.state # Accessed as a property
    
    new_row = pd.DataFrame([{
        "step": len(st.session_state.history),
        "time_sec": state.time,
        "furnace_temp": new_furnace,
        "material_temp": state.temperature,
        "radius": state.radius,
        "target_radius": state.target_radius,
        "reward": getattr(result, 'reward', 0.0)
    }])
    st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True)
    st.session_state.done = getattr(result, 'done', False)

# --- UI LAYOUT ---
st.title("🔬 Heat Treatment Digital Twin")
st.markdown("Monitor real-time continuous ODE thermodynamics and execute SMDP thermal recipes.")

# TOP ROW: Metrics
if not st.session_state.history.empty:
    latest = st.session_state.history.iloc[-1]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Elapsed Time", f"{latest['time_sec'] / 3600:.1f} hrs")
    m2.metric("Core Temp", f"{latest['material_temp']:.1f} °C")
    m3.metric("Precipitate Radius", f"{latest['radius']:.3f} nm", f"Target: {latest['target_radius']:.3f} nm")
    m4.metric("Last Reward", f"{latest['reward']:.2f}")

st.divider()

# MIDDLE ROW: Charts
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("Thermal Inertia (Furnace vs. Material)")
    if not st.session_state.history.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=st.session_state.history["time_sec"]/3600, y=st.session_state.history["furnace_temp"], 
                                 mode='lines+markers', name='Furnace Air', line=dict(color='orange', dash='dot')))
        fig.add_trace(go.Scatter(x=st.session_state.history["time_sec"]/3600, y=st.session_state.history["material_temp"], 
                                 mode='lines+markers', name='Material Core', line=dict(color='red', width=3)))
        fig.update_layout(xaxis_title="Time (Hours)", yaxis_title="Temperature (°C)", height=400)
        st.plotly_chart(fig, width='content')

with col_chart2:
    st.subheader("Nanoprecipitate Growth")
    if not st.session_state.history.empty:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=st.session_state.history["time_sec"]/3600, y=st.session_state.history["radius"], 
                                  mode='lines+markers', name='Current Radius', line=dict(color='blue', width=3)))
        target = st.session_state.history.iloc[-1]["target_radius"]
        fig2.add_hline(y=target, line_dash="dash", line_color="green", annotation_text="Target Radius")
        fig2.update_layout(xaxis_title="Time (Hours)", yaxis_title="Radius (nm)", height=400)
        st.plotly_chart(fig2, width='content')

st.divider()

# BOTTOM ROW: Controls
col_ctrl1, col_ctrl2 = st.columns([1, 2])

with col_ctrl1:
    st.subheader("System Control")
    if st.button("🔄 Initialize / Reset Furnace", width='content'):
        reset_env()
        st.rerun()
        
    if st.session_state.done:
        st.error("Episode Terminated.")

with col_ctrl2:
    st.subheader("Manual Override (SMDP Action)")
    with st.form("action_form"):
        c1, c2 = st.columns(2)
        with c1:
            action_map = {
                0: "Aggressive Cooling (-50°C)",
                1: "Gentle Cooling (-10°C)",
                2: "Hold Temperature (0°C)",
                3: "Gentle Heating (+10°C)",
                4: "Aggressive Heating (+50°C)",
                5: "TERMINATE EPISODE"
            }
            selected_action = st.selectbox("Action", options=list(action_map.keys()), format_func=lambda x: f"{x}: {action_map[x]}")
        with c2:
            duration = st.slider("Duration (Minutes)", min_value=1.0, max_value=300.0, value=60.0, step=1.0)
            
        submitted = st.form_submit_button("Execute Command", disabled=st.session_state.done)
        if submitted:
            step_env(selected_action, duration)
            st.rerun()