"""
TrafficMind AI - Frontend Streamlit Web Application
Developed for the PromptWars x Creative Community VVITU Hackathon.

Technology Stack:
- Streamlit (Web Dashboard, Interactive Sidebar & Reactive State)
- Plotly (Interactive Metropolitan Graph, 12-Hour Forecast, Confusion Matrix)
- Pandas & NumPy (Data Structuring & Feature Visualization)
- Backend Integration: backend.predictor & backend.routing
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx

# Add repository root to Python path for seamless backend imports
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.predictor import (
    generate_synthetic_traffic_dataset,
    train_traffic_model,
    encode_single_input,
    predict_congestion
)
from backend.routing import (
    build_metropolitan_network,
    simulate_network_conditions,
    compute_route_comparison,
    compute_emergency_green_corridor
)

# ==============================================================================
# 1. STREAMLIT PAGE CONFIGURATION & CUSTOM CSS STYLING
# ==============================================================================

st.set_page_config(
    page_title="TrafficMind AI — Intelligent Urban Traffic Engine",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Command Center Aesthetics (Modern Dark Theme, Glassmorphism & Cyber-Urban Accents)
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre, .mono-font {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Gradient Header & Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(13, 22, 38, 0.95) 0%, rgba(18, 30, 49, 0.95) 50%, rgba(24, 43, 73, 0.9) 100%);
        border: 1px solid rgba(0, 229, 255, 0.25);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 229, 255, 0.15);
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00E5FF, #7C4DFF, #00E676);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
        letter-spacing: -0.5px;
    }

    .hero-subtitle {
        color: #B0BEC5;
        font-size: 1.05rem;
        margin-bottom: 14px;
    }

    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-right: 8px;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    .badge-hackathon { background: rgba(124, 77, 255, 0.2); color: #B388FF; border-color: rgba(179, 136, 255, 0.4); }
    .badge-ai { background: rgba(0, 229, 255, 0.15); color: #00E5FF; border-color: rgba(0, 229, 255, 0.35); }
    .badge-synthetic { background: rgba(255, 171, 0, 0.15); color: #FFD54F; border-color: rgba(255, 213, 79, 0.35); }

    /* Glassmorphism Metric Cards */
    .metric-card {
        background: rgba(18, 26, 43, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 14px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(0, 229, 255, 0.35);
        transform: translateY(-2px);
    }
    .metric-label {
        color: #90A4AE;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #ECEFF1;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #78909C;
        margin-top: 2px;
    }

    /* Congestion Alert Banners */
    .congestion-low {
        background: linear-gradient(90deg, rgba(0, 230, 118, 0.15), rgba(0, 230, 118, 0.03));
        border-left: 4px solid #00E676;
        padding: 14px 18px;
        border-radius: 8px;
        color: #B9F6CA;
    }
    .congestion-moderate {
        background: linear-gradient(90deg, rgba(255, 171, 0, 0.18), rgba(255, 171, 0, 0.03));
        border-left: 4px solid #FFAB00;
        padding: 14px 18px;
        border-radius: 8px;
        color: #FFE57F;
    }
    .congestion-severe {
        background: linear-gradient(90deg, rgba(255, 23, 68, 0.2), rgba(255, 23, 68, 0.03));
        border-left: 4px solid #FF1744;
        padding: 14px 18px;
        border-radius: 8px;
        color: #FF8A80;
    }

    /* Callout Box */
    .callout-box {
        background: rgba(30, 41, 59, 0.6);
        border: 1px dashed rgba(148, 163, 184, 0.3);
        border-radius: 10px;
        padding: 14px 18px;
        font-size: 0.88rem;
        color: #CBD5E1;
        margin: 12px 0;
    }

    /* Sidebar Header */
    .sidebar-section-title {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #00E5FF;
        letter-spacing: 1px;
        margin-top: 14px;
        margin-bottom: 8px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# 2. CACHED BACKEND DATA & MODEL LOADERS
# ==============================================================================

@st.cache_data(show_spinner="Generating physics-inspired synthetic traffic dataset...")
def load_dataset():
    return generate_synthetic_traffic_dataset(num_samples=3200, seed=42)


@st.cache_resource(show_spinner="Training TrafficMind Random Forest AI Model...")
def load_model(df: pd.DataFrame):
    return train_traffic_model(df)


@st.cache_data
def load_network():
    return build_metropolitan_network()


# ==============================================================================
# 3. PLOTLY VISUALIZATION ENGINES
# ==============================================================================

def plot_interactive_network(G: nx.Graph, highlighted_route=None, emergency_route=None, title="Metropolitan Road Network"):
    """
    Renders an interactive Plotly graph of the metropolitan road network
    showing live congestion color codes, alternative routes, and emergency corridors.
    """
    fig = go.Figure()

    highlight_edges = set()
    if highlighted_route and len(highlighted_route) > 1:
        for i in range(len(highlighted_route) - 1):
            highlight_edges.add((highlighted_route[i], highlighted_route[i + 1]))
            highlight_edges.add((highlighted_route[i + 1], highlighted_route[i]))

    emergency_edges = set()
    if emergency_route and len(emergency_route) > 1:
        for i in range(len(emergency_route) - 1):
            emergency_edges.add((emergency_route[i], emergency_route[i + 1]))
            emergency_edges.add((emergency_route[i + 1], emergency_route[i]))

    for u, v, data in G.edges(data=True):
        x0, y0 = G.nodes[u]['pos']
        x1, y1 = G.nodes[v]['pos']
        color = data.get('color', '#90A4AE')
        state = data.get('congestion_state', 'Unknown')
        speed = data.get('effective_speed', data['speed_limit'])
        time_min = data.get('dynamic_time_min', data['base_time_min'])
        dist = data['distance_km']
        road_name = data['road_name']

        width = 4.5
        dash = 'solid'
        opacity = 0.85

        is_in_highlight = (u, v) in highlight_edges
        is_in_emergency = (u, v) in emergency_edges

        if is_in_emergency:
            color = '#1DE9B6'  # Neon Emerald
            width = 8.0
            dash = 'solid'
            opacity = 1.0
        elif is_in_highlight:
            color = '#00E5FF'  # Neon Electric Cyan
            width = 7.5
            dash = 'solid'
            opacity = 1.0
        elif highlighted_route or emergency_route:
            opacity = 0.35
            width = 3.0

        hover_text = (
            f"<b>{road_name}</b><br>"
            f"Corridor: {u} ↔ {v}<br>"
            f"Distance: {dist} km<br>"
            f"Live Speed: {speed} km/h (Limit: {data['speed_limit']} km/h)<br>"
            f"Current Travel Time: {time_min} mins<br>"
            f"Congestion Level: <b>{state}</b>"
        )

        fig.add_trace(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=width, color=color, dash=dash),
            hoverinfo='text',
            hovertext=hover_text,
            showlegend=False
        ))

    # Nodes
    node_x = []
    node_y = []
    node_text = []
    node_colors = []

    for node, data in G.nodes(data=True):
        x, y = data['pos']
        node_x.append(x)
        node_y.append(y)
        icon = data.get('icon', '📍')
        ntype = data.get('type', 'Urban Hub')
        node_text.append(f"<b>{icon} {node}</b><br>Zone: {ntype}<br>Grid Coord: ({x}, {y})")

        if emergency_route and node in emergency_route:
            node_colors.append('#1DE9B6')
        elif highlighted_route and node in highlighted_route:
            node_colors.append('#00E5FF')
        elif node == 'City General Trauma Hospital':
            node_colors.append('#FF5252')
        elif node == 'Central Fire & Emergency HQ':
            node_colors.append('#FFAB00')
        else:
            node_colors.append('#ECEFF1')

    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(
            size=19,
            color=node_colors,
            line=dict(width=2.5, color='#0F172A')
        ),
        text=[n.split(' ')[0] for n in G.nodes()],
        textposition="top center",
        textfont=dict(size=10, color="#E2E8F0", family="'Plus Jakarta Sans', sans-serif"),
        hoverinfo='text',
        hovertext=node_text,
        showlegend=False
    ))

    # Legend items
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='#00E676', width=4), name='🟢 Low Congestion'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='#FFB300', width=4), name='🟡 Moderate Congestion'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='#FF1744', width=4), name='🔴 Severe Congestion'))

    if highlighted_route:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='#00E5FF', width=6), name='⚡ TrafficMind AI Route'))
    if emergency_route:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='#1DE9B6', width=7), name='🚑 Emergency Green Corridor'))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#00E5FF")),
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.7)",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#CBD5E1")
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=10, r=10, t=50, b=10),
        height=540
    )

    return fig


def plot_short_term_forecast(current_hour: int, current_prediction: str, avg_speed: float, vehicle_count: int):
    """
    Generates a 12-hour illustrative projection chart.
    Clearly labeled as an illustrative synthetic forecast to maintain scientific integrity.
    """
    forecast_hours = []
    projected_congestion_pct = []
    projected_speed = []
    time_labels = []

    now = datetime.now()

    for step in range(13):
        h = (current_hour + step) % 24
        forecast_time = (now + timedelta(hours=step)).strftime("%I:%M %p")
        time_labels.append(forecast_time)
        forecast_hours.append(h)

        if 7 <= h <= 9:
            rush_bump = 42.0 + (9 - abs(h - 8.2)) * 6.0
        elif 16 <= h <= 19:
            rush_bump = 48.0 + (10 - abs(h - 17.5)) * 5.5
        elif 10 <= h <= 15:
            rush_bump = 24.0
        else:
            rush_bump = 8.0

        if step == 0:
            anchor_val = 25.0 if current_prediction == 'Low' else (55.0 if current_prediction == 'Moderate' else 85.0)
            val = anchor_val
            spd = avg_speed
        else:
            val = np.clip(rush_bump + np.random.normal(0, 3.5), 5.0, 95.0)
            spd = np.clip(85.0 - (val * 0.75) + np.random.normal(0, 2.5), 12.0, 105.0)

        projected_congestion_pct.append(round(val, 1))
        projected_speed.append(round(spd, 1))

    df_forecast = pd.DataFrame({
        'Time': time_labels,
        'HourOfDay': forecast_hours,
        'CongestionIndex': projected_congestion_pct,
        'ProjectedSpeed': projected_speed
    })

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_forecast['Time'],
        y=df_forecast['CongestionIndex'],
        mode='lines+markers',
        name='Projected Congestion Index (%)',
        line=dict(color='#00E5FF', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 229, 255, 0.12)',
        marker=dict(size=7, color='#00E5FF')
    ))

    fig.add_trace(go.Scatter(
        x=df_forecast['Time'],
        y=df_forecast['ProjectedSpeed'],
        mode='lines+markers',
        name='Projected Speed (km/h)',
        yaxis='y2',
        line=dict(color='#FFAB00', width=2, dash='dot'),
        marker=dict(size=6, color='#FFAB00')
    ))

    fig.add_hrect(y0=0, y1=40, fillcolor="#00E676", opacity=0.08, line_width=0, annotation_text="Low Band", annotation_position="top left")
    fig.add_hrect(y0=40, y1=70, fillcolor="#FFB300", opacity=0.08, line_width=0, annotation_text="Moderate Band", annotation_position="top left")
    fig.add_hrect(y0=70, y1=100, fillcolor="#FF1744", opacity=0.08, line_width=0, annotation_text="Severe Congestion Band", annotation_position="top left")

    fig.update_layout(
        title=dict(
            text="Illustrative 12-Hour Traffic Congestion Forecast (Synthetic Simulation Model)",
            font=dict(size=14, color="#E2E8F0")
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.7)",
        xaxis=dict(title="Simulated Forecast Horizon", showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(
            title="Congestion Probability / Index (%)",
            range=[0, 105],
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)"
        ),
        yaxis2=dict(
            title="Projected Speed (km/h)",
            overlaying='y',
            side='right',
            range=[0, 120],
            showgrid=False
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=50, b=10),
        height=380
    )

    return fig


def plot_confusion_matrix_heatmap(cm, class_names):
    """Renders a stylish confusion matrix heatmap with annotations."""
    z_text = [[str(val) for val in row] for row in cm]
    fig = px.imshow(
        cm,
        x=class_names,
        y=class_names,
        labels=dict(x="Predicted Class", y="True Ground-Truth Class", color="Count"),
        color_continuous_scale=[[0, '#0F172A'], [0.5, '#1E3A8A'], [1.0, '#00E5FF']],
        aspect="auto"
    )
    fig.update_traces(text=z_text, texttemplate="%{text}", textfont=dict(size=14, color="#FFFFFF"))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.7)",
        margin=dict(l=20, r=20, t=30, b=20),
        height=320
    )
    return fig


# ==============================================================================
# 4. MAIN APPLICATION WORKFLOW & UI TABS
# ==============================================================================

def main():
    # --- HERO HEADER ---
    st.markdown("""
    <div class="hero-container">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
            <div>
                <div class="hero-title">🚦 TrafficMind AI</div>
                <div class="hero-subtitle">
                    Intelligent Traffic Congestion Prediction & Dynamic Network Rerouting Engine
                </div>
                <div>
                    <span class="badge-pill badge-hackathon">🏆 PromptWars x Creative Community VVITU</span>
                    <span class="badge-pill badge-ai">⚡ Random Forest Classifier</span>
                    <span class="badge-pill badge-ai">🧭 NetworkX Dynamic Routing</span>
                    <span class="badge-pill badge-synthetic">🔬 Synthetic Benchmark Telemetry</span>
                </div>
            </div>
            <div style="text-align: right; margin-top: 10px;">
                <span style="font-size: 0.8rem; color: #94A3B8;">System Mode:</span><br>
                <span style="font-size: 0.95rem; font-weight: 700; color: #00E676;">🟢 LIVE SIMULATION ACTIVE</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load synthetic dataset and model via cached helpers
    df_traffic = load_dataset()
    clf, feature_names, metrics = load_model(df_traffic)
    G_base = load_network()

    # --- SIDEBAR: SIMULATION CONTROLS & PRESET SCENARIOS ---
    st.sidebar.markdown("""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
        <span style="font-size: 1.5rem;">⚙️</span>
        <span style="font-size: 1.15rem; font-weight: 700; color: #ECEFF1;">Control Console</span>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown('<div class="sidebar-section-title">⚡ Quick Scenario Presets</div>', unsafe_allow_html=True)
    preset_choice = st.sidebar.selectbox(
        "Load Demonstration Preset",
        [
            "Custom (Manual Sliders)",
            "Preset 1: Morning Commute Gridlock (8:30 AM, High Volume, Rain)",
            "Preset 2: Major Expressway Accident (Evening Rush, Highway)",
            "Preset 3: Emergency Green-Corridor (Hospital Priority Dispatch)",
            "Preset 4: Off-Peak Smooth Flow (Midnight, Clear)"
        ]
    )

    default_hour = 8
    default_day = "Monday"
    default_road = "Highway Expressway"
    default_volume = 1850
    default_speed = 34.0
    default_weather = "Rain"
    default_incident = "Minor Accident"

    if preset_choice == "Preset 1: Morning Commute Gridlock (8:30 AM, High Volume, Rain)":
        default_hour = 8
        default_day = "Monday"
        default_road = "Highway Expressway"
        default_volume = 2150
        default_speed = 22.0
        default_weather = "Rain"
        default_incident = "None"
    elif preset_choice == "Preset 2: Major Expressway Accident (Evening Rush, Highway)":
        default_hour = 17
        default_day = "Friday"
        default_road = "Highway Expressway"
        default_volume = 2300
        default_speed = 12.0
        default_weather = "Storm"
        default_incident = "Major Accident"
    elif preset_choice == "Preset 3: Emergency Green-Corridor (Hospital Priority Dispatch)":
        default_hour = 14
        default_day = "Wednesday"
        default_road = "Urban Arterial"
        default_volume = 1450
        default_speed = 38.0
        default_weather = "Clear"
        default_incident = "Road Work"
    elif preset_choice == "Preset 4: Off-Peak Smooth Flow (Midnight, Clear)":
        default_hour = 2
        default_day = "Sunday"
        default_road = "Highway Expressway"
        default_volume = 380
        default_speed = 98.0
        default_weather = "Clear"
        default_incident = "None"

    st.sidebar.markdown('<div class="sidebar-section-title">🕒 Temporal & Road Context</div>', unsafe_allow_html=True)
    input_hour = st.sidebar.slider("Hour of Day (24h Clock)", 0, 23, default_hour, help="Time of day determines baseline commuter volume.")
    input_day = st.sidebar.selectbox(
        "Day of Week",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        index=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(default_day)
    )
    is_weekend = input_day in ["Saturday", "Sunday"]

    input_road_type = st.sidebar.selectbox(
        "Corridor Classification",
        ["Highway Expressway", "Urban Arterial", "Downtown Grid", "Harbor Bridge"],
        index=["Highway Expressway", "Urban Arterial", "Downtown Grid", "Harbor Bridge"].index(default_road)
    )

    st.sidebar.markdown('<div class="sidebar-section-title">🚗 Traffic Sensors & Telemetry</div>', unsafe_allow_html=True)
    input_volume = st.sidebar.slider(
        "Vehicle Count (veh / hr / lane)",
        min_value=100,
        max_value=2600,
        value=default_volume,
        step=50,
        help="Volume of vehicles entering the target corridor per hour per lane."
    )
    input_speed = st.sidebar.slider(
        "Observed Average Speed (km/h)",
        min_value=5.0,
        max_value=120.0,
        value=float(default_speed),
        step=1.0,
        help="Live radar or GPS probe measured speed along the corridor."
    )

    st.sidebar.markdown('<div class="sidebar-section-title">🌦️ Environmental & Incident Factors</div>', unsafe_allow_html=True)
    input_weather = st.sidebar.selectbox(
        "Weather Condition",
        ["Clear", "Rain", "Fog", "Storm"],
        index=["Clear", "Rain", "Fog", "Storm"].index(default_weather)
    )
    input_incident = st.sidebar.selectbox(
        "Active Road Incident",
        ["None", "Minor Accident", "Road Work", "Vehicle Breakdown", "Major Accident"],
        index=["None", "Minor Accident", "Road Work", "Vehicle Breakdown", "Major Accident"].index(default_incident)
    )

    scenario_inputs = {
        'hour': input_hour,
        'day': input_day,
        'is_weekend': is_weekend,
        'road_type': input_road_type,
        'vehicle_count': input_volume,
        'avg_speed': input_speed,
        'weather': input_weather,
        'incident': input_incident
    }

    # Execute Model Inference via backend predictor
    X_user = encode_single_input(
        hour=input_hour,
        is_weekend=is_weekend,
        vehicle_count=input_volume,
        avg_speed=input_speed,
        weather=input_weather,
        incident=input_incident,
        road_type=input_road_type,
        feature_names=feature_names
    )

    prediction_result = predict_congestion(clf, X_user)
    pred_class = prediction_result["prediction"]
    pred_probs = prediction_result["probabilities"]
    prob_dict = prediction_result["prob_map"]
    confidence_score = prediction_result["confidence"]

    # Dynamic network simulation via backend routing
    G_dyn = simulate_network_conditions(G_base, scenario_inputs)

    speeds = [d['effective_speed'] for _, _, d in G_dyn.edges(data=True)]
    congestion_states = [d['congestion_state'] for _, _, d in G_dyn.edges(data=True)]
    network_avg_speed = round(float(np.mean(speeds)), 1)
    severe_count = congestion_states.count('Severe')
    moderate_count = congestion_states.count('Moderate')
    low_count = congestion_states.count('Low')

    # Top KPI cards
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    with m_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Predicted Congestion</div>
            <div class="metric-value" style="color: {'#00E676' if pred_class=='Low' else ('#FFAB00' if pred_class=='Moderate' else '#FF1744')};">
                {'🟢 Low' if pred_class=='Low' else ('🟡 Moderate' if pred_class=='Moderate' else '🔴 Severe')}
            </div>
            <div class="metric-sub">Confidence: {round(confidence_score, 1)}%</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Sensor Telemetry</div>
            <div class="metric-value">{input_speed} <span style="font-size:1rem; color:#94A3B8;">km/h</span></div>
            <div class="metric-sub">{input_volume} veh/hr/lane • {input_road_type.split()[0]}</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Metropolitan Network Speed</div>
            <div class="metric-value">{network_avg_speed} <span style="font-size:1rem; color:#94A3B8;">km/h</span></div>
            <div class="metric-sub">Across 20 Primary Urban Arterials</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Network Congestion Status</div>
            <div class="metric-value" style="font-size:1.3rem;">
                <span style="color:#FF1744;">{severe_count} Red</span> | 
                <span style="color:#FFAB00;">{moderate_count} Amber</span> | 
                <span style="color:#00E676;">{low_count} Green</span>
            </div>
            <div class="metric-sub">{input_weather} • Incident: {input_incident}</div>
        </div>
        """, unsafe_allow_html=True)

    # Navigation Tabs
    tab_pred, tab_network, tab_reroute, tab_emergency, tab_eval, tab_about = st.tabs([
        "🚦 Congestion Predictor & Simulator",
        "🗺️ Dynamic City Network Map",
        "🧭 AI Alternative Route Planner",
        "🚑 Emergency Green-Corridor",
        "📊 Model Evaluation & Transparency",
        "📘 Architecture & Hackathon Pitch"
    ])

    # ==========================================================================
    # TAB 1: CONGESTION PREDICTOR & WHAT-IF SIMULATOR
    # ==========================================================================
    with tab_pred:
        p_col1, p_col2 = st.columns([1.1, 1.0])

        with p_col1:
            st.markdown("### 🎯 Live Model Prediction & Driver Explanations")

            if pred_class == 'Low':
                st.markdown("""
                <div class="congestion-low">
                    <b>🟢 Optimal Traffic Flow (Low Congestion)</b><br>
                    Flow velocity is within 85-100% of nominal speed limit. Minimal queueing observed.
                    Dynamic routing recommends standard primary corridors.
                </div>
                """, unsafe_allow_html=True)
            elif pred_class == 'Moderate':
                st.markdown("""
                <div class="congestion-moderate">
                    <b>🟡 Moderate Congestion Warning</b><br>
                    Speed degradation detected due to elevated Volume-to-Capacity ratio. Minor localized bottlenecks.
                    AI proactive rerouting advised for time-sensitive travel.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="congestion-severe">
                    <b>🔴 Critical Severe Congestion Alert</b><br>
                    Gridlock or heavy deceleration detected. Flow velocity collapsed below 30 km/h.
                    Incident or extreme capacity saturation in effect. Dynamic alternative routing strongly advised!
                </div>
                """, unsafe_allow_html=True)

            st.write("")
            st.markdown("#### Model Probability Distribution")

            prob_df = pd.DataFrame({
                'Congestion Level': ['Low', 'Moderate', 'Severe'],
                'Probability (%)': [round(prob_dict.get(c, 0.0) * 100, 1) for c in ['Low', 'Moderate', 'Severe']],
                'Color': ['#00E676', '#FFB300', '#FF1744']
            })

            fig_prob = px.bar(
                prob_df,
                x='Probability (%)',
                y='Congestion Level',
                orientation='h',
                color='Congestion Level',
                color_discrete_map={'Low': '#00E676', 'Moderate': '#FFB300', 'Severe': '#FF1744'},
                text='Probability (%)'
            )
            fig_prob.update_traces(texttemplate='%{text}%', textposition='inside')
            fig_prob.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(15, 23, 42, 0.4)",
                plot_bgcolor="rgba(15, 23, 42, 0.7)",
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10),
                height=180,
                xaxis=dict(range=[0, 105], showgrid=False)
            )
            st.plotly_chart(fig_prob, use_container_width=True)

            # Feature Attribution Breakdown
            st.markdown("#### 🔍 What Drove This Prediction?")
            explanations = []
            if input_volume > 1800:
                explanations.append(f"• **High Volume Squeeze (+{int((input_volume-1800)/15)}% friction)**: Volume ({input_volume} veh/hr) approaches or exceeds corridor design capacity.")
            elif input_volume < 800:
                explanations.append(f"• **Light Traffic Demand**: Low volume ({input_volume} veh/hr) leaves surplus capacity.")

            if input_speed < 30:
                explanations.append(f"• **Critical Velocity Drop**: Observed speed ({input_speed} km/h) signals severe bottleneck or standing queue.")
            elif input_speed > 60:
                explanations.append(f"• **Healthy Velocity Flow**: High observed speed ({input_speed} km/h) strongly pulls prediction toward Low/Moderate.")

            if input_weather != 'Clear':
                explanations.append(f"• **Weather Degradation**: {input_weather} reduces pavement friction and driver headways, elevating congestion risk.")

            if input_incident != 'None':
                explanations.append(f"• **Active Road Incident ({input_incident})**: Physical lane closure causes bottleneck constriction.")

            if not explanations:
                explanations.append("• Balanced operational parameters within nominal tolerances.")

            for exp in explanations:
                st.markdown(exp)

        with p_col2:
            st.markdown("### 📈 Illustrative 12-Hour Forecast")
            fig_fc = plot_short_term_forecast(input_hour, pred_class, input_speed, input_volume)
            st.plotly_chart(fig_fc, use_container_width=True)

            st.markdown("""
            <div class="callout-box">
                <b>ℹ️ Transparency Note on Forecasting:</b><br>
                This short-term projection demonstrates how traffic profiles cycle through diurnal morning and evening peak hours.
                It is computed using the synthetic traffic physics model for rapid prototyping and algorithm verification.
            </div>
            """, unsafe_allow_html=True)

    # ==========================================================================
    # TAB 2: DYNAMIC CITY ROAD NETWORK MAP
    # ==========================================================================
    with tab_network:
        st.markdown("### 🗺️ Real-Time Metropolitan Road Network State")
        st.markdown("Visualizing 12 urban hubs and 20 major transport corridors. Edge colors reflect dynamic live congestion.")

        fig_net = plot_interactive_network(G_dyn, title=f"Metropolitan Road Network — Condition at {input_hour:02d}:00 ({input_weather})")
        st.plotly_chart(fig_net, use_container_width=True)

        st.markdown("#### 📋 Network Corridors Telemetry Inspection")
        corridor_rows = []
        for u, v, d in G_dyn.edges(data=True):
            corridor_rows.append({
                'Corridor Name': d['road_name'],
                'From Zone': u,
                'To Zone': v,
                'Distance (km)': d['distance_km'],
                'Speed Limit (km/h)': d['speed_limit'],
                'Live Speed (km/h)': d['effective_speed'],
                'Live Travel Time (min)': d['dynamic_time_min'],
                'Congestion Level': d['congestion_state']
            })
        df_corr = pd.DataFrame(corridor_rows).sort_values('Live Travel Time (min)', ascending=False)

        st.dataframe(
            df_corr,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Congestion Level": st.column_config.TextColumn(
                    "Congestion",
                    help="Traffic Status: Low, Moderate, Severe"
                )
            }
        )

    # ==========================================================================
    # TAB 3: AI ALTERNATIVE ROUTE PLANNER (NETWORKX)
    # ==========================================================================
    with tab_reroute:
        st.markdown("### 🧭 AI Congestion-Aware Alternative Route Planner")
        st.markdown(
            "Standard GPS often routes drivers along the shortest physical distance, leading straight into gridlocked bottlenecks. "
            "TrafficMind AI models dynamic corridor travel times to calculate the true fastest path."
        )

        node_list = sorted(list(G_dyn.nodes()))
        r_col1, r_col2, r_col3 = st.columns([1, 1, 1])

        with r_col1:
            origin_node = st.selectbox(
                "Origin Location",
                node_list,
                index=node_list.index("West End Industrial Park") if "West End Industrial Park" in node_list else 0
            )

        with r_col2:
            dest_node = st.selectbox(
                "Destination Location",
                node_list,
                index=node_list.index("International Airport Link") if "International Airport Link" in node_list else 1
            )

        with r_col3:
            st.write("")
            st.write("")
            recalc_btn = st.button("🚀 Re-calculate Optimal Path", use_container_width=True, type="primary")

        route_res = compute_route_comparison(G_dyn, origin_node, dest_node)

        if route_res is None:
            st.warning("Origin and Destination cannot be the same. Please choose distinct locations.")
        else:
            rc1, rc2, rc3, rc4 = st.columns(4)

            with rc1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Baseline Shortest Route</div>
                    <div class="metric-value">{route_res['baseline_time']} <span style="font-size:1rem; color:#94A3B8;">mins</span></div>
                    <div class="metric-sub">{route_res['baseline_dist']} km • {route_res['baseline_bottlenecks']} Bottlenecks</div>
                </div>
                """, unsafe_allow_html=True)

            with rc2:
                st.markdown(f"""
                <div class="metric-card" style="border-color: rgba(0, 229, 255, 0.4);">
                    <div class="metric-label">TrafficMind AI Dynamic Route</div>
                    <div class="metric-value" style="color: #00E5FF;">{route_res['ai_time']} <span style="font-size:1rem; color:#94A3B8;">mins</span></div>
                    <div class="metric-sub">{route_res['ai_dist']} km • {route_res['ai_bottlenecks']} Bottlenecks</div>
                </div>
                """, unsafe_allow_html=True)

            with rc3:
                time_saved = route_res['time_saved_min']
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Travel Time Saved</div>
                    <div class="metric-value" style="color: {'#00E676' if time_saved > 0 else '#B0BEC5'};">
                        {'+' if time_saved > 0 else ''}{time_saved} <span style="font-size:1rem; color:#94A3B8;">mins</span>
                    </div>
                    <div class="metric-sub">{route_res['pct_time_saved']}% Faster Arrival</div>
                </div>
                """, unsafe_allow_html=True)

            with rc4:
                rerouted = route_res['is_rerouted']
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">AI Rerouting Decision</div>
                    <div class="metric-value" style="font-size:1.3rem; color: {'#00E5FF' if rerouted else '#B0BEC5'};">
                        {'⚡ Dynamic Bypass Active' if rerouted else '✅ Baseline is Already Optimal'}
                    </div>
                    <div class="metric-sub">{'Circumvented Heavy Congestion' if rerouted else 'No Faster Detour Available'}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### 🗺️ Route Path Visualization")
            fig_route_map = plot_interactive_network(
                G_dyn,
                highlighted_route=route_res['ai_path'],
                title=f"AI Route: {origin_node} ➔ {dest_node} (Highlighted in Neon Cyan)"
            )
            st.plotly_chart(fig_route_map, use_container_width=True)

            t1, t2 = st.columns(2)
            with t1:
                st.markdown("##### 📍 Baseline Route Turn-by-Turn")
                for idx, step in enumerate(route_res['baseline_path']):
                    st.markdown(f"**Step {idx+1}:** {step}")

            with t2:
                st.markdown("##### ⚡ TrafficMind AI Dynamic Route")
                for idx, step in enumerate(route_res['ai_path']):
                    st.markdown(f"**Step {idx+1}:** {step}")

    # ==========================================================================
    # TAB 4: SIMULATED EMERGENCY GREEN-CORRIDOR PLANNER
    # ==========================================================================
    with tab_emergency:
        st.markdown("### 🚑 Emergency Green-Corridor & Signal Preemption Simulator")
        st.markdown(
            "During critical medical emergencies or major structural fires, every minute lost in gridlock costs lives. "
            "TrafficMind AI simulates an intelligent **Green-Corridor Dispatch System**, computing preemptive signal clearance "
            "and dynamic lane-clearing trajectories directly to Trauma Centers and Incident Locations."
        )

        e_col1, e_col2, e_col3 = st.columns(3)

        with e_col1:
            emerg_vehicle = st.selectbox(
                "Emergency Vehicle Unit",
                [
                    "🚑 Critical Trauma Ambulance",
                    "🚒 Heavy Rescue Fire Engine",
                    "🚓 High-Priority Police Escort"
                ]
            )

        with e_col2:
            emerg_origin = st.selectbox(
                "Incident / Dispatch Origin",
                sorted(list(G_dyn.nodes())),
                index=sorted(list(G_dyn.nodes())).index("North Harbor Crossing")
            )

        with e_col3:
            emerg_destination = st.selectbox(
                "Target Emergency Receiving Facility",
                ["City General Trauma Hospital", "Central Fire & Emergency HQ"],
                index=0
            )

        emerg_res = compute_emergency_green_corridor(G_dyn, emerg_origin, emerg_destination)

        if emerg_res is None:
            st.warning("Dispatch Origin and Hospital cannot be identical.")
        else:
            ek1, ek2, ek3, ek4 = st.columns(4)

            with ek1:
                st.markdown(f"""
                <div class="metric-card" style="border-color: rgba(255, 82, 82, 0.4);">
                    <div class="metric-label">Normal Traffic ETA</div>
                    <div class="metric-value" style="color: #FF5252;">{emerg_res['civilian_time']} <span style="font-size:1rem; color:#94A3B8;">mins</span></div>
                    <div class="metric-sub">Without Signal Preemption</div>
                </div>
                """, unsafe_allow_html=True)

            with ek2:
                st.markdown(f"""
                <div class="metric-card" style="border-color: rgba(29, 233, 182, 0.4);">
                    <div class="metric-label">Green-Corridor AI ETA</div>
                    <div class="metric-value" style="color: #1DE9B6;">{emerg_res['emergency_time']} <span style="font-size:1rem; color:#94A3B8;">mins</span></div>
                    <div class="metric-sub">With Priority Signal Wave</div>
                </div>
                """, unsafe_allow_html=True)

            with ek3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Golden Hour Minutes Saved</div>
                    <div class="metric-value" style="color: #00E676;">
                        +{emerg_res['time_saved_min']} <span style="font-size:1rem; color:#94A3B8;">mins</span>
                    </div>
                    <div class="metric-sub">Critical Life-Saving Margin</div>
                </div>
                """, unsafe_allow_html=True)

            with ek4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Signals Preempted</div>
                    <div class="metric-value" style="color: #00E5FF;">
                        {emerg_res['signals_preempted']} <span style="font-size:1rem; color:#94A3B8;">Intersections</span>
                    </div>
                    <div class="metric-sub">Automated Green-Light Wave</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### 🗺️ Priority Green Corridor Live Trajectory")
            fig_emerg_map = plot_interactive_network(
                G_dyn,
                emergency_route=emerg_res['emergency_path'],
                title=f"Green-Corridor Wave: {emerg_origin} ➔ {emerg_destination} (Highlighted in Neon Emerald)"
            )
            st.plotly_chart(fig_emerg_map, use_container_width=True)

            st.markdown("##### 🚨 Green-Corridor Trajectory Intersections")
            path_str = " ➔ ".join([f"`{node}`" for node in emerg_res['emergency_path']])
            st.markdown(path_str)

    # ==========================================================================
    # TAB 5: MODEL EVALUATION & SYNTHETIC DATA TRANSPARENCY
    # ==========================================================================
    with tab_eval:
        st.markdown("### 📊 Model Performance on Held-Out Test Data")

        st.markdown("""
        <div class="callout-box" style="border-color: rgba(255, 171, 0, 0.4);">
            <b>🔬 Scientific Transparency & Synthetic Data Disclosure:</b><br>
            Real municipal induction-loop datasets and city probe telemetry are frequently proprietary, geographically restricted,
            or subject to privacy masking. To ensure <b>100% reproducible evaluation</b> for this hackathon, TrafficMind AI generates
            and benchmarks against a physics-inspired synthetic dataset (Greenshields speed-volume-capacity equations).<br>
            <i>Test metrics below strictly evaluate model fidelity on the synthetic benchmark distribution and should not be presented as validated real-world municipal operational results without field sensor calibration.</i>
        </div>
        """, unsafe_allow_html=True)

        ev_c1, ev_c2, ev_c3, ev_c4 = st.columns(4)

        with ev_c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Test Set Accuracy</div>
                <div class="metric-value" style="color:#00E5FF;">{round(metrics['test_acc']*100, 2)}%</div>
                <div class="metric-sub">Held-out 20% Evaluation Split</div>
            </div>
            """, unsafe_allow_html=True)

        with ev_c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Training Set Accuracy</div>
                <div class="metric-value">{round(metrics['train_acc']*100, 2)}%</div>
                <div class="metric-sub">Controlled Overfitting (max_depth=12)</div>
            </div>
            """, unsafe_allow_html=True)

        with ev_c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Macro F1-Score</div>
                <div class="metric-value" style="color:#00E676;">
                    {round(np.mean(list(metrics['f1'].values())), 3)}
                </div>
                <div class="metric-sub">Balanced Across All 3 Classes</div>
            </div>
            """, unsafe_allow_html=True)

        with ev_c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Synthetic Benchmark Size</div>
                <div class="metric-value">{len(df_traffic):,} <span style="font-size:1rem; color:#94A3B8;">records</span></div>
                <div class="metric-sub">Multi-variable physics simulation</div>
            </div>
            """, unsafe_allow_html=True)

        col_cm, col_fi = st.columns([1, 1])

        with col_cm:
            st.markdown("#### 🎯 Confusion Matrix Heatmap")
            fig_cm = plot_confusion_matrix_heatmap(metrics['confusion_matrix'], metrics['class_names'])
            st.plotly_chart(fig_cm, use_container_width=True)

            st.markdown("##### Class-wise Precision, Recall & F1-Score")
            class_metrics_df = pd.DataFrame({
                'Class': metrics['class_names'],
                'Precision': [round(metrics['precision'][c], 3) for c in metrics['class_names']],
                'Recall': [round(metrics['recall'][c], 3) for c in metrics['class_names']],
                'F1-Score': [round(metrics['f1'][c], 3) for c in metrics['class_names']]
            })
            st.dataframe(class_metrics_df, use_container_width=True, hide_index=True)

        with col_fi:
            st.markdown("#### 🌲 Random Forest Feature Importances (Gini)")
            fi_df = metrics['feature_importances'].head(10).sort_values('importance', ascending=True)

            fig_fi = px.bar(
                fi_df,
                x='importance',
                y='feature',
                orientation='h',
                color='importance',
                color_continuous_scale=[[0, '#1E293B'], [1.0, '#00E5FF']]
            )
            fig_fi.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(15, 23, 42, 0.4)",
                plot_bgcolor="rgba(15, 23, 42, 0.7)",
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10),
                height=320,
                xaxis=dict(title="Relative Importance", showgrid=True),
                yaxis=dict(title="")
            )
            st.plotly_chart(fig_fi, use_container_width=True)

            st.caption("Top predictive features: Average Speed, Vehicle Volume, Road Incidents, and Hour of Day.")

    # ==========================================================================
    # TAB 6: HACKATHON PITCH, ARCHITECTURE & ROADMAP
    # ==========================================================================
    with tab_about:
        st.markdown("### 📘 System Architecture & Hackathon Pitch")

        st.markdown("""
        #### 💡 The Problem
        Urban traffic congestion costs global metropolitan economies over **$300 Billion annually** in wasted fuel, 
        lost productivity, and elevated carbon emissions. Furthermore, emergency vehicles (ambulances, fire engines) 
        lose crucial minutes navigating through static GPS routes that fail to anticipate dynamic gridlocks.

        #### 🚀 The Solution: TrafficMind AI
        TrafficMind AI bridges the gap between **predictive machine learning** and **dynamic graph theory**:
        1. **Predictive Layer**: A Scikit-Learn Random Forest model ingests temporal features (hour, day), physical sensor telemetry (speed, volume), weather conditions, and incident status to predict congestion severity with high fidelity.
        2. **Explainability Layer**: Real-time feature attribution breaks down why congestion is escalating (e.g., volume surge vs. rain-induced capacity drop).
        3. **Dynamic Network Layer**: NetworkX models the city as a weighted graph where edge costs reflect live travel time and delay friction rather than static distance.
        4. **Emergency Green-Corridor**: Preemptively computes signal clearance waves to provide rapid hospital dispatch trajectories.
        """)

        st.markdown("#### 🏗️ Architecture Pipeline")
        st.code("""
  +-------------------------------------------------------------------------+
  |                        1. TELEMETRY INGESTION                           |
  |  Time of Day  |  Vehicle Count  |  Speed  |  Weather  |  Road Incidents |
  +-------------------------------------------------------------------------+
                                      |
                                      v
  +-------------------------------------------------------------------------+
  |                 2. PREDICTIVE AI ENGINE (backend.predictor)             |
  |   Multivariate Feature Extraction -> Classification (Low/Mod/Severe)    |
  |        Confidence Scores -> 12-Hour Illustrative Horizon                |
  +-------------------------------------------------------------------------+
                                      |
                                      v
  +-------------------------------------------------------------------------+
  |              3. DYNAMIC GRAPH & REROUTING ENGINE (backend.routing)      |
  |    Dynamic Edge Cost = Distance / EffectiveSpeed + IncidentPenalties     |
  |        Civilian AI Detour  <---- VS ---->  Emergency Green Corridor     |
  +-------------------------------------------------------------------------+
                                      |
                                      v
  +-------------------------------------------------------------------------+
  |               4. COMMAND CENTER INTERFACE (frontend/app.py)             |
  |     Interactive Network Visualizer | Turn-by-Turn | Model Telemetry     |
  +-------------------------------------------------------------------------+
        """, language="text")

        st.markdown("""
        #### 🔮 Future Municipal Deployment Roadmap
        - **Real-Time Sensor Ingestion**: MQTT / Kafka pipelines connecting inductive loops, CCTV computer vision counters, and connected vehicle probes (V2X).
        - **SUMO / OpenStreetMap Integration**: Microscopic traffic simulation importing real-world OpenStreetMap vector networks.
        - **Reinforcement Learning Signal Control**: Multi-agent reinforcement learning (MARL) for adaptive traffic light timing optimization.
        """)


if __name__ == "__main__":
    main()
