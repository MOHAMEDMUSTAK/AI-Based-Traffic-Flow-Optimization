"""
AI Smart Traffic Management System
===================================
Run with:  streamlit run traffic_app.py
Requires:  pip install streamlit streamlit-autorefresh plotly numpy pandas
No CSV needed — synthetic traffic is generated internally.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math
import time

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Traffic Management",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Compact metric cards */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 12px 16px;
    }
    /* Road signal cards */
    .signal-card {
        border-radius: 12px;
        padding: 16px;
        margin: 4px 0;
        font-family: sans-serif;
    }
    .green-card {
        background: linear-gradient(135deg, #0f3d2e 0%, #1a5c3e 100%);
        border: 1.5px solid #1D9E75;
        box-shadow: 0 0 16px rgba(29,158,117,0.25);
    }
    .red-card {
        background: rgba(226,75,74,0.08);
        border: 1px solid rgba(226,75,74,0.3);
    }
    .section-header {
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #888;
        margin: 1rem 0 0.5rem;
    }
    /* Ticker banner */
    .ticker {
        background: rgba(55,138,221,0.12);
        border: 1px solid rgba(55,138,221,0.3);
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 13px;
        margin-bottom: 12px;
        color: #378ADD;
    }
    .ticker-emergency {
        background: rgba(239,159,39,0.12);
        border: 1px solid rgba(239,159,39,0.3);
        color: #EF9F27;
    }
    .ticker-accident {
        background: rgba(226,75,74,0.12);
        border: 1px solid rgba(226,75,74,0.3);
        color: #E24B4A;
    }
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
ROADS = ["North", "South", "East", "West"]
ROAD_ICONS = {"North": "⬆", "South": "⬇", "East": "➡", "West": "⬅"}
WEATHER_OPTIONS = ["Clear", "Cloudy", "Fog", "Rain", "Snow"]
WEATHER_ICONS = {"Clear": "☀️", "Cloudy": "⛅", "Fog": "🌫️", "Rain": "🌧️", "Snow": "❄️"}
WEATHER_FACTORS = {"Clear": 1.0, "Cloudy": 1.05, "Fog": 1.12, "Rain": 1.28, "Snow": 1.45}
COLORS = {"North": "#378ADD", "South": "#1D9E75", "East": "#D85A30", "West": "#7F77DD"}

SERVICE_RATE = 5
MIN_GREEN = 10
MAX_GREEN = 60
MAX_HISTORY = 40

# ─────────────────────────────────────────────
# TRAFFIC MODEL  (no CSV needed)
# ─────────────────────────────────────────────
def predict_traffic(hour: int, weather: str) -> float:
    """Sinusoidal rush-hour model with dual peaks (8am & 6pm)."""
    morning_peak = 3500 * math.exp(-0.5 * ((hour - 8) / 1.5) ** 2)
    evening_peak = 4200 * math.exp(-0.5 * ((hour - 18) / 1.8) ** 2)
    base = 800 + morning_peak + evening_peak + np.random.normal(0, 150)
    return max(100.0, base * WEATHER_FACTORS.get(weather, 1.0))


def compute_road_traffic(
    manual_vols: Dict[str, int],
    hour: int,
    weather: str,
    accident_road: str,
) -> Dict[str, int]:
    factor = predict_traffic(hour, weather) / 1000.0
    traffic = {}
    for r in ROADS:
        v = manual_vols[r] + factor * 50 + np.random.uniform(-8, 8)
        if accident_road == r:
            v *= 0.5
        traffic[r] = max(1, int(round(v)))
    return traffic


def congestion_level(v: int):
    if v < 40:
        return "Low", "🟢", "#1D9E75"
    elif v < 80:
        return "Medium", "🟡", "#EF9F27"
    else:
        return "High", "🔴", "#E24B4A"


def compute_green_time(queue: int) -> int:
    return int(max(MIN_GREEN, min(MAX_GREEN, queue / SERVICE_RATE)))


def compute_priority(traffic: Dict[str, int], waiting: Dict[str, int], served: Dict[str, int]) -> Dict[str, float]:
    return {r: traffic[r] + waiting[r] + served[r] * 2 for r in ROADS}


def select_green_road(traffic: Dict[str, int], emergency: str) -> str:
    """Give green to the road with the highest vehicle count.

    Emergency override is retained. During normal operation, traffic volume
    alone determines the selected road so a lower-volume road cannot win
    merely because it has a larger fairness/served score.
    """
    if emergency != "None":
        return emergency
    return max(traffic, key=traffic.get)

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
def init_state():
    defaults = {
        "tick": 0,
        "hour": 8,
        "weather": "Clear",
        "served": {r: 0 for r in ROADS},
        "history": [],          # list of dicts with road volumes per tick
        "waiting_history": [],
        "green_road": "North",
        "green_time": 30,
        "last_traffic": {r: 40 for r in ROADS},
        "last_priority": {r: 0.0 for r in ROADS},
        "last_waiting": {r: 15 for r in ROADS},
        "event_log": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("🚦 Traffic Control")

    st.markdown("### ⚙️ Simulation")
    sim_speed = st.slider("Refresh interval (ms)", 500, 5000, 2000, 250)
    auto_weather = st.toggle("Auto weather changes", value=True)
    if not auto_weather:
        manual_weather = st.selectbox("Weather", WEATHER_OPTIONS, index=0)

    st.markdown("### 🎚️ Manual Road Volumes")
    manual_vols = {}
    for r in ROADS:
        manual_vols[r] = st.slider(f"{ROAD_ICONS[r]} {r}", 0, 100, {"North": 40, "South": 30, "East": 20, "West": 10}[r])

    st.markdown("### 🚑 Emergency Override")
    emergency_road = st.selectbox("Emergency road", ["None"] + ROADS)

    st.markdown("### 🚧 Accident")
    accident_road = st.selectbox("Accident road", ["None"] + ROADS)

    st.divider()
    if st.button("🔄 Reset System", use_container_width=True):
        for k in ["tick","hour","weather","served","history","waiting_history",
                  "green_road","green_time","last_traffic","last_priority","last_waiting","event_log"]:
            del st.session_state[k]
        init_state()
        st.rerun()

# ─────────────────────────────────────────────
# AUTO REFRESH
# ─────────────────────────────────────────────
st_autorefresh(interval=sim_speed, key="traffic_refresh")

# ─────────────────────────────────────────────
# SIMULATION STEP
# ─────────────────────────────────────────────
st.session_state.tick += 1
st.session_state.hour = (st.session_state.hour + 1) % 24

# Weather update every 10 ticks
if auto_weather and st.session_state.tick % 10 == 0:
    weights = [0.4, 0.25, 0.1, 0.15, 0.1]
    st.session_state.weather = np.random.choice(WEATHER_OPTIONS, p=weights)
elif not auto_weather:
    st.session_state.weather = manual_weather

weather = st.session_state.weather
hour = st.session_state.hour

traffic = compute_road_traffic(manual_vols, hour, weather, accident_road)
waiting = {r: np.random.randint(5, 35) for r in ROADS}
priority = compute_priority(traffic, waiting, st.session_state.served)
green_road = select_green_road(traffic, emergency_road)
green_time = MAX_GREEN if emergency_road != "None" else compute_green_time(traffic[green_road])
st.session_state.served[green_road] = st.session_state.served.get(green_road, 0) + 1
st.session_state.green_road = green_road
st.session_state.green_time = green_time
st.session_state.last_traffic = traffic
st.session_state.last_priority = priority
st.session_state.last_waiting = waiting

# Append history (keep last MAX_HISTORY rows)
row = {"tick": st.session_state.tick, **{r: traffic[r] for r in ROADS}}
st.session_state.history.append(row)
if len(st.session_state.history) > MAX_HISTORY:
    st.session_state.history.pop(0)

# Event log
event = f"T{st.session_state.tick:03d} | {hour:02d}:00 | {WEATHER_ICONS[weather]} {weather} | 🟢 {green_road} ({green_time}s)"
if emergency_road != "None":
    event += f" | 🚑 Emergency: {emergency_road}"
if accident_road != "None":
    event += f" | ⚠️ Accident: {accident_road}"
st.session_state.event_log.insert(0, event)
if len(st.session_state.event_log) > 12:
    st.session_state.event_log.pop()

# ─────────────────────────────────────────────
# RENDER — HEADER
# ─────────────────────────────────────────────
col_title, col_tick = st.columns([3, 1])
with col_title:
    st.markdown("## 🚦 AI Smart Traffic Management")

with col_tick:
    st.markdown(f"**Tick** `{st.session_state.tick}` &nbsp;|&nbsp; **Hour** `{hour:02d}:00`")

# Ticker banner
ticker_class = "ticker"
ticker_msg = f"🟢 GREEN: **{green_road}** for {green_time}s &nbsp;·&nbsp; {WEATHER_ICONS[weather]} {weather} &nbsp;·&nbsp; Hour {hour:02d}:00"
if emergency_road != "None":
    ticker_class += " ticker-emergency"
    ticker_msg = f"🚑 Emergency override active on **{emergency_road}** — all other roads RED"
elif accident_road != "None":
    ticker_class += " ticker-accident"
    ticker_msg += f" &nbsp;·&nbsp; ⚠️ Accident on **{accident_road}** (50% flow)"
st.markdown(f'<div class="{ticker_class}">{ticker_msg}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# RENDER — SYSTEM METRICS
# ─────────────────────────────────────────────
total_vol = sum(traffic.values())
avg_wait = int(np.mean(list(waiting.values())))
peak_road = max(traffic, key=traffic.get)
throughput = st.session_state.served[green_road]

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total volume", total_vol)
m2.metric("Avg wait", f"{avg_wait}s")
m3.metric("Active green", green_road, f"{green_time}s")
m4.metric("Peak road", peak_road, f"{traffic[peak_road]} veh")
m5.metric("Weather", f"{WEATHER_ICONS[weather]} {weather}")

st.markdown("---")

# ─────────────────────────────────────────────
# RENDER — SIGNAL CARDS
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Live Signal Status</div>', unsafe_allow_html=True)
cols = st.columns(4)
for i, road in enumerate(ROADS):
    is_green = road == green_road
    cong_label, cong_icon, cong_color = congestion_level(traffic[road])
    card_class = "signal-card " + ("green-card" if is_green else "red-card")
    signal = "🟢 GREEN" if is_green else "🔴 RED"
    time_str = f"({green_time}s)" if is_green else ""
    emergency_str = "🚑 Emergency override" if emergency_road == road else ""
    accident_str = "⚠️ Accident — 50% flow" if accident_road == road else ""

    with cols[i]:
        st.markdown(f"""
        <div class="{card_class}">
            <div style="font-size:18px;font-weight:700;">{ROAD_ICONS[road]} {road}</div>
            <div style="font-size:28px;font-weight:700;margin:4px 0;">{traffic[road]}</div>
            <div style="font-size:13px;color:#aaa;">vol &nbsp;|&nbsp; wait: {waiting[road]}s</div>
            <div style="margin-top:8px;font-size:15px;font-weight:600;">{signal} {time_str}</div>
            <div style="font-size:13px;color:{cong_color};margin-top:4px;">{cong_icon} {cong_label} congestion</div>
            <div style="font-size:12px;color:#888;margin-top:4px;">Priority: {priority[road]:.0f} &nbsp;|&nbsp; Served: {st.session_state.served[road]}</div>
            {"<div style='font-size:12px;color:#EF9F27;margin-top:4px;'>"+emergency_str+"</div>" if emergency_str else ""}
            {"<div style='font-size:12px;color:#E24B4A;margin-top:4px;'>"+accident_str+"</div>" if accident_str else ""}
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# RENDER — CHARTS
# ─────────────────────────────────────────────
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown('<div class="section-header">Live Traffic History</div>', unsafe_allow_html=True)
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history).set_index("tick")
        fig_hist = go.Figure()
        for road in ROADS:
            fig_hist.add_trace(go.Scatter(
                x=hist_df.index, y=hist_df[road],
                name=road, line=dict(color=COLORS[road], width=2),
                mode="lines", fill="none",
                hovertemplate=f"{road}: %{{y}}<extra></extra>"
            ))
        fig_hist.update_layout(
            height=250, margin=dict(l=0,r=0,t=10,b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.2, font=dict(size=11)),
            xaxis=dict(title="Tick", gridcolor="rgba(128,128,128,0.15)", color="#888"),
            yaxis=dict(title="Volume", gridcolor="rgba(128,128,128,0.15)", color="#888"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

with chart_col2:
    st.markdown('<div class="section-header">6-Hour Traffic Forecast</div>', unsafe_allow_html=True)
    future_hours = [(hour + i) % 24 for i in range(1, 7)]
    future_vols = [int(predict_traffic(h, weather)) for h in future_hours]
    fig_fore = go.Figure(go.Bar(
        x=[f"{h:02d}:00" for h in future_hours],
        y=future_vols,
        marker=dict(color="#378ADD", opacity=0.7),
        hovertemplate="Hour %{x}: %{y} veh<extra></extra>"
    ))
    fig_fore.update_layout(
        height=250, margin=dict(l=0,r=0,t=10,b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(128,128,128,0.15)", color="#888"),
        yaxis=dict(gridcolor="rgba(128,128,128,0.15)", color="#888"),
    )
    st.plotly_chart(fig_fore, use_container_width=True)

# ─────────────────────────────────────────────
# RENDER — PRIORITY BAR CHART + AI EXPLANATION
# ─────────────────────────────────────────────
detail_col1, detail_col2 = st.columns([1.5, 1])

with detail_col1:
    st.markdown('<div class="section-header">Priority Score Breakdown</div>', unsafe_allow_html=True)
    prio_df = pd.DataFrame({
        "Road": ROADS,
        "Traffic vol": [traffic[r] for r in ROADS],
        "Wait time": [waiting[r] for r in ROADS],
        "Fairness adj": [st.session_state.served[r] * 2 for r in ROADS],
    })
    fig_prio = go.Figure()
    fig_prio.add_trace(go.Bar(name="Traffic vol",  x=prio_df["Road"], y=prio_df["Traffic vol"],  marker_color="#378ADD"))
    fig_prio.add_trace(go.Bar(name="Wait time",    x=prio_df["Road"], y=prio_df["Wait time"],    marker_color="#EF9F27"))
    fig_prio.add_trace(go.Bar(name="Fairness adj", x=prio_df["Road"], y=prio_df["Fairness adj"], marker_color="#7F77DD"))
    fig_prio.update_layout(
        barmode="stack", height=220, margin=dict(l=0,r=0,t=10,b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.25, font=dict(size=11)),
        xaxis=dict(gridcolor="rgba(128,128,128,0.15)", color="#888"),
        yaxis=dict(gridcolor="rgba(128,128,128,0.15)", color="#888"),
    )
    st.plotly_chart(fig_prio, use_container_width=True)

with detail_col2:
    st.markdown('<div class="section-header">AI Decision Summary</div>', unsafe_allow_html=True)
    base_pred = predict_traffic(hour, weather)
    mode = ("Emergency override 🚑" if emergency_road != "None"
            else "Accident management ⚠️" if accident_road != "None"
            else "Normal priority queue ✅")
    st.info(f"""
**Active road:** {green_road} {ROAD_ICONS[green_road]}  
**Green time:** {green_time}s  
**Traffic vol:** {traffic[green_road]}  
**Priority score:** {priority[green_road]:.0f}  
**Base prediction:** {int(base_pred)} veh  
**Weather factor:** ×{WEATHER_FACTORS[weather]:.2f}  
**Mode:** {mode}
""")

# ─────────────────────────────────────────────
# RENDER — EVENT LOG
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">Event Log</div>', unsafe_allow_html=True)
log_text = "\n".join(st.session_state.event_log[:8])
st.code(log_text, language=None)