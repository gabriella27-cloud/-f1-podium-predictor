# =============================================================
#  F1 Podium Predictor — Streamlit App
#  ISB02303402 Data Science Capstone Project
#  Formula 1 World Championship Dataset (Kaggle, 2008-2024)
# =============================================================

import warnings
warnings.filterwarnings("ignore")

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

# ─────────────────────────────────────────────────────────────
# 0. PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Podium Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# 1. CHECK REQUIRED FILES
# ─────────────────────────────────────────────────────────────
REQUIRED_FILES = [
    "f1_podium_rf_model.pkl",
    "driver_lookup.csv",
    "podium_history.csv",
    "season_stats.csv",
]
missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
if missing:
    st.error(f"❌ File tidak ditemukan: {', '.join(missing)}")
    st.stop()

# ─────────────────────────────────────────────────────────────
# 2. LOAD MODEL & DATA
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("f1_podium_rf_model.pkl")

@st.cache_data
def load_data():
    driver_lookup  = pd.read_csv("driver_lookup.csv")
    podium_history = pd.read_csv("podium_history.csv")
    season_stats   = pd.read_csv("season_stats.csv")
    return driver_lookup, podium_history, season_stats

rf = load_model()
driver_lookup, podium_history, season_stats = load_data()

FEATURES = ["grid", "driver_avg_points", "driver_avg_grid", "round", "nationality_encoded"]

# ─────────────────────────────────────────────────────────────
# 3. HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────
def predict_podium(grid_pos, avg_pts, avg_grid, race_round, nat_enc):
    feat = np.array([[grid_pos, avg_pts, avg_grid, race_round, nat_enc]])
    return float(rf.predict_proba(feat)[0][1])

def prob_color(prob):
    if prob >= 0.60: return "#28a745"
    if prob >= 0.35: return "#ffc107"
    return "#dc3545"

# Warna tema — didefinisikan sebagai variabel biasa, TIDAK sebagai dict
DARK_BG   = "#111111"
DARK_PLOT = "#1a1a1a"
GRID_CLR  = "#2a2a2a"
TEXT_CLR  = "#cccccc"

def base_layout(title_text, height=320):
    """Kembalikan dict layout Plotly yang bersih tanpa duplikat key."""
    return dict(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_PLOT,
        font=dict(color=TEXT_CLR, size=11),
        margin=dict(l=10, r=10, t=45, b=10),
        height=height,
        title=dict(text=title_text, font=dict(color="white", size=12)),
        xaxis=dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR),
        yaxis=dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR),
        legend=dict(font=dict(color=TEXT_CLR), bgcolor=DARK_PLOT, bordercolor=GRID_CLR),
        showlegend=True,
    )

# ─────────────────────────────────────────────────────────────
# 4. SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏎️ Race Parameters")
    st.divider()

    driver_list     = sorted(driver_lookup["driver_name"].tolist())
    default_idx     = driver_list.index("Lando Norris") if "Lando Norris" in driver_list else 0
    selected_driver = st.selectbox("👤 Driver", driver_list, index=default_idx)

    st.divider()
    grid_pos   = st.slider("🔢 Grid Position",        1, 20, 5,
                           help="1 = Pole Position, 20 = last")
    race_round = st.slider("📅 Race Round (Season)",  1, 24, 10,
                           help="1 = season opener")

    st.divider()
    st.markdown("**⚙️ Advanced Override**")
    use_custom = st.checkbox("Override driver stats manually", value=False)

    driver_row = driver_lookup[driver_lookup["driver_name"] == selected_driver].iloc[0]

    if use_custom:
        avg_pts  = st.slider("Avg Career Pts / Race", 0.0, 20.0,
                             float(round(driver_row["driver_avg_points"], 2)), 0.1)
        avg_grid = st.slider("Avg Career Grid Pos",   1.0, 20.0,
                             float(round(driver_row["driver_avg_grid"],   2)), 0.1)
    else:
        avg_pts  = float(driver_row["driver_avg_points"])
        avg_grid = float(driver_row["driver_avg_grid"])

    nat_enc = float(driver_row["nationality_encoded"])

    st.divider()
    st.caption("ISB02303402 – Data Science Capstone\n"
               "Formula 1 Dataset · Kaggle · 2008–2024")

# ─────────────────────────────────────────────────────────────
# 5. HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("# 🏎️ F1 Podium Predictor")
st.markdown(
    "**ISB02303402 Data Science Capstone** · "
    "Random Forest Classifier · Formula 1 (2008–2024)"
)
st.divider()

# ─────────────────────────────────────────────────────────────
# 6. PREDICTION
# ─────────────────────────────────────────────────────────────
prob  = predict_podium(grid_pos, avg_pts, avg_grid, race_round, nat_enc)
color = prob_color(prob)

if prob >= 0.60:
    verdict = "🟢 HIGH PROBABILITY"
elif prob >= 0.35:
    verdict = "🟡 MODERATE PROBABILITY"
else:
    verdict = "🔴 LOW PROBABILITY"

# ─────────────────────────────────────────────────────────────
# 7. ROW 1 — Prediction  |  Driver Profile
# ─────────────────────────────────────────────────────────────
col_pred, col_profile = st.columns([1, 1.4])

with col_pred:
    st.subheader(f"Prediction — {selected_driver}")
    st.metric(label=verdict, value=f"{prob * 100:.1f}%")
    st.caption(
        f"Grid **P{grid_pos}** · Round **{race_round}** · "
        f"Avg Pts **{avg_pts:.2f}** · Avg Grid **{avg_grid:.2f}**"
    )

    # Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 38, "color": "white"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": TEXT_CLR,
                     "tickfont": {"color": TEXT_CLR}},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": DARK_PLOT,
            "borderwidth": 0,
            "steps": [
                {"range": [0,  35], "color": "#1a0a0a"},
                {"range": [35, 60], "color": "#1a1500"},
                {"range": [60, 100],"color": "#0a1a0a"},
            ],
        },
        title={"text": "Podium Probability",
               "font": {"color": TEXT_CLR, "size": 13}},
    ))
    fig_gauge.update_layout(
        paper_bgcolor=DARK_BG,
        font={"color": TEXT_CLR},
        height=260,
        margin=dict(l=20, r=20, t=30, b=10),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_profile:
    st.subheader("Driver Profile")

    drv_hist      = podium_history[podium_history["driver_name"] == selected_driver]
    total_podiums = int(drv_hist["podium"].sum())

    m1, m2, m3 = st.columns(3)
    m1.metric("🏆 Podiums",      total_podiums)
    m2.metric("📈 Avg Pts/Race", f"{avg_pts:.2f}")
    m3.metric("🏁 Avg Grid",     f"{avg_grid:.2f}")

    if not drv_hist.empty:
        layout_hist = base_layout(f"Podiums per Season — {selected_driver}", height=230)
        layout_hist["xaxis"]["title"] = "Season"
        layout_hist["yaxis"]["title"] = "Podiums"
        layout_hist["showlegend"]     = False

        fig_hist = go.Figure(go.Bar(
            x=drv_hist["year"],
            y=drv_hist["podium"],
            marker_color="#E10600",
            hovertemplate="Season %{x}<br>Podiums: %{y}<extra></extra>",
        ))
        fig_hist.update_layout(layout_hist)
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("No podium history found for this driver.")

st.divider()

# ─────────────────────────────────────────────────────────────
# 8. ROW 2 — Grid Sensitivity  |  Norris vs Piastri
# ─────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📊 Probability vs Grid Position")

    all_grids = list(range(1, 21))
    all_probs = [
        predict_podium(g, avg_pts, avg_grid, race_round, nat_enc)
        for g in all_grids
    ]

    layout_sens = base_layout(f"{selected_driver} — Grid Sensitivity", height=320)
    layout_sens["xaxis"]["title"]     = "Grid Position"
    layout_sens["xaxis"]["tickmode"]  = "linear"
    layout_sens["yaxis"]["title"]     = "Podium Probability (%)"
    layout_sens["yaxis"]["range"]     = [0, 105]
    layout_sens["showlegend"]         = True

    fig_sens = go.Figure()
    fig_sens.add_trace(go.Bar(
        x=all_grids,
        y=[p * 100 for p in all_probs],
        marker_color=[prob_color(p) for p in all_probs],
        hovertemplate="P%{x}: %{y:.1f}%<extra></extra>",
        showlegend=False,
    ))
    fig_sens.add_trace(go.Scatter(
        x=[grid_pos],
        y=[all_probs[grid_pos - 1] * 100],
        mode="markers",
        marker=dict(size=14, color="white",
                    line=dict(color="#E10600", width=2)),
        name=f"Selected P{grid_pos}",
        hovertemplate=f"P{grid_pos}: {all_probs[grid_pos-1]*100:.1f}%<extra></extra>",
    ))
    fig_sens.update_layout(layout_sens)
    st.plotly_chart(fig_sens, use_container_width=True)

with col_right:
    st.subheader("🔍 McLaren Sanity Check: Norris vs Piastri")

    SPOTLIGHT  = ["Lando Norris", "Oscar Piastri"]
    DRV_COLORS = {"Lando Norris": "#FF8000", "Oscar Piastri": "#E8002D"}
    GRID_RANGE = [1, 3, 5, 8, 10, 15, 20]

    layout_duo = base_layout("Podium Probability by Grid — McLaren Duo 2023–2024", height=320)
    layout_duo["xaxis"]["title"] = "Grid Position"
    layout_duo["yaxis"]["title"] = "Podium Probability (%)"
    layout_duo["yaxis"]["range"] = [0, 105]

    fig_duo = go.Figure()
    for drv in SPOTLIGHT:
        row = driver_lookup[driver_lookup["driver_name"] == drv]
        if row.empty:
            continue
        row = row.iloc[0]
        sp_probs = [
            predict_podium(
                g,
                float(row["driver_avg_points"]),
                float(row["driver_avg_grid"]),
                10,
                float(row["nationality_encoded"])
            ) * 100
            for g in GRID_RANGE
        ]
        fig_duo.add_trace(go.Scatter(
            x=GRID_RANGE,
            y=sp_probs,
            mode="lines+markers",
            name=drv,
            line=dict(color=DRV_COLORS[drv], width=2),
            marker=dict(size=7),
            hovertemplate=f"{drv}<br>P%{{x}}: %{{y:.1f}}%<extra></extra>",
        ))
    fig_duo.update_layout(layout_duo)
    st.plotly_chart(fig_duo, use_container_width=True)

    st.caption(
        "Norris (avg 7.98 pts) dan Piastri (avg 8.07 pts) mendapat probabilitas "
        "hampir identik — model menangkap paritas kompetitif nyata, bukan nama."
    )

st.divider()

# ─────────────────────────────────────────────────────────────
# 9. FEATURE IMPORTANCES
# ─────────────────────────────────────────────────────────────
st.subheader("🧠 Model Feature Importances")

FEAT_LABELS = {
    "grid":                "Starting Grid Position",
    "driver_avg_points":   "Career Avg Points / Race",
    "driver_avg_grid":     "Career Avg Grid Position",
    "round":               "Race Round in Season",
    "nationality_encoded": "Driver Nationality",
}

importances    = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
readable_index = [FEAT_LABELS[f] for f in importances.index]

layout_imp = base_layout("Random Forest — Feature Importances", height=280)
layout_imp["xaxis"]["title"] = "Importance Score"
layout_imp["xaxis"]["range"] = [0, float(importances.max()) * 1.25]
layout_imp["yaxis"]["title"] = ""
layout_imp["showlegend"]     = False

fig_imp = go.Figure(go.Bar(
    x=importances.values,
    y=readable_index,
    orientation="h",
    marker_color="#E10600",
    text=[f"{v:.3f}" for v in importances.values],
    textposition="outside",
    textfont=dict(color="white", size=10),
    hovertemplate="%{y}: %{x:.4f}<extra></extra>",
))
fig_imp.update_layout(layout_imp)
st.plotly_chart(fig_imp, use_container_width=True)

st.caption(
    "Starting Grid Position dan Career Avg Points/Race adalah dua fitur terpenting — "
    "konsisten dengan temuan EDA bahwa posisi start dan performa historis driver "
    "merupakan prediktor podium paling kuat."
)

# ─────────────────────────────────────────────────────────────
# 10. FOOTER
# ─────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#888;font-size:0.8rem'>"
    "ISB02303402 – Data Science Capstone Project &nbsp;|&nbsp; "
    "Model: Random Forest (n=200, depth=8) &nbsp;|&nbsp; "
    "Dataset: Formula 1 · Kaggle · 2008–2024"
    "</div>",
    unsafe_allow_html=True,
)
