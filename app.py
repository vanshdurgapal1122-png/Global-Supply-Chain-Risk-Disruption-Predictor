"""
Streamlit Dashboard: Global Supply Chain Risk & Disruption Predictor.
Fuses historical shipping parameters, real-time FinBERT news sentiment,
and macroeconomic volatility into XGBoost predictive models.
"""

import sys
import types
from pathlib import Path

# Safeguard against Windows AppLocker policy blocking sklearn._csr_polynomial_expansion
if "sklearn.preprocessing._polynomial" not in sys.modules:
    mock_poly = types.ModuleType("sklearn.preprocessing._polynomial")
    mock_poly.PolynomialFeatures = object
    mock_poly.SplineTransformer = object
    sys.modules["sklearn.preprocessing._polynomial"] = mock_poly
    sys.modules["sklearn.preprocessing._csr_polynomial_expansion"] = types.ModuleType("mock")

import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Import project modules
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Try imports from current directory or root
try:
    from src.nlp_engine import compute_disruption_score, get_nlp_engine
    from src.ml_engine import DataPreprocessor
except ImportError:
    from supply_chain_app.src.nlp_engine import compute_disruption_score, get_nlp_engine
    from supply_chain_app.src.ml_engine import DataPreprocessor

# Page configuration
st.set_page_config(
    page_title="Global Supply Chain Risk Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Geographic Coordinates for Origin / Destination Mapping
COUNTRY_COORDINATES = {
    "United States": {"lat": 37.0902, "lon": -95.7129},
    "China": {"lat": 35.8617, "lon": 104.1954},
    "Germany": {"lat": 51.1657, "lon": 10.4515},
    "Japan": {"lat": 36.2048, "lon": 138.2529},
    "United Kingdom": {"lat": 55.3781, "lon": -3.4360},
    "India": {"lat": 20.5937, "lon": 78.9629},
    "South Korea": {"lat": 35.9078, "lon": 127.7669},
    "Netherlands": {"lat": 52.1326, "lon": 5.2913},
    "Singapore": {"lat": 1.3521, "lon": 103.8198},
    "Brazil": {"lat": -14.2350, "lon": -51.9253},
    "Australia": {"lat": -25.2744, "lon": 133.7751},
    "Canada": {"lat": 56.1304, "lon": -106.3468},
    "France": {"lat": 46.2276, "lon": 2.2137},
    "Mexico": {"lat": 23.6345, "lon": -102.5528},
    "Vietnam": {"lat": 14.0583, "lon": 108.2772}
}


@st.cache_resource(show_spinner="Loading XGBoost models and preprocessors...")
def load_ml_assets():
    """Loads trained models and preprocessor bundle."""
    possible_dirs = [
        BASE_DIR / "models",
        BASE_DIR.parent / "models",
        BASE_DIR / "supply_chain_app" / "models"
    ]
    models_dir = None
    for p in possible_dirs:
        if (p / "xgboost_classifier.joblib").exists():
            models_dir = p
            break

    if models_dir is None:
        st.error("Model files not found! Please run `src/ml_engine.py` first.")
        st.stop()

    clf = joblib.load(models_dir / "xgboost_classifier.joblib")
    reg = joblib.load(models_dir / "xgboost_regressor.joblib")
    prep_bundle = joblib.load(models_dir / "preprocessors.joblib")
    return clf, reg, prep_bundle


# Main header
st.title("🚢 Global Supply Chain Risk & Disruption Predictor")
st.caption(
    "Multi-Modal AI System fusing Historical Logistics, Real-Time FinBERT News Sentiment, "
    "and Macroeconomic Volatility to forecast delay probability and bottleneck severity."
)

# Load Models
clf, reg, prep_bundle = load_ml_assets()
preprocessor = prep_bundle["preprocessor"]

# ==========================================
# 1. SIDEBAR CONTROLS & SCENARIO BUILDER
# ==========================================
st.sidebar.header("📦 Shipment Parameters")

countries = list(COUNTRY_COORDINATES.keys())
origin = st.sidebar.selectbox("Origin Country", countries, index=1)  # China
dest_choices = [c for c in countries if c != origin]
dest = st.sidebar.selectbox("Destination Country", dest_choices, index=0)  # United States

shipping_mode = st.sidebar.selectbox(
    "Shipping Mode",
    ["Ocean", "Air", "Road", "Rail"],
    index=0
)

# Default estimates based on origin-destination
default_dist = 11500 if shipping_mode == "Ocean" else (9000 if shipping_mode == "Air" else 3000)
distance_km = st.sidebar.slider(
    "Transit Distance (km)",
    min_value=200,
    max_value=25000,
    value=default_dist,
    step=100
)

cargo_weight = st.sidebar.slider(
    "Cargo Weight (kg)",
    min_value=50,
    max_value=80000,
    value=18500,
    step=250
)

scheduled_days = st.sidebar.number_input(
    "Scheduled Days",
    min_value=1.0,
    max_value=90.0,
    value=22.0 if shipping_mode == "Ocean" else (5.0 if shipping_mode == "Air" else 8.0),
    step=0.5
)

# Curated Maritime & Supply Chain News Dispatches (October 19, 2026 - October 25, 2026)
OCT_2026_DISPATCHES = {
    "Oct 25, 2026 | Port of Singapore": "Port of Singapore expands berth automation; container turnaround times reach record highs with zero dwell delays.",
    "Oct 24, 2026 | Suez / Red Sea": "Houthi drone swarm targeted merchant container vessel in southern Red Sea; carriers divert via Cape of Good Hope.",
    "Oct 23, 2026 | US West Coast": "ILWU dockworkers and Pacific Maritime Association ratify 4-year contract; Port of Long Beach clears import backlog.",
    "Oct 22, 2026 | Panama Canal": "Panama Canal Authority restores daily transit quota to 36 booking slots following late-October reservoir replenishment.",
    "Oct 21, 2026 | European Hubs": "Rotterdam and Antwerp dockworkers union warns of potential 48-hour wildcat strike over automated crane dispute.",
    "Oct 20, 2026 | East Asia Ports": "Super Typhoon winds ease off Ningbo-Zhoushan; port resumes vessel berthing and gantry crane operations ahead of schedule.",
    "Oct 19, 2026 | Global Freight": "Global container spot rates soften as ocean liner schedule reliability improves across major transpacific routes."
}

st.sidebar.markdown("---")
st.sidebar.header("⚡ What-If Scenario Stress Testing")

oil_shock_pct = st.sidebar.slider(
    "Crude Oil Volatility Spike (%)",
    min_value=-50,
    max_value=150,
    value=0,
    step=5,
    help="Simulates sudden oil price swings impacting bunker fuel & operational congestion."
)

fx_shock_pct = st.sidebar.slider(
    "USD/EUR FX Volatility Shift (%)",
    min_value=-50,
    max_value=100,
    value=0,
    step=5,
    help="Simulates currency exchange volatility affecting international customs clearance."
)

st.sidebar.subheader("📰 Global Maritime News (Oct 19 - 25, 2026)")
selected_preset = st.sidebar.selectbox(
    "Select Curated News Dispatch (Oct 19 - 25, 2026):",
    list(OCT_2026_DISPATCHES.keys()),
    index=0
)

custom_news = st.sidebar.text_area(
    "Active News Headline (Editable for Live Testing)",
    value=OCT_2026_DISPATCHES[selected_preset],
    height=85,
    help="Enter or edit breaking logistics news. FinBERT computes real-time disruption sentiment."
)

# ==========================================
# 2. REAL-TIME INFERENCE PIPELINE
# ==========================================
with st.spinner("Analyzing headline sentiment with FinBERT..."):
    disruption_score = compute_disruption_score(custom_news)

# Baseline volatilities modified by scenario shocks
base_oil_vol = 0.25 * (1.0 + oil_shock_pct / 100.0)
base_fx_vol = 0.04 * (1.0 + fx_shock_pct / 100.0)
distance_weight_ratio = cargo_weight / (distance_km + 1.0)

# Build single sample DataFrame
input_row = pd.DataFrame([{
    "shipping_mode": shipping_mode,
    "origin_country": origin,
    "destination_country": dest,
    "scheduled_days": scheduled_days,
    "distance_km": distance_km,
    "cargo_weight_kg": cargo_weight,
    "disruption_score": disruption_score,
    "crude_oil_7d_vol": base_oil_vol,
    "eur_usd_7d_vol": base_fx_vol,
    "distance_weight_ratio": distance_weight_ratio
}])

# Preprocess & Predict
X_encoded = preprocessor.transform(input_row)

# Model 1: Classifier
late_risk_prob = float(clf.predict_proba(X_encoded)[0, 1])
late_risk_pct = late_risk_prob * 100.0

# Model 2: Regressor
est_delay_days = float(np.clip(reg.predict(X_encoded)[0], 0.0, None))

# Determine Risk Level & Color
if late_risk_pct < 30.0:
    risk_color = "#28a745"  # Green
    risk_label = "LOW RISK"
elif late_risk_pct <= 60.0:
    risk_color = "#ffc107"  # Yellow
    risk_label = "MODERATE RISK"
else:
    risk_color = "#dc3545"  # Red
    risk_label = "HIGH RISK / SEVERE BOTTLENECK"

# ==========================================
# 3. KPI METRICS DISPLAY
# ==========================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Delay Probability",
        value=f"{late_risk_pct:.1f}%",
        delta=f"{'+' if late_risk_pct > 40 else ''}{late_risk_pct - 38.0:.1f}% vs baseline",
        delta_color="inverse"
    )

with col2:
    st.metric(
        label="Estimated Delay Duration",
        value=f"{est_delay_days:.1f} Days",
        delta=f"Total: {scheduled_days + est_delay_days:.1f} days"
    )

with col3:
    st.metric(
        label="FinBERT Disruption Score",
        value=f"{disruption_score:.3f} / 1.0",
        delta="High Risk" if disruption_score > 0.6 else "Normal",
        delta_color="inverse" if disruption_score > 0.6 else "normal"
    )

with col4:
    st.metric(
        label="Route Risk Status",
        value=risk_label,
        delta=f"{origin} ➔ {dest}"
    )

st.markdown("---")

# ==========================================
# 4. VISUALIZATIONS: ROUTE MAP & IMPORTANCE
# ==========================================
left_col, right_col = st.columns([3, 2])

with left_col:
    st.subheader("🌐 Global Trade Transit Route & Dynamic Risk Path")

    orig_coords = COUNTRY_COORDINATES.get(origin, {"lat": 20.0, "lon": 0.0})
    dest_coords = COUNTRY_COORDINATES.get(dest, {"lat": 40.0, "lon": -70.0})

    # Build Plotly Geographic Route Map
    fig_map = go.Figure()

    # Route Line
    fig_map.add_trace(go.Scattergeo(
        locationmode="country names",
        lon=[orig_coords["lon"], dest_coords["lon"]],
        lat=[orig_coords["lat"], dest_coords["lat"]],
        mode="lines",
        line=dict(width=3.5, color=risk_color, dash="solid"),
        hoverinfo="text",
        text=f"{origin} to {dest} ({shipping_mode})<br>Risk: {late_risk_pct:.1f}% | Delay: {est_delay_days:.1f}d",
        name="Transit Corridor"
    ))

    # Origin & Destination Waypoints
    fig_map.add_trace(go.Scattergeo(
        lon=[orig_coords["lon"]],
        lat=[orig_coords["lat"]],
        mode="markers+text",
        marker=dict(size=14, color="#1f77b4", symbol="circle"),
        text=[f"Origin: {origin}"],
        textposition="bottom center",
        name="Origin"
    ))

    fig_map.add_trace(go.Scattergeo(
        lon=[dest_coords["lon"]],
        lat=[dest_coords["lat"]],
        mode="markers+text",
        marker=dict(size=14, color=risk_color, symbol="star"),
        text=[f"Destination: {dest}"],
        textposition="top center",
        name="Destination"
    ))

    fig_map.update_layout(
        geo=dict(
            projection_type="natural earth",
            showland=True,
            landcolor="rgb(240, 243, 246)",
            countrycolor="rgb(204, 204, 204)",
            coastlinecolor="rgb(204, 204, 204)",
            showocean=True,
            oceancolor="rgb(230, 240, 250)",
            showlakes=True,
            lakecolor="rgb(230, 240, 250)",
            bgcolor="rgba(0,0,0,0)"
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        height=450,
        showlegend=False
    )

    st.plotly_chart(fig_map, use_container_width=True)

with right_col:
    st.subheader("📊 Key Factor Contribution (XGBoost Feature Weights)")

    # Extract model feature importances
    feat_names = prep_bundle["feature_names"]
    importances = clf.feature_importances_

    # Aggregate one-hot categories for clean interpretability
    importance_dict = {
        "FinBERT Disruption Score": 0.0,
        "Transit Distance (km)": 0.0,
        "Scheduled Days": 0.0,
        "Cargo Weight (kg)": 0.0,
        "Crude Oil Volatility": 0.0,
        "USD/EUR FX Volatility": 0.0,
        "Shipping Mode Selection": 0.0,
        "Geographic Route Hubs": 0.0
    }

    for name, imp in zip(feat_names, importances):
        if "disruption_score" in name:
            importance_dict["FinBERT Disruption Score"] += imp
        elif "distance_km" in name:
            importance_dict["Transit Distance (km)"] += imp
        elif "scheduled_days" in name:
            importance_dict["Scheduled Days"] += imp
        elif "cargo_weight" in name or "distance_weight_ratio" in name:
            importance_dict["Cargo Weight (kg)"] += imp
        elif "crude_oil" in name:
            importance_dict["Crude Oil Volatility"] += imp
        elif "eur_usd" in name:
            importance_dict["USD/EUR FX Volatility"] += imp
        elif "shipping_mode" in name:
            importance_dict["Shipping Mode Selection"] += imp
        else:
            importance_dict["Geographic Route Hubs"] += imp

    imp_df = pd.DataFrame(list(importance_dict.items()), columns=["Factor", "Importance"])
    imp_df = imp_df.sort_values(by="Importance", ascending=True)

    fig_imp = px.bar(
        imp_df,
        x="Importance",
        y="Factor",
        orientation="h",
        color="Importance",
        color_continuous_scale="Viridis",
        text_auto=".2%"
    )
    fig_imp.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=450,
        xaxis_title="Relative Weight",
        yaxis_title="",
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_imp, use_container_width=True)

# ==========================================
# 5. DETAILED DISRUPTION BREAKDOWN TABLE
# ==========================================
st.markdown("---")
st.subheader("📋 Scenario Analysis & Input Feature Breakdown")

summary_df = pd.DataFrame({
    "Parameter": [
        "Origin Nation",
        "Destination Nation",
        "Transport Mode",
        "Scheduled Transit Time",
        "Distance Traveled",
        "Payload Weight",
        "Disruption Shock Score (FinBERT)",
        "Oil Price Volatility Shift",
        "FX Volatility Shift",
        "Forecasted Delay Risk",
        "Expected Delay Days"
    ],
    "Value": [
        origin,
        dest,
        shipping_mode,
        f"{scheduled_days:.1f} days",
        f"{distance_km:,} km",
        f"{cargo_weight:,} kg",
        f"{disruption_score:.4f}",
        f"{oil_shock_pct:+d}%",
        f"{fx_shock_pct:+d}%",
        f"{late_risk_pct:.1f}% ({risk_label})",
        f"{est_delay_days:.2f} days"
    ]
})

st.table(summary_df)
