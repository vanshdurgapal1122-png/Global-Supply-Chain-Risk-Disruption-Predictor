"""
Plotly Analytical Charts & Dashboard Visualizations.
Provides executive-grade visual components for macroeconomic trends,
risk gauges, feature importances, scenario stress tests, and confusion matrices.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Executive Dark Theme Styling Palette
THEME = {
    "bg": "#0B1120",            # Deep slate / navy background
    "card_bg": "#1E293B",       # Surface card background
    "text": "#F8FAFC",          # Bright white text
    "subtext": "#94A3B8",       # Muted slate text
    "grid": "#334155",          # Subtle grid lines
    "accent_cyan": "#06B6D4",   # Cyan highlight
    "accent_blue": "#3B82F6",   # Primary blue
    "green": "#10B981",         # Low risk / positive
    "amber": "#F59E0B",         # Medium risk
    "orange": "#F97316",        # High risk
    "red": "#EF4444"            # Critical risk
}


def create_macro_trend_chart(macro_df: pd.DataFrame) -> go.Figure:
    """Renders dual-axis interactive time series for Brent Crude and FX rates."""
    fig = go.Figure()

    # Crude Oil Brent Line
    if "crude_oil_brent" in macro_df.columns:
        fig.add_trace(go.Scatter(
            x=macro_df.index,
            y=macro_df["crude_oil_brent"],
            name="Brent Crude Oil ($/bbl)",
            line=dict(color="#F59E0B", width=2.5),
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Brent:</b> $%{y:.2f}/bbl<extra></extra>"
        ))

    # WTI Line
    if "crude_oil_wti" in macro_df.columns:
        fig.add_trace(go.Scatter(
            x=macro_df.index,
            y=macro_df["crude_oil_wti"],
            name="WTI Crude ($/bbl)",
            line=dict(color="#FB923C", width=1.5, dash="dot"),
            opacity=0.8,
            hovertemplate="<b>WTI:</b> $%{y:.2f}/bbl<extra></extra>"
        ))

    # EUR/USD secondary axis
    if "eur_usd" in macro_df.columns:
        fig.add_trace(go.Scatter(
            x=macro_df.index,
            y=macro_df["eur_usd"],
            name="EUR / USD Exchange Rate",
            line=dict(color="#38BDF8", width=2.0),
            yaxis="y2",
            hovertemplate="<b>EUR/USD:</b> %{y:.4f}<extra></extra>"
        ))

    fig.update_layout(
        title=dict(text="<b>Global Macroeconomic Indicators (Energy & Currencies)</b>", font=dict(color=THEME["text"], size=16)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=THEME["subtext"], family="Inter, sans-serif"),
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        xaxis=dict(
            showgrid=True,
            gridcolor=THEME["grid"],
            zeroline=False
        ),
        yaxis=dict(
            title=dict(text="Crude Oil Price ($/bbl)", font=dict(color="#F59E0B")),
            showgrid=True,
            gridcolor=THEME["grid"],
            zeroline=False
        ),
        yaxis2=dict(
            title=dict(text="EUR/USD FX Rate", font=dict(color="#38BDF8")),
            overlaying="y",
            side="right",
            showgrid=False
        )
    )

    return fig


def create_delay_gauge(predicted_delay: float, baseline_days: float) -> go.Figure:
    """Renders a circular gauge for predicted route delay in days."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=predicted_delay,
        title=dict(text="<b>Predicted Route Delay</b><br><span style='font-size:12px;color:#94a3b8'>Days Beyond Baseline</span>", font=dict(color=THEME["text"], size=14)),
        delta=dict(reference=0, increasing=dict(color=THEME["red"]), suffix=" d"),
        number=dict(suffix=" d", font=dict(color=THEME["text"], size=28)),
        gauge=dict(
            axis=dict(range=[0, 20], tickcolor=THEME["subtext"]),
            bar=dict(color="#38BDF8", thickness=0.35),
            bgcolor=THEME["card_bg"],
            borderwidth=1,
            bordercolor=THEME["grid"],
            steps=[
                dict(range=[0, 2], color="rgba(16, 185, 129, 0.25)"),
                dict(range=[2, 5], color="rgba(245, 158, 11, 0.25)"),
                dict(range=[5, 9], color="rgba(249, 115, 22, 0.35)"),
                dict(range=[9, 20], color="rgba(239, 68, 68, 0.45)")
            ],
            threshold=dict(
                line=dict(color=THEME["red"], width=4),
                thickness=0.8,
                value=predicted_delay
            )
        )
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        height=220
    )
    return fig


def create_dsi_gauge(dsi_score: float, risk_tier: str, risk_color: str) -> go.Figure:
    """Renders Disruption Severity Index (DSI) composite gauge (0-100)."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=dsi_score,
        title=dict(text=f"<b>Disruption Severity (DSI)</b><br><span style='font-size:12px;color:{risk_color};font-weight:bold'>{risk_tier}</span>", font=dict(color=THEME["text"], size=14)),
        number=dict(suffix="/100", font=dict(color=risk_color, size=28)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=THEME["subtext"]),
            bar=dict(color=risk_color, thickness=0.4),
            bgcolor=THEME["card_bg"],
            borderwidth=1,
            bordercolor=THEME["grid"],
            steps=[
                dict(range=[0, 30], color="rgba(16, 185, 129, 0.2)"),
                dict(range=[30, 55], color="rgba(245, 158, 11, 0.2)"),
                dict(range=[55, 75], color="rgba(249, 115, 22, 0.3)"),
                dict(range=[75, 100], color="rgba(239, 68, 68, 0.4)")
            ]
        )
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        height=220
    )
    return fig


def create_bottleneck_probability_chart(prob_dict: Dict[str, float]) -> go.Figure:
    """Horizontal bar chart showing port bottleneck class probabilities."""
    classes = list(prob_dict.keys())
    probs = [round(v * 100, 1) for v in prob_dict.values()]
    colors = [THEME["green"], THEME["amber"], THEME["orange"], THEME["red"]]

    fig = go.Figure(go.Bar(
        x=probs,
        y=classes,
        orientation="h",
        marker=dict(color=colors, line=dict(color="#FFFFFF", width=1)),
        text=[f"{p}%" for p in probs],
        textposition="auto",
        hovertemplate="<b>%{y}:</b> %{x}% probability<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text="<b>Port Bottleneck Risk Distribution</b>", font=dict(color=THEME["text"], size=14)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=THEME["subtext"], family="Inter, sans-serif"),
        xaxis=dict(title="Probability (%)", range=[0, 100], showgrid=True, gridcolor=THEME["grid"]),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=40, r=20, t=40, b=30),
        height=220
    )
    return fig


def create_feature_importance_chart(metrics: Dict[str, Any]) -> go.Figure:
    """Renders XGBoost feature importances."""
    top_features = metrics.get("top_delay_features", [])
    if not top_features:
        # Fallback dummy importances for initial display
        features = ["rerouted_via_cape", "weather_risk_index", "distance_nm", "fuel_stress_index", "news_disruption_sentiment"]
        importances = [0.38, 0.22, 0.15, 0.12, 0.08]
    else:
        features = [item["feature"] for item in top_features[:10]][::-1]
        importances = [item["importance"] for item in top_features[:10]][::-1]

    fig = go.Figure(go.Bar(
        x=importances,
        y=features,
        orientation="h",
        marker=dict(
            color=importances,
            colorscale="Viridis",
            line=dict(color="rgba(255,255,255,0.2)", width=1)
        ),
        text=[f"{v:.3f}" for v in importances],
        textposition="auto",
        hovertemplate="<b>Feature:</b> %{y}<br><b>Gain / Weight:</b> %{x:.4f}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text="<b>Top Drivers of Delay (XGBoost Feature Importance)</b>", font=dict(color=THEME["text"], size=14)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=THEME["subtext"], family="Inter, sans-serif"),
        xaxis=dict(title="Relative Importance Score", showgrid=True, gridcolor=THEME["grid"]),
        yaxis=dict(tickfont=dict(size=11)),
        margin=dict(l=140, r=20, t=40, b=30),
        height=320
    )
    return fig


def create_scenario_comparison_chart(scenarios: List[Dict[str, Any]]) -> go.Figure:
    """Grouped bar chart for stress test scenario comparisons."""
    names = [s["scenario_name"].split(".")[1].strip() if "." in s["scenario_name"] else s["scenario_name"] for s in scenarios]
    delays = [s["predicted_delay_days"] for s in scenarios]
    dsis = [s["dsi"] for s in scenarios]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Predicted Delay (Days)",
        x=names,
        y=delays,
        marker_color="#38BDF8",
        text=[f"{d}d" for d in delays],
        textposition="auto"
    ))

    fig.add_trace(go.Bar(
        name="Disruption Severity Index (DSI)",
        x=names,
        y=dsis,
        marker_color="#EF4444",
        text=[f"{s}" for s in dsis],
        textposition="auto"
    ))

    fig.update_layout(
        title=dict(text="<b>Stress Testing: Multi-Scenario Disruption Impact</b>", font=dict(color=THEME["text"], size=16)),
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=THEME["subtext"], family="Inter, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        xaxis=dict(showgrid=False, tickfont=dict(size=11, color=THEME["text"])),
        yaxis=dict(title="Value", showgrid=True, gridcolor=THEME["grid"]),
        margin=dict(l=40, r=20, t=50, b=40),
        height=340
    )
    return fig


def create_confusion_matrix_heatmap(cm: List[List[int]], classes: List[str]) -> go.Figure:
    """Renders annotated confusion matrix for port bottleneck classifier."""
    cm_arr = np.array(cm)
    fig = go.Figure(data=go.Heatmap(
        z=cm_arr,
        x=classes,
        y=classes,
        colorscale="Blues",
        text=cm_arr,
        texttemplate="%{text}",
        textfont={"size": 13, "color": "white"},
        hoverongaps=False
    ))

    fig.update_layout(
        title=dict(text="<b>Port Bottleneck Confusion Matrix</b>", font=dict(color=THEME["text"], size=14)),
        xaxis=dict(title="Predicted Class", tickfont=dict(color=THEME["text"])),
        yaxis=dict(title="True Class", tickfont=dict(color=THEME["text"]), autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=THEME["subtext"], family="Inter, sans-serif"),
        margin=dict(l=60, r=20, t=40, b=40),
        height=300
    )
    return fig


def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Renders correlation matrix between shipping variables, macro factors, and delay."""
    cols_to_corr = [
        "distance_nm", "baseline_transit_days", "weather_risk_index",
        "brent_oil_price", "fuel_stress_index", "news_disruption_sentiment",
        "dest_dwell_time_days", "route_delay_days", "disruption_severity_index"
    ]
    existing = [c for c in cols_to_corr if c in df.columns]
    corr = df[existing].corr().round(2)

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale="RdBu_r",
        zmin=-1,
        zmax=1,
        text=corr.values,
        texttemplate="%{text}",
        textfont={"size": 10}
    ))

    fig.update_layout(
        title=dict(text="<b>Cross-Domain Correlation Matrix (Logistics ⨯ Macro ⨯ NLP)</b>", font=dict(color=THEME["text"], size=15)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=THEME["subtext"], family="Inter, sans-serif"),
        xaxis=dict(tickangle=-35),
        margin=dict(l=140, r=20, t=50, b=80),
        height=400
    )
    return fig
