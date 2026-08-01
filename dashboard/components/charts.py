"""
Reusable Plotly chart components for the dashboard.

All charts use a consistent dark-theme style.
"""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd


# Consistent chart theme
DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(15,23,42,0.95)",
    plot_bgcolor="rgba(30,41,59,0.9)",
    font=dict(color="#e2e8f0", family="Inter, sans-serif"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    margin=dict(l=60, r=30, t=50, b=50),
    hovermode="x unified",
)

COLORS = {
    "primary": "#3b82f6",
    "secondary": "#8b5cf6",
    "success": "#22c55e",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "info": "#06b6d4",
    "muted": "#94a3b8",
    "surface": "#1e293b",
}


def spread_timeseries(df: pd.DataFrame, title: str = "Spread Levels") -> go.Figure:
    """Time series of spread levels with multiple maturities."""
    fig = go.Figure()

    cols = [c for c in df.columns if c.startswith("spread_")]
    colors = [COLORS["primary"], COLORS["secondary"], COLORS["info"], COLORS["muted"]]

    for i, col in enumerate(cols):
        fig.add_trace(go.Scatter(
            x=df["date"], y=df[col],
            name=col.replace("spread_", "").upper(),
            line=dict(color=colors[i % len(colors)], width=1.5),
        ))

    fig.update_layout(
        **DARK_LAYOUT,
        title=title,
        xaxis_title="Date",
        yaxis_title="Spread (bps)",
        height=450,
    )
    return fig


def volatility_comparison(
    returns: pd.Series,
    vol_series: dict[str, pd.Series],
    title: str = "Conditional Volatility Comparison",
) -> go.Figure:
    """Compare multiple volatility model outputs."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.6, 0.4],
        subplot_titles=("Conditional Volatility", "Returns"),
    )

    colors = list(COLORS.values())
    for i, (name, vol) in enumerate(vol_series.items()):
        fig.add_trace(
            go.Scatter(
                x=vol.index, y=vol.values,
                name=name, line=dict(color=colors[i % len(colors)], width=1.5),
            ),
            row=1, col=1,
        )

    # Returns as bar chart
    fig.add_trace(
        go.Bar(
            x=returns.index, y=returns.values,
            name="Returns", marker_color=COLORS["muted"], opacity=0.5,
        ),
        row=2, col=1,
    )

    fig.update_layout(**DARK_LAYOUT, title=title, height=600, showlegend=True)
    fig.update_yaxes(title_text="Volatility (bps)", row=1, col=1)
    fig.update_yaxes(title_text="Returns (bps)", row=2, col=1)
    return fig


def var_comparison_chart(var_results: dict[str, dict], confidence: float = 0.99) -> go.Figure:
    """Bar chart comparing VaR across methods."""
    methods = list(var_results.keys())
    vars_ = [r["var"] for r in var_results.values()]
    ess = [r.get("es", r["var"]) for r in var_results.values()]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="VaR", x=methods, y=vars_, marker_color=COLORS["warning"]))
    fig.add_trace(go.Bar(name="ES", x=methods, y=ess, marker_color=COLORS["danger"]))

    fig.update_layout(
        **DARK_LAYOUT,
        title=f"Value-at-Risk Comparison ({confidence:.0%} confidence)",
        barmode="group",
        height=400,
        yaxis_title="bps",
    )
    return fig


def regime_chart(
    volatility: pd.Series,
    labels: np.ndarray,
    regime_means: dict[int, float],
    title: str = "Volatility Regimes",
) -> go.Figure:
    """HMM regime visualization with background shading."""
    fig = go.Figure()

    # Volatility line colored by regime
    regime_colors = [COLORS["success"], COLORS["warning"], COLORS["danger"], COLORS["secondary"]]
    dates = volatility.index

    for r in sorted(set(labels)):
        mask = labels == r
        fig.add_trace(go.Scatter(
            x=dates[mask], y=volatility.values[mask],
            name=f"Regime {r} (μ={regime_means.get(r, 0):.2f})",
            mode="markers",
            marker=dict(color=regime_colors[r % len(regime_colors)], size=2),
        ))

    # Mean lines
    for r, mean_val in regime_means.items():
        fig.add_hline(
            y=mean_val, line_dash="dash",
            line_color=regime_colors[r % len(regime_colors)],
            annotation_text=f"μ_{r}={mean_val:.2f}",
        )

    fig.update_layout(
        **DARK_LAYOUT,
        title=title,
        height=450,
        yaxis_title="Volatility (bps)",
    )
    return fig


def fan_chart(
    fan_data: dict,
    title: str = "Scenario Fan Chart",
) -> go.Figure:
    """Monte Carlo fan chart with percentile bands."""
    dates = fan_data["dates"]
    fig = go.Figure()

    # 5-95 band
    fig.add_trace(go.Scatter(
        x=dates + dates[::-1],
        y=fan_data["p95"] + fan_data["p5"][::-1],
        fill="toself", fillcolor="rgba(59,130,246,0.1)",
        line=dict(color="rgba(0,0,0,0)"),
        name="5-95% band", showlegend=True,
    ))

    # 25-75 band
    fig.add_trace(go.Scatter(
        x=dates + dates[::-1],
        y=fan_data["p75"] + fan_data["p25"][::-1],
        fill="toself", fillcolor="rgba(59,130,246,0.2)",
        line=dict(color="rgba(0,0,0,0)"),
        name="25-75% band",
    ))

    # Median
    fig.add_trace(go.Scatter(
        x=dates, y=fan_data["median"],
        line=dict(color=COLORS["primary"], width=2),
        name="Median",
    ))

    fig.update_layout(
        **DARK_LAYOUT,
        title=title,
        height=500,
        yaxis_title="Spread (bps)",
        xaxis_title="Date",
    )
    return fig


def market_gauge_chart(composite: float, indicators: dict, title: str = "Market Gauge") -> go.Figure:
    """Gauge chart for market stress indicator."""
    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=composite,
        domain={"x": [0.1, 0.9], "y": [0.2, 0.9]},
        number={"suffix": " pts", "font": {"size": 28}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": "darkblue"},
            "bgcolor": "white",
            "steps": [
                {"range": [0, 20], "color": "rgba(34,197,94,0.3)"},
                {"range": [20, 40], "color": "rgba(59,130,246,0.3)"},
                {"range": [40, 60], "color": "rgba(245,158,11,0.3)"},
                {"range": [60, 80], "color": "rgba(249,115,22,0.3)"},
                {"range": [80, 100], "color": "rgba(239,68,68,0.3)"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": composite,
            },
        },
    ))

    fig.update_layout(
        **DARK_LAYOUT,
        title=title,
        height=350,
    )
    return fig
