"""
Risk page: VaR comparison, EVT diagnostics, Hill estimator.
"""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from dashboard.components._data import get_evt_analyzer, get_var_results
from dashboard.components.charts import COLORS, DARK_LAYOUT, var_comparison_chart

dash.register_page(__name__, path="/risk", name="Risk")


def _hill_chart(evt) -> go.Figure:
    """Hill estimator plot: tail index vs k."""
    hill_data = evt.hill_estimator(k_percentile=0.15)
    ks = hill_data.get("k_values", [])
    xis = hill_data.get("xi_values", [])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ks, y=xis,
        mode="lines",
        name="ξ(k)",
        line=dict(color=COLORS["primary"], width=2),
    ))
    if xis:
        mean_xi = sum(xis) / len(xis)
        fig.add_hline(y=mean_xi, line_dash="dash", line_color=COLORS["warning"],
                      annotation_text=f"mean ξ = {mean_xi:.3f}")

    fig.update_layout(
        **DARK_LAYOUT,
        title="Hill Estimator: Tail Index ξ(k)",
        xaxis_title="k (number of upper order statistics)",
        yaxis_title="ξ (tail index)",
        height=350,
    )
    return fig


def _mean_excess_chart(evt) -> go.Figure:
    """Mean excess plot for GPD threshold selection."""
    me_data = evt.mean_excess_data(n_thresholds=50)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=me_data["threshold"], y=me_data["mean_excess"],
        mode="lines+markers",
        name="Mean Excess",
        line=dict(color=COLORS["info"], width=1.5),
        marker=dict(size=3),
    ))

    fig.update_layout(
        **DARK_LAYOUT,
        title="Mean Excess Plot (GPD Threshold Selection)",
        xaxis_title="Threshold u",
        yaxis_title="Mean Excess E(X-u | X>u)",
        height=350,
    )
    return fig


def _var_metric_card(name: str, result: dict) -> dbc.Card:
    var_val = result["var"]
    es_val = result.get("es", var_val)
    method = result.get("method", name)

    extra = []
    if "df" in result:
        extra.append(html.P(f"df = {result['df']:.2f}", className="small text-muted mb-0"))
    if "gpd_shape" in result:
        xi = result.get("gpd_shape", 0)
        extra.append(html.P(f"GPD ξ = {xi:.4f}", className="small text-muted mb-0"))

    return dbc.Card([
        dbc.CardHeader(html.Span(name, className="fw-bold")),
        dbc.CardBody([
            html.P([html.Strong("VaR: "), f"{var_val:.4f} bps"], className="mb-1"),
            html.P([html.Strong("ES: "), f"{es_val:.4f} bps"], className="mb-1"),
            *extra,
        ]),
    ], color="dark", className="border-0 shadow-sm")


def layout():
    var_results = get_var_results()
    evt = get_evt_analyzer()

    # VaR comparison bar chart
    var_chart = var_comparison_chart(var_results, confidence=0.99)

    # EVT diagnostic plots
    hill_fig = _hill_chart(evt)
    me_fig = _mean_excess_chart(evt)

    # Metric cards
    cards = [_var_metric_card(name, r) for name, r in var_results.items()]

    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("风险度量分析", className="fw-bold"),
                html.P(
                    "Value-at-Risk / Expected Shortfall · 历史模拟 / 参数法 / EVT-POT",
                    className="text-muted",
                ),
            ]),
        ], className="mb-4"),

        # VaR comparison
        dbc.Row([
            dbc.Col(dcc.Graph(figure=var_chart), width=8),
            dbc.Col([
                *cards,
            ], width=4),
        ], className="mb-4"),

        # EVT diagnostics
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("EVT诊断", className="mb-0")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col(dcc.Graph(figure=hill_fig), width=6),
                            dbc.Col(dcc.Graph(figure=me_fig), width=6),
                        ]),
                    ]),
                ], color="dark", className="border-0 shadow-sm"),
            ]),
        ]),
    ], fluid=True)
