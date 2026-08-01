"""
Regimes page: HMM regime detection and transition matrix visualization.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from dashboard.components._data import get_regime_result
from dashboard.components.charts import COLORS, DARK_LAYOUT, regime_chart

dash.register_page(__name__, path="/regimes", name="Regimes")


def _transition_heatmap(matrix: np.ndarray, n_regimes: int) -> go.Figure:
    """Heatmap of regime transition probabilities."""
    labels = [f"R{i}" for i in range(n_regimes)]

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=labels,
        y=labels,
        colorscale="Blues",
        reversescale=True,
        text=np.round(matrix, 3),
        texttemplate="%{text:.3f}",
        textfont={"size": 14},
        hoverongaps=False,
    ))

    fig.update_layout(
        **DARK_LAYOUT,
        title="状态转移概率矩阵",
        xaxis_title="To Regime",
        yaxis_title="From Regime",
        height=400,
        width=500,
    )
    return fig


def _regime_stats_table(result) -> dbc.Table:
    """Table of regime statistics."""
    rows = []
    for r in range(result.n_regimes):
        mean_vol = result.regime_means.get(r, 0)
        std_vol = result.regime_stds.get(r, 0)
        count = int(np.sum(result.labels == r))
        pct = count / len(result.labels) * 100

        names = {0: "低波动", 1: "中波动", 2: "高波动", 3: "极端"}
        name = names.get(r, f"R{r}")
        color = ["success", "warning", "danger", "secondary"][r % 4]

        rows.append(html.Tr([
            html.Td(html.Badge(name, color=color)),
            html.Td(f"{mean_vol:.3f}", className="text-end"),
            html.Td(f"{std_vol:.3f}", className="text-end"),
            html.Td(f"{count}", className="text-end"),
            html.Td(f"{pct:.1f}%", className="text-end"),
        ]))

    return dbc.Table(
        [html.Thead(html.Tr([
            html.Th("状态"), html.Th("均值", className="text-end"),
            html.Th("标准差", className="text-end"),
            html.Th("天数", className="text-end"),
            html.Th("占比", className="text-end"),
        ]))] + [html.Tbody(rows)],
        borderless=True, dark=True, size="sm", hover=True,
    )


def layout():
    regime_result, vol_series = get_regime_result()
    rr = regime_result

    # Main regime chart
    reg_chart = regime_chart(
        vol_series,
        rr.labels,
        rr.regime_means,
        title="HMM波动率状态识别",
    )

    # Transition matrix heatmap
    trans_fig = _transition_heatmap(rr.transition_matrix, rr.n_regimes)

    # Stats table
    stats_table = _regime_stats_table(rr)

    # Current regime badge
    current_name = rr.current_regime_name
    color_map = {"Low Vol": "success", "Mid Vol": "warning", "High Vol": "danger", "Extreme": "secondary"}
    badge_color = color_map.get(current_name, "info")

    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("波动率状态分析", className="fw-bold"),
                html.P([
                    "Hidden Markov Model · ",
                    html.Strong(f"当前状态: "),
                    dbc.Badge(current_name, color=badge_color, className="ms-1"),
                ], className="text-muted"),
            ]),
        ], className="mb-4"),

        # Regime chart
        dbc.Row([
            dbc.Col(dcc.Graph(figure=reg_chart), width=12),
        ], className="mb-4"),

        # Transition matrix + stats
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("转移矩阵", className="mb-0")),
                    dbc.CardBody(dcc.Graph(figure=trans_fig)),
                ], color="dark", className="border-0 shadow-sm"),
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("状态统计", className="mb-0")),
                    dbc.CardBody(stats_table),
                ], color="dark", className="border-0 shadow-sm"),
            ], width=6),
        ]),
    ], fluid=True)
