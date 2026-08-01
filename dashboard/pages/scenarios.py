"""
Scenarios page: Fan chart, stress test results, and scenario analysis.
"""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html

from dashboard.components._data import get_scenario_data
from dashboard.components.charts import fan_chart

dash.register_page(__name__, path="/scenarios", name="Scenarios")


def _stress_table(stress: dict) -> dash_table.DataTable:
    """DataTable showing stress test results."""
    rows = []
    for name, s in stress.items():
        rows.append({
            "scenario": name,
            "vol_mult": s["vol_multiplier"],
            "median": round(s["median_final"], 2),
            "p5": round(s["p5_final"], 2),
            "p95": round(s["p95_final"], 2),
            "prob_exceed": f"{s['probability_exceed_threshold']:.1%}",
            "prob_decline": f"{s['probability_decline_threshold']:.1%}",
        })

    cols = [
        {"name": "情景", "id": "scenario"},
        {"name": "波动率倍数", "id": "vol_mult", "type": "numeric"},
        {"name": "中位终值 (bps)", "id": "median", "type": "numeric"},
        {"name": "5%分位", "id": "p5", "type": "numeric"},
        {"name": "95%分位", "id": "p95", "type": "numeric"},
        {"name": "超阈值概率", "id": "prob_exceed"},
        {"name": "跌破阈值概率", "id": "prob_decline"},
    ]

    return dash_table.DataTable(
        data=rows,
        columns=cols,
        style_table={"overflowX": "auto"},
        style_cell={
            "backgroundColor": "#1e293b",
            "color": "#e2e8f0",
            "border": "1px solid #334155",
            "textAlign": "center",
        },
        style_header={"backgroundColor": "#0f172a", "fontWeight": "bold"},
    )


def layout():
    scenario = get_scenario_data()
    fan_data = scenario["fan"]
    stress = scenario["stress"]
    current = scenario["current"]

    # Fan chart
    fan_fig = fan_chart(fan_data, title=f"蒙特卡洛情景扇形图 (当前: {current:.2f} bps)")

    # Stress table
    stress_tbl = _stress_table(stress)

    # Summary metrics
    median_final = fan_data["median"][-1]
    p5_final = fan_data["p5"][-1]
    p95_final = fan_data["p95"][-1]

    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("情景分析与压力测试", className="fw-bold"),
                html.P(
                    "AR(1)+GARCH蒙特卡洛模拟 · 波动率冲击情景 · 252交易日前瞻",
                    className="text-muted",
                ),
            ]),
        ], className="mb-4"),

        # Summary cards
        dbc.Row([
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.P("中位终值", className="text-muted small mb-1"),
                    html.H4(f"{median_final:.2f} bps", className="text-primary fw-bold mb-0"),
                ]), color="dark", className="border-0 shadow-sm"),
            ], width=3),
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.P("5% 分位 (乐观)", className="text-muted small mb-1"),
                    html.H4(f"{p5_final:.2f} bps", className="text-success fw-bold mb-0"),
                ]), color="dark", className="border-0 shadow-sm"),
            ], width=3),
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.P("95% 分位 (悲观)", className="text-muted small mb-1"),
                    html.H4(f"{p95_final:.2f} bps", className="text-danger fw-bold mb-0"),
                ]), color="dark", className="border-0 shadow-sm"),
            ], width=3),
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.P("当前利差", className="text-muted small mb-1"),
                    html.H4(f"{current:.2f} bps", className="text-info fw-bold mb-0"),
                ]), color="dark", className="border-0 shadow-sm"),
            ], width=3),
        ], className="mb-4"),

        # Fan chart
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fan_fig), width=12),
        ], className="mb-4"),

        # Stress test table
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("压力测试结果", className="mb-0")),
                    dbc.CardBody(stress_tbl),
                ], color="dark", className="border-0 shadow-sm"),
            ]),
        ]),
    ], fluid=True)
