"""
Home page: Overview dashboard with spread chart, market gauge, and key statistics.
"""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from dashboard.components._data import (
    get_data,
    get_market_gauge,
    get_returns,
)
from dashboard.components.charts import market_gauge_chart, spread_timeseries

dash.register_page(__name__, path="/", name="Overview")


def _stat_card(label: str, value: str, color: str = "primary") -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            html.P(label, className="text-muted small mb-1"),
            html.H4(value, className=f"text-{color} fw-bold mb-0"),
        ]),
        color="dark",
        className="border-0 shadow-sm",
    )


def layout():
    df = get_data()
    returns = get_returns()
    gauge = get_market_gauge()

    # Key statistics
    spread_col = df["spread_all"]
    latest = float(spread_col.iloc[-1])
    mean_val = float(spread_col.mean())
    std_val = float(returns.std())
    n_obs = len(df)
    date_start = str(df["date"].iloc[0].date())
    date_end = str(df["date"].iloc[-1].date())

    # Market gauge
    composite = gauge["composite"]
    status_eng, status_chn = gauge["status"]
    status_colors = {
        "calm": "success", "normal": "info",
        "elevated": "warning", "stressed": "danger",
    }
    color_key = status_eng.lower()
    badge_color = status_colors.get(color_key, "secondary")

    # Charts
    spread_chart = spread_timeseries(df, title="地方政府债信用利差走势")
    gauge_fig = market_gauge_chart(
        composite,
        indicators={k: v["score"] for k, v in gauge["indicator_scores"].items()},
        title="市场压力仪表盘",
    )

    # Indicator breakdown
    indicator_rows = [
        html.Tr([
            html.Td(k, className="text-muted"),
            html.Td(
                dbc.Progress(
                    value=int(v["score"]),
                    color="danger" if v["score"] > 60 else "warning" if v["score"] > 40 else "success",
                    style={"height": "12px"},
                ),
                style={"width": "60%"},
            ),
            html.Td(f"{v['score']:.0f}", className="text-end fw-bold"),
        ])
        for k, v in gauge["indicator_scores"].items()
    ]

    return dbc.Container([
        # Title
        dbc.Row([
            dbc.Col([
                html.H2("CN Local Gov Spread Analysis", className="fw-bold"),
                html.P(
                    f"Advanced Econometric Framework · {n_obs:,} observations · {date_start} → {date_end}",
                    className="text-muted",
                ),
            ]),
        ], className="mb-4"),

        # Stat cards
        dbc.Row([
            dbc.Col(_stat_card("当前利差", f"{latest:.2f} bps", "primary"), width=3),
            dbc.Col(_stat_card("均值水平", f"{mean_val:.2f} bps", "info"), width=3),
            dbc.Col(_stat_card("波动率(日)", f"{std_val:.3f} bps", "warning"), width=3),
            dbc.Col([
                dbc.Card(
                    dbc.CardBody([
                        html.P("市场状态", className="text-muted small mb-1"),
                        html.H4([
                            html.Span(status_eng.capitalize(), className=f"text-{badge_color} fw-bold"),
                            html.Small(f" ({status_chn})", className="text-muted ms-1"),
                        ], className="mb-0"),
                    ]),
                    color="dark",
                    className="border-0 shadow-sm",
                ),
            ], width=3),
        ], className="mb-4"),

        # Main spread chart
        dbc.Row([
            dbc.Col(dcc.Graph(figure=spread_chart), width=12),
        ], className="mb-4"),

        # Gauge + indicators
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(figure=gauge_fig),
                    ]),
                ], color="dark", className="border-0 shadow-sm"),
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("指标分解", className="mb-0")),
                    dbc.CardBody([
                        dbc.Table(
                            [html.Thead(html.Tr([
                                html.Th("指标"), html.Th("得分", style={"width": "60%"}),
                                html.Th("值", className="text-end"),
                            ]))] + [html.Tbody(indicator_rows)],
                            borderless=True, color="dark", size="sm",
                        ),
                    ]),
                ], color="dark", className="border-0 shadow-sm"),
            ], width=6),
        ]),
    ], fluid=True)
