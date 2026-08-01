"""
Volatility page: Model tournament, comparison chart, and diagnostics.
"""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html

from dashboard.components._data import get_fitted_models, get_returns, get_tournament_df
from dashboard.components.charts import volatility_comparison

dash.register_page(__name__, path="/volatility", name="Volatility")


def _result_card(name: str, result) -> dbc.Card:
    """Compact card showing one model's fit result."""
    metrics = []
    if result.aic is not None:
        metrics.append(html.Li(f"AIC: {result.aic:.2f}"))
    if result.bic is not None:
        metrics.append(html.Li(f"BIC: {result.bic:.2f}"))
    metrics.append(html.Li(f"RMSE: {result.rmse:.4f}"))
    metrics.append(html.Li(f"MAE: {result.mae:.4f}"))
    status = "✓" if result.converged else "✗"
    color = "success" if result.converged else "danger"

    return dbc.Card([
        dbc.CardHeader(html.Span([
            html.Span(name, className="fw-bold"),
            html.Span(f" {status}", className=f"text-{color} ms-2"),
        ])),
        dbc.CardBody(html.Ul(metrics, className="mb-0 small")),
    ], color="dark", className="border-0 shadow-sm mb-3")


def layout():
    returns = get_returns()
    models = get_fitted_models()
    tournament = get_tournament_df()

    # Build vol series dict for comparison chart
    vol_series = {}
    for name, m in models.items():
        try:
            vol_series[name] = m.result.volatility
        except Exception:
            pass

    # Chart
    chart = volatility_comparison(returns, vol_series, title="波动率模型对比")

    # Tournament table
    tour_cols = [
        {"name": "Model", "id": "model"},
        {"name": "AIC", "id": "aic", "type": "numeric", "format": {"specifier": ".2f"}},
        {"name": "BIC", "id": "bic", "type": "numeric", "format": {"specifier": ".2f"}},
        {"name": "RMSE", "id": "rmse", "type": "numeric", "format": {"specifier": ".4f"}},
        {"name": "MAE", "id": "mae", "type": "numeric", "format": {"specifier": ".4f"}},
        {"name": "Converged", "id": "converged"},
    ]
    tour_data = tournament.copy()
    for col in ("aic", "bic", "rmse", "mae"):
        if col in tour_data.columns:
            tour_data[col] = tour_data[col].round(4)

    # Model cards
    model_cards = [_result_card(name, m.result) for name, m in models.items()]

    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("波动率模型分析", className="fw-bold"),
                html.P(
                    "GARCH族 / EWMA / 机器学习模型拟合比较 · 信息准则与预测精度",
                    className="text-muted",
                ),
            ]),
        ], className="mb-4"),

        # Comparison chart
        dbc.Row([
            dbc.Col(dcc.Graph(figure=chart), width=12),
        ], className="mb-4"),

        # Tournament table
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("模型锦标赛排名", className="mb-0")),
                    dbc.CardBody([
                        dash_table.DataTable(
                            data=tour_data.to_dict("records"),
                            columns=tour_cols,
                            sort_action="native",
                            style_table={"overflowX": "auto"},
                            style_cell={
                                "backgroundColor": "#1e293b",
                                "color": "#e2e8f0",
                                "border": "1px solid #334155",
                                "textAlign": "center",
                            },
                            style_header={
                                "backgroundColor": "#0f172a",
                                "fontWeight": "bold",
                            },
                            style_data_conditional=[
                                {
                                    "if": {"column_id": "converged", "filter_query": "{converged} eq True"},
                                    "color": "#22c55e",
                                },
                            ],
                        ),
                    ]),
                ], color="dark", className="border-0 shadow-sm"),
            ]),
        ], className="mb-4"),

        # Individual model cards
        dbc.Row([
            dbc.Col(card, width=3) for card in model_cards
        ]),
    ], fluid=True)
