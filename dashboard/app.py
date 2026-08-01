"""
Dash multi-page dashboard for CN Local Gov Spread.

Professional showcase interface for quinnmacro.com.
Dark theme with Plotly charts and dash-bootstrap-components layout.
"""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import Dash, html

from src.core.config import get_settings
from src.core.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def create_app() -> Dash:
    """Create and configure the Dash application."""
    setup_logging(level="INFO")
    settings = get_settings()

    app = Dash(
        __name__,
        use_pages=True,
        pages_folder="pages",
        external_stylesheets=[dbc.themes.DARKLY],
        title=settings.dashboard.title,
        suppress_callback_exceptions=True,
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1"},
            {"name": "description", "content": "China Local Government Bond Spread Analysis"},
        ],
    )

    app.layout = dbc.Container([
        # Header
        dbc.Navbar(
            dbc.Container([
                dbc.NavbarBrand([
                    html.Span("QuinnMacro", className="fw-bold me-2"),
                    html.Span("CN Local Gov Spread", className="text-muted"),
                ], href="/"),
                dbc.Nav([
                    dbc.NavItem(dbc.NavLink("Overview", href="/")),
                    dbc.NavItem(dbc.NavLink("Volatility", href="/volatility")),
                    dbc.NavItem(dbc.NavLink("Risk", href="/risk")),
                    dbc.NavItem(dbc.NavLink("Regimes", href="/regimes")),
                    dbc.NavItem(dbc.NavLink("Scenarios", href="/scenarios")),
                ], className="ms-auto"),
            ], fluid=True),
            color="dark",
            dark=True,
            className="mb-4",
        ),

        # Page content
        dash.page_container,

        # Footer
        dbc.Container([
            html.Hr(),
            html.P(
                "© 2026 QuinnMacro | Advanced Econometric Framework v4.0",
                className="text-muted text-center small",
            ),
        ], fluid=True),
    ], fluid=True, className="p-3")

    return app


def run_server():
    """Run the dashboard server."""
    settings = get_settings()
    app = create_app()
    logger.info("Starting Dash dashboard on %s:%d", settings.dashboard.host, settings.dashboard.port)
    app.run(
        host=settings.dashboard.host,
        port=settings.dashboard.port,
        debug=settings.dashboard.debug,
    )


if __name__ == "__main__":
    run_server()
