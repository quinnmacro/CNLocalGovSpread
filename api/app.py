"""
FastAPI application — thin REST layer over the quant engine.

Provides endpoints for:
- Health check
- Data summary
- Model fitting and results
- Risk metrics
- Scenario generation
- Report download
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from api.routes import router
from src.core.config import get_settings
from src.core.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    setup_logging(level="INFO")
    logger.info("API starting up (v%s)", get_settings().data.__class__.__module__)
    yield
    logger.info("API shutting down")


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title="CN Local Gov Spread API",
        description="Quantitative analysis framework for China local government bond spreads",
        version="4.0.0",
        lifespan=lifespan,
    )

    # CORS for Dash frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.get("/", response_class=HTMLResponse)
    async def root() -> str:
        return """
        <html>
        <head><title>CN Local Gov Spread API</title></head>
        <body style="font-family: Inter, sans-serif; padding: 2rem; background: #0f172a; color: #f8fafc;">
            <h1 style="color: #3b82f6;">CN Local Gov Spread API v4.0</h1>
            <p>QuinnMacro quantitative analysis framework.</p>
            <ul>
                <li><a href="/docs" style="color: #3b82f6;">API Documentation</a></li>
                <li><a href="/health" style="color: #3b82f6;">Health Check</a></li>
            </ul>
        </body>
        </html>
        """

    return app


app = create_app()
