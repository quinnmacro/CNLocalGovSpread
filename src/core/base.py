"""
Abstract base classes defining the interfaces for all analytical components.

Every volatility model, signal extractor, and risk analyzer must implement
these interfaces. This ensures the tournament/selection layer can compare
any combination of models uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Self

import numpy as np
import pandas as pd

from src.core.types import (
    BacktestResult,
    DiagnosticsResult,
    RiskResult,
    SignalResult,
    VolatilityResult,
)


# --- Exceptions ---


class NotFittedError(RuntimeError):
    """Raised when a method is called before fit()."""


class ConvergenceError(RuntimeError):
    """Raised when model fitting fails to converge."""


class InsufficientDataError(ValueError):
    """Raised when input data is too short for the requested analysis."""


# --- ABC Interfaces ---


class VolatilityModel(ABC):
    """
    Abstract interface for volatility models (GARCH, EGARCH, FIGARCH, EWMA, etc).

    Usage:
        model = GARCHModel(p=1, q=1)
        model.fit(returns)
        result = model.result
    """

    _result: Optional[VolatilityResult] = None

    @property
    def result(self) -> VolatilityResult:
        if self._result is None:
            raise NotFittedError("Call fit() before accessing results.")
        return self._result

    @property
    def is_fitted(self) -> bool:
        return self._result is not None

    @abstractmethod
    def fit(self, returns: pd.Series) -> Self:
        """Fit the model to a return series. Returns self for chaining."""

    @abstractmethod
    def conditional_variance(self) -> pd.Series:
        """Return the in-sample conditional variance series."""

    @abstractmethod
    def forecast(self, steps: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """
        Forecast conditional volatility for `steps` ahead.

        Returns:
            (forecast_mean, forecast_vol) arrays of shape (steps,)
        """

    @abstractmethod
    def diagnose(self) -> DiagnosticsResult:
        """Run Ljung-Box, ARCH-LM, Jarque-Bera on standardized residuals."""

    @property
    def aic(self) -> Optional[float]:
        return self.result.aic if self.is_fitted else None

    @property
    def bic(self) -> Optional[float]:
        return self.result.bic if self.is_fitted else None


class SignalExtractor(ABC):
    """Abstract interface for signal/trend extraction methods."""

    _result: Optional[SignalResult] = None

    @property
    def result(self) -> SignalResult:
        if self._result is None:
            raise NotFittedError("Call fit() before accessing results.")
        return self._result

    @abstractmethod
    def fit(self, spread: pd.Series) -> Self:
        """Extract signal from a spread level series."""

    @abstractmethod
    def get_signal_deviation(self) -> pd.Series:
        """Return deviation of current spread from extracted signal."""


class RiskAnalyzer(ABC):
    """Abstract interface for risk measurement methods (historical, parametric, EVT)."""

    _result: Optional[RiskResult] = None

    @property
    def result(self) -> RiskResult:
        if self._result is None:
            raise NotFittedError("Call fit() before accessing results.")
        return self._result

    @abstractmethod
    def fit(self, returns: pd.Series, confidence: float = 0.99) -> Self:
        """Compute VaR and ES at the given confidence level."""

    @abstractmethod
    def backtest(self, returns: pd.Series, var_series: pd.Series) -> BacktestResult:
        """Run coverage backtest (Kupiec / Christoffersen) on a VaR series."""
