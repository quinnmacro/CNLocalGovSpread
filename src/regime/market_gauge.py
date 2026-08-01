"""
Multi-indicator market stress gauge.

Uses sigmoid/quantile mapping for smooth, monotone scoring (NOT if-elif ladders).
Each indicator produces a score in [0, 100] via sigmoid transformation.
Composite score is a weighted average.

Indicators:
1. Spread Position: percentile rank → sigmoid
2. Volatility Regime: current/mean ratio → sigmoid
3. VaR Breach: return/VaR ratio → sigmoid
4. Signal Deviation: |z-score| → sigmoid
5. Trend Momentum: rolling trend strength → sigmoid
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.core.config import RiskConfig, get_settings
from src.core.logging_config import get_logger

logger = get_logger(__name__)

# Default weights per indicator
DEFAULT_WEIGHTS = {
    "spread_position": 0.20,
    "volatility_regime": 0.25,
    "var_breach": 0.25,
    "signal_deviation": 0.15,
    "trend_momentum": 0.15,
}

# Status classification thresholds
STATUS_THRESHOLDS = [
    (20, "safe", "安全"),
    (40, "watch", "关注"),
    (60, "caution", "警戒"),
    (80, "warning", "预警"),
    (100, "danger", "危险"),
]


def _sigmoid(x: float | np.ndarray, midpoint: float = 1.0, steepness: float = 5.0) -> float | np.ndarray:
    """Smooth sigmoid mapping: 0→0, midpoint→50, ∞→100."""
    return 100.0 / (1.0 + np.exp(-steepness * (x - midpoint)))


class MarketGauge:
    """
    Multi-indicator market stress gauge with sigmoid scoring.

    All indicators are mapped to [0, 100] via smooth sigmoid functions,
    ensuring no discontinuous jumps (unlike the legacy if-elif approach).
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        config: RiskConfig | None = None,
    ) -> None:
        self._weights = weights or DEFAULT_WEIGHTS.copy()
        self._config = config or get_settings().risk

    def compute_indicator_scores(
        self,
        spread: pd.Series,
        returns: pd.Series | None = None,
        vol_series: pd.Series | None = None,
        signal_deviation: pd.Series | None = None,
        var_estimate: float | None = None,
    ) -> dict[str, dict]:
        """
        Compute individual indicator scores.

        Each score is in [0, 100] via sigmoid mapping.
        Returns dict with score + raw values for each indicator.
        """
        scores: dict[str, dict] = {}

        # 1. Spread Position: percentile rank → z-score → sigmoid
        current = float(spread.iloc[-1])
        mean_val = float(spread.mean())
        std_val = float(spread.std())
        z = (current - mean_val) / std_val if std_val > 1e-6 else 0.0
        pct = float((spread < current).sum() / len(spread) * 100)

        # Sigmoid: z=0 → 50, z=2 → ~99.3, z=-2 → ~0.7
        # Use absolute z for risk scoring (both extremes are risky)
        spread_score = float(_sigmoid(abs(z), midpoint=1.5, steepness=3.0))
        scores["spread_position"] = {
            "score": spread_score,
            "z_score": z,
            "percentile": pct,
            "current": current,
            "mean": mean_val,
        }

        # 2. Volatility Regime: current/rolling_mean ratio → sigmoid
        if vol_series is not None and len(vol_series) > 60:
            current_vol = float(vol_series.iloc[-1])
            rolling_mean = float(vol_series.iloc[-60:].mean())
            vol_ratio = current_vol / rolling_mean if rolling_mean > 1e-6 else 1.0
            vol_score = float(_sigmoid(vol_ratio, midpoint=1.0, steepness=5.0))
            scores["volatility_regime"] = {
                "score": vol_score,
                "ratio": vol_ratio,
                "current_vol": current_vol,
                "mean_vol": rolling_mean,
            }
        else:
            scores["volatility_regime"] = {"score": 50.0, "ratio": 1.0}

        # 3. VaR Breach: |return| / VaR ratio → sigmoid
        if var_estimate is not None and var_estimate > 0 and returns is not None:
            current_return = float(abs(returns.iloc[-1]))
            var_ratio = current_return / var_estimate
            var_score = float(_sigmoid(var_ratio, midpoint=1.0, steepness=5.0))
            scores["var_breach"] = {
                "score": var_score,
                "ratio": var_ratio,
                "current_return": current_return,
                "var_estimate": var_estimate,
            }
        else:
            scores["var_breach"] = {"score": 50.0, "ratio": 1.0}

        # 4. Signal Deviation: |z-score| → sigmoid
        if signal_deviation is not None and len(signal_deviation) > 0:
            abs_z = float(abs(signal_deviation.iloc[-1]))
            dev_score = float(_sigmoid(abs_z, midpoint=1.5, steepness=3.0))
            scores["signal_deviation"] = {
                "score": dev_score,
                "z_score": abs_z,
                "raw_deviation": float(signal_deviation.iloc[-1]),
            }
        else:
            scores["signal_deviation"] = {"score": 50.0, "z_score": 0.0}

        # 5. Trend Momentum: rolling trend strength → sigmoid
        if returns is not None and len(returns) > 60:
            recent_20 = float(returns.iloc[-20:].mean())
            older_40 = float(returns.iloc[-60:-20].mean())
            trend_delta = recent_20 - older_40
            std_60 = float(returns.iloc[-60:].std())
            trend_strength = abs(trend_delta) / std_60 if std_60 > 1e-6 else 0.0
            trend_score = float(_sigmoid(trend_strength, midpoint=0.5, steepness=4.0))
            scores["trend_momentum"] = {
                "score": trend_score,
                "delta": trend_delta,
                "strength": trend_strength,
                "direction": "up" if trend_delta > 0 else "down",
            }
        else:
            scores["trend_momentum"] = {"score": 50.0, "delta": 0.0}

        return scores

    def compute_composite(
        self,
        indicator_scores: dict[str, dict],
        weights: dict[str, float] | None = None,
    ) -> float:
        """Weighted average of indicator scores → composite [0, 100]."""
        w = weights or self._weights
        total = 0.0
        total_weight = 0.0
        for key, weight in w.items():
            if key in indicator_scores:
                total += indicator_scores[key]["score"] * weight
                total_weight += weight

        return total / total_weight if total_weight > 0 else 50.0

    def classify(self, composite_score: float) -> tuple[str, str]:
        """
        Classify composite score into status category.

        Returns (english_label, chinese_label).
        """
        for threshold, eng, chn in STATUS_THRESHOLDS:
            if composite_score <= threshold:
                return eng, chn
        return "danger", "危险"

    def assess(
        self,
        spread: pd.Series,
        returns: pd.Series | None = None,
        vol_series: pd.Series | None = None,
        signal_deviation: pd.Series | None = None,
        var_estimate: float | None = None,
    ) -> dict:
        """
        Full market assessment: scores → composite → classification.

        Returns dict with:
        - indicator_scores: per-indicator details
        - composite: float [0, 100]
        - status: (english, chinese) tuple
        """
        scores = self.compute_indicator_scores(
            spread, returns, vol_series, signal_deviation, var_estimate
        )
        composite = self.compute_composite(scores)
        status = self.classify(composite)

        result = {
            "indicator_scores": scores,
            "composite": composite,
            "status": status,
        }

        logger.info(
            "Market gauge: composite=%.1f (%s/%s)",
            composite, status[0], status[1],
        )
        return result
