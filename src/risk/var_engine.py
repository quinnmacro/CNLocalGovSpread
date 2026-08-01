"""
Value-at-Risk computation engine with multiple methods.

Supports: historical simulation, parametric (Student-t), and EVT-based VaR.
Also provides rolling VaR computation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.core.config import get_settings
from src.core.logging_config import get_logger

logger = get_logger(__name__)


class VaREngine:
    """
    Multi-method Value-at-Risk computation.

    Methods
    -------
    historical_var : Simple quantile-based VaR.
    parametric_var : Student-t distribution VaR.
    evt_var : EVT-POT based VaR (delegates to EVTAnalyzer).
    rolling_var : Rolling-window VaR time series.
    """

    @staticmethod
    def historical_var(
        returns: pd.Series | np.ndarray,
        confidence: float = 0.99,
    ) -> dict:
        """
        Historical simulation VaR.

        VaR = quantile(returns, confidence).
        ES = mean of returns exceeding VaR.
        """
        r = np.asarray(returns, dtype=float)
        r = r[~np.isnan(r)]

        var_val = float(np.quantile(r, confidence))
        tail = r[r > var_val]
        es_val = float(tail.mean()) if len(tail) > 0 else var_val

        return {
            "method": "historical",
            "var": var_val,
            "es": es_val,
            "n_obs": len(r),
        }

    @staticmethod
    def parametric_var(
        returns: pd.Series | np.ndarray,
        confidence: float = 0.99,
    ) -> dict:
        """
        Parametric VaR assuming Student-t distribution.

        Fit (df, loc, scale) via MLE, then compute VaR from the fitted CDF.
        """
        r = np.asarray(returns, dtype=float)
        r = r[~np.isnan(r)]

        df, loc, scale = sp_stats.t.fit(r)
        var_val = float(sp_stats.t.ppf(confidence, df=df, loc=loc, scale=scale))

        # ES for Student-t
        # ES = E[X | X > VaR] = loc + scale * (df + t_alpha²) / (df - 1) * pdf(t_alpha) / (1-alpha)
        t_alpha = sp_stats.t.ppf(confidence, df=df)
        if df > 1:
            es_val = loc + scale * (df + t_alpha ** 2) / (df - 1) * sp_stats.t.pdf(t_alpha, df) / (1 - confidence)
        else:
            es_val = var_val  # undefined

        return {
            "method": "parametric_t",
            "var": var_val,
            "es": float(es_val),
            "df": float(df),
            "loc": float(loc),
            "scale": float(scale),
        }

    @staticmethod
    def evt_var(
        returns: pd.Series,
        confidence: float = 0.99,
        threshold_percentile: float | None = None,
    ) -> dict:
        """
        EVT-based VaR via Peaks Over Threshold.

        Delegates to EVTAnalyzer.fit().
        """
        from src.risk.evt import EVTAnalyzer

        analyzer = EVTAnalyzer(threshold_percentile=threshold_percentile)
        analyzer.fit(returns, confidence=confidence)
        result = analyzer.result

        return {
            "method": "evt_pot",
            "var": result.var,
            "es": result.es,
            "gpd_shape": result.gpd_shape,
            "gpd_scale": result.gpd_scale,
            "tail_index": result.tail_index,
            "threshold": result.threshold,
            "n_exceedances": result.n_exceedances,
        }

    @staticmethod
    def rolling_var(
        returns: pd.Series,
        window: int = 252,
        confidence: float = 0.99,
        method: str = "historical",
    ) -> pd.DataFrame:
        """
        Compute rolling-window VaR time series.

        Parameters
        ----------
        returns : pd.Series
            Spread change series.
        window : int
            Rolling window in trading days.
        confidence : float
            VaR confidence level.
        method : str
            "historical" or "parametric".

        Returns
        -------
        pd.DataFrame with columns: var, es.
        """
        engine = VaREngine()
        var_values = []
        es_values = []

        for i in range(window, len(returns)):
            chunk = returns.iloc[i - window:i]

            if method == "parametric":
                result = engine.parametric_var(chunk, confidence)
            else:
                result = engine.historical_var(chunk, confidence)

            var_values.append(result["var"])
            es_values.append(result["es"])

        idx = returns.index[window:]
        return pd.DataFrame(
            {"var": var_values, "es": es_values},
            index=idx,
        )

    @staticmethod
    def compare_methods(
        returns: pd.Series,
        confidence: float = 0.99,
    ) -> pd.DataFrame:
        """
        Compare all VaR methods side by side.

        Returns DataFrame with method as index, var/es as columns.
        """
        engine = VaREngine()

        results = [
            engine.historical_var(returns, confidence),
            engine.parametric_var(returns, confidence),
            engine.evt_var(returns, confidence),
        ]

        df = pd.DataFrame(results).set_index("method")
        logger.info("VaR comparison:\n%s", df.to_string())
        return df
