"""
Hidden Markov Model regime detection for volatility states.

Uses hmmlearn's GaussianHMM to identify distinct volatility regimes
(low, mid, high) in the conditional volatility series.

Regimes are sorted by mean volatility (0=lowest, n-1=highest).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.core.config import get_settings
from src.core.logging_config import get_logger
from src.core.types import RegimeResult

logger = get_logger(__name__)


class HMMRegimeDetector:
    """
    HMM-based volatility regime detector.

    Parameters
    ----------
    n_regimes : int
        Number of hidden states (default from config).
    """

    def __init__(self, n_regimes: int | None = None) -> None:
        self._n_regimes = n_regimes or get_settings().model.n_regimes
        self._model = None
        self._result: Optional[RegimeResult] = None

    def fit(self, volatility_series: pd.Series) -> RegimeResult:
        """
        Fit Gaussian HMM to volatility series.

        Parameters
        ----------
        volatility_series : pd.Series
            Conditional volatility (e.g., from a GARCH model).

        Returns
        -------
        RegimeResult with sorted regimes.
        """
        try:
            from hmmlearn import hmm
        except ImportError:
            logger.warning("hmmlearn not installed. Returning single-regime result.")
            return self._fallback_result(volatility_series)

        X = volatility_series.values.reshape(-1, 1)

        try:
            self._model = hmm.GaussianHMM(
                n_components=self._n_regimes,
                covariance_type="full",
                n_iter=200,
                random_state=42,
                tol=1e-4,
            )
            self._model.fit(X)
            raw_labels = self._model.predict(X)
        except Exception as exc:
            logger.warning("HMM fit failed: %s. Returning fallback.", exc)
            return self._fallback_result(volatility_series)

        # Compute per-regime statistics (before relabeling)
        raw_means = {}
        raw_stds = {}
        for i in range(self._n_regimes):
            mask = raw_labels == i
            if mask.sum() > 0:
                raw_means[i] = float(volatility_series.values[mask].mean())
                raw_stds[i] = float(volatility_series.values[mask].std())
            else:
                raw_means[i] = 0.0
                raw_stds[i] = 0.0

        # Sort regimes by mean volatility (ascending)
        sorted_order = sorted(range(self._n_regimes), key=lambda i: raw_means[i])
        label_map = {old: new for new, old in enumerate(sorted_order)}

        # Relabel
        labels = np.array([label_map[l] for l in raw_labels])

        # Recompute statistics with new labels
        regime_means = {}
        regime_stds = {}
        for i in range(self._n_regimes):
            mask = labels == i
            regime_means[i] = float(volatility_series.values[mask].mean())
            regime_stds[i] = float(volatility_series.values[mask].std()) if mask.sum() > 1 else 0.0

        # Reconstruct transition matrix in new label order
        raw_transmat = self._model.transmat_
        new_transmat = np.zeros_like(raw_transmat)
        for old_from, new_from in label_map.items():
            for old_to, new_to in label_map.items():
                new_transmat[new_from, new_to] = raw_transmat[old_from, old_to]

        current_regime = int(labels[-1])

        self._result = RegimeResult(
            n_regimes=self._n_regimes,
            labels=labels,
            transition_matrix=new_transmat,
            regime_means=regime_means,
            regime_stds=regime_stds,
            current_regime=current_regime,
        )

        logger.info(
            "HMM regimes: %s (current=%d: %s)",
            {i: f"μ={m:.2f}" for i, m in regime_means.items()},
            current_regime, self._result.current_regime_name,
        )
        return self._result

    def _fallback_result(self, vol: pd.Series) -> RegimeResult:
        """Single-regime fallback when HMM fails."""
        n = len(vol)
        return RegimeResult(
            n_regimes=1,
            labels=np.zeros(n, dtype=int),
            transition_matrix=np.array([[1.0]]),
            regime_means={0: float(vol.mean())},
            regime_stds={0: float(vol.std())},
            current_regime=0,
        )

    @property
    def result(self) -> Optional[RegimeResult]:
        return self._result
