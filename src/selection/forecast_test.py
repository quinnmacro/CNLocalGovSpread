"""
Forecast comparison tests: Diebold-Mariano and Model Confidence Set.

References:
- Diebold & Mariano (1995): Comparing predictive accuracy.
- Hansen, Lunde, Nason (2011): The Model Confidence Set.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.core.logging_config import get_logger
from src.core.types import ForecastTestResult

logger = get_logger(__name__)


def _hac_variance(x: np.ndarray, max_lag: int | None = None) -> float:
    """
    Newey-West HAC variance estimator.

    var_hat = gamma_0 + 2 * sum_{j=1}^{h} w_j * gamma_j
    where w_j = 1 - j/(h+1), h = floor(n^{1/3}).
    """
    n = len(x)
    if max_lag is None:
        max_lag = int(n ** (1 / 3))

    x_centered = x - x.mean()
    gamma_0 = np.mean(x_centered ** 2)

    hac = gamma_0
    for j in range(1, max_lag + 1):
        gamma_j = np.mean(x_centered[j:] * x_centered[:-j])
        weight = 1 - j / (max_lag + 1)
        hac += 2 * weight * gamma_j

    return max(hac, 1e-12)  # ensure positive


def diebold_mariano_test(
    errors_a: np.ndarray | pd.Series,
    errors_b: np.ndarray | pd.Series,
    loss: str = "mse",
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
) -> ForecastTestResult:
    """
    Diebold-Mariano test for equal predictive accuracy.

    Parameters
    ----------
    errors_a, errors_b : array-like
        Forecast errors (not losses) from models A and B.
    loss : str
        Loss function: "mse" (squared error) or "mae" (absolute error).
    model_a_name, model_b_name : str
        Names for reporting.

    Returns
    -------
    ForecastTestResult with dm_stat, dm_pvalue, winner.

    Notes
    -----
    H0: E[L(e_A) - L(e_B)] = 0 (equal forecast accuracy).
    DM > 0 => Model A worse than Model B.
    """
    ea = np.asarray(errors_a, dtype=float)
    eb = np.asarray(errors_b, dtype=float)

    if len(ea) != len(eb):
        raise ValueError(f"Error series must have same length: {len(ea)} vs {len(eb)}")

    # Compute losses
    if loss == "mse":
        la = ea ** 2
        lb = eb ** 2
    elif loss == "mae":
        la = np.abs(ea)
        lb = np.abs(eb)
    else:
        raise ValueError(f"Unknown loss: {loss}. Use 'mse' or 'mae'.")

    # Loss differential
    d = la - lb
    n = len(d)

    # DM statistic with HAC standard error
    mean_d = np.mean(d)
    var_d = _hac_variance(d)
    se_d = np.sqrt(var_d / n)

    dm_stat = mean_d / se_d
    dm_pvalue = 2 * (1 - sp_stats.norm.cdf(abs(dm_stat)))

    # Determine winner
    if dm_pvalue < 0.05:
        winner = model_b_name if dm_stat > 0 else model_a_name
    else:
        winner = None  # no significant difference

    result = ForecastTestResult(
        test_name="Diebold-Mariano",
        model_a=model_a_name,
        model_b=model_b_name,
        dm_stat=float(dm_stat),
        dm_pvalue=float(dm_pvalue),
        winner=winner,
    )

    logger.info(
        "DM test: %s vs %s, stat=%.3f, p=%.4f, winner=%s",
        model_a_name, model_b_name, dm_stat, dm_pvalue, winner or "tie",
    )
    return result


class ModelConfidenceSet:
    """
    Hansen (2011) Model Confidence Set.

    Sequentially eliminates the worst-performing model until all
    remaining models have p-value > alpha.

    Parameters
    ----------
    alpha : float
        Significance level for elimination (default 0.1).
    n_boot : int
        Number of bootstrap replications (default 1000).
    """

    def __init__(self, alpha: float = 0.1, n_boot: int = 1000) -> None:
        self._alpha = alpha
        self._n_boot = n_boot
        self._results: dict[str, ForecastTestResult] = {}
        self._confidence_set: list[str] = []
        self._elimination_order: list[str] = []

    def compute(
        self,
        losses: dict[str, pd.Series | np.ndarray],
    ) -> dict[str, ForecastTestResult]:
        """
        Compute the Model Confidence Set.

        Parameters
        ----------
        losses : dict
            {model_name: loss_series} where loss_series is per-observation loss.

        Returns
        -------
        dict : {model_name: ForecastTestResult} with in_confidence_set flag.
        """
        model_names = list(losses.keys())
        loss_arrays = {k: np.asarray(v, dtype=float) for k, v in losses.items()}
        n = len(next(iter(loss_arrays.values())))

        remaining = set(model_names)
        self._elimination_order = []

        while len(remaining) > 1:
            remaining_list = sorted(remaining)
            k = len(remaining_list)

            # Compute pairwise DM statistics
            dm_matrix = np.zeros((k, k))
            for i in range(k):
                for j in range(i + 1, k):
                    d = loss_arrays[remaining_list[i]] - loss_arrays[remaining_list[j]]
                    mean_d = np.mean(d)
                    var_d = _hac_variance(d)
                    se_d = np.sqrt(max(var_d / n, 1e-12))
                    dm_matrix[i, j] = mean_d / se_d
                    dm_matrix[j, i] = -dm_matrix[i, j]

            # Bootstrap p-values for the max statistic
            pvalues = self._bootstrap_pvalues(
                loss_arrays, remaining_list, dm_matrix, n
            )

            # Find worst model (highest p-value for being eliminated)
            min_pvalue = min(pvalues.values())

            if min_pvalue > self._alpha:
                # All remaining models survive
                break

            # Eliminate worst model
            worst_model = min(pvalues, key=pvalues.get)
            self._elimination_order.append(worst_model)
            remaining.remove(worst_model)

            logger.info(
                "MCS: eliminated '%s' (p=%.4f), %d models remaining",
                worst_model, min_pvalue, len(remaining),
            )

        self._confidence_set = sorted(remaining)

        # Build results
        for name in model_names:
            in_set = name in self._confidence_set
            elim_idx = (
                self._elimination_order.index(name) + 1
                if name in self._elimination_order
                else None
            )

            self._results[name] = ForecastTestResult(
                test_name="Model Confidence Set",
                model_a=name,
                model_b="remaining",
                mcs_pvalue=None,
                in_confidence_set=in_set,
            )

        logger.info(
            "MCS complete: confidence set = %s (eliminated %d models)",
            self._confidence_set, len(self._elimination_order),
        )
        return self._results

    def _bootstrap_pvalues(
        self,
        loss_arrays: dict[str, np.ndarray],
        model_names: list[str],
        dm_matrix: np.ndarray,
        n: int,
    ) -> dict[str, float]:
        """Bootstrap p-values for MCS using iid bootstrap of loss differences."""
        k = len(model_names)
        boot_max_stats = np.zeros(self._n_boot)

        # Compute observed max DM statistic
        abs_dm = np.abs(dm_matrix)
        np.fill_diagonal(abs_dm, 0)
        max_dm_per_model = abs_dm.max(axis=1)
        observed_max = max_dm_per_model.max()

        # Bootstrap
        rng = np.random.default_rng(42)
        for b in range(self._n_boot):
            # Resample indices
            idx = rng.integers(0, n, size=n)

            boot_dm = np.zeros((k, k))
            for i in range(k):
                for j in range(i + 1, k):
                    d_boot = (
                        loss_arrays[model_names[i]][idx]
                        - loss_arrays[model_names[j]][idx]
                    )
                    mean_d = np.mean(d_boot)
                    var_d = _hac_variance(d_boot)
                    se_d = np.sqrt(max(var_d / n, 1e-12))
                    boot_dm[i, j] = mean_d / se_d
                    boot_dm[j, i] = -boot_dm[i, j]

            abs_boot = np.abs(boot_dm)
            np.fill_diagonal(abs_boot, 0)
            boot_max_stats[b] = abs_boot.max()

        # Compute p-values per model
        pvalues = {}
        for i, name in enumerate(model_names):
            observed_stat = max_dm_per_model[i]
            pval = np.mean(boot_max_stats >= observed_stat)
            pvalues[name] = max(pval, 1 / self._n_boot)  # floor at 1/B

        return pvalues

    @property
    def confidence_set(self) -> list[str]:
        return self._confidence_set

    @property
    def elimination_order(self) -> list[str]:
        return self._elimination_order
