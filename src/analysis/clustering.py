"""
Spread clustering analysis via Silhouette-optimised KMeans.

Extracts feature representations from spread data and clusters
observations into natural groups (e.g., different market regimes).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import RobustScaler

from src.core.logging_config import get_logger

logger = get_logger(__name__)


class SpreadClustering:
    """
    KMeans clustering with automatic optimal-k selection via Silhouette score.

    Uses robust standardization (median/IQR) to reduce sensitivity to outliers.
    """

    def __init__(self, n_clusters_range: tuple[int, int] = (2, 8)) -> None:
        self._range = n_clusters_range
        self._scaler = RobustScaler()
        self._model: Optional[KMeans] = None
        self._labels: Optional[np.ndarray] = None
        self._optimal_k: int = 2
        self._silhouette_scores: dict[int, float] = {}
        self._feature_names: list[str] = []

    def fit(self, features: pd.DataFrame) -> dict:
        """
        Fit clustering on a feature DataFrame.

        Parameters
        ----------
        features : pd.DataFrame
            Each column is a feature, each row an observation.

        Returns
        -------
        dict with: labels, optimal_k, silhouette_scores, cluster_centers,
             feature_importances.
        """
        self._feature_names = list(features.columns)
        X = features.dropna().values
        X_scaled = self._scaler.fit_transform(X)

        best_k = self._range[0]
        best_score = -1.0
        self._silhouette_scores = {}

        for k in range(self._range[0], min(self._range[1] + 1, len(X_scaled) - 1)):
            km = KMeans(n_clusters=k, n_init=10, random_state=42, max_iter=300)
            labels = km.fit_predict(X_scaled)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(X_scaled, labels)
            self._silhouette_scores[k] = float(score)
            if score > best_score:
                best_score = score
                best_k = k

        self._optimal_k = best_k
        self._model = KMeans(n_clusters=best_k, n_init=10, random_state=42, max_iter=300)
        self._labels = self._model.fit_predict(X_scaled)

        # Cluster centers in original space
        centers_scaled = self._model.cluster_centers_
        centers_original = self._scaler.inverse_transform(centers_scaled)

        # Feature importance: variance ratio per feature across clusters
        feature_importances = {}
        for i, fname in enumerate(self._feature_names):
            between_var = np.var(centers_original[:, i])
            total_var = np.var(X[:, i])
            feature_importances[fname] = float(between_var / total_var) if total_var > 0 else 0.0

        result = {
            "labels": self._labels,
            "optimal_k": self._optimal_k,
            "silhouette_scores": self._silhouette_scores,
            "silhouette_best": best_score,
            "cluster_centers": centers_original,
            "feature_importances": feature_importances,
            "cluster_sizes": {
                int(k): int(v) for k, v in zip(*np.unique(self._labels, return_counts=True))
            },
        }

        logger.info(
            "Clustering: optimal k=%d (silhouette=%.3f), sizes=%s",
            best_k, best_score, result["cluster_sizes"],
        )
        return result

    def fit_on_spreads(
        self,
        spread_df: pd.DataFrame,
        spread_col: str = "spread_all",
        features: list[str] | None = None,
    ) -> dict:
        """
        Auto-extract features from spread data and cluster.

        Features (default):
        - rolling_mean_20: 20-day rolling mean
        - rolling_std_20: 20-day rolling std
        - rolling_skew_60: 60-day rolling skewness
        - level_percentile: percentile rank of current level

        Parameters
        ----------
        spread_df : pd.DataFrame
            Must contain `spread_col` column.
        spread_col : str
            Name of the spread column.
        features : list[str] | None
            Override feature list.
        """
        s = spread_df[spread_col]
        feat_df = pd.DataFrame(index=spread_df.index)

        feat_df["rolling_mean_20"] = s.rolling(20).mean()
        feat_df["rolling_std_20"] = s.rolling(20).std()
        feat_df["rolling_skew_60"] = s.rolling(60).skew()
        feat_df["level_percentile"] = s.rank(pct=True)
        feat_df["rolling_range_60"] = s.rolling(60).max() - s.rolling(60).min()

        # Drop NaN rows (from rolling windows)
        feat_df = feat_df.dropna()

        if features:
            feat_df = feat_df[[f for f in features if f in feat_df.columns]]

        return self.fit(feat_df)

    @property
    def labels(self) -> Optional[np.ndarray]:
        return self._labels

    @property
    def optimal_k(self) -> int:
        return self._optimal_k
