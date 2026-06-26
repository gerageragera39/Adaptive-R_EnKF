"""Offline ML model for scalar adaptive observation error variance r_k."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np

try:  # scikit-learn is listed in requirements; this keeps import errors informative.
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except Exception as exc:  # pragma: no cover - only exercised if dependency missing
    GradientBoostingRegressor = None
    RandomForestRegressor = None
    make_pipeline = None
    StandardScaler = None
    _SKLEARN_IMPORT_ERROR = exc
else:
    _SKLEARN_IMPORT_ERROR = None


FEATURE_NAMES = [
    "innovation_mean_sq",
    "innovation_variance",
    "innovation_abs_mean",
    "ensemble_spread_mean",
    "ensemble_spread_variance",
    "prev_analysis_increment_norm",
    "prev_estimated_r",
]


def diagnostic_features(
    innovation: np.ndarray,
    obs_ensemble: np.ndarray,
    prev_analysis_increment_norm: float,
    prev_estimated_r: float,
) -> np.ndarray:
    """Compute ML features using only quantities available during assimilation.

    The analysis-increment feature is the previous-cycle increment norm. The
    current-cycle increment is not known until after choosing r_k.
    """
    innovation = np.asarray(innovation, dtype=float).ravel()
    obs_ensemble = np.asarray(obs_ensemble, dtype=float)
    if obs_ensemble.ndim != 2:
        raise ValueError("obs_ensemble must have shape (N, obs_dim)")
    spread_by_obs = np.var(obs_ensemble, axis=0, ddof=1) if obs_ensemble.shape[0] > 1 else np.zeros(obs_ensemble.shape[1])
    return np.array(
        [
            float(np.mean(innovation**2)),
            float(np.var(innovation)),
            float(np.mean(np.abs(innovation))),
            float(np.mean(spread_by_obs)),
            float(np.var(spread_by_obs)),
            float(prev_analysis_increment_norm),
            float(prev_estimated_r),
        ],
        dtype=float,
    )


def innovation_based_r(innovation: np.ndarray, obs_ensemble: np.ndarray, min_r: float = 0.05, max_r: float = 5.0) -> float:
    """Simple diagnostic estimate: innovation variance minus ensemble observation spread."""
    f = diagnostic_features(innovation, obs_ensemble, 0.0, 1.0)
    estimate = f[0] - f[3]
    return float(np.clip(estimate, min_r, max_r))


@dataclass
class AdaptiveRModel:
    """A small wrapper around scikit-learn regressors for robust online use."""

    model_type: str = "random_forest"
    min_r: float = 0.05
    max_r: float = 5.0
    smoothing_alpha: float = 0.35
    random_state: int = 0
    model: object | None = None
    last_prediction_: Optional[float] = None

    def _make_model(self):
        if _SKLEARN_IMPORT_ERROR is not None:
            raise ImportError("scikit-learn is required for AdaptiveRModel") from _SKLEARN_IMPORT_ERROR
        model_type = self.model_type.lower().replace("-", "_")
        if model_type in {"random_forest", "rf"}:
            return RandomForestRegressor(n_estimators=120, min_samples_leaf=3, random_state=self.random_state, n_jobs=-1)
        if model_type in {"gradient_boosting", "gb", "gbr"}:
            return make_pipeline(
                StandardScaler(),
                GradientBoostingRegressor(random_state=self.random_state, n_estimators=160, max_depth=2),
            )
        raise ValueError("model_type must be 'random_forest' or 'gradient_boosting'")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaptiveRModel":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"X must have shape (n_samples, {len(FEATURE_NAMES)})")
        if len(y) != X.shape[0]:
            raise ValueError("X and y length mismatch")
        self.model = self._make_model()
        self.model.fit(X, np.clip(y, self.min_r, self.max_r))
        self.last_prediction_ = None
        return self

    def reset(self) -> None:
        self.last_prediction_ = None

    def predict_one(self, features: np.ndarray, smooth: bool = True) -> float:
        if self.model is None:
            raise RuntimeError("AdaptiveRModel must be fitted before prediction")
        raw = float(self.model.predict(np.asarray(features, dtype=float).reshape(1, -1))[0])
        raw = float(np.clip(raw, self.min_r, self.max_r))
        if smooth and self.smoothing_alpha is not None and self.last_prediction_ is not None:
            alpha = float(self.smoothing_alpha)
            raw = alpha * raw + (1.0 - alpha) * self.last_prediction_
        raw = float(np.clip(raw, self.min_r, self.max_r))
        self.last_prediction_ = raw
        return raw


def stack_training_sets(parts: Iterable[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    Xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for X, y in parts:
        if len(X):
            Xs.append(np.asarray(X, dtype=float))
            ys.append(np.asarray(y, dtype=float))
    if not Xs:
        raise ValueError("No training samples were generated")
    return np.vstack(Xs), np.concatenate(ys)
