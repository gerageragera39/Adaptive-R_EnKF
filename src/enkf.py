"""Stochastic Ensemble Kalman Filter for Lorenz'96.

The implementation uses a stochastic EnKF with perturbed observations. This is
simple to explain in a course demo and supports time-varying scalar observation
variance R_k = r_k I. Ensemble arrays always have shape (N, state_dim).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

from .adaptive_r import AdaptiveRModel, diagnostic_features, innovation_based_r
from .lorenz96 import integrate_steps
from .metrics import rmse


@dataclass
class EnKFResult:
    obs_times: np.ndarray
    forecast_mean: np.ndarray
    analysis_mean: np.ndarray
    forecast_rmse: np.ndarray
    analysis_rmse: np.ndarray
    innovations: np.ndarray
    ensemble_spread: np.ndarray
    used_r_values: np.ndarray
    analysis_increment_norm: np.ndarray
    features: np.ndarray


def initialize_ensemble(true_initial_state: np.ndarray, N: int, initial_spread: float = 1.0, seed: Optional[int] = None) -> np.ndarray:
    """Create an initial ensemble around the first true state for a twin experiment."""
    if N < 2:
        raise ValueError("N must be at least 2")
    rng = np.random.default_rng(seed)
    x0 = np.asarray(true_initial_state, dtype=float)
    return x0[None, :] + initial_spread * rng.standard_normal((N, x0.size))


def _analysis_update(
    Xf: np.ndarray,
    y: np.ndarray,
    state_indices: np.ndarray,
    r_value: float,
    rng: np.random.Generator,
    jitter: float = 1e-7,
) -> np.ndarray:
    """Stochastic EnKF analysis update for scalar R = r I."""
    N, K = Xf.shape
    y = np.asarray(y, dtype=float)
    state_indices = np.asarray(state_indices, dtype=int)
    obs_dim = y.size
    if obs_dim != len(state_indices):
        raise ValueError("Observation dimension and state_indices length mismatch")
    r_value = float(r_value)
    if r_value <= 0:
        raise ValueError("r_value must be positive")

    Xmean = Xf.mean(axis=0)
    Af = Xf - Xmean
    Yf = Xf[:, state_indices]
    Ymean = Yf.mean(axis=0)
    Yanom = Yf - Ymean

    cross_cov = (Af.T @ Yanom) / (N - 1)  # K x obs_dim
    obs_cov = (Yanom.T @ Yanom) / (N - 1) + (r_value + jitter) * np.eye(obs_dim)
    # Solve obs_cov * Z = cross_cov.T, then K_gain = Z.T. This is stabler than explicit inverse.
    K_gain = np.linalg.solve(obs_cov, cross_cov.T).T

    perturbed_y = y[None, :] + np.sqrt(r_value) * rng.standard_normal((N, obs_dim))
    innovations_members = perturbed_y - Yf
    return Xf + innovations_members @ K_gain.T


def run_enkf(
    observations: np.ndarray,
    true_states_at_obs: np.ndarray,
    obs_times: np.ndarray,
    state_indices: Optional[np.ndarray] = None,
    N: int = 20,
    dt_model: float = 0.05,
    F: float = 8.0,
    method: Literal["fixed", "oracle", "adaptive", "diagnostic"] = "fixed",
    fixed_r: float = 1.0,
    true_r_values: Optional[np.ndarray] = None,
    adaptive_model: Optional[AdaptiveRModel] = None,
    initial_spread: float = 1.0,
    seed: Optional[int] = None,
    min_r: float = 0.05,
    max_r: float = 5.0,
) -> EnKFResult:
    """Run EnKF over observation times using one R strategy.

    ``true_r_values`` is required for ``oracle`` and for R-tracking metrics but
    is never used by ``adaptive`` prediction.
    """
    y_all = np.asarray(observations, dtype=float)
    truth = np.asarray(true_states_at_obs, dtype=float)
    obs_times = np.asarray(obs_times, dtype=float)
    if y_all.ndim != 2 or truth.ndim != 2:
        raise ValueError("observations and true_states_at_obs must be 2D")
    if y_all.shape[0] != truth.shape[0] or len(obs_times) != y_all.shape[0]:
        raise ValueError("Observation, truth, and time lengths must match")
    K = truth.shape[1]
    state_indices = np.arange(K, dtype=int) if state_indices is None else np.asarray(state_indices, dtype=int)
    if y_all.shape[1] != len(state_indices):
        raise ValueError("observations second dimension must equal number of observed state indices")
    if fixed_r <= 0:
        raise ValueError("fixed_r must be positive")
    if method in {"oracle"} and true_r_values is None:
        raise ValueError("true_r_values is required for oracle mode")
    if method == "adaptive" and adaptive_model is None:
        raise ValueError("adaptive_model is required for adaptive mode")

    rng = np.random.default_rng(seed)
    if adaptive_model is not None:
        adaptive_model.reset()

    n_obs = y_all.shape[0]
    obs_dim = y_all.shape[1]
    forecast_mean = np.empty((n_obs, K), dtype=float)
    analysis_mean = np.empty((n_obs, K), dtype=float)
    forecast_rmse = np.empty(n_obs, dtype=float)
    analysis_rmse = np.empty(n_obs, dtype=float)
    innovations = np.empty((n_obs, obs_dim), dtype=float)
    ensemble_spread = np.empty(n_obs, dtype=float)
    used_r = np.empty(n_obs, dtype=float)
    increment_norm = np.empty(n_obs, dtype=float)
    features = np.empty((n_obs, 7), dtype=float)

    X = initialize_ensemble(truth[0], N=N, initial_spread=initial_spread, seed=seed)
    prev_increment = 0.0
    prev_r = float(fixed_r)

    for k in range(n_obs):
        if k > 0:
            delta_t = float(obs_times[k] - obs_times[k - 1])
            steps = int(round(delta_t / dt_model))
            if steps < 1 or not np.isclose(steps * dt_model, delta_t):
                raise ValueError("Observation time gaps must be positive multiples of dt_model")
            X = integrate_steps(X, steps, dt=dt_model, F=F)

        xf_mean = X.mean(axis=0)
        Yf = X[:, state_indices]
        obs_forecast_mean = Yf.mean(axis=0)
        innovation = y_all[k] - obs_forecast_mean
        feature_vec = diagnostic_features(innovation, Yf, prev_increment, prev_r)

        if method == "fixed":
            r_k = fixed_r
        elif method == "oracle":
            r_k = float(true_r_values[k])
        elif method == "adaptive":
            r_k = adaptive_model.predict_one(feature_vec, smooth=True)
        elif method == "diagnostic":
            r_k = innovation_based_r(innovation, Yf, min_r=min_r, max_r=max_r)
        else:
            raise ValueError(f"Unknown EnKF method '{method}'")
        r_k = float(np.clip(r_k, min_r, max_r))

        Xa = _analysis_update(X, y_all[k], state_indices=state_indices, r_value=r_k, rng=rng)
        xa_mean = Xa.mean(axis=0)

        forecast_mean[k] = xf_mean
        analysis_mean[k] = xa_mean
        forecast_rmse[k] = rmse(xf_mean, truth[k])
        analysis_rmse[k] = rmse(xa_mean, truth[k])
        innovations[k] = innovation
        ensemble_spread[k] = float(np.sqrt(np.mean(np.var(X, axis=0, ddof=1))))
        used_r[k] = r_k
        increment_norm[k] = float(np.linalg.norm(xa_mean - xf_mean) / np.sqrt(K))
        features[k] = feature_vec

        X = Xa
        prev_increment = increment_norm[k]
        prev_r = r_k

    return EnKFResult(
        obs_times=obs_times,
        forecast_mean=forecast_mean,
        analysis_mean=analysis_mean,
        forecast_rmse=forecast_rmse,
        analysis_rmse=analysis_rmse,
        innovations=innovations,
        ensemble_spread=ensemble_spread,
        used_r_values=used_r,
        analysis_increment_norm=increment_norm,
        features=features,
    )


def collect_training_features(
    observations: np.ndarray,
    true_states_at_obs: np.ndarray,
    obs_times: np.ndarray,
    true_r_values: np.ndarray,
    state_indices: Optional[np.ndarray] = None,
    N: int = 20,
    dt_model: float = 0.05,
    F: float = 8.0,
    seed: Optional[int] = None,
    fixed_r: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect supervised examples from a fixed-R EnKF training run.

    The target is synthetic true r_k. Features are the same diagnostics used in
    evaluation and never include true state or current true R.
    """
    result = run_enkf(
        observations=observations,
        true_states_at_obs=true_states_at_obs,
        obs_times=obs_times,
        state_indices=state_indices,
        N=N,
        dt_model=dt_model,
        F=F,
        method="fixed",
        fixed_r=fixed_r,
        seed=seed,
    )
    return result.features, np.asarray(true_r_values, dtype=float)
