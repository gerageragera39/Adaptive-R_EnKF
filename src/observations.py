"""Synthetic observation generation for Lorenz'96 twin experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class ObservationData:
    obs_times: np.ndarray
    obs_indices: np.ndarray
    state_indices: np.ndarray
    y: np.ndarray
    true_r: np.ndarray
    true_obs: np.ndarray


def true_r_values(
    times: np.ndarray,
    mode: str = "sinusoidal",
    base: float = 1.0,
    amplitude: float = 0.7,
    period: float = 5.0,
    minimum: float = 0.05,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Return scalar observation-error variances r(t), clipped positive."""
    times = np.asarray(times, dtype=float)
    mode = mode.lower()
    if mode == "constant":
        r = np.full_like(times, float(base), dtype=float)
    elif mode == "sinusoidal":
        r = base + amplitude * np.sin(2.0 * np.pi * times / period)
    elif mode == "stepwise":
        half_period = max(period / 2.0, 1e-12)
        high = base + amplitude
        low = base - amplitude
        r = np.where((np.floor(times / half_period) % 2) == 0, high, low)
    elif mode == "noisy":
        rng = np.random.default_rng(seed)
        smooth = base + amplitude * np.sin(2.0 * np.pi * times / period)
        r = smooth + 0.15 * amplitude * rng.standard_normal(times.shape)
    else:
        raise ValueError(f"Unknown R mode '{mode}'. Choose constant, sinusoidal, stepwise, or noisy.")
    return np.clip(r, minimum, None).astype(float)


def observation_indices(times: np.ndarray, dt_obs: float) -> np.ndarray:
    """Return model-time indices matching the requested observation frequency."""
    if dt_obs <= 0:
        raise ValueError("dt_obs must be positive")
    times = np.asarray(times, dtype=float)
    if len(times) < 2:
        return np.array([0], dtype=int)
    dt = float(times[1] - times[0])
    stride = int(round(dt_obs / dt))
    if stride < 1 or not np.isclose(stride * dt, dt_obs):
        raise ValueError(f"dt_obs={dt_obs} must be an integer multiple of model dt={dt}")
    return np.arange(0, len(times), stride, dtype=int)


def generate_observations(
    times: np.ndarray,
    true_states: np.ndarray,
    dt_obs: float = 0.05,
    state_indices: Optional[Sequence[int]] = None,
    r_mode: str = "sinusoidal",
    base: float = 1.0,
    amplitude: float = 0.7,
    period: float = 5.0,
    min_r: float = 0.05,
    seed: Optional[int] = None,
) -> ObservationData:
    """Generate observations y_k = H x_k + N(0, r_k I).

    Full observations are the default. Partial observations are supported by
    passing ``state_indices`` but the project focuses on H=I.
    """
    times = np.asarray(times, dtype=float)
    true_states = np.asarray(true_states, dtype=float)
    if true_states.ndim != 2:
        raise ValueError("true_states must have shape (time, state_dim)")
    if len(times) != true_states.shape[0]:
        raise ValueError("times and true_states length mismatch")

    K = true_states.shape[1]
    if state_indices is None:
        state_indices_arr = np.arange(K, dtype=int)
    else:
        state_indices_arr = np.asarray(state_indices, dtype=int)
        if np.any(state_indices_arr < 0) or np.any(state_indices_arr >= K):
            raise ValueError("state_indices out of range")

    obs_idx = observation_indices(times, dt_obs)
    obs_times = times[obs_idx]
    true_obs = true_states[obs_idx][:, state_indices_arr]
    r = true_r_values(obs_times, mode=r_mode, base=base, amplitude=amplitude, period=period, minimum=min_r, seed=seed)

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(true_obs.shape) * np.sqrt(r)[:, None]
    y = true_obs + noise
    return ObservationData(obs_times=obs_times, obs_indices=obs_idx, state_indices=state_indices_arr, y=y, true_r=r, true_obs=true_obs)
