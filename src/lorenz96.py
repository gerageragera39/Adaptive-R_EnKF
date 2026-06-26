"""Lorenz'96 model utilities.

The course project uses the standard 40-variable Lorenz'96 system

    dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F

with cyclic indexing, forcing F=8, and RK4 integration with dt=0.05.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def lorenz96_derivative(x: np.ndarray, F: float = 8.0) -> np.ndarray:
    """Return the Lorenz'96 derivative for one state or an ensemble.

    Parameters
    ----------
    x:
        State array with shape ``(K,)`` or ensemble array ``(..., K)``.
    F:
        Constant forcing.
    """
    x = np.asarray(x, dtype=float)
    return (np.roll(x, -1, axis=-1) - np.roll(x, 2, axis=-1)) * np.roll(x, 1, axis=-1) - x + F


def rk4_step(x: np.ndarray, dt: float = 0.05, F: float = 8.0) -> np.ndarray:
    """Advance one RK4 step. Works for a single state or an ensemble."""
    x = np.asarray(x, dtype=float)
    k1 = lorenz96_derivative(x, F=F)
    k2 = lorenz96_derivative(x + 0.5 * dt * k1, F=F)
    k3 = lorenz96_derivative(x + 0.5 * dt * k2, F=F)
    k4 = lorenz96_derivative(x + dt * k3, F=F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_steps(x0: np.ndarray, n_steps: int, dt: float = 0.05, F: float = 8.0) -> np.ndarray:
    """Integrate ``x0`` forward by ``n_steps`` RK4 steps and return the final state."""
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")
    x = np.asarray(x0, dtype=float).copy()
    for _ in range(n_steps):
        x = rk4_step(x, dt=dt, F=F)
    return x


def default_initial_state(K: int = 40) -> np.ndarray:
    """Return a common deterministic non-equilibrium Lorenz'96 initial state."""
    x0 = np.full(K, 8.0, dtype=float)
    x0[0] += 0.01
    return x0


def simulate_trajectory(
    T: float,
    dt: float = 0.05,
    K: int = 40,
    F: float = 8.0,
    x0: Optional[np.ndarray] = None,
    spinup_time: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a Lorenz'96 trajectory after optional spin-up.

    Returns
    -------
    times, states:
        ``times`` has shape ``(n_steps + 1,)`` and ``states`` has shape
        ``(n_steps + 1, K)``.
    """
    if T < 0:
        raise ValueError("T must be non-negative")
    if dt <= 0:
        raise ValueError("dt must be positive")
    n_steps = int(round(T / dt))
    if not np.isclose(n_steps * dt, T):
        raise ValueError(f"T={T} must be an integer multiple of dt={dt}")

    state = default_initial_state(K) if x0 is None else np.asarray(x0, dtype=float).copy()
    if state.shape != (K,):
        raise ValueError(f"x0 must have shape ({K},), got {state.shape}")

    spinup_steps = int(round(spinup_time / dt))
    state = integrate_steps(state, spinup_steps, dt=dt, F=F)

    states = np.empty((n_steps + 1, K), dtype=float)
    states[0] = state
    for i in range(1, n_steps + 1):
        states[i] = rk4_step(states[i - 1], dt=dt, F=F)
    times = np.arange(n_steps + 1, dtype=float) * dt
    return times, states
