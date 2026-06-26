"""Experiment orchestration for fixed, adaptive, and oracle EnKF comparisons."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .adaptive_r import AdaptiveRModel, stack_training_sets
from .enkf import EnKFResult, collect_training_features, run_enkf
from .metrics import summarize_assimilation
from .observations import ObservationData, generate_observations
from .lorenz96 import simulate_trajectory
from .plotting import plot_grid_bar, plot_r_tracking, plot_rmse_comparison, plot_state_heatmap


@dataclass
class ExperimentConfig:
    N: int = 20
    dt_obs: float = 0.1
    T: float = 10.0
    seed: int = 1
    r_mode: str = "sinusoidal"
    ml_model_type: str = "random_forest"
    K: int = 40
    F: float = 8.0
    dt_model: float = 0.05
    spinup_time: float = 5.0
    r_base: float = 1.0
    r_amplitude: float = 0.7
    r_period: float = 5.0
    n_train_runs: int = 4
    T_train: float = 10.0


def make_truth_and_obs(config: ExperimentConfig) -> Tuple[np.ndarray, np.ndarray, ObservationData]:
    times, truth = simulate_trajectory(
        T=config.T,
        dt=config.dt_model,
        K=config.K,
        F=config.F,
        spinup_time=config.spinup_time,
    )
    obs = generate_observations(
        times,
        truth,
        dt_obs=config.dt_obs,
        r_mode=config.r_mode,
        base=config.r_base,
        amplitude=config.r_amplitude,
        period=config.r_period,
        seed=config.seed,
    )
    return times, truth, obs


def build_training_dataset(config: ExperimentConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Generate independent synthetic runs for supervised adaptive-R training."""
    parts = []
    # Deterministic but different training cases: varied phases via observation noise seed and varied periods/amplitudes.
    for j in range(config.n_train_runs):
        train_seed = 10_000 + 97 * config.seed + j
        T_train = max(config.T_train, config.dt_obs)
        times, truth = simulate_trajectory(
            T=T_train,
            dt=config.dt_model,
            K=config.K,
            F=config.F,
            spinup_time=config.spinup_time + 0.4 * j,
        )
        obs = generate_observations(
            times,
            truth,
            dt_obs=config.dt_obs,
            r_mode="sinusoidal" if config.r_mode == "constant" else config.r_mode,
            base=config.r_base,
            amplitude=config.r_amplitude * (0.75 + 0.15 * (j % 4)),
            period=max(config.r_period * (0.8 + 0.15 * (j % 5)), config.dt_obs),
            seed=train_seed,
        )
        X, y = collect_training_features(
            observations=obs.y,
            true_states_at_obs=truth[obs.obs_indices],
            obs_times=obs.obs_times,
            true_r_values=obs.true_r,
            state_indices=obs.state_indices,
            N=config.N,
            dt_model=config.dt_model,
            F=config.F,
            seed=train_seed + 1,
            fixed_r=config.r_base,
        )
        # Drop the first sample because previous-cycle features are initialized constants.
        parts.append((X[1:], y[1:]))
    return stack_training_sets(parts)


def train_adaptive_model(config: ExperimentConfig) -> AdaptiveRModel:
    X_train, y_train = build_training_dataset(config)
    model = AdaptiveRModel(model_type=config.ml_model_type, random_state=config.seed, smoothing_alpha=0.35)
    model.fit(X_train, y_train)
    return model


def run_single_experiment(config: ExperimentConfig, save_outputs: bool = False) -> Dict[str, object]:
    """Run fixed, adaptive, and oracle EnKF on one synthetic twin experiment."""
    times, truth, obs = make_truth_and_obs(config)
    truth_obs = truth[obs.obs_indices]
    adaptive_model = train_adaptive_model(config)

    results: Dict[str, EnKFResult] = {}
    for method in ["fixed", "adaptive", "oracle"]:
        results[method] = run_enkf(
            observations=obs.y,
            true_states_at_obs=truth_obs,
            obs_times=obs.obs_times,
            state_indices=obs.state_indices,
            N=config.N,
            dt_model=config.dt_model,
            F=config.F,
            method=method,
            fixed_r=config.r_base,
            true_r_values=obs.true_r,
            adaptive_model=adaptive_model if method == "adaptive" else None,
            seed=config.seed + {"fixed": 11, "adaptive": 22, "oracle": 33}[method],
        )

    rows = [summarize_assimilation(result, obs.true_r, method, config.N, config.dt_obs, config.seed) for method, result in results.items()]
    summary = pd.DataFrame(rows)

    output = {"config": config, "times": times, "truth": truth, "obs": obs, "results": results, "summary": summary}
    if save_outputs:
        save_single_outputs(output)
    return output


def save_single_outputs(output: Dict[str, object], figures_dir: str | Path = "figures", results_dir: str | Path = "results") -> None:
    figures_dir = Path(figures_dir)
    results_dir = Path(results_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    config: ExperimentConfig = output["config"]
    obs: ObservationData = output["obs"]
    results: Dict[str, EnKFResult] = output["results"]
    summary: pd.DataFrame = output["summary"]
    tag = f"N{config.N}_dt{config.dt_obs:g}_seed{config.seed}_{config.r_mode}"

    summary.to_csv(results_dir / f"single_summary_{tag}.csv", index=False)
    # Also write the conventional latest summary for easy UI loading.
    summary.to_csv(results_dir / "latest_single_summary.csv", index=False)

    r_table = pd.DataFrame({"time": obs.obs_times, "true_r": obs.true_r, **{f"{m}_r": r.used_r_values for m, r in results.items()}})
    r_table.to_csv(results_dir / f"r_tracking_{tag}.csv", index=False)

    plot_rmse_comparison(results, figures_dir / f"rmse_comparison_{tag}.png")
    plot_r_tracking(obs.obs_times, obs.true_r, results, figures_dir / f"r_tracking_{tag}.png")
    plot_state_heatmap(output["times"], output["truth"], figures_dir / f"truth_heatmap_{tag}.png")


def run_grid_experiment(
    ensemble_sizes: Sequence[int] = (10, 20, 40),
    dt_obs_values: Sequence[float] = (0.05, 0.1, 0.2),
    T: float = 10.0,
    seed: int = 1,
    r_mode: str = "sinusoidal",
    ml_model_type: str = "random_forest",
    save_outputs: bool = True,
) -> pd.DataFrame:
    rows: List[dict] = []
    for N in ensemble_sizes:
        for dt_obs in dt_obs_values:
            cfg = ExperimentConfig(N=N, dt_obs=float(dt_obs), T=T, seed=seed, r_mode=r_mode, ml_model_type=ml_model_type)
            out = run_single_experiment(cfg, save_outputs=False)
            rows.extend(out["summary"].to_dict(orient="records"))
    summary = pd.DataFrame(rows)
    if save_outputs:
        Path("results").mkdir(exist_ok=True)
        Path("figures").mkdir(exist_ok=True)
        summary.to_csv("results/experiment_summary.csv", index=False)
        plot_grid_bar(summary, "figures/grid_mean_analysis_rmse.png")
    return summary
