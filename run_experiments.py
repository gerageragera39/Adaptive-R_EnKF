"""Command-line runner for Project 7 experiments."""
from __future__ import annotations

import argparse

from src.experiments import ExperimentConfig, run_grid_experiment, run_single_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Run Lorenz'96 EnKF adaptive-R experiments")
    parser.add_argument("--mode", choices=["single", "grid"], default="single")
    parser.add_argument("--N", type=int, default=20, help="Ensemble size for single mode")
    parser.add_argument("--dt-obs", type=float, default=0.1, help="Observation frequency for single mode")
    parser.add_argument("--T", type=float, default=10.0, help="Experiment length")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--r-mode", choices=["constant", "sinusoidal", "stepwise", "noisy"], default="sinusoidal")
    parser.add_argument("--ml-model", choices=["random_forest", "gradient_boosting"], default="random_forest")
    parser.add_argument("--quick", action="store_true", help="Use shorter settings for a fast smoke run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    T = 4.0 if args.quick else args.T
    if args.mode == "single":
        cfg = ExperimentConfig(N=args.N, dt_obs=args.dt_obs, T=T, seed=args.seed, r_mode=args.r_mode, ml_model_type=args.ml_model)
        out = run_single_experiment(cfg, save_outputs=True)
        print(out["summary"].to_string(index=False))
        print("Saved single-run CSVs in results/ and figures in figures/.")
    else:
        sizes = (10, 20) if args.quick else (10, 20, 40)
        dts = (0.1, 0.2) if args.quick else (0.05, 0.1, 0.2)
        summary = run_grid_experiment(sizes, dts, T=T, seed=args.seed, r_mode=args.r_mode, ml_model_type=args.ml_model, save_outputs=True)
        print(summary.to_string(index=False))
        print("Saved results/experiment_summary.csv and figures/grid_mean_analysis_rmse.png")


if __name__ == "__main__":
    main()
