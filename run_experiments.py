"""Command-line runner for Project 7 experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.experiments import ExperimentConfig, run_grid_experiment, run_multi_seed_grid_experiment, run_single_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Run Lorenz'96 EnKF adaptive-R experiments")
    parser.add_argument("--mode", choices=["single", "grid"], default="single")
    parser.add_argument("--N", type=int, default=20, help="Ensemble size for single mode")
    parser.add_argument("--dt-obs", type=float, default=0.1, help="Observation frequency for single mode")
    parser.add_argument("--T", type=float, default=10.0, help="Experiment length")
    parser.add_argument("--seed", type=int, default=1, help="Single random seed; used when --seeds is omitted")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="One or more random seeds, e.g. --seeds 1 2 3 4 5. Overrides --seed when provided.",
    )
    parser.add_argument("--r-mode", choices=["constant", "sinusoidal", "stepwise", "noisy"], default="sinusoidal")
    parser.add_argument("--ml-model", choices=["random_forest", "gradient_boosting"], default="random_forest")
    parser.add_argument("--quick", action="store_true", help="Use shorter settings for a fast smoke run")
    return parser.parse_args()


def _selected_seeds(args) -> list[int]:
    return [int(s) for s in (args.seeds if args.seeds is not None else [args.seed])]


def main() -> None:
    args = parse_args()
    seeds = _selected_seeds(args)
    T = 4.0 if args.quick else args.T
    sizes = (10, 20) if args.quick else (10, 20, 40)
    dts = (0.1, 0.2) if args.quick else (0.05, 0.1, 0.2)

    if args.mode == "single":
        rows = []
        for seed in seeds:
            cfg = ExperimentConfig(N=args.N, dt_obs=args.dt_obs, T=T, seed=seed, r_mode=args.r_mode, ml_model_type=args.ml_model)
            out = run_single_experiment(cfg, save_outputs=True)
            rows.extend(out["summary"].to_dict(orient="records"))
        summary = pd.DataFrame(rows)
        print(summary.to_string(index=False))
        if len(seeds) > 1:
            Path("results").mkdir(exist_ok=True)
            summary.to_csv("results/single_summary_multi_seed.csv", index=False)
            print("Saved results/single_summary_multi_seed.csv plus per-seed single-run CSVs/figures.")
        else:
            print("Saved single-run CSVs in results/ and figures in figures/.")
    else:
        if len(seeds) == 1:
            summary = run_grid_experiment(sizes, dts, T=T, seed=seeds[0], r_mode=args.r_mode, ml_model_type=args.ml_model, save_outputs=True)
            print(summary.to_string(index=False))
            print("Saved results/experiment_summary.csv and figures/grid_mean_analysis_rmse.png")
        else:
            # In quick mode, keep training short too so CLI smoke checks remain fast.
            n_train_runs = 2 if args.quick else 4
            T_train = max(T, 0.5) if args.quick else 10.0
            raw, aggregated = run_multi_seed_grid_experiment(
                seeds=seeds,
                ensemble_sizes=sizes,
                dt_obs_values=dts,
                T=T,
                r_mode=args.r_mode,
                ml_model_type=args.ml_model,
                save_outputs=True,
                n_train_runs=n_train_runs,
                T_train=T_train,
                spinup_time=0.2 if args.quick else 5.0,
            )
            print("Raw per-seed results:")
            print(raw.to_string(index=False))
            print("\nAggregated over seeds:")
            print(aggregated.to_string(index=False))
            print(
                "Saved results/experiment_summary_raw.csv, results/experiment_summary_aggregated.csv, "
                "figures/grid_mean_analysis_rmse_aggregated.png, and figures/grid_adaptive_improvement_aggregated.png"
            )


if __name__ == "__main__":
    main()
