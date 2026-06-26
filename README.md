# Project 7: EnKF with Adaptive Observation Error (Lorenz'96)

This project implements a reproducible synthetic twin experiment for a university Data Assimilation course. It compares three stochastic Ensemble Kalman Filter (EnKF) variants on the 40-variable Lorenz'96 model:

- **fixed R**: scalar observation-error variance `r = 1.0`, so `R = r I`.
- **adaptive R**: an offline machine-learning model predicts `r_k` from assimilation diagnostics.
- **oracle R**: uses the synthetic true `r(t)` as an upper-bound reference, not as a realistic operational method.

The observation-error covariance is intentionally implemented as a scalar `R_k = r_k I` for clarity and presentation value.

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick sanity experiment

```bash
python run_experiments.py --mode single --quick --N 10 --dt-obs 0.1 --seed 1
```

This writes CSV files under `results/` and PNG figures under `figures/`.

## Grid experiments

Single-seed grid:

```bash
python run_experiments.py --mode grid --T 10 --seed 1
```

Multi-seed grid:

```bash
python run_experiments.py --mode grid --T 20 --seeds 1 2 3 4 5
```

The grid varies:

- ensemble size `N in {10, 20, 40}`
- observation frequency `dt_obs in {0.05, 0.1, 0.2}`

Project 7 requires varying `N` and `dt_obs`. Multi-seed averaging is an extra robustness check: because stochastic EnKF updates, observation noise, initial ensembles, and ML training all depend on random seeds, one seed can make a method look unusually good or bad. Averaging across seeds gives a more honest comparison.

Single-seed output: `results/experiment_summary.csv` and `figures/grid_mean_analysis_rmse.png`.

Multi-seed output:

- `results/experiment_summary_raw.csv`: one row per seed / N / dt_obs / method
- `results/experiment_summary_aggregated.csv`: grouped by N, dt_obs, and method with mean/std metrics
- `figures/grid_mean_analysis_rmse_aggregated.png`: mean analysis RMSE with error bars over seeds
- `figures/grid_adaptive_improvement_aggregated.png`: adaptive-vs-fixed analysis RMSE improvement


## Streamlit UI

```bash
streamlit run app.py
```

The UI provides sidebar controls for ensemble size, observation interval, total time, random seed, optional multi-seed grid input, true `R(t)` mode, and ML model type. Tabs show an overview, single-run metrics, RMSE plots, R tracking, grid results, exported CSVs, and in-app documentation. The grid tab prefers aggregated multi-seed CSVs when they exist and can also show raw per-seed rows.

## Scientific setup

- Model: Lorenz'96 with `K=40`, forcing `F=8.0`.
- Integrator: fourth-order Runge-Kutta (RK4), model timestep `dt=0.05`.
- Observations: full-state operator `H=I` by default.
- Synthetic observations: `y_k = H x_true(t_k) + epsilon_k`, `epsilon_k ~ N(0, r_k I)`.
- True `r(t)` modes: constant, sinusoidal, stepwise, or noisy.

## Adaptive-R model

Training data are generated from separate synthetic runs. The supervised target is the synthetic true `r_k`, which is allowed in the twin-experiment training setup. During evaluation, the model does **not** receive the true state or true `r(t)`.

Features used during evaluation are available from the assimilation cycle:

- mean innovation squared
- innovation variance
- innovation absolute mean
- mean ensemble spread in observation space
- variance of ensemble spread in observation space
- previous analysis-increment norm
- previous estimated `r`

Predictions are clamped to `[0.05, 5.0]` and smoothed with an exponential moving average.

## Metrics

Each raw experiment row reports:

- mean forecast/background RMSE
- mean analysis RMSE
- median analysis RMSE
- `r` estimation MAE
- correlation between estimated/used `r_k` and true `r(t)`
- innovation mean and variance
- analysis improvement over forecast

For multi-seed grids, the aggregated CSV groups by `N`, `dt_obs`, and `method`, then reports mean and standard deviation over seeds for mean analysis RMSE, mean forecast RMSE, median analysis RMSE, R MAE, R correlation, and analysis improvement. It also includes `n_seeds`.

## Important caution

Adaptive `R` is not guaranteed to improve RMSE. In some seeds/configurations, fixed `R=1.0` can perform similarly if it is close to the average true observation error. If adaptive `R` tracks the true sinusoidal pattern but RMSE improvement is small, report that honestly. Multi-seed averages should also be interpreted honestly: positive adaptive-vs-fixed improvement means adaptive was better on average for that setting, while negative values mean fixed was better on average.

## Tests

```bash
pytest -q
```

The tests cover Lorenz'96 shapes, observation shapes and positivity, EnKF output shapes, positive `R`, non-NaN RMSE, and a short end-to-end experiment.
