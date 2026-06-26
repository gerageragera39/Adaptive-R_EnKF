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

## Full grid experiment

```bash
python run_experiments.py --mode grid --T 10 --seed 1
```

The grid varies:

- ensemble size `N in {10, 20, 40}`
- observation frequency `dt_obs in {0.05, 0.1, 0.2}`

Main output: `results/experiment_summary.csv` and `figures/grid_mean_analysis_rmse.png`.

## Streamlit UI

```bash
streamlit run app.py
```

The UI provides sidebar controls for ensemble size, observation interval, total time, random seed, true `R(t)` mode, and ML model type. Tabs show an overview, single-run metrics, RMSE plots, R tracking, grid results, and exported CSVs.

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

Each experiment reports:

- mean forecast/background RMSE
- mean analysis RMSE
- median analysis RMSE
- `r` estimation MAE
- correlation between estimated/used `r_k` and true `r(t)`
- innovation mean and variance
- analysis improvement over forecast

## Important caution

Adaptive `R` is not guaranteed to improve RMSE. In some seeds/configurations, fixed `R=1.0` can perform similarly if it is close to the average true observation error. If adaptive `R` tracks the true sinusoidal pattern but RMSE improvement is small, report that honestly.

## Tests

```bash
pytest -q
```

The tests cover Lorenz'96 shapes, observation shapes and positivity, EnKF output shapes, positive `R`, non-NaN RMSE, and a short end-to-end experiment.
