# Project: Adaptive Observation Error Estimation for EnKF on Lorenz'96

This repository implements **Project: EnKF with Adaptive Observation Error (Lorenz'96)** for a university Data Assimilation course.

The project studies whether an Ensemble Kalman Filter (EnKF) can benefit from estimating the observation-error variance dynamically instead of using a fixed observation-error covariance. The experiments are performed in a reproducible synthetic **twin-experiment** setup using the 40-dimensional Lorenz'96 model.

## Main idea

In a twin experiment, the true state is generated artificially by the model itself. Synthetic observations are then created by adding controlled Gaussian noise to the true state. Since the true state is known, the quality of the data assimilation algorithm can be evaluated directly using RMSE.

This project compares three EnKF variants:

* **fixed R**: uses a constant scalar observation-error variance `r = 1.0`, so `R = r I`.
* **adaptive R**: predicts a time-dependent scalar observation-error variance `r_k` from assimilation diagnostics using an offline ML model.
* **oracle R**: uses the synthetic true `r(t)` and serves only as an upper-bound reference, not as a realistic operational method.

For clarity and presentation purposes, the observation-error covariance is implemented as a scalar matrix:

```text
R_k = r_k I
```

## Repository structure

```text
Adaptive-R_EnKF/
  src/
    adaptive_r.py       # ML model for adaptive R estimation
    enkf.py             # stochastic Ensemble Kalman Filter
    experiments.py      # single-run, grid, and multi-seed experiments
    lorenz96.py         # Lorenz'96 model and RK4 integration
    metrics.py          # RMSE and diagnostic metrics
    observations.py     # synthetic observations and true r(t)
    plotting.py         # figures and visualization
  tests/                # unit and sanity tests
  figures/              # generated plots
  results/              # generated CSV result tables
  app.py                # Streamlit UI
  run_experiments.py    # command-line experiment runner
  presentation_notes.md # short notes for presentation
  requirements.txt
  README.md
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick sanity check

Run the tests:

```bash
pytest -q
```

Run a small quick experiment:

```bash
python run_experiments.py --mode single --quick --N 10 --dt-obs 0.1 --seed 1
```

This checks that the simulation, observation generation, EnKF, adaptive-R model, metrics, and plotting pipeline work correctly.

## Main single experiment

A representative single run can be started with:

```bash
python run_experiments.py --mode single --N 40 --dt-obs 0.1 --T 20 --seed 1
```

This generates time-dependent plots such as:

* Lorenz'96 true-state heatmap
* EnKF analysis RMSE comparison
* true `r(t)` vs. fixed/adaptive/oracle used `r`

The single run is useful for explaining how the method behaves over time.

Example outputs:

```text
figures/truth_heatmap_N40_dt0.1_seed1_sinusoidal.png
figures/rmse_comparison_N40_dt0.1_seed1_sinusoidal.png
figures/r_tracking_N40_dt0.1_seed1_sinusoidal.png
```

## Full multi-seed grid experiment

The main experiment should be run across several random seeds:

```bash
python run_experiments.py --mode grid --T 20 --seeds 1 2 3 4 5
```

The grid varies:

```text
N      ∈ {10, 20, 40}
dt_obs ∈ {0.05, 0.1, 0.2}
```

For each combination, the experiment compares:

```text
fixed R
adaptive R
oracle R
```

The multi-seed setup is used because a single stochastic EnKF run can be strongly affected by random initialization and observation noise. Averaging over several seeds gives a more reliable comparison.

Main outputs:

```text
results/experiment_summary_raw.csv
results/experiment_summary_aggregated.csv
figures/grid_mean_analysis_rmse_aggregated.png
figures/grid_adaptive_improvement_aggregated.png
```

## Streamlit UI

The project also includes a Streamlit interface:

```bash
streamlit run app.py
```

The UI is mainly intended for inspection and presentation. It allows interactive single runs, grid runs, and visualization of saved results.

The command-line runner is recommended for final reproducible experiments, while the UI is useful for demonstrating the project and quickly checking plots.

## Scientific setup

* Model: Lorenz'96 with `K = 40` variables.
* Forcing: `F = 8.0`.
* Integrator: fourth-order Runge-Kutta.
* Model timestep: `dt = 0.05`.
* Observation operator: full-state observation, `H = I`.
* Synthetic observations:

```text
y_k = H x_true(t_k) + epsilon_k
epsilon_k ~ N(0, r_k I)
```

The true observation-error variance `r(t)` can be generated using different modes:

* constant
* sinusoidal
* stepwise
* noisy

The sinusoidal mode is the main scenario used to test adaptive observation-error estimation.

## Adaptive-R model

The adaptive method trains an offline machine-learning model on separate synthetic training runs. The supervised target is the synthetic true `r_k`, which is allowed in the twin-experiment setup.

During evaluation, the adaptive model does **not** receive the true state or the true `r(t)` as input.

Features are based only on quantities available during assimilation, including:

* mean squared innovation
* innovation variance
* mean absolute innovation
* mean ensemble spread in observation space
* variance of ensemble spread in observation space
* previous analysis-increment norm
* previous estimated `r`

Predictions are clamped to a safe positive range and optionally smoothed to avoid unstable jumps.

## Metrics

Each experiment reports:

* mean forecast/background RMSE
* mean analysis RMSE
* median analysis RMSE
* `r` estimation MAE
* correlation between estimated/used `r_k` and true `r(t)`
* innovation mean
* innovation variance
* analysis improvement over forecast

For multi-seed experiments, the aggregated table reports mean and standard deviation across seeds.

## Results summary

In the multi-seed grid experiment, adaptive `R` reduced the mean analysis RMSE compared to fixed `R` in most tested configurations.

The improvement was most visible for larger ensembles, especially `N = 40`. This is expected because a larger ensemble provides more reliable spread and innovation diagnostics for adaptive observation-error estimation.

However, the improvement is not uniform across all configurations. The standard deviations show noticeable seed-to-seed variability, and adaptive `R` is not guaranteed to improve every individual run.

A suitable final interpretation is:

> Adaptive observation-error estimation can improve EnKF performance when the true observation error changes over time, especially when the ensemble is large enough to provide reliable diagnostics. However, the benefit depends on ensemble size, observation frequency, and stochastic variability.

## Important limitations

This project is a simplified research-style implementation.

Important simplifications:

* `R_k` is represented as a scalar covariance `r_k I`, not a full covariance matrix.
* Observations are synthetic and generated from the known true state.
* The adaptive model is trained in a twin-experiment setting where the true `r_k` is known during training.
* The oracle method is not realistic; it is included only as an upper-bound reference.
* Adaptive `R` does not always improve RMSE, especially when fixed `R = 1.0` is already close to the average true observation error.

## Recommended workflow

For checking the project:

```bash
pytest -q
python run_experiments.py --mode single --quick --N 10 --dt-obs 0.1 --seed 1
```

For generating representative plots:

```bash
python run_experiments.py --mode single --N 40 --dt-obs 0.1 --T 20 --seed 1
```

For final results:

```bash
python run_experiments.py --mode grid --T 20 --seeds 1 2 3 4 5
```

For interactive visualization:

```bash
streamlit run app.py
```

## Presentation notes

The project can be explained using the following structure:

1. Lorenz'96 generates the synthetic true state.
2. Observations are created by adding Gaussian noise with time-varying variance `r(t)`.
3. Standard EnKF assumes a fixed observation-error covariance.
4. Adaptive EnKF estimates `r_k` from innovations and ensemble diagnostics.
5. Oracle EnKF uses the true `r(t)` only as a reference.
6. RMSE and `R`-tracking plots compare the methods.
7. Multi-seed grid results show that adaptive `R` improves mean RMSE in most configurations, especially for larger ensembles.
8. The result is useful but not universal: adaptive `R` helps on average, but not in every single stochastic run.
