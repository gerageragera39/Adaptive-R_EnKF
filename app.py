"""Streamlit UI for Project 7: EnKF with adaptive observation error."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.experiments import ExperimentConfig, run_grid_experiment, run_multi_seed_grid_experiment, run_single_experiment
from src.plotting import (
    plot_adaptive_improvement_aggregated,
    plot_grid_bar,
    plot_grid_bar_aggregated,
    plot_r_tracking,
    plot_rmse_comparison,
)


st.set_page_config(page_title="Lorenz'96 Adaptive-R EnKF", layout="wide")
st.title("Project 7: EnKF with Adaptive Observation Error (Lorenz'96)")


def parse_seed_text(seed_text: str) -> list[int]:
    """Parse comma/space separated seed text into unique integers, preserving order."""
    parts = seed_text.replace(",", " ").split()
    seeds: list[int] = []
    for part in parts:
        seed = int(part)
        if seed not in seeds:
            seeds.append(seed)
    if not seeds:
        raise ValueError("Please provide at least one seed.")
    return seeds


st.sidebar.header("Controls")
N = st.sidebar.selectbox("Ensemble size N", [10, 20, 40], index=1)
dt_obs = st.sidebar.selectbox("Observation frequency dt_obs", [0.05, 0.1, 0.2], index=1)
T = st.sidebar.slider("Total time T", min_value=2.0, max_value=20.0, value=8.0, step=1.0)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=100000, value=1, step=1)
use_multiple_seeds = st.sidebar.checkbox("Use multiple seeds", value=False)
seed_text = st.sidebar.text_input("Seeds for grid", value="1,2,3", help="Comma or space separated, e.g. 1,2,3 or 1 2 3")
r_mode = st.sidebar.selectbox("True R(t) mode", ["sinusoidal", "constant", "stepwise", "noisy"], index=0)
ml_model_type = st.sidebar.selectbox("ML model type", ["random_forest", "gradient_boosting"], index=0)
run_single = st.sidebar.button("Run single experiment", type="primary")
run_grid = st.sidebar.button("Run grid over N and dt_obs")

if "single_output" not in st.session_state:
    st.session_state.single_output = None
if "grid_summary" not in st.session_state:
    st.session_state.grid_summary = None
if "grid_raw_summary" not in st.session_state:
    st.session_state.grid_raw_summary = None
if "grid_source" not in st.session_state:
    st.session_state.grid_source = None

if run_single:
    cfg = ExperimentConfig(N=int(N), dt_obs=float(dt_obs), T=float(T), seed=int(seed), r_mode=r_mode, ml_model_type=ml_model_type)
    with st.spinner("Running truth simulation, ML training, and fixed/adaptive/oracle EnKFs..."):
        st.session_state.single_output = run_single_experiment(cfg, save_outputs=True)
    st.success("Single experiment complete. Outputs saved to results/ and figures/.")

if run_grid:
    try:
        seeds = parse_seed_text(seed_text) if use_multiple_seeds else [int(seed)]
    except ValueError as exc:
        st.error(f"Invalid seeds: {exc}")
        seeds = []
    if seeds:
        if use_multiple_seeds:
            with st.spinner("Running multi-seed grid (N={10,20,40}, dt_obs={0.05,0.1,0.2})..."):
                raw_summary, aggregated_summary = run_multi_seed_grid_experiment(
                    seeds=seeds,
                    T=float(T),
                    r_mode=r_mode,
                    ml_model_type=ml_model_type,
                    save_outputs=True,
                )
            st.session_state.grid_raw_summary = raw_summary
            st.session_state.grid_summary = aggregated_summary
            st.session_state.grid_source = f"current session, aggregated over seeds {seeds}"
            st.success("Multi-seed grid complete. Saved raw and aggregated CSVs plus aggregated figures.")
        else:
            with st.spinner("Running grid (N={10,20,40}, dt_obs={0.05,0.1,0.2})..."):
                st.session_state.grid_summary = run_grid_experiment(T=float(T), seed=seeds[0], r_mode=r_mode, ml_model_type=ml_model_type, save_outputs=True)
            st.session_state.grid_raw_summary = st.session_state.grid_summary
            st.session_state.grid_source = f"current session, single seed {seeds[0]}"
            st.success("Grid complete. Saved results/experiment_summary.csv.")

overview_tab, single_tab, rmse_tab, r_tab, grid_tab, exported_tab, docs_tab = st.tabs(
    [
        "Overview",
        "Single Run",
        "RMSE Comparison",
        "R Tracking",
        "Experiment Grid",
        "Exported Results",
        "Documentation",
    ]
)

with overview_tab:
    st.markdown(
        """
        This demo is a synthetic twin experiment for the 40-variable Lorenz'96 model.

        **Methods compared**
        - **fixed**: stochastic EnKF with scalar observation variance `r = 1.0`.
        - **adaptive**: offline ML model predicts `r_k` from innovations, ensemble spread, previous increment, and previous estimate.
        - **oracle**: EnKF uses the known synthetic `r(t)` as an upper-bound reference.

        The adaptive model is **not** given the true state or true `r(t)` during evaluation. True `r(t)` is used only as the supervised training target in separate synthetic runs.
        """
    )
    st.info("Use the sidebar to run a single experiment. For a presentation, T=6–10 is usually fast and illustrative.")

out = st.session_state.single_output
if out is not None:
    summary = out["summary"]
    obs = out["obs"]
    results = out["results"]

    with single_tab:
        st.subheader("Single-run metrics")
        st.dataframe(summary, width="stretch")
        best = summary.sort_values("mean_analysis_rmse").iloc[0]
        st.metric("Best mean analysis RMSE", f"{best.mean_analysis_rmse:.3f}", help=f"Method: {best.method}")
        st.caption("Adaptive-R may or may not improve RMSE; the table reports what happened for this seed/configuration.")

    with rmse_tab:
        st.subheader("Analysis RMSE comparison")
        st.pyplot(plot_rmse_comparison(results))
        method = st.selectbox("Show forecast/background vs analysis for", list(results.keys()))
        from src.plotting import plot_forecast_analysis_rmse

        st.pyplot(plot_forecast_analysis_rmse(results[method], method))

    with r_tab:
        st.subheader("R tracking")
        st.pyplot(plot_r_tracking(obs.obs_times, obs.true_r, results))
        r_df = pd.DataFrame({"time": obs.obs_times, "true_r": obs.true_r, **{f"{m}_used_r": r.used_r_values for m, r in results.items()}})
        st.dataframe(r_df, width="stretch")
else:
    with single_tab:
        st.warning("Run a single experiment from the sidebar first.")
    with rmse_tab:
        st.warning("Run a single experiment from the sidebar first.")
    with r_tab:
        st.warning("Run a single experiment from the sidebar first.")

with grid_tab:
    st.subheader("Experiment grid")
    aggregated_csv = Path("results/experiment_summary_aggregated.csv")
    raw_csv = Path("results/experiment_summary_raw.csv")
    legacy_csv = Path("results/experiment_summary.csv")

    grid_summary = st.session_state.grid_summary
    raw_summary = st.session_state.grid_raw_summary
    source = st.session_state.grid_source

    if grid_summary is None:
        if aggregated_csv.exists():
            grid_summary = pd.read_csv(aggregated_csv)
            source = f"loaded aggregated CSV: {aggregated_csv}"
            if raw_csv.exists():
                raw_summary = pd.read_csv(raw_csv)
        elif legacy_csv.exists():
            grid_summary = pd.read_csv(legacy_csv)
            raw_summary = grid_summary
            source = f"loaded single-seed/raw CSV: {legacy_csv}"

    if grid_summary is not None and not grid_summary.empty:
        st.caption(f"Result source: {source or 'current session'}")
        is_aggregated = "mean_analysis_rmse_mean" in grid_summary.columns
        if is_aggregated:
            view = st.radio("Grid table view", ["Aggregated over seeds", "Raw per-seed results"], horizontal=True)
            if view == "Aggregated over seeds":
                st.dataframe(grid_summary, width="stretch")
                st.pyplot(plot_grid_bar_aggregated(grid_summary))
                try:
                    st.pyplot(plot_adaptive_improvement_aggregated(grid_summary))
                except ValueError as exc:
                    st.info(str(exc))
            else:
                if raw_summary is None and raw_csv.exists():
                    raw_summary = pd.read_csv(raw_csv)
                if raw_summary is not None and not raw_summary.empty:
                    st.dataframe(raw_summary, width="stretch")
                    st.pyplot(plot_grid_bar(raw_summary))
                else:
                    st.info("Raw per-seed results are not available in this session or results/experiment_summary_raw.csv.")
        else:
            st.dataframe(grid_summary, width="stretch")
            st.pyplot(plot_grid_bar(grid_summary))
    else:
        st.info("Run a grid from the sidebar, or generate grid CSVs with run_experiments.py.")

with exported_tab:
    st.subheader("Exported results")
    result_files = sorted(Path("results").glob("*.csv"))
    if not result_files:
        st.info("No CSV files found yet in results/.")
    else:
        selected = st.selectbox("CSV file", result_files, format_func=lambda p: p.name)
        df = pd.read_csv(selected)
        st.dataframe(df, width="stretch")
        st.download_button("Download selected CSV", data=df.to_csv(index=False), file_name=selected.name, mime="text/csv")

with docs_tab:
    st.header("Full project documentation")
    st.markdown(
        """
        ## 1. Project goal

        This project demonstrates **data assimilation with an Ensemble Kalman Filter (EnKF)** on the
        chaotic 40-variable **Lorenz'96** model. The central question is:

        > What happens when the observation-error variance changes over time, and can an adaptive
        > machine-learning estimate of that variance help the EnKF?

        The app compares three methods:

        | Method | What it uses | Purpose |
        |---|---|---|
        | **fixed** | Constant `R = 1.0 I` | Baseline EnKF |
        | **adaptive** | ML-predicted scalar `r_k`, so `R_k = r_k I` | Practical adaptive method |
        | **oracle** | Synthetic true `r(t)` | Upper-bound reference only |

        This is a **synthetic twin experiment**: the code creates a true Lorenz'96 trajectory, generates
        noisy observations from it, and then tests whether the filters can recover the true state.

        ## 2. Scientific model

        The Lorenz'96 model is

        ```text
        dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F
        ```

        with cyclic indexing. In this project:

        - number of variables: `K = 40`
        - forcing: `F = 8.0`
        - model timestep: `dt = 0.05`
        - integration scheme: fourth-order Runge-Kutta (`RK4`)
        - main observation operator: full observation, `H = I`

        ## 3. Observation generation

        At observation times, synthetic observations are generated as

        ```text
        y_k = H x_true(t_k) + noise_k
        noise_k ~ N(0, r_k I)
        ```

        The true scalar variance `r_k` can follow several modes:

        - **sinusoidal**: smoothly varying error variance; best for demonstrating tracking.
        - **constant**: fixed true error variance.
        - **stepwise**: abrupt switches between lower and higher variance.
        - **noisy**: sinusoidal pattern plus random perturbations.

        ## 4. EnKF implementation

        The code uses a **stochastic EnKF with perturbed observations**. Each ensemble member is forecast
        through Lorenz'96, then updated using the ensemble-estimated covariance and the selected observation
        error covariance `R_k = r_k I`.

        Returned diagnostics include:

        - forecast/background mean state
        - analysis mean state
        - forecast RMSE
        - analysis RMSE
        - innovations
        - ensemble spread
        - used `r_k` values
        - analysis increment norm

        ## 5. Adaptive observation error

        The adaptive method trains an offline scikit-learn regression model on separate synthetic runs.
        The supervised target is the true synthetic `r_k`, but **only during training**.

        During evaluation, the adaptive model does **not** receive:

        - the true state
        - the true `r(t)`
        - oracle-only information

        It uses only diagnostics available during assimilation:

        - mean innovation squared
        - innovation variance
        - mean absolute innovation
        - mean ensemble spread in observation space
        - variance of ensemble spread in observation space
        - previous analysis increment norm
        - previous estimated `r`

        Predictions are clamped to a safe positive range `[0.05, 5.0]` and smoothed to avoid unstable jumps.

        ## 6. What each sidebar control does

        - **Ensemble size N**: number of ensemble members. Larger ensembles usually estimate covariances
          better but run slower. Options: `10`, `20`, `40`.
        - **Observation frequency dt_obs**: time between observations. Smaller values mean more frequent
          observations. Options: `0.05`, `0.1`, `0.2`.
        - **Total time T**: length of the experiment in model time units.
        - **Random seed**: controls reproducible observation noise, ensemble initialization, and ML training.
        - **True R(t) mode**: chooses how the synthetic observation-error variance changes over time.
        - **ML model type**: chooses the regression model for adaptive `r_k` prediction:
          `random_forest` or `gradient_boosting`.
        - **Run single experiment**: runs one selected configuration and saves single-run CSV/figure outputs.
        - **Use multiple seeds**: when enabled, the grid runs once for every seed in the seed list and aggregates results.
        - **Seeds for grid**: comma- or space-separated seed list, for example `1,2,3`. Used only when multiple seeds are enabled.
        - **Run grid over N and dt_obs**: runs all combinations of `N in {10,20,40}` and
          `dt_obs in {0.05,0.1,0.2}` with the current `T`, R mode, ML model, and selected seed behavior.

        ## 7. What each tab shows

        - **Overview**: short summary of the project and methods.
        - **Single Run**: table of metrics for fixed, adaptive, and oracle methods.
        - **RMSE Comparison**: plots analysis RMSE over time and forecast-vs-analysis RMSE for one method.
        - **R Tracking**: compares true `r(t)` with the values used by fixed, adaptive, and oracle filters.
        - **Experiment Grid**: loads or runs the grid experiment. It prefers aggregated multi-seed results when available, and can also show raw per-seed rows.
        - **Exported Results**: browses CSV files saved in `results/` and lets the user download them.
        - **Documentation**: this explanation page.

        ## 8. How to interpret results

        Lower **analysis RMSE** means the filter estimate is closer to the true Lorenz'96 state after
        assimilating observations. The **oracle** method is not realistic, but it is useful as a reference.

        Important scientific caution:

        - Adaptive `R` does **not** always improve RMSE.
        - If fixed `R=1.0` is close to the average true error variance, fixed-R can perform similarly.
        - Adaptive `R` may track the pattern of `r(t)` while producing only a small RMSE gain.
        - Results depend on ensemble size, observation frequency, random seed, and the true R scenario.
        - Multi-seed averaging is useful because one seed can make a method look unusually good or bad.

        ## 9. Files produced by the app/scripts

        Main source files:

        - `src/lorenz96.py`: Lorenz'96 derivative, RK4, trajectory simulation.
        - `src/observations.py`: synthetic observations and true `r(t)` modes.
        - `src/enkf.py`: stochastic EnKF implementation.
        - `src/adaptive_r.py`: ML features and adaptive-R regressor.
        - `src/metrics.py`: RMSE and summary metrics.
        - `src/plotting.py`: Matplotlib figures.
        - `src/experiments.py`: single and grid experiment orchestration.
        - `run_experiments.py`: command-line runner.
        - `app.py`: Streamlit UI.

        Output folders:

        - `results/`: CSV tables, including `experiment_summary.csv`, `experiment_summary_raw.csv`, and `experiment_summary_aggregated.csv`.
        - `figures/`: PNG plots for presentation/demo use.

        ## 10. Command-line usage

        Quick single experiment:

        ```bash
        python run_experiments.py --mode single --quick --N 10 --dt-obs 0.1 --seed 1
        ```

        Single-seed grid experiment:

        ```bash
        python run_experiments.py --mode grid --T 10 --seed 1
        ```

        Multi-seed grid experiment:

        ```bash
        python run_experiments.py --mode grid --T 20 --seeds 1 2 3 4 5
        ```

        Run tests:

        ```bash
        pytest -q
        ```

        Launch this UI:

        ```bash
        streamlit run app.py
        ```
        """
    )

