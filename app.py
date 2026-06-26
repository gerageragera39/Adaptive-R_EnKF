"""Streamlit UI for Project 7: EnKF with adaptive observation error."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.experiments import ExperimentConfig, run_grid_experiment, run_single_experiment
from src.plotting import plot_grid_bar, plot_r_tracking, plot_rmse_comparison


st.set_page_config(page_title="Lorenz'96 Adaptive-R EnKF", layout="wide")
st.title("Project 7: EnKF with Adaptive Observation Error (Lorenz'96)")

st.sidebar.header("Controls")
N = st.sidebar.selectbox("Ensemble size N", [10, 20, 40], index=1)
dt_obs = st.sidebar.selectbox("Observation frequency dt_obs", [0.05, 0.1, 0.2], index=1)
T = st.sidebar.slider("Total time T", min_value=2.0, max_value=20.0, value=8.0, step=1.0)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=100000, value=1, step=1)
r_mode = st.sidebar.selectbox("True R(t) mode", ["sinusoidal", "constant", "stepwise", "noisy"], index=0)
ml_model_type = st.sidebar.selectbox("ML model type", ["random_forest", "gradient_boosting"], index=0)
run_single = st.sidebar.button("Run single experiment", type="primary")
run_grid = st.sidebar.button("Run quick grid")

if "single_output" not in st.session_state:
    st.session_state.single_output = None
if "grid_summary" not in st.session_state:
    st.session_state.grid_summary = None

if run_single:
    cfg = ExperimentConfig(N=int(N), dt_obs=float(dt_obs), T=float(T), seed=int(seed), r_mode=r_mode, ml_model_type=ml_model_type)
    with st.spinner("Running truth simulation, ML training, and fixed/adaptive/oracle EnKFs..."):
        st.session_state.single_output = run_single_experiment(cfg, save_outputs=True)
    st.success("Single experiment complete. Outputs saved to results/ and figures/.")

if run_grid:
    with st.spinner("Running quick grid (N={10,20,40}, dt_obs={0.05,0.1,0.2})..."):
        st.session_state.grid_summary = run_grid_experiment(T=float(T), seed=int(seed), r_mode=r_mode, ml_model_type=ml_model_type, save_outputs=True)
    st.success("Grid complete. Saved results/experiment_summary.csv.")

overview_tab, single_tab, rmse_tab, r_tab, grid_tab, exported_tab = st.tabs(
    ["Overview", "Single Run", "RMSE Comparison", "R Tracking", "Experiment Grid", "Exported Results"]
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
        st.dataframe(summary, use_container_width=True)
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
        st.dataframe(r_df, use_container_width=True)
else:
    with single_tab:
        st.warning("Run a single experiment from the sidebar first.")
    with rmse_tab:
        st.warning("Run a single experiment from the sidebar first.")
    with r_tab:
        st.warning("Run a single experiment from the sidebar first.")

with grid_tab:
    st.subheader("Experiment grid")
    grid_summary = st.session_state.grid_summary
    csv_path = Path("results/experiment_summary.csv")
    if grid_summary is None and csv_path.exists():
        grid_summary = pd.read_csv(csv_path)
    if grid_summary is not None and not grid_summary.empty:
        st.dataframe(grid_summary, use_container_width=True)
        st.pyplot(plot_grid_bar(grid_summary))
    else:
        st.info("Run a quick grid from the sidebar, or generate results/experiment_summary.csv with run_experiments.py.")

with exported_tab:
    st.subheader("Exported results")
    result_files = sorted(Path("results").glob("*.csv"))
    if not result_files:
        st.info("No CSV files found yet in results/.")
    else:
        selected = st.selectbox("CSV file", result_files, format_func=lambda p: p.name)
        df = pd.read_csv(selected)
        st.dataframe(df, use_container_width=True)
        st.download_button("Download selected CSV", data=df.to_csv(index=False), file_name=selected.name, mime="text/csv")
