from src.experiments import ExperimentConfig, run_single_experiment


def test_short_single_experiment_summary_methods():
    cfg = ExperimentConfig(N=10, dt_obs=0.1, T=0.5, seed=2, n_train_runs=2, T_train=0.5, spinup_time=0.2)
    out = run_single_experiment(cfg, save_outputs=False)
    summary = out["summary"]
    assert set(summary["method"]) == {"fixed", "adaptive", "oracle"}
    assert summary["mean_analysis_rmse"].notna().all()


def test_multi_seed_grid_aggregation_tiny_case():
    from src.experiments import run_multi_seed_grid_experiment

    raw, aggregated = run_multi_seed_grid_experiment(
        seeds=[1, 2],
        ensemble_sizes=(10,),
        dt_obs_values=(0.1,),
        T=0.5,
        save_outputs=False,
        n_train_runs=2,
        T_train=0.5,
        spinup_time=0.2,
    )

    assert len(raw) == 2 * 1 * 1 * 3  # seeds * N values * dt_obs values * methods
    assert set(raw["method"]) == {"fixed", "adaptive", "oracle"}
    assert len(aggregated) == 3
    assert set(aggregated["method"]) == {"fixed", "adaptive", "oracle"}
    assert (aggregated["n_seeds"] == 2).all()
    for column in [
        "mean_analysis_rmse_mean",
        "mean_analysis_rmse_std",
        "mean_forecast_rmse_mean",
        "r_mae_mean",
        "analysis_improvement_mean",
    ]:
        assert column in aggregated.columns
        assert aggregated[column].notna().all()
