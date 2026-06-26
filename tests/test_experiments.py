from src.experiments import ExperimentConfig, run_single_experiment


def test_short_single_experiment_summary_methods():
    cfg = ExperimentConfig(N=10, dt_obs=0.1, T=0.5, seed=2, n_train_runs=2, T_train=0.5, spinup_time=0.2)
    out = run_single_experiment(cfg, save_outputs=False)
    summary = out["summary"]
    assert set(summary["method"]) == {"fixed", "adaptive", "oracle"}
    assert summary["mean_analysis_rmse"].notna().all()
