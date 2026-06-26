import numpy as np

from src.adaptive_r import AdaptiveRModel
from src.enkf import collect_training_features, run_enkf
from src.lorenz96 import simulate_trajectory
from src.observations import generate_observations


def small_case():
    times, truth = simulate_trajectory(T=0.5, dt=0.05, spinup_time=0.2)
    obs = generate_observations(times, truth, dt_obs=0.1, r_mode="sinusoidal", seed=4)
    return truth, obs


def test_enkf_output_shapes_and_no_nan_rmse():
    truth, obs = small_case()
    result = run_enkf(
        observations=obs.y,
        true_states_at_obs=truth[obs.obs_indices],
        obs_times=obs.obs_times,
        state_indices=obs.state_indices,
        N=10,
        method="fixed",
        fixed_r=1.0,
        seed=5,
    )
    n_obs = len(obs.obs_times)
    assert result.forecast_mean.shape == (n_obs, 40)
    assert result.analysis_mean.shape == (n_obs, 40)
    assert result.innovations.shape == (n_obs, 40)
    assert result.used_r_values.shape == (n_obs,)
    assert np.all(result.used_r_values > 0)
    assert not np.isnan(result.analysis_rmse).any()


def test_collect_training_features_and_adaptive_positive_predictions():
    truth, obs = small_case()
    X, y = collect_training_features(
        observations=obs.y,
        true_states_at_obs=truth[obs.obs_indices],
        obs_times=obs.obs_times,
        true_r_values=obs.true_r,
        state_indices=obs.state_indices,
        N=10,
        seed=6,
    )
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == 7
    model = AdaptiveRModel(model_type="random_forest", random_state=7).fit(X, y)
    result = run_enkf(
        observations=obs.y,
        true_states_at_obs=truth[obs.obs_indices],
        obs_times=obs.obs_times,
        state_indices=obs.state_indices,
        N=10,
        method="adaptive",
        adaptive_model=model,
        true_r_values=obs.true_r,
        seed=8,
    )
    assert np.all(result.used_r_values > 0)
    assert not np.isnan(result.analysis_rmse).any()
