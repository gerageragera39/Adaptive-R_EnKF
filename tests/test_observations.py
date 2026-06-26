import numpy as np

from src.lorenz96 import simulate_trajectory
from src.observations import generate_observations, true_r_values


def test_observation_generation_shapes_and_positive_r():
    times, states = simulate_trajectory(T=0.4, dt=0.05, spinup_time=0.1)
    obs = generate_observations(times, states, dt_obs=0.1, r_mode="sinusoidal", seed=3)
    assert obs.y.shape == (5, 40)
    assert obs.true_obs.shape == (5, 40)
    assert obs.true_r.shape == (5,)
    assert np.all(obs.true_r > 0)


def test_true_r_values_positive_for_all_modes():
    t = np.linspace(0, 2, 20)
    for mode in ["constant", "sinusoidal", "stepwise", "noisy"]:
        r = true_r_values(t, mode=mode, seed=1)
        assert np.all(r > 0)
