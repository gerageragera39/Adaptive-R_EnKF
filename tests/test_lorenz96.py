import numpy as np

from src.lorenz96 import lorenz96_derivative, rk4_step, simulate_trajectory


def test_lorenz96_output_shape_single_and_ensemble():
    x = np.ones(40)
    assert lorenz96_derivative(x).shape == (40,)
    ens = np.ones((10, 40))
    assert lorenz96_derivative(ens).shape == (10, 40)


def test_rk4_and_simulate_shapes():
    x = np.ones(40)
    x_next = rk4_step(x, dt=0.05)
    assert x_next.shape == (40,)
    times, states = simulate_trajectory(T=0.2, dt=0.05, K=40, spinup_time=0.1)
    assert times.shape == (5,)
    assert states.shape == (5, 40)
    assert np.all(np.isfinite(states))
