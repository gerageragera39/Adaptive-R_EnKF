# Presentation Notes

- This is a synthetic twin experiment: the Lorenz'96 nature run creates the “truth,” and noisy observations are generated from it.
- The EnKF uses 40 variables, RK4 integration with `dt=0.05`, and full observations `H=I` for a clear baseline demonstration.
- The baseline assumes fixed scalar observation variance `R = 1.0 I` even when the synthetic true observation error varies over time.
- The adaptive method predicts a scalar `r_k` from operational diagnostics such as innovations, ensemble spread, previous increment norm, and previous estimated `r`.
- The true `r(t)` is used only as a supervised target in separate training runs; it is not used as an evaluation feature.
- The oracle run uses true `r(t)` only as an upper-bound reference to show what would happen if the observation error were known.
- Compare RMSE and R-tracking separately: good tracking of `r(t)` does not always imply a large RMSE improvement.
- If fixed-R performs similarly, explain that `R=1.0` is near the mean of the sinusoidal true error in the default setup.
