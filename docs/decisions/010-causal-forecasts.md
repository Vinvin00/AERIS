# ADR 010 — Independent per-axis Kalman forecasters with continuous-time noise

Status: proposed for AERIS-010 review. Date: 2026-10-06.

Forecast models are self-contained Kalman filters over measured centers, not read-outs of
the tracker state, so a model can be compared on identical inputs and its causality is
enforced by the API (monotone updates, no issue before a received measurement). Process noise
is the continuous white-noise form integrated over the real delta time (verified by numerical
integration), unlike the tracker's discrete form, because forecast horizons span 0.1-2 s and
measurement spacing is irregular. Acceleration is a separate three-state filter with a jerk
noise model, as CONTRACTS.md requires. Unavailable forecasts are emitted with a reason so
misses stay visible to the AERIS-005 evaluator. Interval coverage is empirical and conditional;
no calibration claim is made, and axis independence (diagonal covariance) is accepted for now.
Alternatives not taken: reusing tracker covariance (couples evaluation to tracker tuning) and
a two-frame finite-difference acceleration (explicitly rejected by the contract).
