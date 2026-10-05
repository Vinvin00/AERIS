# ADR 009 — Constant-velocity Kalman tracker, Mahalanobis-gated Hungarian association

Status: proposed for AERIS-009 review. Date: 2026-10-02.

A constant-velocity Kalman filter over `[cx, cy, vx, vy, w, h]` with
delta-time-dependent discrete white-noise-acceleration Q, Joseph updates and explicit
PSD checks, as required by CONTRACTS.md. Association is optimal (SciPy Hungarian) on
squared Mahalanobis distance, not IoU, so tiny, fast aircraft whose boxes do not
overlap between frames can still match; it is gated by a chi-square quantile and is class
aware. All timeouts are in seconds so behaviour is frame-rate independent.

Rejected for now: IoU-only association (fails on tiny/fast targets), acceleration
and turn models (CONTRACTS.md makes acceleration a separate model), appearance
re-identification, and mapping detector confidence to R (no validated method). SciPy is
added as an extra pin only for the assignment solver.

Parameters are untuned engineering defaults. AC3's real-video evaluation is blocked by the
absence of identity-labeled real footage; synthetic scenes only exercise the evaluator
path, and the evaluator returns unavailable metrics when identities are absent.
