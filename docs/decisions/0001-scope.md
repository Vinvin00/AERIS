# ADR 0001 — Baseline scope

Status: accepted for scaffold. Date: 2026-09-24.

Use recorded RGB video and fixed cameras for the first release. Preserve camera-relative
pixel motion and optional calibrated rays. Separate experimental metric ranging and
NVIDIA optimization from release requirements. Build evaluators and regression fixtures
early. Adopt one active writer per checkout and independent review per task.

Reason: data quality, observability, and reproducibility constrain valid claims more
than model choice. This keeps a usable CPU pipeline possible on the current Mac.

Consequence: C++ dependencies and ML versions are selected in explicit later tasks.
The scaffold checks are not evidence of real aircraft detection accuracy.
