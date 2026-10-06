# Publication evidence and contribution record

Codex performed the October 6, 2026 publication maintenance: moved the C++
foundation CI job to macOS, corrected the README's implementation status,
added the reproducible ONNX smoke command, and coordinated independent review.
Earlier history contains substantial AI-authored implementation and review,
including Claude co-author trailers. Those trailers are preserved. Git's author
field alone does not establish who wrote or understands the implementation.
The owner should describe their own design decisions, debugging and validation
experience when presenting the project; AI output is not evidence of that experience.

At inspection, `main` was ten commits ahead of its locally recorded `origin/main`.
There was no README-specific task or failed README review in this checkout's
registry or saved runs. MAINT-001 covers the current correction; it does not claim
to close a missing task or reset another task's review budget.

## Run the actual model locally

Use the existing environment and hash-pinned weights described in
[the detector guide](detection.md) and [weight retrieval instructions](../models/README.md).
The local FGVC sample is described in [data QA](data_qa.md). This command downloads
nothing and refuses to overwrite a previous run:

```sh
PYTHONPATH=python .venv/bin/python scripts/smoke_onnx.py \
  --output-dir benchmarks/runs/onnx-smoke-2026-10-06-03
```

The output directory contains `input.mp4`, `overlay.mp4` and `report.json`.
Videos and photographs remain Git-ignored because the sample's terms do not
permit redistribution. Only the measured JSON record is published as
`benchmarks/onnx_smoke.json`. Its source fingerprint captures the code at execution;
its Git revision names the parent commit because execution preceded the maintenance
commit. Hardware and timings describe this local run only.

The model is YOLOX-Tiny trained upstream on COCO, using ONNX Runtime's CPU provider.
The input is six crops from one still photograph, encoded at 10 fps. Detection
records use original decoded-frame pixels. This is a smoke demonstration of
inference and rendering, with no tracking, forecasting, or real-video accuracy claim.
Weight license terms remain unverified; no project distribution license is selected.

## CI and task evidence

The foundation workflow runs Python 3.11 and 3.12 on Ubuntu and C++20 contract tests
on macOS. C++ is still a foundation test, not a port of the detector or tracker.
Real-model tests skip in CI when the external weights and sample are absent.
Local task checks and the independent review are stored under the Git-ignored
`orchestration/runs/MAINT-001/` directory. A local pass does not prove a GitHub Actions
pass; inspect the workflow run for the published commit before claiming green CI.
