# Revised roadmap

Tasks and exact dependencies are in `orchestration/tasks.json`. A milestone is
complete only after its tasks have evidence and independent review.

Milestone membership: G1 = 001–003; G2 = 004–007 plus deferred 022 (real measured baseline); G3 = 008–009 plus deferred 021 (real-video MOT evidence);
G4 = 010–011; G5 = 012, 013, 015, 017; G6 = 014, 016; G7 = 018.
All IDs use the `AERIS-` prefix. Requests through a gate include all earlier gates.
IDs 019 and 020 are optional extensions and are not included in G7. Unlike them, deferred
021 and 022 gate G5-G7 (through 015, 017, 016 and 018), so those stay blocked until real tracking and baseline evidence exist.

| Gate | Work | Exit evidence | Approximate focused effort |
|---|---|---|---|
| G0 | Foundation and orchestration (this delivery) | Offline checks, C++ contract check, tested review loop | Delivered scaffold |
| G1 | Contracts, data audit, adapters, QA | Licensed/provenanced manifest, split report, validator tests, real visual samples | 30–50 h |
| G2 | Evaluation, baseline model, early export | Synthetic analytical fixtures, real baseline (requires deferred AERIS-022; not met until the owner supplies complete labels), export parity evidence | 35–60 h |
| G3 | Video and tracking | Correct timestamp/dropout behavior, MOT report on real identities (requires deferred AERIS-021; not met until the owner supplies labeled footage) | 40–70 h |
| G4 | Forecasts and calibrated geometry | Causal horizon evaluation, pixel mode, optional calibrated rays | 35–60 h |
| G5 | Python MVP, robustness, regression | End-to-end artifacts, versioned benchmark, failure examples | 60–110 h |
| G6 | C++ parity and performance | Matched fixtures, runtime integration, CPU ONNX report | 70–140 h |
| G7 | Release report | Real/synthetic evidence separated, clean reproduction, limitations | 30–60 h |

Ranges are estimates; tasks overlap in learning and integration. Do not add these
to the original calendar as commitments. The first baseline determines a revised
budget and feasible accuracy/runtime requirements.

Optional extensions after G7: conditional range research, NVIDIA TensorRT/Jetson,
moving-camera compensation. Each requires a new resource-aware task with evidence
criteria. AERIS-019 and AERIS-020 are deliberately `deferred` and cannot be dispatched.

## Dependency spine

Contract -> data audit/loader -> validator -> real detector baseline

Contract -> analytical fixtures -> metric adapters -> baseline benchmark

Baseline + benchmark -> video -> tracker -> prediction -> Python integration

Prediction + calibration -> angular geometry -> Python integration

Python integration -> robustness -> regression release gates

Early export + Python integration -> C++ parity -> reproducible performance

Regression + C++ performance -> final v1 report

Fine-tuning is an experiment after the baseline. The baseline remains acceptable
if it meets the declared use case; an unsuccessful experiment is still valid work
when its protocol and negative result are sound. Release performance criteria must
be selected after baseline measurement and recorded before candidate comparison.
