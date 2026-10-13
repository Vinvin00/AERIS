# ADR 012 - One CLI composing the accepted modules, with atomic immutable runs

Status: proposed for AERIS-012 review. Date: 2026-10-07.

`aeris.cli` composes the accepted modules and changes none. Tracking and forecasting run after decoding,
feeding detections in timestamp order, which keeps them causal while leaving the video pipeline untouched.
The overlay is a second decoding pass because the accepted pipeline only draws detections. Forecasters get
the raw detection center matched to each observed track rather than the tracker's filtered center, to avoid
double filtering. A replay detector makes runs deterministic and weight-free for CI. Runs go to a new
directory via a staged sibling directory and `os.replace`, so failures leave nothing and runs are never
overwritten. Truth-dependent metrics are computed only from supplied labels; otherwise they are null with
reasons. The detector operating score is an untuned default recorded in `metrics.json`. Not done: a
real-time source, multi-sequence runs, apparent-size-change labels in the CSV, and any accuracy claim.
