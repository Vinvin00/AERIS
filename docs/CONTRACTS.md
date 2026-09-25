# Shared contracts — version 1

Changes require a separate reviewed task and migration note. These are the initial
specifications to implement, not a claim that their runtime implementations exist.

## Coordinates, frames, and annotations

- Image origin is top-left; x right, y down. Bboxes are float `xyxy`, 0-based,
  half-open pixel extents in the **original decoded frame**, not model input space.
  Require finite `0 <= x_min < x_max <= width` and equivalent y bounds.
- Each frame has `schema_version`, `sequence_id`, integer `frame_index`, finite
  nonnegative `timestamp_s`, `timestamp_source`, width, height, image path/checksum,
  camera ID (nullable), and `annotation_status` (`complete`, `partial`, `unlabeled`).
- Video timestamps strictly increase within a sequence. Images without a time series
  can use null timestamp; do not fabricate temporal metrics. Decode PTS is preferred;
  frame-index/FPS fallback must be marked inferred. Reject zero/negative delta time.
- Ground-truth objects contain annotation ID, class ID, bbox, nullable track ID,
  `ignore`, `occluded`, and `truncated`. No detector confidence on ground truth.
  Negative frames have `complete` status and an empty object list.
- Track ID is unique within a sequence, never global; no duplicate identity in a
  frame. Missing identity is null, not a made-up ID. Classes use a versioned mapping.
- Manifest includes source URI, provenance, rights/terms, checksum, grouping ID,
  split, sensor/modality, annotation capabilities, and known pretraining overlap.
- Split by recording/session/source group first, then frame. Exact duplicates and
  source-related derivatives cannot cross splits. Near-duplicates require audit.

## Runtime boundaries

`Frame -> preprocess -> Detector -> Detections -> Tracker -> TrackStates -> Predictor`

Frame carries pixels, timing, dimensions, transforms, and permitted calibration.
Detector returns original-frame bbox, class, score in [0,1], and model provenance.
Tracker consumes only current/past detections and timestamps. Evaluator alone sees
truth and future labels. Simulator rendering and truth storage are separate outputs.

Track state: sequence ID, ID, timestamp, bbox, center, velocity (px/s), covariance,
lifecycle, age_s, time_since_seen_s, last measurement time, and observed/predicted flag.
State order starts `[cx, cy, vx, vy, w, h]`. Q depends on delta time; R describes bbox
measurement error, not detector confidence directly. Use numerically stable covariance
updates, validate PSD/finite covariance, gate association and handle empty inputs.
Map detector score to R only through a documented, validated method.

Prediction: issue time, horizon_s, target time, predicted original-frame center,
covariance, and model name. Horizons: 0.1, 0.25, 0.5, 1, 2 seconds. No future detector
results during prediction. Constant velocity is the first baseline. Acceleration
is a separate model with defined state/noise assumptions, not a noisy two-frame difference.

## Geometry and availability

Calibration includes K, distortion model/coefficients, original resolution, lens/zoom,
calibration ID and quality. Undistort before ray computation; carry crop/resize
transforms explicitly. Camera-coordinate ray is normalized `K^-1 [u,v,1]` only after
the correct distortion treatment. Angular bbox extent uses the angle between edge
rays; it is not a physical dimension. Angular rate is camera-relative rad/s internally,
degrees/s for display. Image heading uses `atan2(-vy, vx)` counterclockwise from image
right, range [0,360); near-zero speed has undefined heading. This is not aircraft yaw.

Unknown fields serialize as null plus `availability_reason`; JSON NaN/Infinity is
forbidden. Apparent expansion labels: expanding/shrinking/stable/uncertain. Pose,
camera movement, and zoom invalidate approach/recede conclusions without extra evidence.

## Outputs and reproducibility

Planned `aeris input.mp4 --output DIR` writes overlay video, `tracks.csv`,
`predictions.jsonl`, `metrics.json`, `benchmark.json`, and `run_manifest.json`.
Without ground truth, detection/tracking/forecast scores are unavailable, while
counts and measured timings remain available. CLI is implemented in AERIS-012.

Run manifest: code revision plus dirty/content fingerprint, config and model hashes,
dataset/split IDs and hashes, random seeds, dependency lock digest, OS/CPU/GPU,
actual execution provider, precision, input resolution, warmup/sample counts,
timestamp policy, command, and artifact hashes. Preserve baseline runs immutably.
Large files live outside Git with manifests and reproducible retrieval instructions.

## Evaluation

Detection: COCO AP50 and AP@[.50:.95]; precision/recall/F1 at a validation-selected
score threshold and stated IoU threshold. Custom width bins in original pixels:
`[0,16)`, `[16,32)`, `[32,64)`, `[64,infinity)`. Do not call these COCO size bins.
Report per-bin support, false positives per frame, and annotated negative-frame errors.
Partial annotations need explicit ignore policy, never implicit true negatives.

Tracking: TrackEval HOTA and IDF1 on eligible identity-labeled sequences; ID switches,
fragmentation, and retention with named definitions. Detector-only data cannot score
tracking. Report predicted-only gaps separately from observed positions.

Forecast: evaluate the same eligible ground-truth identities/horizons for all models.
Report matched position RMSE, ADE/FDE where relevant, valid count, coverage, and
unmatched/missed forecasts. Missing forecasts must not silently improve an error score.
Future truth interpolation is evaluator-only, with bounded gap policy and no
extrapolation beyond truth. Bootstrap confidence intervals by sequence, not correlated frames.

Uncertainty: distinguish detector confidence, state covariance, forecast interval,
and physical-size prior. Report empirical coverage at stated levels (e.g. 50/90/95%),
interval width, support, and residual checks. Test conditional regimes; a filter's
covariance is not automatically a calibrated real-world uncertainty estimate.

Latency: per-stage and end-to-end wall-clock P50/P95/P99, warmup excluded explicitly,
sample count, dropped frames, queueing, total throughput, hardware/provider/precision.
Synchronize asynchronous GPU execution. Headless inference and rendered pipeline
are separate runs. Shared CI runners cannot enforce a fixed hardware latency target.

Robustness: fixed seeds/severities and transform manifests; image transforms also
transform boxes and calibration as applicable. Temporal drops retain actual times;
geometric shake changes projected truth. Photometric changes retain labels but may
alter visibility; report the policy. Derivatives retain their source split.

Regression: development set only, immutable approved baseline, same protocol and
hardware for latency. Thresholds set from observed repeatability before evaluating
a candidate. Final holdout is not repeatedly tuned against. Missing required evidence
is a blocked release, never a fabricated pass.
