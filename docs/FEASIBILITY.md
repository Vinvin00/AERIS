# Feasibility and plan integrity review

Reviewed 2026-09-24. The supplied plan is archived in `docs/source/project-plan.txt`.

## Verdict

The image-space perception project is feasible. A reliable detector, timestamp-aware
tracker, short-horizon predictor, measured uncertainty, and reproducible evaluation
form a coherent portfolio project. The original schedule is an ordering suggestion,
not a supported delivery estimate. Dataset acquisition/annotation and cross-language
integration are major dependencies that the original plan underestimates.

Planning estimate (engineering judgment, not measured velocity): approximately
200–350 focused hours for a Python research MVP; an additional 100–200 for a complete
C++ runtime, robustness campaign, and presentation. Model training and annotation
can extend elapsed time substantially. At 10 hours/week this is a multi-month
project; 8–12 weeks is plausible only for a reduced scope or substantial weekly
availability, suitable data, and existing Python/C++ experience. Re-estimate after
the dataset audit and first real baseline. Hardware and weekly hours remain unknown.

## Required corrections

| Original issue | Consequence | Revised requirement |
|---|---|---|
| Ground truth arrives in phase 7 | Algorithms can develop around unverifiable claims | Add analytic fixtures and evaluation contracts before the first model |
| Regression arrives in phase 11 | Earlier changes can silently break behavior | Unit/fixture gates immediately; real benchmark gates after baseline approval |
| A still-image aircraft dataset is assumed sufficient | Does not establish tiny-object or multi-object video performance | Separate still-image baseline, identity-annotated video, negatives, and synthetic geometry evidence |
| Frame schema omits sequence/camera/split context | ID collisions, leakage, and timestamp ambiguity | Versioned sequence-aware contracts with null IDs for untracked images |
| Camera motion is unspecified | Pan/zoom can be mistaken for aircraft motion or approach | Fixed-camera MVP; camera-relative outputs; reject or flag motion/zoom for geometry |
| Bounding-box expansion implies approach | Pose, occlusion, zoom, and detector jitter confound range | Output expanding/shrinking/stable/uncertain; approach is hypothesis-only extension |
| Calibration implies metric 3D motion | Intrinsics produce rays, not depth | No absolute range or world velocity claim without additional assumptions/truth |
| Range interval lacks probability semantics | A plausible interval can appear calibrated | State conditioning assumptions, coverage level, method, and measured coverage |
| Tracker uses frame ages and fixed transitions | Variable frame rate/dropouts change physical behavior | Use seconds for delta time, loss timeout, and forecast horizons |
| Generic tracking score | Results cannot be reproduced or compared | HOTA/IDF1 plus explicitly defined error/retention metrics |
| C++ port happens after algorithm completion | Export or preprocessing incompatibility discovered late | ONNX feasibility probe early; port only frozen, fixture-tested interfaces |
| TensorRT assumed available | Local Mac cannot establish NVIDIA runtime performance | CPU is required baseline; CoreML optional; NVIDIA/Jetson results require that hardware |
| Accuracy and latency examples resemble targets | Invented numbers can turn into false claims | Treat every number in the source plan as illustrative, never a baseline |
| `metrics.json` on arbitrary video | Ground-truth-dependent scores are unknowable | Emit metric availability/reason; benchmark scores require matching labels |
| Many directories before a working pipeline | Infrastructure consumes the project | Thin executable slices, bounded tasks, measurable acceptance |

## Scope and release boundary

**MVP:** recorded optical RGB video; fixed camera and no changing zoom; aircraft
category explicitly mapped by dataset; detection, multiple identities, image-space
motion, timestamp-based forecasts, overlay, tracks, performance timings, and
evaluation against versioned labels. Calibrated angular outputs are conditional on
valid calibration; the pipeline remains usable without them. Synthetic fixtures
validate math and plumbing only. At least one real held-out video set is required
before claiming real tracking performance.

**v1 engineering extension:** parity-tested C++ inference/tracking/runtime,
controlled robustness campaign, reproducible CPU ONNX benchmark, final limitations
and results report. TensorRT/Jetson and monocular ranging are optional extensions,
not release blockers. Live streaming, moving-camera compensation, re-identification
across long gaps, metric reconstruction, and control integration are out of scope.

## Data decision

Start with a small rights-cleared RGB collection covering negatives, tiny aircraft,
occlusions, crossings, and changing illumination. Audit before committing to a model.
The audit must distinguish fully annotated negative frames from unlabeled frames.

FGVC-Aircraft is a candidate for a still-image smoke baseline, not the intended video
benchmark: its official format has one aircraft box per image, 1-based coordinates,
and a copyright banner that requires handling. Its images are offered for
non-commercial research; do not bundle them into this repository.
[Official dataset description](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/).

DUT Anti-UAV is a visible-light UAV detection/tracking candidate, but UAV identity,
scale, domain, annotation completeness, and permissions must be audited before use.
It cannot silently substitute for evidence about manned aircraft.
[Dataset authors' paper](https://arxiv.org/abs/2205.10851).

Do not assume footage captured *from* drones is footage *of* aircraft. Preserve
source grouping across splits; adjacent frames and degraded derivatives must never
cross split boundaries. Detector pretraining overlap with any test data must be
recorded as a limitation. Keep a development regression set distinct from the final
holdout, which is only opened for release evaluation.

## Model and hardware decision

Select the baseline through a short spike: reproducible weights, class mapping,
permitted use, Python inference, CPU export, and documented input/output behavior.
Ultralytics publishes AGPL and enterprise options; record the chosen code and weight
terms before adoption rather than choosing a project license by accident.
[Vendor licensing documentation](https://docs.ultralytics.com/).

Use ONNX Runtime CPU as the portable deployment baseline. Execution providers are
hardware-specific; list the providers actually used and inspect fallback instead
of labeling every run “accelerated.” CoreML is an optional Apple path. TensorRT is
an NVIDIA path and needs its compatible CUDA/runtime stack and representative hardware.
[Provider reference](https://onnxruntime.ai/docs/execution-providers/),
[TensorRT requirements](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html).

## Scientific decision

Use the COCO evaluator for detection AP and TrackEval for tracking metrics, retaining
the exact versions/configurations and conversion tests. Do not replace standardized
metrics with home-grown approximations carrying the same name.
[COCO evaluator implementation](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py),
[TrackEval](https://github.com/JonathonLuiten/TrackEval).

Intrinsics and lens distortion determine viewing rays. Cropping/resizing and camera
zoom change the pixel-to-ray interpretation. Calibration must identify resolution,
lens configuration, reprojection error, and valid operating conditions.
[OpenCV camera geometry reference](https://docs.opencv.org/4.8.0/d9/d0c/group__calib3d.html).

Bounding-box width is an apparent extent, not a measured physical wingspan. Even
under pinhole assumptions, range scales with assumed projected size. Pose and
visibility uncertainty can dominate detector uncertainty. No algorithm can remove
that ambiguity from a generic monocular bounding box. Document assumption failures
and abstain; reserve numerical ranging claims for a separately validated extension.

## Risks and stop conditions

| Risk | Early test | Response |
|---|---|---|
| No usable sequence identities | Audit one representative clip | Annotate a bounded licensed subset; defer real MOT claims |
| Tiny-target recall inadequate | Size-binned recall and false positives | Resolution/tiling experiment with latency budget; retain baseline |
| No calibration | Verify camera metadata | Continue pixel-space mode; angular/range values unavailable |
| No training hardware | Time a small CPU inference sample | Use pretrained baseline; schedule training only after budget decision |
| Export unsupported | Early export parity sample | Change supported ops/model; do not defer incompatibility to C++ phase |
| Synthetic-only success | Compare a real held-out sequence | Report domain gap and avoid generalization claims |
| Agents repeatedly fail acceptance | Three recorded attempts | Stop task and request a concrete scope/contract/resource decision |

Evidence gates control readiness. They cannot guarantee correctness or eliminate
the need to review major scientific and resource decisions.
