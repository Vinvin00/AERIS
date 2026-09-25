"""AERIS evaluation entry points (AERIS-005); protocol in docs/evaluation.md, ADR 005.

- ``evaluate_detection``: pinned pycocotools COCOeval AP50 and AP@[.50:.95] on complete
  frames, plus threshold precision/recall/F1, contract width-bin recall with support,
  and negative-frame false positives.
- ``evaluate_tracking``: pinned official TrackEval HOTA, CLEAR and Identity on
  identity-labelled complete sequences.
- ``evaluate_forecasts``: causal horizon evaluation with a model-independent eligible
  set, misses, coverage, interval coverage and sequence-bootstrap intervals.

Every unavailable number is null with an ``availability_reason`` entry.
"""
from aeris.evaluation._common import (
    MIN_SEQUENCES_FOR_INTERVALS, PYCOCOTOOLS_VERSION, TRACKEVAL_COMMIT, TRACKEVAL_SOURCE,
    WIDTH_BINS, EvaluationError, EvaluatorVersionError, evaluator_versions,
    require_pinned_evaluators, sequence_bootstrap,
)
from aeris.evaluation.detection import evaluate_detection
from aeris.evaluation.forecast import evaluate_forecasts
from aeris.evaluation.tracking import TrackerBox, evaluate_tracking

__all__ = [
    "MIN_SEQUENCES_FOR_INTERVALS", "PYCOCOTOOLS_VERSION", "TRACKEVAL_COMMIT", "TRACKEVAL_SOURCE",
    "WIDTH_BINS", "EvaluationError", "EvaluatorVersionError", "TrackerBox", "evaluate_detection",
    "evaluate_forecasts", "evaluate_tracking", "evaluator_versions", "require_pinned_evaluators",
    "sequence_bootstrap",
]
