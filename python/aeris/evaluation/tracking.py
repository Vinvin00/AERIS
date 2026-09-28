"""TrackEval tracking evaluation (official upstream, pinned) over contract-v1 truth.

Protocol (docs/evaluation.md):
- Only identity-labelled sequences are eligible: every frame ``complete`` and every
  non-ignored object carries a track ID. Detector-only or partial data cannot score
  tracking; ineligible sequences are listed with reasons.
- Distractors follow TrackEval's MOTChallenge preprocessing: one Hungarian assignment of
  tracker boxes over all truth (IoU below 0.5 zeroed); only boxes assigned to an ignored
  object are removed, so a box on a labelled object that also overlaps an ignored one stays.
- Similarity is TrackEval's own box IoU; HOTA, CLEAR (MOTA, IDSW, Frag, MT/PT/ML) and
  Identity (IDF1) are TrackEval's metric classes and sequence combination.
- Predicted-only tracker boxes (``observed=False``) are excluded unless requested, and are
  always counted separately.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aeris.data.contracts import AnnotationRecord, BBox, ContractError
from aeris.evaluation._common import (
    EvaluationError, evaluator_versions, numpy_scalar_aliases, quiet, require_pinned_evaluators,
)

DISTRACTOR_IOU = 0.5


@dataclass(frozen=True)
class TrackerBox:
    """One tracker output box in original-frame pixels; observed=False marks a prediction."""

    sequence_id: str
    frame_index: int
    track_id: str
    bbox: BBox
    observed: bool = True

    def __post_init__(self) -> None:
        for name in ("sequence_id", "track_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or any(c.isspace() for c in value):
                raise ContractError(f"{name}: expected nonempty identifier without whitespace")
        if type(self.frame_index) is not int or self.frame_index < 0:
            raise ContractError("frame_index: expected integer >= 0")
        if not isinstance(self.bbox, BBox):
            raise ContractError("bbox: expected BBox")
        if type(self.observed) is not bool:
            raise ContractError("observed: expected bool")


def _boxes(items) -> np.ndarray:
    return np.array([[b.x_min, b.y_min, b.x_max, b.y_max] for b in items], dtype=float).reshape(-1, 4)


def _sequence_data(records, boxes_by_frame, box_iou, assign):
    """TrackEval's per-sequence data dict, plus the count of distractor-matched boxes."""
    gt_names = sorted({item.track_id for r in records for item in r.objects if not item.ignore})
    tracker_names = sorted({b.track_id for items in boxes_by_frame.values() for b in items})
    gt_ids, tracker_ids = {n: i for i, n in enumerate(gt_names)}, {n: i for i, n in enumerate(tracker_names)}
    data = {"num_timesteps": len(records), "num_gt_ids": len(gt_names),
            "num_tracker_ids": len(tracker_names), "num_gt_dets": 0, "num_tracker_dets": 0,
            "gt_ids": [], "tracker_ids": [], "similarity_scores": []}
    removed = 0
    eps = np.finfo(float).eps
    for record in records:
        truth = [item for item in record.objects if not item.ignore]
        ignored = [item.bbox for item in record.objects if item.ignore]
        tracked = list(boxes_by_frame.get(record.frame.frame_index, []))
        if ignored and tracked:
            # MOTChallenge preprocessing (TrackEval mot_challenge_2d_box): one Hungarian
            # assignment over all truth, labelled rows first, with IoU below 0.5 zeroed;
            # remove only tracker boxes assigned to an ignored row.
            scores = box_iou(_boxes([i.bbox for i in truth] + ignored), _boxes([b.bbox for b in tracked]),
                             box_format="x0y0x1y1")
            scores[scores < DISTRACTOR_IOU - eps] = 0
            rows, cols = assign(-scores)
            drop = {c for r, c in zip(rows, cols) if scores[r, c] > eps and r >= len(truth)}
            removed += len(drop)
            tracked = [b for i, b in enumerate(tracked) if i not in drop]
        similarity = (box_iou(_boxes([i.bbox for i in truth]), _boxes([b.bbox for b in tracked]),
                              box_format="x0y0x1y1") if truth and tracked
                      else np.zeros((len(truth), len(tracked))))
        data["gt_ids"].append(np.array([gt_ids[i.track_id] for i in truth], dtype=int))
        data["tracker_ids"].append(np.array([tracker_ids[b.track_id] for b in tracked], dtype=int))
        data["similarity_scores"].append(similarity)
        data["num_gt_dets"] += len(truth)
        data["num_tracker_dets"] += len(tracked)
    return data, removed


def _summary(results: dict) -> dict:
    hota, clear, identity = results["HOTA"], results["CLEAR"], results["Identity"]
    mean = lambda value: float(np.mean(value))  # noqa: E731  HOTA fields are per-alpha arrays
    return {
        "HOTA": mean(hota["HOTA"]), "DetA": mean(hota["DetA"]), "AssA": mean(hota["AssA"]),
        "LocA": mean(hota["LocA"]),
        "MOTA": float(clear["MOTA"]), "MOTP": float(clear["MOTP"]), "IDSW": int(clear["IDSW"]),
        "Frag": int(clear["Frag"]), "MT": int(clear["MT"]), "PT": int(clear["PT"]),
        "ML": int(clear["ML"]), "CLR_TP": int(clear["CLR_TP"]), "CLR_FN": int(clear["CLR_FN"]),
        "CLR_FP": int(clear["CLR_FP"]),
        "IDF1": float(identity["IDF1"]), "IDP": float(identity["IDP"]),
        "IDR": float(identity["IDR"]), "IDTP": int(identity["IDTP"]),
        "IDFN": int(identity["IDFN"]), "IDFP": int(identity["IDFP"]),
    }


def evaluate_tracking(truth, tracker_boxes, *, include_predicted: bool = False,
                      check_versions: bool = True) -> dict:
    """HOTA, CLEAR and Identity per eligible sequence and combined."""
    versions = require_pinned_evaluators() if check_versions else evaluator_versions()
    with quiet():  # the package __init__ reports optional dataset loaders it cannot import
        from scipy.optimize import linear_sum_assignment
        from trackeval.datasets._base_dataset import _BaseDataset
        from trackeval.metrics import CLEAR, HOTA, Identity
    sequences: dict[str, list[AnnotationRecord]] = {}
    frames = {}
    for record in truth:
        if not isinstance(record, AnnotationRecord):
            raise EvaluationError("truth: expected AnnotationRecord entries")
        key = (record.frame.sequence_id, record.frame.frame_index)
        if key in frames:
            raise EvaluationError(f"truth: duplicate frame {key}")
        frames[key] = record
        sequences.setdefault(record.frame.sequence_id, []).append(record)
    ineligible = {}
    for sequence_id, records in sequences.items():
        records.sort(key=lambda r: r.frame.frame_index)
        if any(r.frame.annotation_status != "complete" for r in records):
            ineligible[sequence_id] = "partial or unlabeled frames: identities and false positives are not fully annotated"
        elif any(item.track_id is None for r in records for item in r.objects if not item.ignore):
            ineligible[sequence_id] = "objects without track IDs: detector-only labels cannot score tracking"
    boxes = {}
    counts = {"tracker_boxes": 0, "predicted_only_boxes": 0, "predicted_only_excluded": 0,
              "unscored_on_ineligible_sequences": 0, "removed_as_distractor_matches": 0}
    for box in tracker_boxes:
        if not isinstance(box, TrackerBox):
            raise EvaluationError("tracker_boxes: expected TrackerBox entries")
        key = (box.sequence_id, box.frame_index)
        if key not in frames:
            raise EvaluationError(f"tracker box on frame {key} has no truth record")
        try:
            box.bbox.validate_bounds(frames[key].frame.width, frames[key].frame.height)
        except ContractError as error:
            raise EvaluationError(f"tracker box on frame {key}: {error}") from error
        counts["tracker_boxes"] += 1
        counts["predicted_only_boxes"] += not box.observed
        if box.sequence_id in ineligible:
            counts["unscored_on_ineligible_sequences"] += 1
        elif box.observed or include_predicted:
            boxes.setdefault(box.sequence_id, {}).setdefault(box.frame_index, []).append(box)
        else:
            counts["predicted_only_excluded"] += 1
    for sequence_id, items in boxes.items():
        for frame_index, frame_boxes in items.items():
            names = [b.track_id for b in frame_boxes]
            if len(names) != len(set(names)):
                raise EvaluationError(f"tracker: duplicate track ID in frame {(sequence_id, frame_index)}")
    metrics = (HOTA(), CLEAR({"PRINT_CONFIG": False}), Identity({"PRINT_CONFIG": False}))
    per_sequence, raw = {}, {}
    with numpy_scalar_aliases(), quiet():
        for sequence_id in sorted(set(sequences) - set(ineligible)):
            data, removed = _sequence_data(sequences[sequence_id], boxes.get(sequence_id, {}),
                                           _BaseDataset._calculate_box_ious, linear_sum_assignment)
            counts["removed_as_distractor_matches"] += removed
            raw[sequence_id] = {m.get_name(): m.eval_sequence(data) for m in metrics}
            per_sequence[sequence_id] = {**_summary(raw[sequence_id]),
                                         "gt_detections": data["num_gt_dets"],
                                         "tracker_detections": data["num_tracker_dets"]}
        combined = ({m.get_name(): m.combine_sequences({s: raw[s][m.get_name()] for s in raw})
                     for m in metrics} if raw else None)
    reasons = {}
    if combined is None:
        reasons["combined"] = ("no identity-labelled complete sequence: tracking metrics are "
                               "unavailable (detector-only data cannot score tracking)")
    return {
        "evaluator": {"name": "TrackEval HOTA/CLEAR/Identity", "versions": versions,
                      "clear_identity_iou_threshold": 0.5},
        "include_predicted": include_predicted,
        "eligible_sequences": sorted(raw), "ineligible_sequences": dict(sorted(ineligible.items())),
        "counts": counts,
        "per_sequence": per_sequence,
        "combined": _summary(combined) if combined else None,
        "availability_reason": reasons,
    }
