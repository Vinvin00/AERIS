"""Offline, original-pixel evaluation adapters. Synthetic fixtures are not benchmarks."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from contextlib import redirect_stdout
from io import StringIO
import math

import numpy as np

from aeris.data.contracts import AnnotationRecord, BBox, DetectionRecord, PredictionRecord

COCO_VERSION = "2.0.11"
TRACKEVAL_VERSION = "1.3.0"
WIDTH_BINS = ((0, 16), (16, 32), (32, 64), (64, math.inf))


class EvaluationError(ValueError):
    """Inputs violate the declared evaluation protocol."""


def _iou(a: BBox, b: BBox) -> float:
    intersection = max(0, min(a.x_max, b.x_max) - max(a.x_min, b.x_min)) * max(
        0, min(a.y_max, b.y_max) - max(a.y_min, b.y_min))
    area_a = (a.x_max - a.x_min) * (a.y_max - a.y_min)
    area_b = (b.x_max - b.x_min) * (b.y_max - b.y_min)
    return intersection / (area_a + area_b - intersection)


def _xywh(box: BBox) -> list[float]:
    return [box.x_min, box.y_min, box.x_max - box.x_min, box.y_max - box.y_min]


def _frame_key(record: AnnotationRecord | DetectionRecord) -> tuple[str, int]:
    if isinstance(record, AnnotationRecord):
        record = record.frame
    return record.sequence_id, record.frame_index


def detection_metrics(
    annotations: list[AnnotationRecord], detections: list[DetectionRecord], *,
    score_threshold: float, iou_threshold: float = 0.5, max_detections: int = 100,
) -> dict:
    """COCO AP plus fixed-threshold counts. Partial/unlabeled frames are excluded.

    ``score_threshold`` must have been selected on validation data before this call.
    COCO's standard maxDets=[1,10,max_detections] applies to AP; count metrics
    also use the same per-image detection budget to prevent box-spam recall.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not 0 <= score_threshold <= 1 or not 0 < iou_threshold <= 1:
        raise EvaluationError("thresholds outside valid range")
    if max_detections < 10:
        raise EvaluationError("max_detections must be >= 10 for COCO maxDets")
    frames = {_frame_key(a): a for a in annotations}
    if len(frames) != len(annotations):
        raise EvaluationError("duplicate annotation frame")
    selected = {key: ann for key, ann in frames.items() if ann.frame.annotation_status == "complete"}
    skipped = {"partial": sum(a.frame.annotation_status == "partial" for a in annotations),
               "unlabeled": sum(a.frame.annotation_status == "unlabeled" for a in annotations)}
    by_frame = defaultdict(list)
    for det in detections:
        key = _frame_key(det)
        if key not in frames:
            raise EvaluationError(f"detection has no frame: {key}")
        det.validate_frame(frames[key].frame)
        if det.class_mapping_version != frames[key].class_mapping_version:
            raise EvaluationError("detection class mapping differs from truth")
        if key in selected:
            by_frame[key].append(det)
    if not selected:
        return {"available": False, "reason": "no_complete_annotations", "skipped": skipped,
                "ap50": None, "ap50_95": None}
    keys = sorted(selected)
    image_ids = {key: i + 1 for i, key in enumerate(keys)}
    classes = sorted({obj.class_id for ann in selected.values() for obj in ann.objects}
                     | {d.class_id for ds in by_frame.values() for d in ds})
    if not classes:
        return {"available": True, "backend": f"pycocotools=={COCO_VERSION}",
                "ap50": None, "ap50_95": None, "ap_reason": "no_categories",
                "score_threshold": score_threshold, "iou_threshold": iou_threshold,
                "max_detections_per_frame": max_detections,
                "counts": {"tp": 0, "fp": 0, "fn": 0},
                "precision": None, "recall": None, "f1": None,
                "width_bins": [{"width_px": [lo, None if math.isinf(hi) else hi],
                                "support": 0, "matched": 0, "recall": None}
                               for lo, hi in WIDTH_BINS],
                "false_positives_per_complete_frame": 0.0,
                "negative_frame_errors": 0, "negative_frame_count": len(keys),
                "complete_frame_count": len(keys), "skipped": skipped}
    coco_gt = COCO()
    coco_gt.dataset = {"info": {}, "licenses": [],
        "images": [{"id": image_ids[k], "width": selected[k].frame.width,
                    "height": selected[k].frame.height} for k in keys],
        "categories": [{"id": c + 1, "name": str(c)} for c in classes], "annotations": []}
    aid = 1
    for key in keys:
        for obj in selected[key].objects:
            box = _xywh(obj.bbox)
            coco_gt.dataset["annotations"].append({"id": aid, "image_id": image_ids[key],
                "category_id": obj.class_id + 1, "bbox": box, "area": box[2] * box[3],
                "iscrowd": int(obj.ignore)})
            aid += 1
    with redirect_stdout(StringIO()):
        coco_gt.createIndex()
    limited = {k: sorted(by_frame[k], key=lambda d: d.score, reverse=True)[:max_detections]
               for k in keys}
    results = [{"image_id": image_ids[k], "category_id": d.class_id + 1,
                "bbox": _xywh(d.bbox), "score": d.score} for k in keys for d in limited[k]]
    with redirect_stdout(StringIO()):
        coco_dt = coco_gt.loadRes(results) if results else COCO()
        if not results:
            coco_dt.dataset = {**coco_gt.dataset, "annotations": []}
            coco_dt.createIndex()
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.imgIds = list(image_ids.values())
        evaluator.params.catIds = [c + 1 for c in classes]
        evaluator.params.maxDets = [1, 10, max_detections]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    def ap(x):
        return None if x < 0 else float(x)
    counts = {"tp": 0, "fp": 0, "fn": 0}
    bins = [{"width_px": [lo, None if math.isinf(hi) else hi], "support": 0,
             "matched": 0} for lo, hi in WIDTH_BINS]
    negative_errors = 0
    for key in keys:
        ann = selected[key]
        truth = [o for o in ann.objects if not o.ignore]
        ignored = [o for o in ann.objects if o.ignore]
        for obj in truth:
            width = obj.bbox.x_max - obj.bbox.x_min
            next(b for b, (lo, hi) in zip(bins, WIDTH_BINS) if lo <= width < hi)["support"] += 1
        matched = set()
        for det in limited[key]:
            if det.score < score_threshold:
                continue
            candidates = [(i, _iou(det.bbox, obj.bbox)) for i, obj in enumerate(truth)
                          if i not in matched and det.class_id == obj.class_id]
            if candidates and max(v for _, v in candidates) >= iou_threshold:
                idx, _ = max(candidates, key=lambda pair: pair[1])
                matched.add(idx)
                counts["tp"] += 1
                width = truth[idx].bbox.x_max - truth[idx].bbox.x_min
                next(b for b, (lo, hi) in zip(bins, WIDTH_BINS) if lo <= width < hi)["matched"] += 1
            elif not any(det.class_id == obj.class_id and _iou(det.bbox, obj.bbox) >= iou_threshold
                         for obj in ignored):
                counts["fp"] += 1
                if ann.is_negative:
                    negative_errors += 1
        counts["fn"] += len(truth) - len(matched)
    for b in bins:
        b["recall"] = b["matched"] / b["support"] if b["support"] else None
    p = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else None
    r = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else None
    return {"available": True, "backend": f"pycocotools=={COCO_VERSION}",
            "ap50_95": ap(evaluator.stats[0]), "ap50": ap(evaluator.stats[1]),
            "score_threshold": score_threshold, "iou_threshold": iou_threshold,
            "max_detections_per_frame": max_detections, "counts": counts,
            "precision": p, "recall": r, "f1": 2*p*r/(p+r) if p is not None and r is not None and p+r else None,
            "width_bins": bins, "false_positives_per_complete_frame": counts["fp"]/len(keys),
            "negative_frame_errors": negative_errors,
            "negative_frame_count": sum(selected[k].is_negative for k in keys),
            "complete_frame_count": len(keys), "skipped": skipped}


@dataclass(frozen=True)
class TrackObservation:
    sequence_id: str
    frame_index: int
    track_id: str
    bbox: BBox
    class_id: int
    observed: bool = True


def tracking_metrics(annotations: list[AnnotationRecord], tracks: list[TrackObservation]) -> dict:
    """TrackEval HOTA/IDF1; only complete timed identity-labeled sequences."""
    from trackeval.metrics import HOTA, Identity

    frames = {_frame_key(a): a for a in annotations}
    if len(frames) != len(annotations):
        raise EvaluationError("duplicate annotation frame")
    sequences = defaultdict(list)
    for ann in annotations:
        sequences[ann.frame.sequence_id].append(ann)
    eligible = {}
    for seq, anns in sequences.items():
        if (all(a.frame.annotation_status == "complete" and a.frame.timestamp_s is not None for a in anns)
            and all(o.track_id is not None for a in anns for o in a.objects if not o.ignore)
            and any(not o.ignore for a in anns for o in a.objects)):
            eligible[seq] = sorted(anns, key=lambda a: a.frame.frame_index)
    if not eligible:
        return {"available": False, "reason": "no_complete_timed_identity_sequence",
                "hota": None, "idf1": None}
    by_frame = defaultdict(list)
    predicted_only = 0
    for track in tracks:
        key = (track.sequence_id, track.frame_index)
        if key not in frames:
            raise EvaluationError("track has no annotation frame")
        track.bbox.validate_bounds(frames[key].frame.width, frames[key].frame.height)
        if track.observed and track.sequence_id in eligible:
            if any(t.track_id == track.track_id for t in by_frame[key]):
                raise EvaluationError("duplicate tracker identity in frame")
            by_frame[key].append(track)
        elif not track.observed:
            predicted_only += 1
    hota_metric, identity_metric = HOTA(), Identity({"PRINT_CONFIG": False})
    results = []
    for seq, anns in eligible.items():
        gt_ids = {o.track_id for a in anns for o in a.objects if not o.ignore}
        tr_ids = {t.track_id for a in anns for t in by_frame[(seq, a.frame.frame_index)]}
        gi = {v: i for i, v in enumerate(sorted(gt_ids))}
        ti = {v: i for i, v in enumerate(sorted(tr_ids))}
        payload = {"gt_ids": [], "tracker_ids": [], "similarity_scores": [],
                   "num_gt_ids": len(gi), "num_tracker_ids": len(ti),
                   "num_gt_dets": 0, "num_tracker_dets": 0}
        for ann in anns:
            truth = [o for o in ann.objects if not o.ignore]
            observed = by_frame[(seq, ann.frame.frame_index)]
            payload["gt_ids"].append(np.asarray([gi[o.track_id] for o in truth], dtype=int))
            payload["tracker_ids"].append(np.asarray([ti[t.track_id] for t in observed], dtype=int))
            payload["similarity_scores"].append(np.asarray(
                [[_iou(o.bbox, t.bbox) if o.class_id == t.class_id else 0 for t in observed]
                 for o in truth], dtype=float).reshape(len(truth), len(observed)))
            payload["num_gt_dets"] += len(truth)
            payload["num_tracker_dets"] += len(observed)
        results.append((hota_metric.eval_sequence(payload), identity_metric.eval_sequence(payload)))
    h = hota_metric.combine_sequences({str(i): pair[0] for i, pair in enumerate(results)})
    ident = identity_metric.combine_sequences({str(i): pair[1] for i, pair in enumerate(results)})
    return {"available": True, "backend": f"trackeval=={TRACKEVAL_VERSION}",
            "hota": float(h["HOTA"].mean()), "idf1": float(ident["IDF1"]),
            "idtp": int(ident["IDTP"]), "idfp": int(ident["IDFP"]),
            "idfn": int(ident["IDFN"]), "eligible_sequences": sorted(eligible),
            "predicted_only_positions_excluded": predicted_only}


def forecast_metrics(annotations: list[AnnotationRecord], predictions: list[PredictionRecord], *,
                     models: list[str], horizons_s: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0),
                     max_interpolation_gap_s: float = 0.5) -> dict:
    """Evaluate declared models on identical issue/identity/horizon keys.

    Issues require an observed, nonignored identity at a complete timed frame.
    Future truth may be interpolated only between complete observed samples in the
    same sequence/identity; it is never extrapolated. No runtime predictor is called.
    """
    if not models or len(set(models)) != len(models):
        raise EvaluationError("models must be a nonempty unique list")
    if max_interpolation_gap_s <= 0:
        raise EvaluationError("max_interpolation_gap_s must be positive")
    if not horizons_s or any(not h > 0 for h in horizons_s):
        raise EvaluationError("horizons_s must be nonempty and strictly positive")
    by_track = defaultdict(list)
    issues = []
    for ann in annotations:
        if ann.frame.annotation_status != "complete" or ann.frame.timestamp_s is None:
            continue
        for obj in ann.objects:
            if obj.ignore or obj.track_id is None:
                continue
            key = (ann.frame.sequence_id, obj.track_id)
            center = ((obj.bbox.x_min + obj.bbox.x_max)/2,
                      (obj.bbox.y_min + obj.bbox.y_max)/2)
            by_track[key].append((ann.frame.timestamp_s, center))
            issues.append((key, ann.frame.timestamp_s))
    for values in by_track.values():
        values.sort()
        if len({t for t, _ in values}) != len(values):
            raise EvaluationError("duplicate identity timestamp")
    eligible = {}
    for key, issue in issues:
        values = by_track[key]
        for horizon in horizons_s:
            target = issue + horizon
            exact = next((c for t, c in values if abs(t - target) <= max(1e-9, 4 * math.ulp(target))), None)
            if exact is not None:
                eligible[(key[0], key[1], issue, horizon)] = exact
                continue
            for (t0, c0), (t1, c1) in zip(values, values[1:]):
                if t0 < target < t1 and t1 - t0 <= max_interpolation_gap_s:
                    weight = (target-t0)/(t1-t0)
                    eligible[(key[0], key[1], issue, horizon)] = (
                        c0[0]+weight*(c1[0]-c0[0]), c0[1]+weight*(c1[1]-c0[1]))
                    break
    supplied = {}
    for p in predictions:
        if p.model_name not in models:
            raise EvaluationError("prediction model not declared")
        key = (p.sequence_id, p.track_id, p.issue_time_s, p.horizon_s)
        if (p.model_name, key) in supplied:
            raise EvaluationError("duplicate forecast")
        supplied[(p.model_name, key)] = p
    result = {}
    unmatched_forecasts = {model: sum(1 for name, key in supplied
                                      if name == model and key not in eligible)
                           for model in models}
    for model in models:
        per_horizon = {}
        for horizon in horizons_s:
            keys = [key for key in eligible if key[3] == horizon]
            errors_sq = []
            for key in keys:
                p = supplied.get((model, key))
                if p is not None and p.center is not None:
                    truth = eligible[key]
                    errors_sq.append((p.center[0]-truth[0])**2 + (p.center[1]-truth[1])**2)
            matched = len(errors_sq)
            per_horizon[str(horizon)] = {"support": len(keys), "matched": matched,
                "missed": len(keys)-matched, "coverage": matched/len(keys) if keys else None,
                "rmse_px": math.sqrt(sum(errors_sq)/matched) if matched else None,
                "mean_error_px": sum(math.sqrt(e) for e in errors_sq)/matched if matched else None}
        result[model] = per_horizon
    return {"available": bool(eligible), "reason": None if eligible else "no_eligible_future_truth",
            "interpolation_max_gap_s": max_interpolation_gap_s, "models": result,
            "eligible_keys": len(eligible),
            "forecasts_without_eligible_truth": unmatched_forecasts}
