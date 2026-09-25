"""COCO detection evaluation (pycocotools, pinned) over contract-v1 truth and detections.

Protocol (docs/evaluation.md):
- Unlabeled frames are excluded; detections on them are counted, never scored.
- AP50 and AP@[.50:.95] are standard COCOeval results over *complete* frames only.
  Partial labels cannot score false positives, so AP is unavailable without complete frames.
- AERIS ``ignore`` objects become COCO crowd regions (iscrowd=1), COCO's own ignore
  mechanism: detections matched to them are neither true nor false positives.
- On partial frames a full-frame crowd region per category leaves unmatched detections
  unscored; their labelled objects still count toward recall.
- Precision/recall/F1 need a pre-fixed score threshold (validation-selected, supplied by
  the caller) and use COCOeval matches at one stated IoU threshold. Width bins are the
  contract's original-pixel bins, never COCO area ranges.
"""
from __future__ import annotations

import numpy as np

from aeris.data.contracts import AnnotationRecord, ContractError, DetectionRecord
from aeris.evaluation._common import (
    WIDTH_BINS, EvaluationError, evaluator_versions, quiet, ratio, require_pinned_evaluators,
    width_bin,
)

COCO_MAX_DETECTIONS = 100


def _xywh(box) -> list[float]:
    return [box.x_min, box.y_min, box.x_max - box.x_min, box.y_max - box.y_min]


def _index(truth, detections, max_detections_per_frame):
    frames = {}
    for record in truth:
        if not isinstance(record, AnnotationRecord):
            raise EvaluationError("truth: expected AnnotationRecord entries")
        key = (record.frame.sequence_id, record.frame.frame_index)
        if key in frames:
            raise EvaluationError(f"truth: duplicate frame {key}")
        frames[key] = record
    mappings = {record.class_mapping_version for record in frames.values()}
    if len(mappings) > 1:
        raise EvaluationError(f"truth: mixed class_mapping_version {sorted(mappings)}")
    by_frame = {}
    for detection in detections:
        if not isinstance(detection, DetectionRecord):
            raise EvaluationError("detections: expected DetectionRecord entries")
        key = (detection.sequence_id, detection.frame_index)
        if key not in frames:
            raise EvaluationError(f"detection on frame {key} has no truth record")
        try:
            detection.validate_frame(frames[key].frame)
        except ContractError as error:
            raise EvaluationError(f"detection on frame {key}: {error}") from error
        if mappings and detection.class_mapping_version not in mappings:
            raise EvaluationError(f"detection on frame {key}: class_mapping_version "
                                  f"{detection.class_mapping_version} differs from truth")
        by_frame.setdefault(key, []).append(detection)
    dropped = 0
    for key, items in by_frame.items():
        ranked = sorted(items, key=lambda d: -d.score)  # stable: ties keep input order
        dropped += max(0, len(ranked) - max_detections_per_frame)
        by_frame[key] = ranked[:max_detections_per_frame]
    return frames, by_frame, dropped


def _coco_sets(frames, keys, by_frame, categories):
    images, annotations, results, gt_width = [], [], [], {}
    image_ids = {key: number for number, key in enumerate(keys, 1)}
    for key in keys:
        record, image_id = frames[key], image_ids[key]
        frame = record.frame
        images.append({"id": image_id, "width": frame.width, "height": frame.height})
        for item in record.objects:
            x, y, w, h = _xywh(item.bbox)
            annotations.append({"id": len(annotations) + 1, "image_id": image_id,
                                "category_id": item.class_id + 1, "bbox": [x, y, w, h],
                                "area": w * h, "iscrowd": int(item.ignore)})
            gt_width[len(annotations)] = w
        if frame.annotation_status == "partial":
            for category in categories:
                annotations.append({"id": len(annotations) + 1, "image_id": image_id,
                                    "category_id": category,
                                    "bbox": [0.0, 0.0, float(frame.width), float(frame.height)],
                                    "area": float(frame.width * frame.height), "iscrowd": 1})
        results += [{"image_id": image_id, "category_id": d.class_id + 1, "bbox": _xywh(d.bbox),
                     "score": d.score} for d in by_frame.get(key, [])]
    ground_truth = {"images": images, "annotations": annotations,
                    "categories": [{"id": c, "name": f"class-{c - 1}"} for c in categories]}
    return ground_truth, results, gt_width, {number: key for key, number in image_ids.items()}


def _cocoeval(ground_truth, results, iou_thresholds=None):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with quiet():
        coco = COCO()
        coco.dataset = ground_truth
        coco.createIndex()
        evaluation = COCOeval(coco, coco.loadRes(results), "bbox")
        if iou_thresholds is not None:
            evaluation.params.iouThrs = np.array(iou_thresholds, dtype=float)
        evaluation.evaluate()
        evaluation.accumulate()
        if iou_thresholds is None:
            evaluation.summarize()
    return evaluation


def _average_precision(frames, by_frame, categories, reasons) -> tuple:
    complete = [key for key, record in frames.items() if record.frame.annotation_status == "complete"]
    if not complete:
        why = "no complete frames: partial labels cannot score false positives"
        reasons.update(ap50=why, ap_50_95=why)
        return None, None
    ground_truth, results, _, _ = _coco_sets(frames, complete, by_frame, categories)
    if not any(a["iscrowd"] == 0 for a in ground_truth["annotations"]):
        why = "no non-ignored ground truth in complete frames"
        reasons.update(ap50=why, ap_50_95=why)
        return None, None
    if not results:
        return 0.0, 0.0  # COCO semantics: labelled objects and no detections give AP 0
    stats = _cocoeval(ground_truth, results).stats
    return float(stats[1]), float(stats[0])


def _matches(frames, by_frame, categories, iou_threshold, score_threshold):
    """Count COCOeval matches at one IoU threshold for detections scoring >= threshold."""
    keys = [key for key, record in frames.items() if record.frame.annotation_status != "unlabeled"]
    ground_truth, results, gt_width, key_of = _coco_sets(frames, keys, by_frame, categories)
    counts = {"tp_complete": 0, "fp_complete": 0, "fn_complete": 0, "tp_all": 0, "gt_all": 0,
              "ignored_detections": 0}
    bins = {name: {"support": 0, "matched": 0} for name, _ in WIDTH_BINS}
    fp_by_frame = dict.fromkeys(keys, 0)
    gt_counted = set()
    evaluation = _cocoeval(ground_truth, results, [iou_threshold]) if results else None
    for entry in (evaluation.evalImgs if evaluation else []):
        if entry is None or entry["aRng"] != evaluation.params.areaRng[0]:
            continue
        key = key_of[entry["image_id"]]
        complete = frames[key].frame.annotation_status == "complete"
        scores = dict(zip(entry["dtIds"], entry["dtScores"]))
        for number, score in enumerate(entry["dtScores"]):
            if score < score_threshold:
                continue
            if entry["dtIgnore"][0][number]:
                counts["ignored_detections"] += 1
            elif entry["dtMatches"][0][number] > 0:
                counts["tp_complete"] += complete
            elif complete:
                counts["fp_complete"] += 1
                fp_by_frame[key] += 1
        for number, gt_id in enumerate(entry["gtIds"]):
            if entry["gtIgnore"][number]:
                continue
            gt_counted.add(gt_id)
            match = int(entry["gtMatches"][0][number])
            found = match > 0 and scores[match] >= score_threshold
            name = width_bin(gt_width[gt_id])
            bins[name]["support"] += 1
            bins[name]["matched"] += found
            counts["gt_all"] += 1
            counts["tp_all"] += found
            if complete and not found:
                counts["fn_complete"] += 1
    # Labelled objects on frames without any detection never reach COCO: count them missed.
    for annotation in ground_truth["annotations"]:
        if annotation["iscrowd"] or annotation["id"] in gt_counted:
            continue
        key = key_of[annotation["image_id"]]
        name = width_bin(gt_width[annotation["id"]])
        bins[name]["support"] += 1
        counts["gt_all"] += 1
        if frames[key].frame.annotation_status == "complete":
            counts["fn_complete"] += 1
    return counts, bins, fp_by_frame


def evaluate_detection(truth, detections, *, iou_threshold: float = 0.5,
                       score_threshold: float | None = None,
                       max_detections_per_frame: int = COCO_MAX_DETECTIONS,
                       check_versions: bool = True) -> dict:
    """COCO AP plus threshold metrics, width-bin recall and negative-frame errors."""
    if not (isinstance(iou_threshold, float) and 0 < iou_threshold <= 1):
        raise EvaluationError("iou_threshold must be a float in (0, 1]")
    if score_threshold is not None and not (isinstance(score_threshold, (int, float))
                                            and 0 <= score_threshold <= 1):
        raise EvaluationError("score_threshold must be None or in [0, 1]")
    if type(max_detections_per_frame) is not int or not 1 <= max_detections_per_frame <= COCO_MAX_DETECTIONS:
        raise EvaluationError(f"max_detections_per_frame must be an integer in [1, {COCO_MAX_DETECTIONS}]")
    versions = require_pinned_evaluators() if check_versions else evaluator_versions()
    frames, by_frame, dropped = _index(truth, detections, max_detections_per_frame)
    categories = sorted({item.class_id + 1 for r in frames.values() for item in r.objects}
                        | {d.class_id + 1 for items in by_frame.values() for d in items})
    status = [record.frame.annotation_status for record in frames.values()]
    negatives = [key for key, record in frames.items() if record.is_negative]
    unlabeled = {key for key, record in frames.items() if record.frame.annotation_status == "unlabeled"}
    labelled = [item for key, r in frames.items() if key not in unlabeled for item in r.objects]
    reasons: dict[str, str] = {}
    ap50, ap = _average_precision(frames, by_frame, categories, reasons)
    report = {
        "evaluator": {"name": "pycocotools COCOeval (bbox)", "versions": versions},
        "iou_threshold": iou_threshold, "score_threshold": score_threshold,
        "max_detections_per_frame": max_detections_per_frame,
        "frames": {"complete": status.count("complete"), "partial": status.count("partial"),
                   "unlabeled_excluded": status.count("unlabeled"), "negative": len(negatives)},
        "support": {"ground_truth": sum(not i.ignore for i in labelled),
                    "ignored_ground_truth": sum(i.ignore for i in labelled),
                    "detections": sum(len(v) for k, v in by_frame.items() if k not in unlabeled),
                    "detections_on_unlabeled_frames": sum(len(v) for k, v in by_frame.items() if k in unlabeled),
                    "detections_dropped_by_budget": dropped},
        "ap50": ap50, "ap_50_95": ap50 if ap50 is None else ap,
    }
    names = ("precision", "recall", "f1", "true_positives", "false_positives", "false_negatives",
             "recall_including_partial", "false_positives_per_complete_frame")
    if score_threshold is None:
        why = "no validation-selected score threshold was supplied"
        report.update(dict.fromkeys(names))
        reasons.update(dict.fromkeys(names, why))
        bins = {name: {"support": None, "recall": None, "availability_reason": {
            "support": why, "recall": why}} for name, _ in WIDTH_BINS}
        report["negative_frames"] = {"count": len(negatives), "with_false_positives": None,
                                     "false_positives_per_negative_frame": None,
                                     "availability_reason": {"with_false_positives": why,
                                                             "false_positives_per_negative_frame": why}}
    else:
        counts, raw_bins, fp_by_frame = _matches(frames, by_frame, categories, iou_threshold,
                                                 score_threshold)
        tp, fp, fn = counts["tp_complete"], counts["fp_complete"], counts["fn_complete"]
        complete = status.count("complete")
        no_complete = "no complete frames: partial labels cannot score false positives"
        precision = ratio(tp, tp + fp, "precision", reasons,
                          no_complete if complete == 0 else "no detections at or above the threshold")
        recall = ratio(tp, tp + fn, "recall", reasons,
                       no_complete if complete == 0 else "no non-ignored ground truth in complete frames")
        f1 = None
        if precision is None or recall is None:
            reasons["f1"] = "precision or recall is unavailable"
        else:
            f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        report.update(precision=precision, recall=recall, f1=f1, true_positives=tp,
                      false_positives=fp, false_negatives=fn,
                      recall_including_partial=ratio(counts["tp_all"], counts["gt_all"],
                                                     "recall_including_partial", reasons,
                                                     "no non-ignored labelled ground truth"),
                      false_positives_per_complete_frame=ratio(fp, complete,
                                                               "false_positives_per_complete_frame",
                                                               reasons, no_complete))
        if complete == 0:
            for name in ("true_positives", "false_positives", "false_negatives"):
                report[name] = None
                reasons[name] = no_complete
        report["support"]["detections_matched_to_ignored_regions"] = counts["ignored_detections"]
        bins = {}
        for name, _ in WIDTH_BINS:
            entry = {"support": raw_bins[name]["support"], "availability_reason": {}}
            entry["recall"] = ratio(raw_bins[name]["matched"], entry["support"], "recall",
                                    entry["availability_reason"],
                                    "zero support: no non-ignored labelled objects in this bin")
            bins[name] = entry
        negative_fp = [fp_by_frame.get(key, 0) for key in negatives]
        negative_reasons = {}
        report["negative_frames"] = {
            "count": len(negatives), "with_false_positives": sum(n > 0 for n in negative_fp),
            "false_positives_per_negative_frame": ratio(sum(negative_fp), len(negatives),
                                                        "false_positives_per_negative_frame",
                                                        negative_reasons,
                                                        "no annotated negative frames"),
            "availability_reason": negative_reasons}
    report["width_bins"] = bins
    report["availability_reason"] = reasons
    return report
