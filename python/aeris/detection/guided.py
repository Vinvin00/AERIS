"""Tracker-guided full-resolution crops (AERIS-034, PLAN-011).

Each frame, after the normal frame pass, the detector also runs on up to ``MAX_CROPS``
model-input-sized windows cut at native resolution around confirmed or lost tracks. A window is
centred on the track's position in its last emitted ``TrackState`` propagated by that state's
velocity to the current timestamp, so it uses past information only. A small, distant aircraft
that the downscaled frame pass misses is seen at full resolution inside the window.

Crop detections are *extend-only*: the tracker offers them only to confirmed or lost tracks still
unmatched after its high- and low-score passes, so a crop can never start or confirm a track. Boxes
that touch an interior window edge are dropped as fragments (as in tiling), crop boxes are merged by
NMS, and a crop box overlapping a frame-pass box with IoU >= 0.5 is dropped as a duplicate.
"""
from __future__ import annotations

import numpy as np

from aeris.data.contracts import BBox, DetectionRecord
from aeris.tracking.tracker import CONFIRMED, LOST

MAX_CROPS = 4          # pre-registered in PLAN-011
DUPLICATE_IOU = 0.5    # pre-registered in PLAN-011 [Q3]


def crop_windows(states, timestamp_s: float, width: int, height: int, size: int,
                 max_crops: int = MAX_CROPS) -> list[tuple[str, int, int, int, int]]:
    """``[(track_id, x0, y0, w, h)]`` for the most recently observed confirmed or lost tracks
    (ties by ``track_id``). Windows are ``size`` square (smaller only when the frame is), shifted
    to lie inside the frame."""
    eligible = sorted((s for s in states if s.lifecycle in (CONFIRMED, LOST)),
                      key=lambda s: (-s.last_measurement_time_s, s.track_id))
    cw, ch = min(size, width), min(size, height)
    out = []
    for s in eligible[:max_crops]:
        dt = timestamp_s - s.timestamp_s
        if dt < 0:
            raise ValueError("crop state is from the future")
        cx = s.center[0] + s.velocity_px_s[0] * dt
        cy = s.center[1] + s.velocity_px_s[1] * dt
        x0 = int(round(min(max(cx - cw / 2, 0.0), width - cw)))
        y0 = int(round(min(max(cy - ch / 2, 0.0), height - ch)))
        out.append((s.track_id, x0, y0, cw, ch))
    return out


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def crop_detections(image: np.ndarray, windows, detect_window, frame_detections, *, sequence_id: str,
                    frame_index: int, timestamp_s: float, provenance: str, class_mapping_version: str,
                    nms, nms_iou: float) -> list[DetectionRecord]:
    """Run ``detect_window(crop) -> [(xyxy_in_crop, class_id, score)]`` on each window and return
    original-frame detections after fragment, NMS and frame-pass duplicate filtering."""
    height, width = image.shape[:2]
    found = []
    for _, x0, y0, w, h in windows:
        m = 0.02 * max(w, h)
        for box, class_id, score in detect_window(image[y0:y0 + h, x0:x0 + w]):
            if ((x0 > 0 and box[0] <= m) or (x0 + w < width and box[2] >= w - m)
                    or (y0 > 0 and box[1] <= m) or (y0 + h < height and box[3] >= h - m)):
                continue     # cut by an interior window edge: a fragment of something larger
            found.append(([box[0] + x0, box[1] + y0, box[2] + x0, box[3] + y0], class_id, float(score)))
    if not found:
        return []
    keep = nms(np.array([f[0] for f in found], dtype=np.float32),
               np.array([f[2] for f in found], dtype=np.float32), nms_iou)
    frame_boxes = [(d.bbox.x_min, d.bbox.y_min, d.bbox.x_max, d.bbox.y_max) for d in frame_detections]
    out = []
    for k in keep:
        box, class_id, score = found[k]
        if any(_iou(box, f) >= DUPLICATE_IOU for f in frame_boxes):
            continue
        box = [min(max(box[0], 0.0), width), min(max(box[1], 0.0), height),
               min(max(box[2], 0.0), width), min(max(box[3], 0.0), height)]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        out.append(DetectionRecord(1, sequence_id, frame_index, timestamp_s, width, height, BBox(*box),
                                   class_id, class_mapping_version, score, provenance, {}))
    return out


class GuidedCrops:
    """Binds a ``BaselineStages`` detector to :func:`crop_windows` and :func:`crop_detections`."""

    def __init__(self, stages, max_crops: int = MAX_CROPS):
        self.stages, self.max_crops = stages, max_crops
        cfg = stages.config
        self.size = cfg["input"]["size"]
        self.provenance = (f"{cfg['model']['name']}@{cfg['model']['sha256'][:12]}/"
                           f"{stages.runner.name}+guided-crop")
        self.crops_run = 0

    def _detect_window(self, crop):
        b, cfg = self.stages._b, self.stages.config
        tensor, ratio = b.preprocess(crop, cfg["input"]["size"], cfg["input"]["pad_value"])
        raw = self.stages.runner(tensor)
        return [(box, class_id, score) for box, class_id, score, _ in
                b.postprocess(raw, ratio, crop.shape[1], crop.shape[0], cfg)]

    def __call__(self, image, states, frame_detections, *, sequence_id, frame_index, timestamp_s):
        height, width = image.shape[:2]
        windows = crop_windows(states, timestamp_s, width, height, self.size, self.max_crops)
        self.crops_run += len(windows)
        cfg = self.stages.config
        return crop_detections(image, windows, self._detect_window, frame_detections,
                               sequence_id=sequence_id, frame_index=frame_index, timestamp_s=timestamp_s,
                               provenance=self.provenance, class_mapping_version=cfg["class_mapping_version"],
                               nms=self.stages._b.nms, nms_iou=cfg["postprocess"]["nms_iou"]), len(windows)
