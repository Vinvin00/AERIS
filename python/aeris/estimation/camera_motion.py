"""Frame-to-frame camera (background) motion, so direction can be measured against the scene.

Each processed frame gets a 3x3 transform mapping its original-frame pixels into the pixels of
the first processed frame (the *reference*). Motion between consecutive processed frames is a
RANSAC similarity (translation, rotation, uniform scale) fitted to sparse Lucas-Kanade tracks of
background corners; detection boxes are masked out so aircraft do not count as background.
Estimation runs on a copy downscaled to ``max_side`` pixels for speed.

When too few background corners exist (a clear sky, a blank frame) the step is recorded as
failed and treated as *no camera motion*, which equals the uncompensated behaviour for that step;
it is not a measurement, and callers report the failure count. A *successful* fit can still be
wrong (moving water or clouds, undetected aircraft, parallax) and is then worse than not
estimating. Errors accumulate along the chain. Parameters are untuned engineering defaults.
"""
from __future__ import annotations

import cv2
import numpy as np


class CameraMotion:
    def __init__(self, max_side: int = 640, min_inliers: int = 20, box_margin: float = 0.2):
        self.max_side, self.min_inliers, self.box_margin = max_side, min_inliers, box_margin
        self.to_ref = np.eye(3)            # current frame -> reference pixels
        self.counters = {"frames": 0, "estimated": 0, "failed": 0}
        self._prev = None                  # (gray, mask, scale) of the previous processed frame

    def update(self, frame_bgr: np.ndarray, boxes=()) -> bool:
        """Advance to ``frame_bgr`` (``boxes``: BBox-like objects to exclude). Returns whether the
        motion from the previous frame was estimated; the first frame defines the reference."""
        height, width = frame_bgr.shape[:2]
        scale = min(1.0, self.max_side / max(height, width))
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if scale < 1.0:
            gray = cv2.resize(gray, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
        mask = np.full(gray.shape, 255, np.uint8)
        for b in boxes:
            mx, my = (b.x_max - b.x_min) * self.box_margin, (b.y_max - b.y_min) * self.box_margin
            cv2.rectangle(mask, (int((b.x_min - mx) * scale), int((b.y_min - my) * scale)),
                          (int(np.ceil((b.x_max + mx) * scale)), int(np.ceil((b.y_max + my) * scale))), 0, -1)
        self.counters["frames"] += 1
        prev, self._prev = self._prev, (gray, mask, scale)
        if prev is None:
            return True
        m = self._similarity(prev[0], prev[1], gray)
        if m is None:
            self.counters["failed"] += 1
            return False
        s = np.diag([scale, scale, 1.0])            # same scale every frame (fixed frame size)
        prev_to_cur = np.linalg.inv(s) @ np.vstack([m, [0, 0, 1]]) @ s
        self.to_ref = self.to_ref @ np.linalg.inv(prev_to_cur)
        self.counters["estimated"] += 1
        return True

    def _similarity(self, prev_gray, prev_mask, gray):
        corners = cv2.goodFeaturesToTrack(prev_gray, 400, 0.01, 7, mask=prev_mask, blockSize=7)
        if corners is None or len(corners) < self.min_inliers:
            return None
        moved, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, corners, None,
                                                    winSize=(21, 21), maxLevel=3)
        good = status.ravel() == 1
        if good.sum() < self.min_inliers:
            return None
        m, inliers = cv2.estimateAffinePartial2D(corners[good], moved[good], method=cv2.RANSAC,
                                                 ransacReprojThreshold=2.0, maxIters=2000)
        if m is None or inliers is None or int(inliers.sum()) < self.min_inliers:
            return None
        return m

    def to_reference(self, xy) -> tuple[float, float]:
        return apply_transform(self.to_ref, xy)

    def from_reference(self, xy) -> tuple[float, float]:
        return apply_transform(np.linalg.inv(self.to_ref), xy)


def apply_transform(h: np.ndarray, xy) -> tuple[float, float]:
    x, y, w = h @ np.array([xy[0], xy[1], 1.0])
    return float(x / w), float(y / w)
