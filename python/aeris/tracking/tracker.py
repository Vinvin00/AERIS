"""Timestamp-aware constant-velocity Kalman tracker with gated Hungarian association.

State order ``[cx, cy, vx, vy, w, h]`` in original-frame pixels and px/s. Prediction
and update use the actual delta time to each detection timestamp; the tracker only
ever sees the current and past detections. Parameters in ``configs/tracker/`` are
engineering defaults, not tuned on real video.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from aeris.data.contracts import BBox, DetectionRecord

CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "tracker" / "baseline.json"
TENTATIVE, CONFIRMED, LOST = "tentative", "confirmed", "lost"
_BIG = 1e9


class TrackerError(ValueError):
    """Inputs or numerical state violate the tracker contract."""


def load_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(Path(path).read_text())


@dataclass(frozen=True)
class TrackState:
    sequence_id: str
    track_id: str
    class_id: int
    timestamp_s: float
    lifecycle: str
    center: tuple[float, float]
    velocity_px_s: tuple[float, float]
    size: tuple[float, float]
    bbox: BBox | None
    covariance: tuple[tuple[float, ...], ...]
    age_s: float
    time_since_seen_s: float
    last_measurement_time_s: float
    observed: bool
    availability_reason: dict = field(default_factory=dict)
    measurement_center_px: tuple[float, float] | None = None


@dataclass
class _Track:
    track_id: str
    class_id: int
    x: np.ndarray
    p: np.ndarray
    born_s: float
    last_seen_s: float
    hits: int = 1
    lifecycle: str = TENTATIVE
    observed: bool = True
    measurement_center_px: tuple[float, float] | None = None


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of ``[cx, cy, w, h]`` boxes."""
    ax0, ax1, ay0, ay1 = a[0] - a[2] / 2, a[0] + a[2] / 2, a[1] - a[3] / 2, a[1] + a[3] / 2
    bx0, bx1, by0, by1 = b[0] - b[2] / 2, b[0] + b[2] / 2, b[1] - b[3] / 2, b[1] + b[3] / 2
    inter = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


class KalmanTracker:
    """One tracker per sequence. Call :meth:`step` once per frame in time order."""

    H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]], dtype=float)

    def __init__(self, sequence_id: str, config: dict | None = None):
        self.sequence_id = sequence_id
        self.cfg = load_config() if config is None else config
        if not isinstance(self.cfg, dict):
            raise TrackerError("tracker config must be a JSON object")
        for key in ("accel_sigma_px_s2", "size_sigma_px_sqrt_s", "initial_velocity_sigma_px_s",
                    "r_position_frac", "r_size_frac", "r_floor_px", "gate_chi2", "confirm_hits",
                    "max_coast_s", "tentative_timeout_s", "duplicate_iou", "birth_suppress_iou"):
            value = self.cfg.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value) or value <= 0:
                raise TrackerError(f"config {key} must be a positive finite number")
        # Optional AERIS-033 keys; their defaults reproduce the AERIS-009 model exactly.
        frac = self.cfg.get("size_sigma_frac_sqrt_s", 0.0)
        if isinstance(frac, bool) or not isinstance(frac, (int, float)) or not math.isfinite(frac) or frac < 0:
            raise TrackerError("config size_sigma_frac_sqrt_s must be a finite number >= 0")
        if self.cfg.get("gate_mode", "full") not in ("full", "position"):
            raise TrackerError("config gate_mode must be 'full' or 'position'")
        if self.cfg.get("gate_mode") == "position":
            g = self.cfg.get("gate_chi2_position")
            if isinstance(g, bool) or not isinstance(g, (int, float)) or not math.isfinite(g) or g <= 0:
                raise TrackerError("config gate_chi2_position must be a positive finite number")
        self._tracks: list[_Track] = []
        self._next_id = 1
        self._time: float | None = None
        self._size: tuple[int, int] | None = None
        self.counters = {"steps": 0, "births": 0, "deleted": 0, "duplicates_dropped": 0,
                         "births_suppressed": 0, "gated_pairs": 0, "low_score_matches": 0}

    # -- model ---------------------------------------------------------------
    def _transition(self, dt: float, size=(0.0, 0.0)):
        """``size`` (w, h) scales the optional size-proportional noise ``size_sigma_frac_sqrt_s``:
        a box that grows or shrinks fast as an aircraft closes in is then not gated out."""
        f = np.eye(6)
        f[0, 2] = f[1, 3] = dt
        q = np.zeros((6, 6))
        a2 = self.cfg["accel_sigma_px_s2"] ** 2
        block = a2 * np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
        for pos, vel in ((0, 2), (1, 3)):
            q[np.ix_([pos, vel], [pos, vel])] = block
        frac = self.cfg.get("size_sigma_frac_sqrt_s", 0.0)
        q[4, 4] = (self.cfg["size_sigma_px_sqrt_s"] ** 2 + (frac * size[0]) ** 2) * dt
        q[5, 5] = (self.cfg["size_sigma_px_sqrt_s"] ** 2 + (frac * size[1]) ** 2) * dt
        return f, q

    def _measurement_noise(self, w: float, h: float) -> np.ndarray:
        floor = self.cfg["r_floor_px"]
        pos, size = self.cfg["r_position_frac"], self.cfg["r_size_frac"]
        sd = [max(floor, pos * w), max(floor, pos * h), max(floor, size * w), max(floor, size * h)]
        return np.diag(np.square(sd))

    @staticmethod
    def _check(p: np.ndarray, label: str) -> None:
        if not np.all(np.isfinite(p)) or not np.allclose(p, p.T, rtol=1e-9, atol=1e-9):
            raise TrackerError(f"{label}: covariance not finite and symmetric")
        if np.linalg.eigvalsh(p).min() < -1e-8 * max(1.0, float(np.abs(p).max())):
            raise TrackerError(f"{label}: covariance not positive semi-definite")

    def _predict(self, track: _Track, dt: float) -> None:
        f, q = self._transition(dt, (track.x[4], track.x[5]))
        track.x = f @ track.x
        track.p = f @ track.p @ f.T + q
        track.p = (track.p + track.p.T) / 2
        track.x[4:] = np.maximum(track.x[4:], 1e-3)
        self._check(track.p, f"track {track.track_id} predict")

    def _update(self, track: _Track, z: np.ndarray, r: np.ndarray) -> None:
        s = self.H @ track.p @ self.H.T + r
        gain = track.p @ self.H.T @ np.linalg.inv(s)
        track.x = track.x + gain @ (z - self.H @ track.x)
        ikh = np.eye(6) - gain @ self.H
        track.p = ikh @ track.p @ ikh.T + gain @ r @ gain.T   # Joseph form
        track.p = (track.p + track.p.T) / 2
        track.x[4:] = np.maximum(track.x[4:], 1e-3)
        self._check(track.p, f"track {track.track_id} update")

    # -- public --------------------------------------------------------------
    def step(self, timestamp_s: float, detections: list[DetectionRecord],
             high_score: float = 0.0, extend_only: list[DetectionRecord] | None = None) -> list[TrackState]:
        """Advance to ``timestamp_s`` using only detections at this time (possibly none).

        Detections scoring below ``high_score`` are used only in a second association pass
        (ByteTrack-style) to keep confirmed or lost tracks measured; they never start a track
        or confirm a tentative one. The default 0.0 makes every detection high-score.

        ``extend_only`` detections (AERIS-034 guided crops) are offered in a third pass only to
        confirmed or lost tracks still unmatched after the first two; they never start or confirm
        a track. ``None`` (the default) leaves the previous behaviour untouched.
        """
        if not isinstance(high_score, (int, float)) or not 0 <= high_score <= 1:
            raise TrackerError("high_score must be in [0, 1]")
        if not isinstance(timestamp_s, (int, float)) or isinstance(timestamp_s, bool) \
                or not math.isfinite(timestamp_s) or timestamp_s < 0:
            raise TrackerError("timestamp_s must be finite and nonnegative")
        if self._time is not None and timestamp_s <= self._time:
            raise TrackerError(f"timestamp {timestamp_s} does not exceed previous {self._time}")
        for d in [*detections, *(extend_only or [])]:
            if d.sequence_id != self.sequence_id or d.timestamp_s != timestamp_s:
                raise TrackerError("detection sequence or timestamp does not match this step")
            if self._size is None:
                self._size = (d.width, d.height)
            elif self._size != (d.width, d.height):
                raise TrackerError("detection frame size changes within a sequence")
        dets = self._deduplicate(detections)
        dt = None if self._time is None else timestamp_s - self._time
        self._time = timestamp_s
        self.counters["steps"] += 1
        for track in self._tracks:
            self._predict(track, dt)
            track.observed = False
            track.measurement_center_px = None
        # Timeout in seconds, evaluated before association so a long gap cannot revive a track.
        alive = []
        for track in self._tracks:
            if timestamp_s - track.last_seen_s > self.cfg["max_coast_s"]:
                self.counters["deleted"] += 1
            else:
                alive.append(track)
        self._tracks = alive
        high = [d for d in dets if d.score >= high_score]
        low = [d for d in dets if d.score < high_score]
        matches, unmatched_tracks, unmatched_dets = self._associate(high)
        second = [ti for ti in unmatched_tracks if self._tracks[ti].lifecycle != TENTATIVE]
        low_matches, _, _ = self._associate(low, second)
        self.counters["low_score_matches"] += len(low_matches)
        rescued = {ti for ti, _ in low_matches}
        unmatched_tracks = [ti for ti in unmatched_tracks if ti not in rescued]
        pairs = [(ti, high[di]) for ti, di in matches] + [(ti, low[di]) for ti, di in low_matches]
        if extend_only is not None:
            extra = self._deduplicate(extend_only)
            third = [ti for ti in unmatched_tracks if self._tracks[ti].lifecycle != TENTATIVE]
            extend_matches, _, _ = self._associate(extra, third)
            self.counters["extend_only_matches"] = self.counters.get("extend_only_matches", 0) + len(extend_matches)
            extended = {ti for ti, _ in extend_matches}
            unmatched_tracks = [ti for ti in unmatched_tracks if ti not in extended]
            pairs += [(ti, extra[di]) for ti, di in extend_matches]
        for ti, det in pairs:
            track = self._tracks[ti]
            z = np.array([(det.bbox.x_min + det.bbox.x_max) / 2, (det.bbox.y_min + det.bbox.y_max) / 2,
                          det.bbox.x_max - det.bbox.x_min, det.bbox.y_max - det.bbox.y_min])
            self._update(track, z, self._measurement_noise(z[2], z[3]))
            track.last_seen_s, track.observed = timestamp_s, True
            track.measurement_center_px = (float(z[0]), float(z[1]))
            track.hits += 1
            if track.hits >= self.cfg["confirm_hits"] or track.lifecycle in (CONFIRMED, LOST):
                track.lifecycle = CONFIRMED
        for ti in unmatched_tracks:
            track = self._tracks[ti]
            if track.lifecycle == TENTATIVE:
                if timestamp_s - track.last_seen_s > self.cfg["tentative_timeout_s"]:
                    track.lifecycle = "dead"
            elif track.lifecycle == CONFIRMED:
                track.lifecycle = LOST
        dead = [t for t in self._tracks if t.lifecycle == "dead"]
        self.counters["deleted"] += len(dead)
        self._tracks = [t for t in self._tracks if t.lifecycle != "dead"]
        for di in unmatched_dets:
            self._maybe_birth(high[di], timestamp_s)
        return [self._state(t, timestamp_s) for t in self._tracks]

    # -- internals -----------------------------------------------------------
    def _deduplicate(self, detections):
        kept = []
        for det in sorted(detections, key=lambda d: -d.score):
            box = _box(det)
            if any(k.class_id == det.class_id and _iou(_box(k), box) >= self.cfg["duplicate_iou"]
                   for k in kept):
                self.counters["duplicates_dropped"] += 1
            else:
                kept.append(det)
        return kept

    def _associate(self, dets, track_ids=None):
        """Gated Hungarian matching of ``dets`` to ``track_ids`` (indices into the track list;
        all tracks by default). Returns (track index, det index) pairs and the unmatched of each."""
        track_ids = list(range(len(self._tracks))) if track_ids is None else list(track_ids)
        n_t, n_d = len(track_ids), len(dets)
        if not n_t or not n_d:
            return [], track_ids, list(range(n_d))
        cost = np.full((n_t, n_d), _BIG)
        position_only = self.cfg.get("gate_mode", "full") == "position"
        gate = self.cfg["gate_chi2_position"] if position_only else self.cfg["gate_chi2"]
        axes = slice(0, 2) if position_only else slice(0, 4)
        for i, track in enumerate(self._tracks[ti] for ti in track_ids):
            predicted = self.H @ track.x
            for j, det in enumerate(dets):
                if det.class_id != track.class_id:
                    continue
                box = _box(det)
                s = self.H @ track.p @ self.H.T + self._measurement_noise(box[2], box[3])
                innovation = (box - predicted)[axes]
                d2 = float(innovation @ np.linalg.solve(s[axes, axes], innovation))
                if d2 <= gate:
                    cost[i, j] = d2
                else:
                    self.counters["gated_pairs"] += 1
        rows, cols = linear_sum_assignment(cost)
        matches = [(track_ids[r], int(c)) for r, c in zip(rows, cols) if cost[r, c] < _BIG]
        matched_t, matched_d = {r for r, _ in matches}, {c for _, c in matches}
        return (matches, [ti for ti in track_ids if ti not in matched_t],
                [j for j in range(n_d) if j not in matched_d])

    def _maybe_birth(self, det, timestamp_s):
        box = _box(det)
        for track in self._tracks:
            if track.class_id == det.class_id and _iou(track.x[[0, 1, 4, 5]], box) >= self.cfg["birth_suppress_iou"]:
                self.counters["births_suppressed"] += 1
                return
        sd_v = self.cfg["initial_velocity_sigma_px_s"]
        r = self._measurement_noise(box[2], box[3])
        p = np.diag([r[0, 0], r[1, 1], sd_v**2, sd_v**2, r[2, 2], r[3, 3]])
        x = np.array([box[0], box[1], 0.0, 0.0, box[2], box[3]])
        self._tracks.append(_Track(f"t{self._next_id}", det.class_id, x, p, timestamp_s, timestamp_s,
                                   measurement_center_px=(float(box[0]), float(box[1]))))
        self._next_id += 1
        self.counters["births"] += 1

    def _state(self, track: _Track, timestamp_s: float) -> TrackState:
        cx, cy, vx, vy, w, h = (float(v) for v in track.x)
        width, height = self._size or (None, None)
        bbox, reason = None, {}
        if width is not None:
            x0, x1 = max(0.0, cx - w / 2), min(float(width), cx + w / 2)
            y0, y1 = max(0.0, cy - h / 2), min(float(height), cy + h / 2)
            if x1 > x0 and y1 > y0:
                bbox = BBox(x0, y0, x1, y1)
        if bbox is None:
            reason = {"bbox": "predicted box lies outside the frame or no frame size known yet"}
        return TrackState(self.sequence_id, track.track_id, track.class_id, timestamp_s, track.lifecycle,
                          (cx, cy), (vx, vy), (w, h), bbox,
                          tuple(tuple(float(v) for v in row) for row in track.p),
                          timestamp_s - track.born_s, timestamp_s - track.last_seen_s,
                          track.last_seen_s, track.observed, reason,
                          track.measurement_center_px)


def _box(det: DetectionRecord) -> np.ndarray:
    b = det.bbox
    return np.array([(b.x_min + b.x_max) / 2, (b.y_min + b.y_max) / 2, b.x_max - b.x_min, b.y_max - b.y_min])
