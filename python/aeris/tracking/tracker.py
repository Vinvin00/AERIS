"""Timestamp-aware Kalman tracking and gated Hungarian association (AERIS-009).

One `Tracker` instance follows one sequence. Each call to `step()` gives it every
detection of one frame (a `DetectionRecord`, possibly none) and that frame's
timestamp; it suppresses duplicate detections of the same object (IoU greedy NMS,
highest score kept), predicts every live track forward by the elapsed time, and solves
gated Hungarian assignment in two passes: already-confirmed tracks match first against
every detection, then tentative tracks match against whatever is left. It then updates
matched tracks, starts a tentative track for each still-unmatched detection, and
returns every live track's current `TrackState`. Tracks consume only the current and
past frames of their own sequence: `step()` refuses a timestamp that does not strictly
increase, so nothing can look ahead (CONTRACTS.md "Runtime boundaries").

State is `[cx, cy, vx, vy, w, h]` (CONTRACTS.md), a constant-velocity model for the box
centre plus a slowly-varying size. Measurements are `[cx, cy, w, h]`: width and height
are tracked in the state and matched into the gate and Kalman update, because it is
often the cheapest way to keep two crossing tracks' identities apart when their
predicted centres coincide but their sizes do not (see docs/tracking.md and ADR 009).

Process noise Q(dt) is a discretized white-noise-acceleration model for each of the x
and y position/velocity pairs, plus a size term whose variance grows linearly with dt;
both explicitly depend on the elapsed time, not the frame count. Measurement noise R is
fixed from configured standard deviations. CONTRACTS.md requires any mapping from
detector score to R to be "documented and validated"; no such validation exists yet
(the config's `scope_note` says so), so score is not used to modulate R at all.
Covariance updates use Joseph's form and are symmetrized and eigenvalue-clipped every
step so that floating-point drift cannot make a returned covariance indefinite.

Lifecycle is `tentative -> confirmed -> lost -> deleted`, in elapsed seconds, never
frames: a track becomes confirmed once its number of matched frames reaches
`min_hits_to_confirm` (it can never move back to tentative); a confirmed track that
goes unmatched becomes `lost` and flips back to `confirmed` the moment it matches
again; a track (tentative or lost) is deleted, dropped from the tracker's state and
left out of that step's return, once its time since last matched exceeds
`lost_timeout_s`.

Usage:
    tracker = Tracker("seq-1", config_from_dict(load_config()))
    for timestamp_s, detections in frames:                 # strictly increasing times
        states = tracker.step(timestamp_s, detections)      # every live TrackState
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

from aeris.data.contracts import BBox, DetectionRecord

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs" / "tracker" / "baseline.json"
STATE_DIM = 6    # [cx, cy, vx, vy, w, h]
MEASURE_DIM = 4  # [cx, cy, w, h]
CX, CY, VX, VY, W, H_ = range(STATE_DIM)
# Measurement matrix: rows/cols of the state a measurement observes directly.
H_MATRIX = np.zeros((MEASURE_DIM, STATE_DIM))
H_MATRIX[0, CX] = H_MATRIX[1, CY] = H_MATRIX[2, W] = H_MATRIX[3, H_] = 1.0
LIFECYCLES = ("tentative", "confirmed", "lost")
UNGATED_COST = 1e12  # far above any gated cost; keeps linear_sum_assignment well-posed
PSD_EPS = -1e-9      # eigenvalues no more negative than this are float noise, clipped to 0


class TrackingError(ValueError):
    """A tracker call violates its contract (mixed sequence, non-causal timestamp, ...)."""


@dataclass(frozen=True)
class TrackerConfig:
    """The parts of configs/tracker/*.json that change numbers."""

    accel_std_px_s2: float
    size_std_px_sqrt_s: float
    position_std_px: float
    size_std_px: float
    initial_velocity_std_px_s: float
    gate_false_alarm_rate: float
    duplicate_suppression_iou: float
    min_hits_to_confirm: int
    lost_timeout_s: float
    min_size_px: float

    @property
    def gate_chi2(self) -> float:
        """Mahalanobis^2 gate: reject a match beyond the configured false-alarm rate
        under MEASURE_DIM degrees of freedom (a correct match almost never exceeds it)."""
        return float(chi2.ppf(1.0 - self.gate_false_alarm_rate, MEASURE_DIM))


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    return json.loads(Path(path).read_text())


def config_from_dict(config: dict) -> TrackerConfig:
    process, measurement = config["process_noise"], config["measurement_noise"]
    return TrackerConfig(
        accel_std_px_s2=process["accel_std_px_s2"], size_std_px_sqrt_s=process["size_std_px_sqrt_s"],
        position_std_px=measurement["position_std_px"], size_std_px=measurement["size_std_px"],
        initial_velocity_std_px_s=config["initial_uncertainty"]["velocity_std_px_s"],
        gate_false_alarm_rate=config["association"]["gate_false_alarm_rate"],
        duplicate_suppression_iou=config["association"]["duplicate_suppression_iou"],
        min_hits_to_confirm=config["lifecycle"]["min_hits_to_confirm"],
        lost_timeout_s=config["lifecycle"]["lost_timeout_s"],
        min_size_px=config["geometry"]["min_size_px"])


@dataclass(frozen=True)
class TrackState:
    """One track's estimate at one timestamp (CONTRACTS.md "Track state").

    `covariance` is the full 6x6 state covariance (row-major nested tuples, px^2 and
    (px/s)^2 units). `observed` is True only on a step where a detection was matched to
    this track this frame; otherwise the state is a pure prediction.
    """

    sequence_id: str
    track_id: str
    timestamp_s: float
    bbox: BBox
    center: tuple[float, float]
    velocity: tuple[float, float]
    covariance: tuple[tuple[float, ...], ...]
    lifecycle: str
    age_s: float
    time_since_seen_s: float
    last_measurement_time_s: float | None
    observed: bool

    def __post_init__(self) -> None:
        if self.lifecycle not in LIFECYCLES:
            raise TrackingError(f"lifecycle: expected one of {LIFECYCLES}")
        if type(self.observed) is not bool:
            raise TrackingError("observed: expected bool")
        if not isinstance(self.bbox, BBox):
            raise TrackingError("bbox: expected BBox")


def _measurement(detection: DetectionRecord) -> np.ndarray:
    b = detection.bbox
    return np.array([(b.x_min + b.x_max) / 2, (b.y_min + b.y_max) / 2,
                     b.x_max - b.x_min, b.y_max - b.y_min])


def _iou(a: BBox, b: BBox) -> float:
    width = min(a.x_max, b.x_max) - max(a.x_min, b.x_min)
    height = min(a.y_max, b.y_max) - max(a.y_min, b.y_min)
    if width <= 0 or height <= 0:
        return 0.0
    inter = width * height
    area_a = (a.x_max - a.x_min) * (a.y_max - a.y_min)
    area_b = (b.x_max - b.x_min) * (b.y_max - b.y_min)
    return inter / (area_a + area_b - inter)


def _suppress_duplicates(detections: list[DetectionRecord], threshold: float) -> list[DetectionRecord]:
    """Greedy IoU NMS within one frame's own detections, highest score first.

    Two detections of the same physical object in one frame (a duplicate, or two
    detectors' outputs merged upstream) are not the tracker's association problem to
    solve with gating alone: an established track's tight covariance can score a small
    residual to the true detection as *more* costly than a fresh, wide-covariance track
    spawned from the duplicate scores to it, handing the identity to the newcomer. This
    removes the redundant candidate before either sees the association step.
    """
    order = sorted(range(len(detections)), key=lambda i: detections[i].score, reverse=True)
    kept: list[int] = []
    for index in order:
        if all(_iou(detections[index].bbox, detections[j].bbox) <= threshold for j in kept):
            kept.append(index)
    keep = set(kept)
    return [d for i, d in enumerate(detections) if i in keep]


def _symmetrize_psd(matrix: np.ndarray) -> np.ndarray:
    """Force exact symmetry and clip float-noise-negative eigenvalues to 0."""
    matrix = (matrix + matrix.T) / 2
    values, vectors = np.linalg.eigh(matrix)
    if values.min() < PSD_EPS:
        raise TrackingError(f"covariance: eigenvalue {values.min()!r} is negative beyond float noise")
    values = np.clip(values, 0, None)
    return vectors @ np.diag(values) @ vectors.T


def _process_noise(dt: float, config: TrackerConfig) -> np.ndarray:
    """White-noise-acceleration Q for (cx, vx) and (cy, vy), plus a dt-scaled size term."""
    q = config.accel_std_px_s2 ** 2
    block = q * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
    Q = np.zeros((STATE_DIM, STATE_DIM))
    for pos, vel in ((CX, VX), (CY, VY)):
        for i, a in enumerate((pos, vel)):
            for j, b in enumerate((pos, vel)):
                Q[a, b] = block[i, j]
    size_var = (config.size_std_px_sqrt_s ** 2) * dt
    Q[W, W] = Q[H_, H_] = size_var
    return Q


def _transition(dt: float) -> np.ndarray:
    F = np.eye(STATE_DIM)
    F[CX, VX] = F[CY, VY] = dt
    return F


class _KalmanTrack:
    """Mutable internal Kalman state for one track; `TrackState` snapshots are read-only."""

    def __init__(self, track_id: str, timestamp_s: float, measurement: np.ndarray,
                config: TrackerConfig):
        self.track_id = track_id
        self.x = np.array([measurement[0], measurement[1], 0.0, 0.0, measurement[2], measurement[3]])
        self.P = np.diag([config.position_std_px ** 2, config.position_std_px ** 2,
                          config.initial_velocity_std_px_s ** 2, config.initial_velocity_std_px_s ** 2,
                          config.size_std_px ** 2, config.size_std_px ** 2]).astype(float)
        self._r = np.diag([config.position_std_px ** 2, config.position_std_px ** 2,
                           config.size_std_px ** 2, config.size_std_px ** 2]).astype(float)
        self.created_s = timestamp_s
        self.last_timestamp_s = timestamp_s
        self.last_measurement_time_s = timestamp_s
        self.time_since_seen_s = 0.0
        self.hits = 1
        self.confirmed = False
        self.observed = True

    def predict(self, timestamp_s: float, config: TrackerConfig) -> None:
        dt = timestamp_s - self.last_timestamp_s
        if dt <= 0:
            raise TrackingError(f"timestamp_s: must strictly increase (got dt={dt!r})")
        F = _transition(dt)
        self.x = F @ self.x
        self.P = _symmetrize_psd(F @ self.P @ F.T + _process_noise(dt, config))
        self.last_timestamp_s = timestamp_s
        self.time_since_seen_s += dt

    def innovation(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y = measurement - H_MATRIX @ self.x
        S = H_MATRIX @ self.P @ H_MATRIX.T + self._r
        return y, S

    def gated_cost(self, measurement: np.ndarray) -> float:
        y, S = self.innovation(measurement)
        return float(y @ np.linalg.solve(S, y))

    def update(self, measurement: np.ndarray, config: TrackerConfig) -> None:
        y, S = self.innovation(measurement)
        K = self.P @ H_MATRIX.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        joseph = np.eye(STATE_DIM) - K @ H_MATRIX
        self.P = _symmetrize_psd(joseph @ self.P @ joseph.T + K @ self._r @ K.T)
        self.last_measurement_time_s = self.last_timestamp_s
        self.time_since_seen_s = 0.0
        self.hits += 1
        self.observed = True
        if self.hits >= config.min_hits_to_confirm:
            self.confirmed = True

    def miss(self) -> None:
        self.observed = False

    def to_state(self, sequence_id: str, config: TrackerConfig) -> TrackState:
        cx, cy, vx, vy, w, h = (float(v) for v in self.x)
        w, h = max(w, config.min_size_px), max(h, config.min_size_px)
        # BBox requires a non-negative origin (CONTRACTS.md coordinates); a track whose
        # estimated centre sits close to (or, from noise, just past) the image's top-left
        # edge would otherwise produce cx - w/2 < 0. Shift the box to start at 0 rather
        # than shrink it, keeping the estimated width and height exact.
        x_min, y_min = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
        bbox = BBox(x_min, y_min, x_min + w, y_min + h)
        lifecycle = "tentative" if not self.confirmed else ("confirmed" if self.observed else "lost")
        return TrackState(
            sequence_id=sequence_id, track_id=self.track_id, timestamp_s=self.last_timestamp_s,
            bbox=bbox, center=(float(cx), float(cy)), velocity=(float(vx), float(vy)),
            covariance=tuple(tuple(float(v) for v in row) for row in self.P),
            lifecycle=lifecycle, age_s=self.last_timestamp_s - self.created_s,
            time_since_seen_s=self.time_since_seen_s,
            last_measurement_time_s=self.last_measurement_time_s, observed=self.observed)


class Tracker:
    """Kalman tracking and gated Hungarian association for one sequence."""

    def __init__(self, sequence_id: str, config: TrackerConfig):
        if not isinstance(sequence_id, str) or not sequence_id:
            raise TrackingError("sequence_id: expected nonempty string")
        self.sequence_id = sequence_id
        self.config = config
        self._tracks: dict[str, _KalmanTrack] = {}
        self._last_timestamp_s: float | None = None
        self._next_id = 1

    def _new_id(self) -> str:
        track_id = f"track-{self._next_id:06d}"
        self._next_id += 1
        return track_id

    def step(self, timestamp_s: float, detections: list[DetectionRecord]) -> list[TrackState]:
        """Advance by one frame; returns every live track's state, tentative ones included."""
        if not np.isfinite(timestamp_s):
            raise TrackingError("timestamp_s: expected a finite number")
        if self._last_timestamp_s is not None and timestamp_s <= self._last_timestamp_s:
            raise TrackingError(
                f"timestamp_s: must strictly increase within a sequence (previous "
                f"{self._last_timestamp_s!r}, got {timestamp_s!r}); the tracker never revisits "
                "an earlier time")
        for detection in detections:
            if not isinstance(detection, DetectionRecord):
                raise TrackingError("detections: expected DetectionRecord entries")
            if detection.sequence_id != self.sequence_id:
                raise TrackingError(
                    f"detections: sequence_id {detection.sequence_id!r} does not match "
                    f"this tracker's {self.sequence_id!r}")
            if detection.timestamp_s != timestamp_s:
                raise TrackingError("detections: timestamp_s does not match the step's timestamp_s")
        detections = _suppress_duplicates(detections, self.config.duplicate_suppression_iou)
        for track in self._tracks.values():
            track.predict(timestamp_s, self.config)
        self._last_timestamp_s = timestamp_s

        measurements = [_measurement(d) for d in detections]
        track_ids = list(self._tracks)
        matched_tracks: set[str] = set()
        remaining = list(range(len(measurements)))
        # Confirmed tracks get first refusal on every detection, then tentative ones compete
        # for what is left. Without this cascade, a track just spawned from a duplicate or a
        # false positive can occasionally look marginally cheaper (in Mahalanobis terms) than
        # an established track whose covariance has grown a little since its last match, and
        # would otherwise be able to steal the established track's next detection outright.
        for confirmed_only in (True, False):
            ids = [tid for tid in track_ids if self._tracks[tid].confirmed == confirmed_only]
            if not ids or not remaining:
                continue
            cost = np.full((len(ids), len(remaining)), UNGATED_COST)
            for i, track_id in enumerate(ids):
                track = self._tracks[track_id]
                for j, detection_index in enumerate(remaining):
                    value = track.gated_cost(measurements[detection_index])
                    if value <= self.config.gate_chi2:
                        cost[i, j] = value
            rows, cols = linear_sum_assignment(cost)
            used = set()
            for row, col in zip(rows, cols):
                if cost[row, col] < UNGATED_COST:
                    track_id = ids[row]
                    detection_index = remaining[col]
                    self._tracks[track_id].update(measurements[detection_index], self.config)
                    matched_tracks.add(track_id)
                    used.add(col)
            remaining = [d for j, d in enumerate(remaining) if j not in used]
        for track_id in track_ids:
            if track_id not in matched_tracks:
                self._tracks[track_id].miss()
        for detection_index in remaining:
            track_id = self._new_id()
            self._tracks[track_id] = _KalmanTrack(
                track_id, timestamp_s, measurements[detection_index], self.config)
        deleted = [track_id for track_id, track in self._tracks.items()
                  if track.time_since_seen_s > self.config.lost_timeout_s]
        for track_id in deleted:
            del self._tracks[track_id]
        return [track.to_state(self.sequence_id, self.config) for track in self._tracks.values()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the AERIS-009 tracker over a frames.jsonl/detections.jsonl pair.")
    parser.add_argument("frames", type=Path, help="FrameRecord JSON Lines, one sequence")
    parser.add_argument("detections", type=Path, help="DetectionRecord JSON Lines for those frames")
    parser.add_argument("--out", type=Path, required=True, help="output directory (not in Git)")
    args = parser.parse_args(argv)
    frames = [json.loads(line) for line in args.frames.read_text().splitlines()]
    detections_by_frame: dict[int, list[DetectionRecord]] = {}
    for line in args.detections.read_text().splitlines():
        record = DetectionRecord.from_json(line)
        detections_by_frame.setdefault(record.frame_index, []).append(record)
    frames.sort(key=lambda f: f["frame_index"])
    sequence_ids = {f["sequence_id"] for f in frames}
    if len(sequence_ids) != 1:
        print(f"frames: expected one sequence_id, found {sorted(sequence_ids)}", file=sys.stderr)
        return 2
    tracker = Tracker(next(iter(sequence_ids)), config_from_dict(load_config()))
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "tracks.jsonl", "w") as handle:
        for frame in frames:
            states = tracker.step(frame["timestamp_s"], detections_by_frame.get(frame["frame_index"], []))
            for state in states:
                handle.write(json.dumps({**state.__dict__, "frame_index": frame["frame_index"]},
                                        default=list) + "\n")
    print(json.dumps({"frames": len(frames), "sequence_id": tracker.sequence_id}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
