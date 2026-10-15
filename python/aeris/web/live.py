"""Live detection loop: camera / stream / paced file -> detect -> track -> forecast -> MJPEG.

Timestamps are seconds since the session started and strictly increasing. Paced files
use decoded presentation times relative to the first frame, with an explicit
frame/FPS fallback when PTS are unavailable; cameras and streams use monotonic
arrival time. If inference is slower than the
source, frames are dropped (camera: always the newest; file: skipped to stay real time)
and counted. Nothing here is a benchmark; rates are wall-clock on this machine.
"""
from __future__ import annotations

from threading import Condition, Event, Thread
import math
import time
from pathlib import Path

import cv2
import numpy as np

from aeris.cli import DEFAULT_MIN_SCORE, draw_states, heading_resolved, measurement_sigma_px
from aeris.estimation.camera_motion import CameraMotion
from aeris.estimation.camera import CameraModel, FrameContext
from aeris.estimation.trajectory import TrajectoryForecaster, forecast_vs_detections
from aeris.estimation.world_bearing import FixedCameraPose, world_bearing_from_pixel
from aeris.tracking.tracker import TENTATIVE, KalmanTracker
from aeris.video.pipeline import timestamp_plan


SCORE_WINDOW_S = 60.0  # scored history kept; bounds memory on endless streams
SCORE_EVERY_S = 1.0

def load_stages(profile: str):
    from aeris.detection import profiles
    return profiles.build(profile)


def fixed_camera_stability(motion: CameraMotion, estimated: bool, width: int, height: int) -> bool | None:
    """Whether background evidence still supports the original fixed camera pose.

    The first frame and failed estimation are unknown. A transform that moves any
    frame corner by more than 1 original pixel is treated as movement. This is an
    untuned suppression threshold, not a measured mount-stability guarantee.
    """
    if motion.counters["frames"] < 2 or not estimated:
        return None
    corners = ((0, 0), (width, 0), (0, height), (width, height))
    for x, y in corners:
        out = motion.to_ref @ np.array([x, y, 1.0])
        if not np.all(np.isfinite(out)) or abs(out[2]) < 1e-12:
            return None
        if math.hypot(out[0] / out[2] - x, out[1] / out[2] - y) > 1.0:
            return False
    return True


class LiveSession:
    def __init__(self, source, stages, *, paced: bool, label: str,
                 camera: CameraModel | None = None, pose: FixedCameraPose | None = None,
                 current_zoom: float | None = None):
        self.source, self.stages, self.paced, self.label = source, stages, paced, label
        self.camera = camera or CameraModel(None)
        self.pose = pose
        self.current_zoom = current_zoom
        self.cond = Condition()
        self.jpeg: bytes | None = None
        self.seq = 0
        self.stats = {"state": "starting", "message": "Opening source…", "frames": 0, "dropped": 0,
                      "fps": None, "latency_ms": None, "detections": 0, "tracks": 0, "source": label}
        self._stop = Event()
        self._thread = Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    @property
    def active(self) -> bool:
        return self._thread.is_alive()

    def _set(self, **values):
        with self.cond:
            self.stats.update(values)
            self.cond.notify_all()

    def _run(self):
        capture = cv2.VideoCapture(self.source)
        try:
            if not capture.isOpened():
                self._set(state="failed", message=f"Cannot open source: {self.label}")
                return
            fps = capture.get(cv2.CAP_PROP_FPS) if self.paced else 0.0
            pts, timestamp_source, timestamp_reason = (
                timestamp_plan(Path(self.source), fps, "pts_or_inferred") if self.paced
                else (None, "monotonic_arrival", None))
            tracker = KalmanTracker("live")
            # Forecasts run in reference-frame (scene-relative) pixels so camera pans do not
            # masquerade as aircraft motion; everything shown is mapped back to the current frame.
            forecaster = TrajectoryForecaster("live", "constant_velocity")
            motion = CameraMotion()
            start = time.monotonic()
            last_t, index, position, ema = -1.0, 0, 0, None
            seen_n: dict[str, int] = {}
            measured: dict[str, list] = {}
            issued: list = []
            score, last_score_t = None, -1e9
            pose_invalidated = False
            self._set(state="running", message="Live", timestamp_source=timestamp_source,
                      timestamp_reason=timestamp_reason)
            while not self._stop.is_set():
                if self.paced and pts is not None and position < len(pts):
                    elapsed = time.monotonic() - start
                    next_time = pts[position] - pts[0]
                    if elapsed < next_time:
                        time.sleep(min(next_time - elapsed, 0.05))
                        continue
                    while position + 1 < len(pts) and pts[position + 1] - pts[0] <= elapsed:
                        if not capture.grab():
                            raise RuntimeError("decoder ended before probed PTS count")
                        position += 1
                        self.stats["dropped"] += 1
                elif self.paced and pts is None and fps > 0:
                    target = int((time.monotonic() - start) * fps)
                    if position > target:
                        time.sleep((position - target) / fps)
                        continue
                    while position < target and capture.grab():  # drop to stay real time
                        position += 1
                        self.stats["dropped"] += 1
                ok, frame = capture.read()
                if not ok:
                    self._set(state="ended" if self.paced else "failed",
                              message="End of file." if self.paced else "Source stopped delivering frames.")
                    return
                position += 1
                t0 = time.perf_counter()
                if pts is not None:
                    if position - 1 >= len(pts):
                        raise RuntimeError("decoded more frames than probed PTS count")
                    t = pts[position - 1] - pts[0]
                else:
                    t = (position - 1) / fps if self.paced else time.monotonic() - start
                t = max(t, last_t + 1e-6)
                last_t = t
                height, width = frame.shape[:2]
                prepared = self.stages.preprocess(frame)
                raw = self.stages.infer(prepared)
                found = self.stages.postprocess(raw, prepared, width=width, height=height,
                                                sequence_id="live", frame_index=index, timestamp_s=t)
                dets = [d for d in found if d.score >= DEFAULT_MIN_SCORE]
                states = tracker.step(t, found, high_score=DEFAULT_MIN_SCORE)
                motion_estimated = motion.update(frame, [d.bbox for d in found])
                stable = fixed_camera_stability(motion, motion_estimated, width, height)
                if motion.counters["frames"] > 1 and stable is not True:
                    pose_invalidated = True  # a later stable image cannot restore a surveyed pose
                if pose_invalidated:
                    stable = False
                to_view = np.linalg.inv(motion.to_ref)[:2, :2]      # reference -> current axes
                forecasts, panel = {}, []
                for s in states:
                    if s.observed and s.measurement_center_px is not None:
                        seen_n[s.track_id] = seen_n.get(s.track_id, 0) + 1
                        z = motion.to_reference(s.measurement_center_px)
                        measured.setdefault(s.track_id, []).append((t, z))
                        forecaster.update(s.track_id, t, z, measurement_sigma_px(s, tracker.cfg))
                for s in states:
                    if s.lifecycle != TENTATIVE:
                        recs = forecaster.forecast(s.track_id, t)
                        row = {"id": s.track_id, "lifecycle": s.lifecycle, "observations": seen_n.get(s.track_id, 0),
                               "speed_px_s": None, "heading_deg": None, "forecast": [],
                               "observed_bearing": None, "forecast_bearings": []}
                        bearing_args = dict(camera=self.camera, pose=self.pose, sequence_id="live",
                                            track_id=s.track_id, issue_time_s=t, frame_size=(width, height),
                                            context=FrameContext(zoom=self.current_zoom,
                                                                 camera_moved=stable is False),
                                            camera_stable=stable)
                        if s.observed and s.measurement_center_px is not None:
                            row["observed_bearing"] = world_bearing_from_pixel(
                                pixel=s.measurement_center_px, **bearing_args).to_dict()
                        if all(r.center for r in recs):  # velocity() only valid when forecasts exist
                            vx, vy = to_view @ np.array(forecaster.velocity(s.track_id, t)[0])
                            row["speed_px_s"] = math.hypot(vx, vy)
                            if heading_resolved(forecaster, s.track_id, t):
                                # same convention as tracks.csv (camera.image_heading): CCW from image right
                                row["heading_deg"] = math.degrees(math.atan2(-vy, vx)) % 360
                        for r in recs:
                            if r.center:  # sigma = 1-sigma position std dev at this horizon
                                x, y = motion.from_reference(r.center)
                                row["forecast"].append({"horizon_s": r.horizon_s, "x": x, "y": y,
                                                        "sigma_px": math.sqrt(max(r.covariance[0][0], r.covariance[1][1]))})
                                row["forecast_bearings"].append(world_bearing_from_pixel(
                                    pixel=(x, y), horizon_s=r.horizon_s, **bearing_args).to_dict())
                            if r.horizon_s == 1.0 and r.center and row["heading_deg"] is not None:
                                forecasts[s.track_id] = motion.from_reference(r.center)
                        panel.append(row)
                        issued.extend(recs)
                if t - last_score_t >= SCORE_EVERY_S:
                    last_score_t, cutoff = t, t - SCORE_WINDOW_S
                    issued = [p for p in issued if p.issue_time_s >= cutoff]
                    measured = {k: v for k, v in ((k, [m for m in v if m[0] >= cutoff]) for k, v in measured.items()) if v}
                    score = forecast_vs_detections(measured, issued, models=["constant_velocity"])
                draw_states(frame, [(s, forecasts.get(s.track_id)) for s in states])
                dt = time.perf_counter() - t0
                ema = dt if ema is None else 0.9 * ema + 0.1 * dt
                cv2.putText(frame, f"LIVE t={t:6.1f}s  {1 / ema:4.1f} fps  det={len(dets)}",
                            (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                index += 1
                with self.cond:
                    if ok:
                        self.jpeg, self.seq = buf.tobytes(), self.seq + 1
                    self.stats.update(frames=index, fps=1 / ema, latency_ms=dt * 1000,
                                      detections=len(dets), panel=panel, media_t=t, score=score,
                                      camera_motion=dict(motion.counters),
                                      tracks=sum(s.lifecycle != TENTATIVE for s in states))
                    self.cond.notify_all()
        except Exception as exc:  # surfaced to the page; the loop must not die silently
            self._set(state="failed", message=f"Live loop stopped: {exc}")
        finally:
            capture.release()
            if self.stats["state"] in {"starting", "running"}:
                self._set(state="stopped", message="Stopped.")

    def frames(self):
        """Yield JPEG bytes as they are produced; ends when the session is no longer running."""
        seen = 0
        while True:
            with self.cond:
                self.cond.wait_for(lambda: self.seq != seen or self.stats["state"] not in
                                   {"starting", "running"}, timeout=2)
                if self.seq != seen and self.jpeg:
                    seen, jpeg = self.seq, self.jpeg
                elif self.stats["state"] not in {"starting", "running"}:
                    return
                else:
                    continue
            yield jpeg
