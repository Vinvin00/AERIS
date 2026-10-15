"""Live detection loop: camera / stream / paced file -> detect -> track -> forecast -> MJPEG.

Timestamps are seconds since the session started and strictly increasing. Paced files
use media time (frame position / container fps) so inference jitter does not turn into
apparent motion; cameras and streams use monotonic arrival time. If inference is slower than the
source, frames are dropped (camera: always the newest; file: skipped to stay real time)
and counted. Nothing here is a benchmark; rates are wall-clock on this machine.
"""
from __future__ import annotations

from threading import Condition, Event, Thread
import time

import cv2

from aeris.cli import DEFAULT_MIN_SCORE, draw_states, heading_resolved, measurement_sigma_px
from aeris.estimation.trajectory import TrajectoryForecaster
from aeris.tracking.tracker import TENTATIVE, KalmanTracker


TILES = 2  # 2x2 crops + full frame: distant aircraft vanish at 416 px full-frame (hkia-early: 0 vs 6/6 probed frames)


def load_stages():
    from aeris.detection import baseline
    from aeris.video import pipeline
    config = baseline.load_config()
    runner = baseline.OnnxRuntimeRunner(baseline.verified_model_path(config))
    return pipeline.BaselineStages(config, runner, tiles=TILES)


class LiveSession:
    def __init__(self, source, stages, *, paced: bool, label: str):
        self.source, self.stages, self.paced, self.label = source, stages, paced, label
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
            tracker = KalmanTracker("live")
            forecaster = TrajectoryForecaster("live", "constant_velocity")
            start = time.monotonic()
            last_t, index, position, ema = -1.0, 0, 0, None
            self._set(state="running", message="Live")
            while not self._stop.is_set():
                if self.paced and fps > 0:
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
                t = (position - 1) / fps if fps > 0 else time.monotonic() - start
                t = max(t, last_t + 1e-6)
                last_t = t
                height, width = frame.shape[:2]
                prepared = self.stages.preprocess(frame)
                raw = self.stages.infer(prepared)
                dets = [d for d in self.stages.postprocess(
                    raw, prepared, width=width, height=height, sequence_id="live",
                    frame_index=index, timestamp_s=t) if d.score >= DEFAULT_MIN_SCORE]
                states = tracker.step(t, dets)
                forecasts = {}
                for s in states:
                    if s.observed and s.measurement_center_px is not None:
                        forecaster.update(s.track_id, t, s.measurement_center_px,
                                          measurement_sigma_px(s, tracker.cfg))
                for s in states:
                    if s.lifecycle != TENTATIVE:
                        for r in forecaster.forecast(s.track_id, t):
                            if r.horizon_s == 1.0 and r.center and heading_resolved(forecaster, s.track_id, t):
                                forecasts[s.track_id] = r.center
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
                                      detections=len(dets),
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
