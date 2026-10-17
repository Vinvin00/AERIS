"""Focused local web UI checks without downloading media or loading the model."""
from __future__ import annotations

from http.server import ThreadingHTTPServer
import io
import json
import math
import copy
from pathlib import Path
import tempfile
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import unittest
from unittest import mock

from aeris.web.server import DemoState, handler_for


class WebUITests(unittest.TestCase):
    def setUp(self):
        self.state = DemoState()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(self.state))
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)

    def request(self, path, data=None, *, origin=True, headers=None):
        headers = dict(headers or {})
        if data is not None and origin:
            headers["Origin"] = self.base
        request = Request(self.base + path, data=data, headers=headers)
        return urlopen(request, timeout=3)

    def test_page_and_sample_catalog(self):
        with self.request("/") as response:
            page = response.read().decode()
        self.assertIn("Analysis console", page)
        self.assertIn("Accuracy stays blank", page)
        with self.request("/api/samples") as response:
            samples = json.load(response)["samples"]
        self.assertEqual(len(samples), 7)
        self.assertTrue(all(s["page"].startswith("https://commons.wikimedia.org/") for s in samples))

    def test_post_requires_local_origin_and_rejects_unknown_sample(self):
        body = json.dumps({"slug": "unknown"}).encode()
        with self.assertRaises(HTTPError) as context:
            self.request("/api/select", body, origin=False)
        self.assertEqual(context.exception.code, 403)
        with self.assertRaises(HTTPError) as context:
            self.request("/api/select", body)
        self.assertEqual(context.exception.code, 400)

    def test_get_rejects_unexpected_host(self):
        with self.assertRaises(HTTPError) as context:
            self.request("/api/state", headers={"Host": "untrusted.example"})
        self.assertEqual(context.exception.code, 403)

    def test_upload_and_byte_range_preview(self):
        body = b"0123456789"
        with self.request("/api/upload", body, headers={"X-Filename": "own%20video.mp4"}) as response:
            state = json.load(response)
        self.assertEqual(state["selected_name"], "own video.mp4")
        self.assertEqual(state["status"], "ready")
        with self.request("/api/video", headers={"Range": "bytes=2-5"}) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.read(), b"2345")
        self.state.selected_video.unlink()

    def test_failed_upload_clears_previous_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            old = Path(directory) / "old.mp4"
            old.write_bytes(b"old")
            self.state.selected_video = old
            self.state.preview_video = old
            self.state.selected_name = "old.mp4"
            self.state.run_dir = Path(directory)
            self.state.status = "complete"
            with self.assertRaises(ValueError):
                self.state.select_upload("new.mp4", io.BytesIO(b"short"), 10)
            snapshot = self.state.snapshot()
            self.assertEqual(snapshot["status"], "failed")
            self.assertIsNone(snapshot["selected_name"])
            self.assertIsNone(snapshot["run_dir"])
            self.assertIsNone(snapshot["video_url"])
            self.assertIn("Could not prepare video", snapshot["message"])

    def test_run_requires_model_and_has_no_stale_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.mp4"
            path.write_bytes(b"video")
            self.state.selected_video = path
            self.state.selected_name = "clip.mp4"
            self.state.status = "ready"
            with mock.patch("aeris.web.server.shutil.which", return_value=None):
                with self.assertRaises(HTTPError) as context:
                    self.request("/api/run", b"{}")
            self.assertEqual(context.exception.code, 400)
            self.assertIsNone(self.state.snapshot()["metrics"])


if __name__ == "__main__":
    unittest.main()


class LiveTimingTests(unittest.TestCase):
    def test_paced_file_uses_irregular_presentation_times(self):
        import cv2
        import numpy as np
        from aeris.web.live import LiveSession

        class Stages:
            def __init__(self):
                self.times = []
            def preprocess(self, frame):
                return frame
            def infer(self, prepared):
                return None
            def postprocess(self, raw, prepared, *, timestamp_s, **_):
                self.times.append(timestamp_s)
                return []

        pts = [1.25, 1.30, 1.45, 1.50, 1.70, 1.75]
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "vfr-plan.avi")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (64, 48))
            for _ in pts:
                writer.write(np.zeros((48, 64, 3), np.uint8))
            writer.release()
            stages = Stages()
            with mock.patch("aeris.web.live.timestamp_plan", return_value=(pts, "decode_pts", None)):
                session = LiveSession(path, stages, paced=True, label="vfr-plan")
                session.start()
                session._thread.join(timeout=5)
        self.assertEqual(session.stats["state"], "ended")
        self.assertEqual(session.stats["timestamp_source"], "decode_pts")
        self.assertEqual(len(stages.times), len(pts))
        for actual, source in zip(stages.times, pts):
            self.assertAlmostEqual(actual, source - pts[0], places=9)

    def test_paced_file_uses_media_time_despite_slow_inference(self):
        import time
        import cv2
        import numpy as np
        from aeris.web.live import LiveSession

        class SlowStages:
            def __init__(self):
                self.times = []
            def preprocess(self, frame):
                return frame
            def infer(self, prepared):
                time.sleep(0.07)                     # slower than 20 fps; frames must be dropped
            def postprocess(self, raw, prepared, *, timestamp_s, **_):
                self.times.append(timestamp_s)
                return []

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "clip.avi")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (64, 48))
            for _ in range(12):
                writer.write(np.zeros((48, 64, 3), np.uint8))
            writer.release()
            stages = SlowStages()
            no_pts = (None, "inferred_fps", "forced: tests the fixed-fps fallback")
            with mock.patch("aeris.web.live.timestamp_plan", return_value=no_pts):
                session = LiveSession(path, stages, paced=True, label="clip")
                session.start()
                session._thread.join(timeout=10)
        self.assertEqual(session.stats["state"], "ended")
        self.assertEqual(session.stats["timestamp_source"], "inferred_fps")
        self.assertGreater(len(stages.times), 3)
        self.assertGreater(session.stats["dropped"], 0)
        for t in stages.times:                        # every timestamp is a frame time k / 20
            self.assertAlmostEqual(t * 20, round(t * 20), places=6)
        self.assertEqual(stages.times[0], 0.0)        # first frame is media time zero, not 1/fps
        self.assertTrue(all(b > a for a, b in zip(stages.times, stages.times[1:])))

    def test_forecast_sigma_follows_box_size(self):
        from types import SimpleNamespace
        from aeris.cli import measurement_sigma_px
        cfg = {"r_floor_px": 1.0, "r_position_frac": 0.05}
        self.assertEqual(measurement_sigma_px(SimpleNamespace(size=(200.0, 80.0)), cfg), 10.0)
        self.assertEqual(measurement_sigma_px(SimpleNamespace(size=(8.0, 4.0)), cfg), 1.0)


class LiveBearingTests(unittest.TestCase):
    def test_server_loads_operator_files_at_startup(self):
        from aeris.web.server import main

        calibration = {"camera_id": "fixture", "width": 64, "height": 48,
                       "K": [[80, 0, 32], [0, 80, 24], [0, 0, 1]],
                       "distortion_model": "none", "distortion_coefficients": [],
                       "calibration_id": "fixture-cal", "quality": "analytic",
                       "lens": "fixture", "zoom": 1.0}
        pose = {"schema_version": 1, "pose_id": "fixture-pose", "camera_id": "fixture",
                "calibration_id": "fixture-cal", "reference_id": "fixture-enu",
                "position_enu_m": [0, 0, 0],
                "rotation_camera_to_enu": [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
                "surveyed_at_utc": "2026-10-15T09:00:00Z", "fixed_mount": True}
        with tempfile.TemporaryDirectory() as tmp:
            cal_path, pose_path = Path(tmp) / "cal.json", Path(tmp) / "pose.json"
            cal_path.write_text(json.dumps(calibration))
            pose_path.write_text(json.dumps(pose))
            captured = []
            with mock.patch("aeris.web.server.handler_for", side_effect=lambda state: captured.append(state)), \
                 mock.patch("aeris.web.server.ThreadingHTTPServer") as server:
                server.return_value.server_port = 8765
                self.assertEqual(main(["--no-open", "--calibration", str(cal_path), "--pose",
                                       str(pose_path), "--current-zoom", "1"]), 0)
            self.assertEqual(captured[0].camera.calibration.calibration_id, "fixture-cal")
            self.assertEqual(captured[0].pose.pose_id, "fixture-pose")
            self.assertEqual(captured[0].current_zoom, 1.0)

    def test_stability_assessment_distinguishes_first_failed_and_moved_frames(self):
        import numpy as np
        from aeris.web.live import fixed_camera_stability

        motion = mock.Mock()
        motion.counters = {"frames": 1}
        motion.to_ref = np.eye(3)
        self.assertIsNone(fixed_camera_stability(motion, True, 64, 48))
        motion.counters["frames"] = 2
        self.assertIsNone(fixed_camera_stability(motion, False, 64, 48))
        self.assertTrue(fixed_camera_stability(motion, True, 64, 48))
        motion.to_ref[0, 2] = 2.0
        self.assertFalse(fixed_camera_stability(motion, True, 64, 48))

    def test_live_loop_separates_observed_and_causal_forecast_bearings(self):
        import cv2
        import numpy as np
        from aeris.data.contracts import BBox, DetectionRecord
        from aeris.estimation.camera import Calibration, CameraModel
        from aeris.estimation.world_bearing import FixedCameraPose
        from aeris.web.live import LiveSession

        calibration = Calibration.from_dict({
            "camera_id": "fixture", "width": 64, "height": 48,
            "K": [[80, 0, 32], [0, 80, 24], [0, 0, 1]],
            "distortion_model": "none", "distortion_coefficients": [],
            "calibration_id": "fixture-cal", "quality": "analytic", "lens": "fixture", "zoom": 1.0})
        pose = FixedCameraPose.from_dict({
            "schema_version": 1, "pose_id": "fixture-pose", "camera_id": "fixture",
            "calibration_id": "fixture-cal", "reference_id": "fixture-enu",
            "position_enu_m": [0, 0, 0],
            "rotation_camera_to_enu": [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
            "surveyed_at_utc": "2026-10-15T09:00:00Z", "fixed_mount": True})

        class Stages:
            def __init__(self, future_shift=0):
                self.future_shift = future_shift
                self.session = None
                self.prefix = None
            def preprocess(self, frame):
                return frame
            def infer(self, prepared):
                return None
            def postprocess(self, raw, prepared, *, frame_index, timestamp_s, **_):
                if frame_index == 8:
                    self.prefix = copy.deepcopy(self.session.stats.get("panel"))
                x = 17.0 + frame_index * 0.6 + (self.future_shift if frame_index >= 8 else 0)
                return [DetectionRecord(1, "live", frame_index, timestamp_s, 64, 48,
                                        BBox(x, 18, x + 8, 26), 4, "fixture-v1", 0.95,
                                        "analytic fixture", {})]

        def motion_class(fail=False, move=False):
            class Motion:
                def __init__(self):
                    self.to_ref = np.eye(3)
                    self.to_ref[0, 2] = -0.5  # nonidentity scene reference, within stability gate
                    self.counters = {"frames": 0, "estimated": 0, "failed": 0}
                def update(self, frame, boxes):
                    self.counters["frames"] += 1
                    if fail and self.counters["frames"] == 2:
                        self.counters["failed"] += 1
                        return False
                    if move:
                        self.to_ref[0, 2] = -2.0 if self.counters["frames"] == 5 else -0.5
                    self.counters["estimated"] += 1
                    return True
                def to_reference(self, xy):
                    return (xy[0] + self.to_ref[0, 2], xy[1])
                def from_reference(self, xy):
                    return (xy[0] - self.to_ref[0, 2], xy[1])
            return Motion

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "bearing.avi")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (64, 48))
            for _ in range(14):
                writer.write(np.zeros((48, 64, 3), np.uint8))
            writer.release()
            first_prefix = None
            for fail, future_shift, move, zoom in ((False, 0, False, 1.0), (False, 8, False, 1.0),
                                                   (True, 0, False, 1.0), (False, 0, True, 1.0),
                                                   (False, 0, False, 1.1)):
                with self.subTest(failed_motion=fail, future_shift=future_shift, move=move, zoom=zoom), \
                     mock.patch("aeris.web.live.CameraMotion", motion_class(fail, move)):
                    stages = Stages(future_shift)
                    session = LiveSession(path, stages, paced=True, label="fixture",
                                          camera=CameraModel(calibration), pose=pose, current_zoom=zoom)
                    stages.session = session
                    session.start()
                    session._thread.join(timeout=5)
                    self.assertEqual(session.stats["state"], "ended")
                    self.assertGreater(session.stats["frames"], 5)
                    self.assertTrue(stages.prefix and stages.prefix[0]["forecast_bearings"])
                    if not fail and not move and zoom == 1.0 and future_shift == 0:
                        first_prefix = stages.prefix
                    elif future_shift:
                        self.assertEqual(stages.prefix, first_prefix)  # later detections cannot alter past
                        continue
                    row = session.stats["panel"][0]
                    observed = row["observed_bearing"]
                    self.assertEqual(observed["issue_time_s"], session.stats["media_t"])
                    self.assertTrue(row["forecast_bearings"])
                    for b in row["forecast_bearings"]:
                        self.assertAlmostEqual(b["target_time_s"], b["issue_time_s"] + b["horizon_s"])
                        self.assertEqual(b["kind"], "forecast")
                    if fail or move or zoom != 1.0:
                        self.assertIsNone(observed["direction_enu"])
                        self.assertTrue(observed["availability_reason"])
                        self.assertTrue(all(b["direction_enu"] is None for b in row["forecast_bearings"]))
                    else:
                        self.assertIsNotNone(observed["azimuth_deg"])
                        self.assertTrue(all(b["direction_enu"] is not None for b in row["forecast_bearings"]))
                        by_horizon = {b["horizon_s"]: b for b in row["forecast_bearings"]}
                        for forecast in row["forecast"]:
                            expected = math.degrees(math.atan((forecast["x"] - 32) / 80)) % 360
                            self.assertAlmostEqual(by_horizon[forecast["horizon_s"]]["azimuth_deg"],
                                                   expected, places=6)


class LivePanelTests(unittest.TestCase):
    def test_panel_heading_uses_the_tracks_csv_convention(self):
        """A target moving up and to the right reads 45 degrees (counterclockwise from image right,
        as camera.image_heading), not 315. Blank frames also exercise the live camera-motion
        fallback (no background corners: counted as failed, treated as no motion)."""
        import cv2
        import numpy as np
        from aeris.data.contracts import BBox, DetectionRecord
        from aeris.web.live import LiveSession

        class Mover:
            def preprocess(self, frame):
                return frame
            def infer(self, prepared):
                return None
            def postprocess(self, raw, prepared, *, width, height, sequence_id, frame_index, timestamp_s):
                x, y = 40 + 3 * frame_index, 200 - 3 * frame_index
                return [DetectionRecord(1, sequence_id, frame_index, timestamp_s, width, height,
                                        BBox(x - 12, y - 6, x + 12, y + 6), 0, "aeris-aircraft-v1", 0.9, "test", {})]

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "blank.avi")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (320, 240))
            for _ in range(30):
                writer.write(np.zeros((240, 320, 3), np.uint8))
            writer.release()
            with mock.patch("aeris.web.live.timestamp_plan", return_value=(None, "inferred_fps", "test")):
                # paced: media-time timestamps (1/20 s); unpaced wall-clock deltas depend on machine speed
                session = LiveSession(path, Mover(), paced=True, label="blank")
                session.start()
                session._thread.join(timeout=10)
        stats = session.stats
        self.assertIn(stats["state"], {"ended", "failed"})
        row = stats["panel"][0]
        self.assertAlmostEqual(row["heading_deg"], 45.0, delta=2.0)
        self.assertEqual(stats["camera_motion"]["estimated"], 0)
        self.assertGreater(stats["camera_motion"]["failed"], 20)


class ForecastNoiseCallSiteTests(unittest.TestCase):
    """AERIS-024 AC2 at the call sites: the batch CLI and the live loop must pass
    measurement_sigma_px to the forecasters (0.05 x 200 px = 10 px for a 200x80 box),
    not the old fixed 1 px. A formula-only test cannot catch a call site reverting."""

    @staticmethod
    def _clip(tmp, frames=8):
        import cv2
        import numpy as np
        path = Path(tmp) / "blank.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (320, 240))
        for _ in range(frames):
            writer.write(np.zeros((240, 320, 3), np.uint8))
        writer.release()
        return path

    @staticmethod
    def _box(i):
        x = 40 + 4 * i
        return [x, 80, x + 200, 160]                      # 200x80: larger side 200 px

    @staticmethod
    def _spy():
        from aeris.estimation.trajectory import TrajectoryForecaster
        sigmas, original = [], TrajectoryForecaster.update

        def spy(forecaster, track_id, timestamp_s, center, sigma_px=1.0):
            sigmas.append(sigma_px)
            return original(forecaster, track_id, timestamp_s, center, sigma_px)
        return sigmas, mock.patch.object(TrajectoryForecaster, "update", spy)

    def test_live_loop_passes_box_relative_noise(self):
        from aeris.data.contracts import BBox, DetectionRecord
        from aeris.web.live import LiveSession
        box = self._box

        class Stages:
            def preprocess(self, frame):
                return frame
            def infer(self, prepared):
                return None
            def postprocess(self, raw, prepared, *, width, height, sequence_id, frame_index, timestamp_s):
                return [DetectionRecord(1, sequence_id, frame_index, timestamp_s, width, height,
                                        BBox(*box(frame_index)), 0, "aeris-aircraft-v1", 0.9, "test", {})]

        sigmas, patch = self._spy()
        with tempfile.TemporaryDirectory() as tmp, patch:
            session = LiveSession(str(self._clip(tmp)), Stages(), paced=False, label="blank")
            session.start()
            session._thread.join(timeout=10)
        self.assertGreater(len(sigmas), 3)
        self.assertTrue(all(abs(s - 10.0) < 1e-6 for s in sigmas), sigmas)

    def test_batch_cli_passes_box_relative_noise(self):
        from aeris import cli
        sigmas, patch = self._spy()
        with tempfile.TemporaryDirectory() as tmp:
            replay = Path(tmp) / "replay.json"
            replay.write_text(json.dumps({"frames": {str(i): [{"bbox": self._box(i), "class_id": 0, "score": 0.9}]
                                                     for i in range(8)}}))
            with patch:
                cli.run_pipeline(self._clip(tmp), Path(tmp) / "out", stages=cli.ReplayStages(replay),
                                 render=False, command=["test"])
        self.assertGreaterEqual(len(sigmas), 8)
        self.assertTrue(all(abs(s - 10.0) < 1e-6 for s in sigmas), sigmas)


class LiveScenePanelTests(unittest.TestCase):
    def test_live_panel_is_scene_relative_under_a_pan(self):
        """Textured clip panning 5 px/frame; the aircraft moves +3 px/frame in the scene (60 px/s at
        20 fps), so in the image it drifts -2 px/frame. The panel heading must read about 0 deg
        (scene motion to the right) and the 1 s forecast, in the panel and as the drawn arrow, must
        lie about +60 px from the aircraft in current-frame pixels, not at image drift (-40 px) or in
        reference-frame coordinates. The stub detector is keyed on media time, not on the processed
        frame count, so frames dropped by the paced loop cannot desynchronise box and image."""
        import cv2
        import numpy as np
        from aeris.data.contracts import BBox, DetectionRecord
        from aeris.web.live import LiveSession
        frames, width, height = 30, 320, 240
        noise = (np.random.default_rng(4).random((height, width + 5 * frames)) * 255).astype(np.uint8)
        scene = cv2.cvtColor(cv2.normalize(cv2.GaussianBlur(noise, (0, 0), 2), None, 0, 255,
                                           cv2.NORM_MINMAX), cv2.COLOR_GRAY2BGR)

        def image_x(i):
            return 120 - 2 * i                                 # scene x 120 + 3i minus camera x 5i

        class Stages:
            def preprocess(self, frame):
                return frame
            def infer(self, prepared):
                return None
            def postprocess(self, raw, prepared, *, width, height, sequence_id, frame_index, timestamp_s):
                x = image_x(round(timestamp_s * 20))           # decoded frame position from media time
                return [DetectionRecord(1, sequence_id, frame_index, timestamp_s, width, height,
                                        BBox(x - 12, 100, x + 12, 112), 0, "aeris-aircraft-v1", 0.9, "test", {})]

        import aeris.web.live as live
        from aeris.estimation.camera_motion import CameraMotion
        drawn, boxes = [], []
        original_draw, original_update = live.draw_states, CameraMotion.update

        def draw_spy(frame, items):
            drawn.append([f for _, f in items])
            return original_draw(frame, items)

        def update_spy(motion, frame_bgr, boxes_arg=()):
            boxes.append(len(list(boxes_arg)))
            return original_update(motion, frame_bgr, boxes_arg)

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(live, "draw_states", draw_spy), \
                mock.patch.object(CameraMotion, "update", update_spy):
            path = str(Path(tmp) / "pan.avi")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (width, height))
            for i in range(frames):
                image = scene[:, 5 * i:5 * i + width].copy()
                cv2.rectangle(image, (image_x(i) - 12, 100), (image_x(i) + 12, 112), (255, 255, 255), -1)
                writer.write(image)
            writer.release()
            session = LiveSession(path, Stages(), paced=True, label="pan")
            session.start()
            session._thread.join(timeout=10)
        stats = session.stats
        self.assertGreater(stats["camera_motion"]["estimated"], frames // 2)
        row = stats["panel"][0]
        heading = (row["heading_deg"] + 180) % 360 - 180               # wrap to (-180, 180]
        self.assertLess(abs(heading), 10, row)
        one_s = [f for f in row["forecast"] if f["horizon_s"] == 1.0][0]
        last = image_x(round(stats["media_t"] * 20))
        self.assertLess(abs(one_s["x"] - (last + 60)), 3, (one_s, last))
        self.assertLess(abs(drawn[-1][0][0] - (last + 60)), 3, (drawn[-1], last))   # drawn arrow
        self.assertTrue(boxes and min(boxes) >= 1, boxes)          # detections are masked out
