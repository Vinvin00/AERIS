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
                session = LiveSession(path, Mover(), paced=False, label="blank")
                session.start()
                session._thread.join(timeout=10)
        stats = session.stats
        self.assertIn(stats["state"], {"ended", "failed"})
        row = stats["panel"][0]
        self.assertAlmostEqual(row["heading_deg"], 45.0, delta=2.0)
        self.assertEqual(stats["camera_motion"]["estimated"], 0)
        self.assertGreater(stats["camera_motion"]["failed"], 20)
