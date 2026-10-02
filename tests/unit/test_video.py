"""Video pipeline tests: generated synthetic clips + a deterministic fake detector.

The synthetic clips test decoding, timestamps, timings and plumbing, not detection
quality. A local-only test pans a real FGVC aircraft photo through the real baseline
(skipped when the Git-ignored weights/images are absent).
"""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from aeris.data.contracts import BBox, DetectionRecord
from aeris.video import pipeline as p

ROOT = Path(__file__).resolve().parents[2]
FPS, W, H, N = 10, 64, 48, 12
HAVE_FFPROBE = shutil.which("ffprobe") is not None
HAVE_FFMPEG = shutil.which("ffmpeg") is not None


class FakeStages:
    """Deterministic detector: one box at x = 5 + 2*frame_index, no real inference."""
    name = "fake"

    def __init__(self, fail_at=None, wrong_frame=False):
        self.fail_at, self.wrong_frame = fail_at, wrong_frame

    def preprocess(self, image_bgr):
        return image_bgr.shape

    def infer(self, prepared):
        return prepared

    def postprocess(self, raw, prepared, *, width, height, sequence_id, frame_index, timestamp_s):
        if frame_index == self.fail_at:
            raise RuntimeError("boom")
        x = 5 + 2 * frame_index
        return [DetectionRecord(1, sequence_id, frame_index + self.wrong_frame, timestamp_s, width,
                                height, BBox(x, 10, x + 10, 30), 0, "aeris-aircraft-v1", 0.9, "fake", {})]


def write_clip(path, frames=N, fps=FPS, size=(W, H)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert writer.isOpened()
    for i in range(frames):
        image = np.full((size[1], size[0], 3), 30, np.uint8)
        cv2.rectangle(image, (5 + 2 * i, 10), (15 + 2 * i, 30), (200, 200, 200), -1)
        writer.write(image)
    writer.release()


class TimestampTests(unittest.TestCase):
    def test_valid_and_invalid_sequences(self):
        p.validate_timestamps([0.0, 0.1, 0.25])
        for bad in ([0.0, 0.0], [0.0, 0.1, 0.05], [0.1, float("nan")], [-0.1, 0.0],
                    [0.0, float("inf")], [True, 1.0]):
            with self.assertRaises(p.VideoError):
                p.validate_timestamps(bad)

    def test_mixed_timestamp_sources_are_rejected(self):
        assert p.require_single_timestamp_source(["decode_pts"] * 3) == "decode_pts"
        with self.assertRaises(p.VideoError):
            p.require_single_timestamp_source(["decode_pts", "inferred_fps"])
        with self.assertRaises(p.VideoError):
            p.require_single_timestamp_source([])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.clip = self.tmp / "clip.mp4"
        write_clip(self.clip)

    def run_clip(self, **kw):
        return p.process_video(self.clip, FakeStages(), sequence_id="syn", **kw)

    def test_fake_detector_end_to_end_with_render(self):
        out = self.tmp / "overlay.mp4"
        result = self.run_clip(output_video=out)
        assert (result["width"], result["height"]) == (W, H)
        assert result["decoded_frames"] == result["processed_frames"] == N
        assert len(result["detections"]) == N and result["skipped_frames"] == {"stride": 0}
        times = [f["timestamp_s"] for f in result["frames"]]
        p.validate_timestamps(times)
        np.testing.assert_allclose(np.diff(times), 1 / FPS, atol=0.02)
        assert result["detections"][3].bbox.x_min == 11
        for name in p.STAGES:
            summary = result["stage_summary"][name]
            assert summary["samples"] == N - 1 and summary["warmup_excluded"] == 1
            assert summary["p50_s"] >= 0 and summary["total_s"] >= 0
        assert all(f["timings_s"]["render"] > 0 for f in result["frames"])
        check = cv2.VideoCapture(str(out))
        assert (int(check.get(cv2.CAP_PROP_FRAME_WIDTH)), int(check.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (W, H)
        assert int(check.get(cv2.CAP_PROP_FRAME_COUNT)) == N
        check.release()

    def test_total_is_sum_of_stages_and_run_totals_match_frames(self):
        result = self.run_clip(output_video=self.tmp / "o.mp4")
        stage_names = [n for n in p.STAGES if n != "total"]
        for f in result["frames"]:
            t = f["timings_s"]
            assert abs(t["total"] - sum(t[n] for n in stage_names)) < 1e-12
            assert t["render"] > 0
        for name in p.STAGES:
            frames_sum = sum(f["timings_s"][name] for f in result["frames"])
            assert abs(result["stage_summary"][name]["total_s"] - frames_sum) < 1e-9

    def test_extra_pts_fail_even_when_max_frames_is_set_but_video_ends(self):
        extra = [i * 0.1 for i in range(N + 3)]
        with mock.patch.object(p, "probe_pts", return_value=extra):
            with self.assertRaises(p.VideoError):
                self.run_clip(max_frames=100)
            assert self.run_clip(max_frames=5)["decoded_frames"] == 5   # truncated, not EOF

    def test_ffprobe_errors_are_treated_as_unavailable(self):
        for error in (subprocess.TimeoutExpired("ffprobe", 1), OSError("gone")):
            with mock.patch.object(p.subprocess, "run", side_effect=error), \
                    mock.patch.object(p.shutil, "which", return_value="/x/ffprobe"):
                assert p.probe_pts(self.clip) is None
        bad = mock.Mock(returncode=0, stdout="{not json")
        with mock.patch.object(p.subprocess, "run", return_value=bad), \
                mock.patch.object(p.shutil, "which", return_value="/x/ffprobe"):
            assert p.probe_pts(self.clip) is None
            with self.assertRaises(p.VideoError):
                self.run_clip(timestamp_policy="pts")

    def test_unreadable_or_short_overlay_is_detected(self):
        class DeadWriter:
            def __init__(self, *a):
                pass

            def isOpened(self):
                return True

            def write(self, frame):
                pass

            def release(self):
                pass

        with mock.patch.object(p.cv2, "VideoWriter", DeadWriter):
            with self.assertRaises(p.VideoError):
                self.run_clip(output_video=self.tmp / "dead.mp4")

    def test_overlay_with_fewer_frames_than_input_is_detected(self):
        real_writer = cv2.VideoWriter

        class ShortWriter:
            def __init__(self, *args):
                self.inner, self.count = real_writer(*args), 0

            def isOpened(self):
                return self.inner.isOpened()

            def write(self, frame):
                self.count += 1
                if self.count <= 2:
                    self.inner.write(frame)

            def release(self):
                self.inner.release()

        with mock.patch.object(p.cv2, "VideoWriter", ShortWriter):
            with self.assertRaises(p.VideoError):
                self.run_clip(output_video=self.tmp / "short.mp4")

    def test_render_disabled_is_recorded(self):
        result = self.run_clip()
        assert result["rendered"] is False and "render disabled" in result["render_note"]
        assert all(f["timings_s"]["render"] == 0.0 for f in result["frames"])

    @unittest.skipUnless(HAVE_FFPROBE, "ffprobe absent")
    def test_container_pts_drive_timestamps(self):
        result = self.run_clip()
        assert result["timestamp_source"] == "decode_pts" and result["timestamp_reason"] is None

    def test_inferred_fallback_is_marked(self):
        with mock.patch.object(p, "probe_pts", return_value=None):
            result = self.run_clip()
        assert result["timestamp_source"] == "inferred_fps"
        assert "inferred from nominal" in result["timestamp_reason"]
        np.testing.assert_allclose([f["timestamp_s"] for f in result["frames"]],
                                   [i / FPS for i in range(N)])

    def test_strict_policy_rejects_missing_pts(self):
        with mock.patch.object(p, "probe_pts", return_value=None):
            with self.assertRaises(p.VideoError):
                self.run_clip(timestamp_policy="pts")

    def test_zero_negative_and_miscounted_pts_fail(self):
        for bad in ([i * 0.1 for i in range(N - 1)] + [(N - 2) * 0.1],       # zero delta
                    [i * 0.1 for i in range(N - 1)] + [0.05],                  # negative delta
                    [i * 0.1 for i in range(N - 1)],                           # fewer than decoded
                    [i * 0.1 for i in range(N + 1)]):                          # more than decoded
            with mock.patch.object(p, "probe_pts", return_value=bad):
                with self.assertRaises(p.VideoError):
                    self.run_clip()

    def test_stride_records_skipped_frames_and_keeps_timestamps(self):
        result = self.run_clip(stride=3)
        assert result["processed_frames"] == 4 and result["skipped_frames"] == {"stride": 8}
        assert [f["frame_index"] for f in result["frames"]] == [0, 3, 6, 9]

    def test_max_frames_truncates_without_false_count_error(self):
        result = self.run_clip(max_frames=5)
        assert result["decoded_frames"] == 5

    def test_failure_paths(self):
        with self.assertRaises(p.VideoError):
            p.process_video(self.tmp / "missing.mp4", FakeStages(), sequence_id="s")
        bad = self.tmp / "garbage.mp4"
        bad.write_bytes(b"not a video" * 50)
        with self.assertRaises(p.VideoError):
            p.process_video(bad, FakeStages(), sequence_id="s")
        with self.assertRaises(p.VideoError) as ctx:
            p.process_video(self.clip, FakeStages(fail_at=4), sequence_id="s")
        assert "frame 4" in str(ctx.exception)
        with self.assertRaises(p.VideoError):
            p.process_video(self.clip, FakeStages(wrong_frame=True), sequence_id="s")
        for kw in ({"stride": 0}, {"max_frames": 0}, {"warmup_frames": -1},
                   {"timestamp_policy": "bogus"}):
            with self.assertRaises(p.VideoError):
                self.run_clip(**kw)
        with self.assertRaises(p.VideoError):
            self.run_clip(output_video=self.tmp / "no_such_dir" / "o.mp4")

    def test_dimension_change_and_non_color_frames_fail(self):
        class FakeCapture:
            def __init__(self, frames):
                self.frames = list(frames)

            def isOpened(self):
                return True

            def get(self, prop):
                return float(FPS)

            def read(self):
                return (True, self.frames.pop(0)) if self.frames else (False, None)

            def release(self):
                pass

        color = np.zeros((H, W, 3), np.uint8)
        for frames in ([color, np.zeros((H + 2, W, 3), np.uint8)], [color, np.zeros((H, W), np.uint8)]):
            with mock.patch.object(p, "probe_pts", return_value=None), \
                    mock.patch.object(p.cv2, "VideoCapture", return_value=FakeCapture(frames)):
                with self.assertRaises(p.VideoError):
                    p.process_video(self.clip, FakeStages(), sequence_id="s")

    @unittest.skipUnless(HAVE_FFMPEG and HAVE_FFPROBE, "ffmpeg/ffprobe absent")
    def test_variable_frame_rate_pts_are_used_not_nominal_fps(self):
        vfr = self.tmp / "vfr.mp4"
        done = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(self.clip), "-vf",
             "select='lt(mod(n\\,4)\\,3)'", "-fps_mode", "passthrough",
             "-video_track_timescale", "10000", str(vfr)],
            capture_output=True, text=True)
        if done.returncode != 0:
            self.skipTest(f"ffmpeg could not build VFR clip: {done.stderr[:120]}")
        result = p.process_video(vfr, FakeStages(), sequence_id="vfr")
        times = [f["timestamp_s"] for f in result["frames"]]
        assert result["timestamp_source"] == "decode_pts" and len(times) == 9
        np.testing.assert_allclose(times, [0, .1, .2, .4, .5, .6, .8, .9, 1.0], atol=1e-4)  # not i/fps


class RealBaselineTests(unittest.TestCase):
    """Pan a real FGVC aircraft photo through the real detector (local-only)."""
    image = ROOT / "data/raw/g1-bounded-sample/images/1143132.jpg"
    config = json.loads((ROOT / "configs/detector/baseline.json").read_text())
    model = ROOT / config["model"]["path"]

    @unittest.skipUnless(image.is_file() and model.is_file(), "local weights/FGVC image absent")
    def test_real_baseline_on_panned_photo(self):
        from aeris.detection import baseline as b
        photo = cv2.imread(str(self.image))
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "pan.mp4"
            width, height = 640, 400
            writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height))
            for i in range(6):
                writer.write(photo[100:100 + height, 4 * i:4 * i + width].copy())
            writer.release()
            runner = b.OnnxRuntimeRunner(b.verified_model_path(self.config))
            out = Path(tmp) / "o.mp4"
            result = p.process_video(clip, p.BaselineStages(self.config, runner),
                                     sequence_id="pan", output_video=out)
        assert result["processed_frames"] == 6 and (result["width"], result["height"]) == (width, height)
        assert result["stage_summary"]["inference"]["p50_s"] > 0
        assert len(result["detections"]) >= 5
        assert all(d.class_id == 0 and d.timestamp_s is not None for d in result["detections"])
        xs = [d.bbox.x_min for d in result["detections"]]
        assert max(xs) - min(xs) < 60   # box tracks the slow pan; no coordinate blow-up


if __name__ == "__main__":
    unittest.main()
