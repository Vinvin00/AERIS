"""Timestamp-aware video pipeline tests (AERIS-008).

Unit tests use SYNTHETIC inputs only: short videos written at test time with OpenCV's
VideoWriter into a temporary directory (nothing is committed), a fake capture that feeds
chosen frames and timestamps, and the deterministic FakeDetector. Expected values are
worked out in the comments.

Tests marked REAL need the pinned Commons clip in data/raw/video/ (fetched with
`python -m aeris.video.samples fetch`) and, for the ONNX tests, the AERIS-006 export.
They skip with the reason when those are absent; AERIS_REQUIRE_REAL_VIDEO=1 turns the
skips into failures. None of this measures detector quality.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from aeris.data.contracts import DetectionRecord, FrameRecord
from aeris.detection import baseline as det
from aeris.detection.samples import SampleError
from aeris.video import pipeline as vp
from aeris.video import samples

ROOT = Path(__file__).resolve().parents[2]
STRICT = os.environ.get("AERIS_REQUIRE_REAL_VIDEO") == "1"


def real(ready: bool, reason: str):
    """Skip (or, under AERIS_REQUIRE_REAL_VIDEO=1, fail) when REAL inputs are missing."""
    if ready:
        return
    if STRICT:
        raise AssertionError(f"REAL evidence required but unavailable: {reason}")
    raise unittest.SkipTest(f"REAL: {reason}")


def write_video(path: Path, frames: int, fps: float, size=(64, 48), fourcc="MJPG") -> Path:
    """SYNTHETIC video: a bright 8x8 square moving right on a grey background."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
    assert writer.isOpened(), fourcc
    for i in range(frames):
        image = np.full((size[1], size[0], 3), 90, np.uint8)
        image[20:28, 4 * i % (size[0] - 8):4 * i % (size[0] - 8) + 8] = 250
        writer.write(image)
    writer.release()
    return path


class TempDir(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())


# ------------------------------------------------------------------ timestamps

class TimestampPolicyTests(unittest.TestCase):
    def feed(self, policy, values):
        return [policy(i, v) for i, v in enumerate(values)]

    def test_pts_seconds_from_first_frame(self):
        policy = vp.TimestampPolicy("pts")
        # Container milliseconds 500, 533, 567, 600 -> 0, 0.033, 0.067, 0.1 s.
        out = self.feed(policy, [500.0, 533.0, 567.0, 600.0])
        for got, want in zip(out, [0.0, 0.033, 0.067, 0.1]):
            self.assertAlmostEqual(got, want, places=12)
        self.assertEqual(policy.source, "decode_pts")
        summary = policy.summary()
        self.assertEqual((summary["first_pts_ms"], summary["deltas"]), (500.0, 3))
        self.assertAlmostEqual(summary["delta_s_min"], 0.033, places=12)
        self.assertAlmostEqual(summary["delta_s_max"], 0.034, places=12)

    def test_variable_frame_rate_is_kept(self):
        # Deltas 10, 40, 1 ms stay as they are; nothing assumes a constant rate.
        out = self.feed(vp.TimestampPolicy("pts"), [0.0, 10.0, 50.0, 51.0])
        self.assertEqual([round(t * 1000, 9) for t in out], [0.0, 10.0, 50.0, 51.0])

    def test_zero_and_negative_deltas_fail(self):
        for values, frame in (([0.0, 33.0, 33.0], 2), ([0.0, 33.0, 20.0], 2),
                              ([0.0, 0.0], 1)):  # OpenCV reports 0 when a frame has no timestamp
            policy = vp.TimestampPolicy("pts")
            with self.subTest(values=values):
                with self.assertRaisesRegex(vp.TimestampError, f"frame {frame}: timestamp delta"):
                    self.feed(policy, values)

    def test_missing_timestamp_fails_with_hint(self):
        for bad in (None, float("nan"), float("inf"), -1.0):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(vp.TimestampError, "inferred_fps --fps"):
                    self.feed(vp.TimestampPolicy("pts"), [0.0, bad])

    def test_inferred_fps_is_explicit_and_marked(self):
        policy = vp.TimestampPolicy("inferred_fps", 25.0)
        # index / 25, whatever the container says (even no timestamp at all).
        self.assertEqual(self.feed(policy, [None, 999.0, None]), [0.0, 0.04, 0.08])
        self.assertEqual(policy.source, "inferred_fps")
        for fps in (None, 0, -5.0, float("nan"), float("inf"), True, "25"):
            with self.subTest(fps=fps), self.assertRaises(vp.PipelineError):
                vp.TimestampPolicy("inferred_fps", fps)
        with self.assertRaises(vp.PipelineError):
            vp.TimestampPolicy("pts", 25.0)  # an fps is never silently used in PTS mode
        with self.assertRaises(vp.PipelineError):
            vp.TimestampPolicy("frame_index")


# ------------------------------------------------------------------ Matroska timeline reader

def _id(element: int) -> bytes:
    return element.to_bytes((element.bit_length() + 7) // 8, "big")


def _el(element: int, payload: bytes) -> bytes:
    return _id(element) + (0x0100000000000000 | len(payload)).to_bytes(8, "big") + payload


def _uint_el(element: int, value: int) -> bytes:
    return _el(element, value.to_bytes(max(1, (value.bit_length() + 7) // 8), "big"))


UNKNOWN_SIZE = b"\x01\xff\xff\xff\xff\xff\xff\xff"


def simple_block(track: int, relative: int, flags: int = 0x80) -> bytes:
    return _el(0xA3, bytes([0x80 | track]) + relative.to_bytes(2, "big", signed=True) + bytes([flags]) + b"payload")


def cluster(time: int, *blocks: bytes, unknown: bool = False, time_first: bool = True) -> bytes:
    children = (_uint_el(0xE7, time) + b"".join(blocks)) if time_first else (b"".join(blocks) + _uint_el(0xE7, time))
    return (_id(0x1F43B675) + UNKNOWN_SIZE + children) if unknown else _el(0x1F43B675, children)


def mkv(tracks, *clusters: bytes, scale: int = 1_000_000) -> bytes:
    """SYNTHETIC Matroska bytes: EBML header, unknown-size Segment, Info, Tracks, clusters."""
    header = _el(0x1A45DFA3, _el(0x4282, b"webm"))
    info = _el(0x1549A966, _uint_el(0x2AD7B1, scale))
    entries = _el(0x1654AE6B, b"".join(_el(0xAE, _uint_el(0xD7, n) + _uint_el(0x83, kind)) for n, kind in tracks))
    return header + _id(0x18538067) + UNKNOWN_SIZE + info + entries + b"".join(clusters)


class MatroskaTimelineTests(TempDir):
    def timeline(self, data: bytes) -> dict:
        path = self.dir / "t.webm"
        path.write_bytes(data)
        return vp.matroska_timeline(path)

    def test_times_scale_and_presentation_order(self):
        # Clusters at 0 and 67 ms; blocks stored in decode order (B-frame style: 0, 67, 33).
        data = mkv([(1, 1)], cluster(0, simple_block(1, 0), simple_block(1, 67), simple_block(1, 33)),
                   cluster(100, simple_block(1, 0), simple_block(1, 33)))
        self.assertEqual(self.timeline(data)["times_ms"], [0.0, 33.0, 67.0, 100.0, 133.0])
        # A 0.1 ms timecode scale: 5 units = 0.5 ms.
        scaled = self.timeline(mkv([(1, 1)], cluster(10, simple_block(1, 0), simple_block(1, 5)), scale=100_000))
        self.assertEqual((scaled["times_ms"], scaled["timecode_scale_ns"]), ([1.0, 1.5], 100_000))

    def test_other_tracks_block_groups_and_unknown_size_clusters(self):
        audio = simple_block(2, 5)  # track 2 is audio (type 2): ignored
        group = _el(0xA0, _el(0xA1, bytes([0x81]) + (40).to_bytes(2, "big") + b"\x00payload"))
        data = mkv([(1, 1), (2, 2)], cluster(0, simple_block(1, 0), audio, group, unknown=True),
                   cluster(80, simple_block(1, 0)))
        info = self.timeline(data)
        self.assertEqual((info["track"], info["times_ms"]), (1, [0.0, 40.0, 80.0]))

    def test_refusals(self):
        cases = {
            "two video tracks": mkv([(1, 1), (2, 1)], cluster(0, simple_block(1, 0))),
            "no video track": mkv([(2, 2)], cluster(0, simple_block(2, 0))),
            "Xiph lacing": mkv([(1, 1)], cluster(0, simple_block(1, 0, 0x82))),
            "fixed lacing": mkv([(1, 1)], cluster(0, simple_block(1, 0, 0x84))),
            "EBML lacing": mkv([(1, 1)], cluster(0, simple_block(1, 0, 0x86))),
            "invisible": mkv([(1, 1)], cluster(0, simple_block(1, 0, 0x88))),
            "block before cluster time": mkv([(1, 1)], cluster(0, simple_block(1, 0), time_first=False)),
            "not matroska": b"\x00" * 64,
        }
        for name, data in cases.items():
            with self.subTest(name), self.assertRaisesRegex(vp.PipelineError, "Matroska"):
                self.timeline(data)

    def test_truncated_file_keeps_what_was_read(self):
        data = mkv([(1, 1)], cluster(0, simple_block(1, 0), simple_block(1, 33)), cluster(67, simple_block(1, 0)))
        # The last SimpleBlock is 20 bytes: ID, 8-byte size, track, 2-byte time, flags, 7-byte
        # payload. Cutting 6 bytes leaves its header, so its time counts (whether the frame
        # decodes is the decoder's business); cutting 16 bytes removes the header.
        self.assertEqual(self.timeline(data[:-6])["times_ms"], [0.0, 33.0, 67.0])
        self.assertEqual(self.timeline(data[:-16])["times_ms"], [0.0, 33.0])


# ------------------------------------------------------------------ fake capture seam

class FakeCapture:
    """Stands in for cv2.VideoCapture: yields given (image, pts_ms) pairs."""

    def __init__(self, frames, fps=10.0, reported=None):
        self.frames, self.i, self.pts = list(frames), 0, 0.0
        self.props = {cv2.CAP_PROP_FPS: fps, cv2.CAP_PROP_FRAME_COUNT: reported or len(self.frames),
                      cv2.CAP_PROP_FRAME_WIDTH: self.frames[0][0].shape[1],
                      cv2.CAP_PROP_FRAME_HEIGHT: self.frames[0][0].shape[0],
                      cv2.CAP_PROP_FOURCC: 0}

    def isOpened(self):
        return True

    def getBackendName(self):
        return "FAKE"

    def read(self):
        if self.i >= len(self.frames):
            return False, None
        image, self.pts = self.frames[self.i]
        self.i += 1
        return True, image

    def get(self, prop):
        return self.pts if prop == cv2.CAP_PROP_POS_MSEC else self.props[prop]

    def release(self):
        pass


def grey(width=32, height=24, value=100):
    return np.full((height, width, 3), value, np.uint8)


class FakeCaptureTests(TempDir):
    def run_fake(self, frames, reported=None, **kwargs):
        dummy = self.dir / "input.mp4"
        dummy.write_bytes(bytes(4) + b"ftypisom" + bytes(400))  # MP4 magic; frames come from the fake
        capture = FakeCapture(frames, reported=reported)
        with mock.patch.object(vp.cv2, "VideoCapture", return_value=capture):
            return vp.run(dummy, vp.FakeDetector(), self.dir / "out", **kwargs)

    def read_run(self):
        return json.loads((self.dir / "out" / "run.json").read_text())

    def test_duplicate_timestamp_stops_the_run_and_is_recorded(self):
        frames = [(grey(), 0.0), (grey(), 40.0), (grey(), 40.0), (grey(), 80.0)]
        with self.assertRaisesRegex(vp.PipelineError, "frame 2: timestamp delta 0.000 ms"):
            self.run_fake(frames)
        run = self.read_run()
        self.assertEqual(run["status"], "failed")
        self.assertIn("frame 2", run["error"])
        self.assertEqual((run["counts"]["decoded"], run["counts"]["processed"]), (3, 2))
        self.assertEqual(len((self.dir / "out" / "frames.jsonl").read_text().splitlines()), 2)

    def test_frame_size_change_fails(self):
        frames = [(grey(), 0.0), (grey(40, 24), 40.0)]
        with self.assertRaisesRegex(vp.PipelineError, "frame 1: size"):
            self.run_fake(frames)

    def test_early_end_is_warned(self):
        frames = [(grey(), 40.0 * i) for i in range(5)]
        run = self.run_fake(frames, reported=10)
        self.assertEqual(run["status"], "ok")
        self.assertEqual(run["warnings"], ["stream ended after 5 frames; the container reports 10"])
        # Stopping at max_frames is not an early end.
        run = self.run_fake(frames, reported=10, max_frames=2)
        self.assertEqual((run["counts"]["processed"], run["counts"]["stopped_at_max_frames"]), (2, True))
        self.assertEqual(run["warnings"], [])

    def test_total_spans_record_building_and_render(self):
        # Review finding N2: record building (to_records) now advances the clock by 5 ms and
        # render is on. Ticks per frame: start, decode end, pre start/end, inference
        # start/end, post start/end (+5 ms in to_records), render start/end, frame end ->
        # render 1 ms, total 15 ms, other = 15 - 5 x 1 ms = 10 ms.
        clock = {"t": 0.0}

        def tick():
            now = clock["t"]
            clock["t"] += 0.001
            return now
        original = vp.det.to_records

        def slow_records(*args, **kwargs):
            clock["t"] += 0.005
            return original(*args, **kwargs)
        frames = [(grey(), 100.0 * i) for i in range(3)]
        with mock.patch.object(vp.det, "to_records", slow_records):
            self.run_fake(frames, clock=tick, render=True, warmup_frames=0)
        timings = json.loads((self.dir / "out" / "timings.json").read_text())
        for row in timings["per_frame"]:
            self.assertAlmostEqual(row["render"], 0.001)
            self.assertAlmostEqual(row["total"], 0.015)
            self.assertAlmostEqual(row["other"], 0.010)

    def test_inferred_fps_run(self):
        frames = [(grey(), 0.0), (grey(), 0.0), (grey(), 0.0)]  # no usable timestamps
        with self.assertRaises(vp.TimestampError):
            self.run_fake(frames)
        run = self.run_fake(frames, timestamps="inferred_fps", fps=5.0)
        records = [FrameRecord.from_json(line) for line in
                   (self.dir / "out" / "frames.jsonl").read_text().splitlines()]
        self.assertEqual([r.timestamp_s for r in records], [0.0, 0.2, 0.4])
        self.assertEqual({r.timestamp_source for r in records}, {"inferred_fps"})
        self.assertEqual(run["timestamps"]["source"], "inferred_fps")

    def test_odd_dimensions_refuse_rendering_but_headless_still_runs(self):
        # AERIS-021 (Codex finding C008-1): OpenCV's mp4v overlay writer silently rounds an
        # odd width or height down for 4:2:0 chroma subsampling (measured directly against
        # this build: 65x49 -> 64x48, 63x47 -> 62x46; an even dimension is unaffected).
        # OpenCV's own VideoWriter cannot be made to emit a genuinely odd-dimension file
        # (it performs the same rounding on write), so the odd "decoded" frame is supplied
        # directly through the fake capture rather than a real encode.
        for width, height in ((33, 24), (32, 25), (33, 25)):  # odd width, odd height, both
            with self.subTest(width=width, height=height):
                frames = [(grey(width, height), 0.0), (grey(width, height), 40.0)]
                with self.assertRaisesRegex(vp.PipelineError,
                                            f"cannot render an overlay for {width}x{height} frames.*"
                                            "mp4v needs an even width and height"):
                    self.run_fake(frames, render=True)
                self.assertFalse((self.dir / "out" / "overlay.mp4").exists())
                run = self.read_run()
                self.assertEqual(run["status"], "failed")
                self.assertIn(f"{width}x{height}", run["error"])
                # The same frames process fine without --render.
                headless = self.run_fake(frames)
                self.assertEqual(headless["status"], "ok")
                self.assertIsNone(headless["overlay"])

    def test_stage_timings_with_a_scripted_clock(self):
        # Each clock() call advances 1 ms. Per processed frame the clock is read at: frame
        # start, decode end, preprocess start/end, inference start/end, postprocess
        # start/end, frame end -> each named stage 1 ms, total 8 ms, other = 8 - 4 = 4 ms.
        ticks = iter(range(10_000))
        frames = [(grey(), 100.0 * i) for i in range(4)]
        run = self.run_fake(frames, clock=lambda: next(ticks) / 1000.0, warmup_frames=1)
        timings = json.loads((self.dir / "out" / "timings.json").read_text())
        for row in timings["per_frame"]:
            for stage in ("decode", "preprocess", "inference", "postprocess"):
                self.assertAlmostEqual(row[stage], 0.001)
            self.assertIsNone(row["render"])
            self.assertAlmostEqual(row["other"], 0.004)
            self.assertAlmostEqual(row["total"], 0.008)
        # Loop: start tick 0, four frames of 9 ticks (1-36), a failed read (37, 38), end 39.
        self.assertAlmostEqual(timings["loop_wall_s"], 0.039)
        self.assertAlmostEqual(timings["processed_frames_per_second"], 4 / 0.039)
        self.assertEqual(timings["warmup_frames_excluded"], 1)
        self.assertEqual(timings["summary"]["total"]["count"], 3)  # 4 processed - 1 warm-up
        self.assertEqual(timings["summary"]["render"]["count"], 0)
        self.assertAlmostEqual(timings["summary"]["inference"]["p95"], 0.001)
        self.assertEqual(timings["decode_all_decoded_frames"]["count"], 4)
        self.assertEqual(run["counts"]["processed"], 4)


# ------------------------------------------------------------------ real containers, fake detector

class StoredTimeTests(TempDir):
    """pts mode checks each decoded Matroska frame against the file's stored block time."""

    def run_with(self, video, pts_values):
        capture = FakeCapture([(grey(64, 48), pts) for pts in pts_values])
        with mock.patch.object(vp.cv2, "VideoCapture", return_value=capture):
            return vp.run(video, vp.FakeDetector(), self.dir / "out")

    def test_decoded_time_must_equal_the_stored_time(self):
        video = write_video(self.dir / "clip.mkv", 5, 10.0, fourcc="mp4v")  # stored 0, 100, ... 400 ms
        run = self.run_with(video, [0.0, 100.0, 200.0, 300.0, 400.0])
        self.assertIn("stored block times", run["timestamps"]["stored_times_checked"])
        with self.assertRaisesRegex(vp.TimestampError, "frame 2: decoded time 250.000 ms differs from "
                                    "the stored block time 200.000 ms"):
            self.run_with(video, [0.0, 100.0, 250.0, 300.0, 400.0])  # increasing, but not what is stored
        with self.assertRaisesRegex(vp.TimestampError, "frame 5: the decoder produced more frames"):
            self.run_with(video, [0.0, 100.0, 200.0, 300.0, 400.0, 500.0])

    def test_laced_or_invisible_blocks_are_refused(self):
        video = write_video(self.dir / "clip.mkv", 5, 10.0, fourcc="mp4v")
        data = video.read_bytes()
        _, blocks = matroska_blocks(data)
        flags_at = blocks[2][2] + 2  # flags byte follows the 16-bit relative time
        for name, bits in (("laced", 0x06), ("invisible", 0x08)):
            with self.subTest(name):
                copy = bytearray(data)
                copy[flags_at] |= bits
                path = self.dir / f"{name}.mkv"
                path.write_bytes(copy)
                with self.assertRaisesRegex(vp.PipelineError, f"Matroska: {name} video blocks.*inferred_fps"):
                    vp.run(path, vp.FakeDetector(), self.dir / "out")
                # The explicit fallback still runs.
                run = vp.run(path, vp.FakeDetector(), self.dir / "out", timestamps="inferred_fps", fps=10.0)
                self.assertIsNone(run["timestamps"]["stored_times_checked"])


class ContainerTests(TempDir):
    def records(self, name):
        text = (self.dir / "out" / name).read_text().splitlines()
        cls = FrameRecord if name == "frames.jsonl" else DetectionRecord
        return [cls.from_json(line) for line in text]

    def test_mp4_timestamps_detections_and_overlay(self):
        video = write_video(self.dir / "clip.mp4", 9, 10.0, fourcc="mp4v")
        run = vp.run(video, vp.FakeDetector(), self.dir / "out", render=True)
        self.assertEqual(run["input"]["container"], "mp4")
        frames = self.records("frames.jsonl")
        # 9 frames at 10 fps in MP4: container times 0, 100, ..., 800 ms.
        self.assertEqual([round(f.timestamp_s, 9) for f in frames], [i / 10 for i in range(9)])
        self.assertEqual({f.timestamp_source for f in frames}, {"decode_pts"})
        self.assertEqual({(f.width, f.height, f.annotation_status) for f in frames}, {(64, 48, "unlabeled")})
        # FakeDetector: one box per frame except k % 4 == 3 (frames 3 and 7) -> 7 detections.
        detections = self.records("detections.jsonl")
        self.assertEqual([d.frame_index for d in detections], [0, 1, 2, 4, 5, 6, 8])
        by_frame = {f.frame_index: f for f in frames}
        for d in detections:
            d.validate_frame(by_frame[d.frame_index])
            k = d.frame_index
            self.assertEqual((d.bbox.x_min, d.bbox.y_min), ((37 * k) % 45, (23 * k) % 39))
            self.assertEqual((d.bbox.x_max - d.bbox.x_min, d.bbox.y_max - d.bbox.y_min), (20.0, 10.0))
            self.assertAlmostEqual(d.score, 0.5 + (k % 5) / 10)
            self.assertEqual(d.class_mapping_version, "aeris-aircraft-v1")
        # The frame checksum is the SHA-256 of the decoded pixels.
        cap = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG)
        ok, first = cap.read()
        cap.release()
        self.assertEqual(frames[0].image_checksum, hashlib.sha256(first.tobytes()).hexdigest())
        # Overlay: same size as the input, one frame per processed frame. Decode the file
        # itself rather than trusting Overlay.close()'s return value (AERIS-021, C008-1):
        # OpenCV's mp4v writer can silently write a different size than requested.
        overlay = cv2.VideoCapture(str(self.dir / "out" / "overlay.mp4"), cv2.CAP_FFMPEG)
        count, shapes = 0, set()
        while True:
            ok, image = overlay.read()
            if not ok:
                break
            count += 1
            shapes.add(image.shape)
        actual_w, actual_h = overlay.get(cv2.CAP_PROP_FRAME_WIDTH), overlay.get(cv2.CAP_PROP_FRAME_HEIGHT)
        overlay.release()
        self.assertEqual((count, shapes), (9, {(48, 64, 3)}))
        self.assertEqual(run["overlay"]["frames"], 9)
        self.assertEqual((run["overlay"]["width"], run["overlay"]["height"]), (actual_w, actual_h))
        # run.json digests match the files written.
        for name, digest in run["outputs"].items():
            self.assertEqual(hashlib.sha256((self.dir / "out" / name).read_bytes()).hexdigest(), digest)
        self.assertEqual(run["input"]["sha256"], hashlib.sha256(video.read_bytes()).hexdigest())

    def test_mp4_timestamps_at_another_rate(self):
        video = write_video(self.dir / "clip.mp4", 5, 12.5, fourcc="mp4v")
        vp.run(video, vp.FakeDetector(), self.dir / "out")
        # 12.5 fps -> 80 ms steps from the container.
        self.assertEqual([round(f.timestamp_s * 1000, 6) for f in self.records("frames.jsonl")],
                         [0.0, 80.0, 160.0, 240.0, 320.0])

    def test_stride_and_skipped_frames(self):
        video = write_video(self.dir / "clip.mp4", 10, 10.0, fourcc="mp4v")
        run = vp.run(video, vp.FakeDetector(), self.dir / "out", every=3)
        # Frames 0, 3, 6, 9 processed; the other 6 decoded and skipped by stride.
        self.assertEqual([f.frame_index for f in self.records("frames.jsonl")], [0, 3, 6, 9])
        self.assertEqual(run["counts"], {"decoded": 10, "processed": 4, "skipped": {"stride": 6},
                                         "stopped_at_max_frames": False, "detections": 3})
        timings = json.loads((self.dir / "out" / "timings.json").read_text())
        self.assertEqual(timings["decode_all_decoded_frames"]["count"], 10)
        self.assertEqual(len(timings["per_frame"]), 4)
        # Times still come from the container: 0, 0.3, 0.6, 0.9 s.
        self.assertEqual([round(f.timestamp_s, 9) for f in self.records("frames.jsonl")], [0.0, 0.3, 0.6, 0.9])

    def test_detector_failure_is_recorded(self):
        video = write_video(self.dir / "clip.mp4", 6, 10.0, fourcc="mp4v")
        with self.assertRaisesRegex(vp.PipelineError, "frame 2: RuntimeError: fake detector failure"):
            vp.run(video, vp.FakeDetector(fail_at=2), self.dir / "out")
        run = json.loads((self.dir / "out" / "run.json").read_text())
        self.assertEqual((run["status"], run["counts"]["processed"]), ("failed", 2))

    def test_unreadable_inputs_fail(self):
        garbage = self.dir / "garbage.mp4"
        garbage.write_bytes(b"not a video at all")
        empty = self.dir / "empty.webm"
        empty.write_bytes(b"")
        broken = self.dir / "broken.webm"
        broken.write_bytes(b"\x1a\x45\xdf\xa3" + b"\xff" * 300)  # Matroska magic, nothing valid after
        for path, message in ((self.dir / "missing.mp4", "no such file"),
                              (garbage, "unknown file does not store a presentation time"),
                              (empty, "unknown file does not store a presentation time"),
                              (broken, "Matroska: ")):
            with self.subTest(path=path.name):
                with self.assertRaisesRegex(vp.PipelineError, message):
                    vp.run(path, vp.FakeDetector(), self.dir / "out")

    def test_bad_arguments(self):
        video = write_video(self.dir / "clip.mp4", 2, 10.0, fourcc="mp4v")
        for kwargs in ({"every": 0}, {"every": 1.5}, {"max_frames": 0}, {"warmup_frames": -1},
                       {"timestamps": "pts", "fps": 30.0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(vp.PipelineError):
                vp.run(video, vp.FakeDetector(), self.dir / "out", **kwargs)

    def test_overlay_writer_failure(self):
        video = write_video(self.dir / "clip.mp4", 2, 10.0, fourcc="mp4v")

        class Closed:
            def __init__(self, *args):
                pass

            def isOpened(self):
                return False
        with mock.patch.object(vp.cv2, "VideoWriter", Closed):
            with self.assertRaisesRegex(vp.PipelineError, "cannot open an mp4v video writer"):
                vp.run(video, vp.FakeDetector(), self.dir / "out", render=True)

    def test_containers_without_frame_times_need_the_explicit_fallback(self):
        # AVI stores no per-frame times (FFmpeg derives them from index x rate); MPEG-TS needs
        # a PTS only every 0.7 s; a raw MJPEG stream has no container at all (FFmpeg assumes
        # 25 fps). All are refused in pts mode.
        avi = write_video(self.dir / "clip.avi", 4, 10.0)
        ts = write_video(self.dir / "clip.ts", 4, 10.0, fourcc="mp4v")
        raw = self.dir / "clip.mjpeg"
        raw.write_bytes(b"".join(cv2.imencode(".jpg", grey(64, 48, 40 * i))[1].tobytes() for i in range(4)))
        for path, container in ((avi, "avi"), (ts, "mpegts"), (raw, "unknown")):
            with self.subTest(container=container):
                self.assertEqual(vp.container_format(path), container)
                with self.assertRaisesRegex(vp.PipelineError, f"a {container} file does not store a "
                                            "presentation time for every frame.*inferred_fps --fps"):
                    vp.run(path, vp.FakeDetector(), self.dir / "out")
        # With the explicit fallback the AVI runs, marked inferred_fps: 0, 0.2, 0.4, 0.6 s at 5 fps.
        run = vp.run(avi, vp.FakeDetector(), self.dir / "out", timestamps="inferred_fps", fps=5.0)
        frames = self.records("frames.jsonl")
        self.assertEqual([f.timestamp_s for f in frames], [0.0, 0.2, 0.4, 0.6])
        self.assertEqual({f.timestamp_source for f in frames}, {"inferred_fps"})
        self.assertEqual(run["input"]["container"], "avi")

    def test_container_magic_bytes(self):
        cases = {"matroska": b"\x1a\x45\xdf\xa3" + bytes(10), "mp4": bytes(4) + b"ftypisom",
                 "mpegts": b"".join(b"\x47" + bytes(187) for _ in range(3)),
                 "ogg": b"OggS" + bytes(10), "avi": b"RIFF\0\0\0\0AVI LIST", "unknown": b"\0" * 20}
        for name, head in cases.items():
            path = self.dir / f"x.{name}"
            path.write_bytes(head)
            self.assertEqual(vp.container_format(path), name)

    def test_cli(self):
        video = write_video(self.dir / "clip.mp4", 3, 10.0, fourcc="mp4v")
        self.assertEqual(vp.main(["run", str(video), "--out", str(self.dir / "cli"), "--detector", "fake"]), 0)
        run = json.loads((self.dir / "cli" / "run.json").read_text())
        self.assertEqual(run["argv"][-4:], ["--out", str(self.dir / "cli"), "--detector", "fake"])
        self.assertEqual(vp.main(["run", str(self.dir / "nope.mp4"), "--out", str(self.dir / "x"),
                                  "--detector", "fake"]), 2)


# ------------------------------------------------------------------ sample fetch

class SampleTests(TempDir):
    def setUp(self):
        super().setUp()
        self.data = b"\x1aE\xdf\xa3 synthetic webm bytes"
        self.entry = {"id": "clip", "title": "File:Clip.webm", "license_code": "cc-by-sa-4.0",
                      "sha1": hashlib.sha1(self.data).hexdigest(),
                      "sha256": hashlib.sha256(self.data).hexdigest()}
        self.manifest = {"allowed_licenses": ["cc0", "cc-by-sa-[0-9]*"], "max_file_bytes": 1000,
                         "entries": [self.entry]}

    def fake_get(self, licence="cc-by-sa-4.0", sha1=None, data=None):
        api = {"query": {"pages": [{"title": "File:Clip.webm", "imageinfo": [{
            "url": "https://upload.example/Clip.webm", "descriptionurl": "https://commons.example/Clip",
            "sha1": sha1 or self.entry["sha1"], "size": len(self.data), "mime": "video/webm",
            "extmetadata": {"License": {"value": licence}, "Artist": {"value": "Someone"}}}]}]}}

        def get(url, limit):
            if "api.php" in url:
                return json.dumps(api).encode()
            return data if data is not None else self.data
        return get

    def test_fetch_and_reuse(self):
        result = samples.fetch_entry(self.entry, self.manifest, self.dir, self.fake_get())
        self.assertFalse(result["reused"])
        self.assertEqual((self.dir / "clip.webm").read_bytes(), self.data)
        again = samples.fetch_entry(self.entry, self.manifest, self.dir, lambda *a: self.fail("no request"))
        self.assertTrue(again["reused"])

    def test_refusals(self):
        cases = {"licence not allowed": self.fake_get(licence="cc-by-nc-4.0"),
                 "licence changed": self.fake_get(licence="cc0"),
                 "commons sha1 differs": self.fake_get(sha1="0" * 40),
                 "bytes differ": self.fake_get(data=b"other bytes")}
        for name, get in cases.items():
            with self.subTest(name), self.assertRaises(SampleError):
                samples.fetch_entry(self.entry, self.manifest, self.dir / name.replace(" ", "_"), get)

    def test_manifest_is_pinned(self):
        manifest = samples.load_manifest()
        for entry in manifest["entries"]:
            self.assertRegex(entry["sha256"], "^[0-9a-f]{64}$")
            self.assertRegex(entry["sha1"], "^[0-9a-f]{40}$")
            self.assertTrue(any(__import__("fnmatch").fnmatchcase(entry["license_code"], p)
                                for p in manifest["allowed_licenses"]))
        bad = self.dir / "bad.json"
        bad.write_text(json.dumps({"entries": [{"id": "x", "title": "File:x.webm", "sha1": "abc", "sha256": "0" * 64}]}))
        with self.assertRaises(SampleError):
            samples.load_manifest(bad)


# ------------------------------------------------------------------ REAL: pinned Commons clip

MANIFEST = samples.load_manifest()
CLIP = MANIFEST["entries"][0]
CLIP_PATH = samples.sample_path(CLIP)
HAVE_CLIP = samples.verified(CLIP_PATH, CLIP)
DETECTOR_CONFIG = det.load_config()
ONNX = ROOT / DETECTOR_CONFIG["export"]["aeris_export"]["path"]


def _vint(data: bytes, pos: int, keep_marker: bool = False) -> tuple[int, int]:
    """An EBML variable-length integer at pos: (value, byte length)."""
    first, length, mask = data[pos], 1, 0x80
    while not first & mask:
        mask >>= 1
        length += 1
    value = first if keep_marker else first & (mask - 1)
    for i in range(1, length):
        value = (value << 8) | data[pos + i]
    return value, length


def matroska_blocks(data: bytes) -> tuple[int, list[tuple[int, int, int]]]:
    """Independent Matroska reader for REAL checks: the TimecodeScale (ns) and, per
    SimpleBlock/Block, (absolute timecode in scale units, track, offset of its 16-bit
    relative timecode). Only the containers needed to reach blocks are entered."""
    masters = {0x18538067, 0x1549A966, 0xA0}  # Segment, Info, BlockGroup
    scale, blocks = [1_000_000], []

    def walk(start, end, cluster_time):
        pos = start
        while pos < end:
            element, a = _vint(data, pos, keep_marker=True)
            size, b = _vint(data, pos + a)
            body = pos + a + b
            if size == (1 << (7 * b)) - 1:  # unknown size: runs to the parent's end
                size = end - body
            if element == 0x2AD7B1:  # TimecodeScale
                scale[0] = int.from_bytes(data[body:body + size], "big")
            elif element == 0x1F43B675:  # Cluster: its Timecode comes first
                time, p = 0, body
                while p < body + size:
                    child, c = _vint(data, p, keep_marker=True)
                    child_size, d = _vint(data, p + c)
                    if child == 0xE7:
                        time = int.from_bytes(data[p + c + d:p + c + d + child_size], "big")
                        break
                    p += c + d + child_size
                walk(body, body + size, time)
            elif element in masters:
                walk(body, body + size, cluster_time)
            elif element in (0xA3, 0xA1):  # SimpleBlock, Block
                track, t = _vint(data, body)
                relative = int.from_bytes(data[body + t:body + t + 2], "big", signed=True)
                blocks.append((cluster_time + relative, track, body + t))
            pos = body + size

    walk(0, len(data), 0)
    return scale[0], blocks


class RealClipTests(TempDir):
    def setUp(self):
        super().setUp()
        real(HAVE_CLIP, f"{CLIP_PATH} missing or not the pinned file; run python -m aeris.video.samples fetch")

    def test_container_timestamps_are_used(self):
        run = vp.run(CLIP_PATH, vp.FakeDetector(), self.dir / "out")
        measured = CLIP["measured_2026_09_27"]
        self.assertEqual(run["counts"]["decoded"], measured["frames_decoded_opencv_ffmpeg"])
        self.assertEqual((run["input"]["width"], run["input"]["height"]), (measured["width"], measured["height"]))
        frames = [FrameRecord.from_json(line) for line in (self.dir / "out" / "frames.jsonl").read_text().splitlines()]
        stamps_ms = [f.timestamp_s * 1000 for f in frames]
        # The WebM time base is 1 ms, so real PTS are whole milliseconds with 33/34 ms steps;
        # index / 29.97 would give 33.366... ms steps instead.
        self.assertTrue(all(abs(t - round(t)) < 1e-6 for t in stamps_ms))
        self.assertEqual({round(b - a) for a, b in zip(stamps_ms, stamps_ms[1:])}, {33, 34})
        self.assertEqual(round(stamps_ms[-1]), round(measured["last_timestamp_ms"]))
        self.assertGreater(abs(stamps_ms[-1] - 391 / measured["nominal_fps"] * 1000), 0.3)
        # And they equal the container's own block times, read without FFmpeg.
        scale, blocks = matroska_blocks(CLIP_PATH.read_bytes())
        self.assertEqual({track for _, track, _ in blocks}, {1})
        block_ms = sorted(time * scale / 1e6 for time, _, _ in blocks)
        self.assertEqual(len(block_ms), len(stamps_ms))
        for got, ms in zip(stamps_ms, block_ms):
            self.assertAlmostEqual(got, ms - block_ms[0], places=6)


    def test_tampered_container_times_stop_the_run(self):
        # Rewrite frame 10's block time in a copy. Equal to frame 9: zero delta at frame 10,
        # also when frame 10 is skipped by stride. 5 ms before frame 9: the stored times,
        # sorted into presentation order, put 295 ms at position 9, so the decoded 300 ms at
        # frame 9 already differs from its stored time. A later, still increasing time is kept.
        data = CLIP_PATH.read_bytes()
        scale, blocks = matroska_blocks(data)
        self.assertEqual(scale, 1_000_000)  # 1 ms units
        (time9, _, _), (time10, _, offset10) = blocks[9], blocks[10]
        relative10 = int.from_bytes(data[offset10:offset10 + 2], "big", signed=True)
        for name, target, every, error in (
                ("zero", time9, 1, "frame 10: timestamp delta 0.000 ms"),
                ("zero", time9, 4, "frame 10: timestamp delta 0.000 ms"),
                ("negative", time9 - 5, 1, "frame 9: decoded time 300.000 ms differs from the stored block time 295.000 ms"),
                ("later", time10 + 20, 1, None)):
            with self.subTest(name=name, every=every):
                copy = bytearray(data)
                copy[offset10:offset10 + 2] = (relative10 + target - time10).to_bytes(2, "big", signed=True)
                path = self.dir / f"{name}-{every}.webm"
                path.write_bytes(copy)
                out = self.dir / f"out-{name}-{every}"
                if error is None:
                    run = vp.run(path, vp.FakeDetector(), out, every=every, max_frames=20)
                    self.assertEqual(run["status"], "ok")
                    self.assertAlmostEqual(run["timestamps"]["delta_s_max"], 0.054)  # 33/34 + 20 ms
                    continue
                with self.assertRaisesRegex(vp.TimestampError, error):
                    vp.run(path, vp.FakeDetector(), out, every=every)
                run = json.loads((out / "run.json").read_text())
                self.assertEqual((run["status"], run["counts"]["decoded"]),
                                 ("failed", 10 if error.startswith("frame 9") else 11))

    def test_truncated_clip_warns(self):
        cut = self.dir / "cut.webm"
        cut.write_bytes(CLIP_PATH.read_bytes()[:CLIP["size_bytes"] // 3])
        run = vp.run(cut, vp.FakeDetector(), self.dir / "out")
        self.assertEqual(run["status"], "ok")
        self.assertLess(run["counts"]["decoded"], 392)
        self.assertEqual(len(run["warnings"]), 1)
        self.assertIn("the container reports 392", run["warnings"][0])

    def test_onnx_stages_match_the_accepted_detector(self):
        real(ONNX.is_file(), f"{ONNX} missing (AERIS-006 export)")
        detector = vp.OnnxDetector(threads=1)
        run = vp.run(CLIP_PATH, detector, self.dir / "out", every=40, max_frames=10, render=True)
        self.assertEqual(run["counts"]["processed"], 10)
        self.assertEqual(run["counts"]["skipped"], {"stride": 351})
        self.assertEqual(run["detector"]["providers"], ["CPUExecutionProvider"])
        recorded = [DetectionRecord.from_json(line) for line in
                    (self.dir / "out" / "detections.jsonl").read_text().splitlines()]
        frames = {f.frame_index: f for f in (FrameRecord.from_json(line) for line in
                  (self.dir / "out" / "frames.jsonl").read_text().splitlines())}
        # The split stages must give exactly what the accepted detect() gives on the same frames.
        settings = detector.settings
        cap = cv2.VideoCapture(str(CLIP_PATH), cv2.CAP_FFMPEG)
        expected, index = [], 0
        while len(expected) < 10 * 1000:
            ok, image = cap.read()
            if not ok or index > 360:
                break
            if index % 40 == 0:
                result = det.detect(detector.model, image, settings)
                expected += det.to_records(result.detections, frames[index], result.letterbox,
                                           settings, detector.provenance)
            index += 1
        cap.release()
        self.assertEqual([r.to_json() for r in recorded], [r.to_json() for r in expected])
        # Output geometry: every box inside the 1920 x 1080 frame; overlay at full size.
        for r in recorded:
            self.assertTrue(0 <= r.bbox.x_min < r.bbox.x_max <= 1920 and 0 <= r.bbox.y_min < r.bbox.y_max <= 1080)
        self.assertEqual((run["overlay"]["width"], run["overlay"]["height"], run["overlay"]["frames"]), (1920, 1080, 10))
        timings = json.loads((self.dir / "out" / "timings.json").read_text())
        for stage in ("decode", "preprocess", "inference", "postprocess", "render", "other", "total"):
            self.assertEqual(timings["summary"][stage]["count"], 9)  # 10 processed - 1 warm-up
        for row in timings["per_frame"]:
            named = sum(row[k] for k in ("decode", "preprocess", "inference", "postprocess", "render"))
            self.assertGreaterEqual(row["other"], 0.0)
            self.assertAlmostEqual(row["total"], named + row["other"], places=9)
            self.assertTrue(math.isfinite(timings["summary"][stage]["p50"]))
