"""End-to-end CLI tests on generated clips with a deterministic replay detector.

The synthetic clips test plumbing, artifacts and failure paths. Nothing here measures real
detection, tracking or forecasting accuracy. The real-model test runs only when the Git-ignored
weights and the FGVC sample exist, on a clip derived from one still photograph.
"""
import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from aeris import cli
from aeris.data.contracts import (
    AnnotationRecord, BBox, FrameRecord, GroundTruthObject, PredictionRecord,
)
from aeris.video import pipeline

ROOT = Path(__file__).resolve().parents[2]
W, H, N, FPS = 160, 120, 30, 10
CALIBRATION = {"camera_id": "synthetic", "width": W, "height": H,
               "K": [[200.0, 0.0, 80.0], [0.0, 200.0, 60.0], [0.0, 0.0, 1.0]],
               "distortion_model": "none", "distortion_coefficients": [],
               "calibration_id": "synthetic-pinhole", "quality": "synthetic", "lens": "fixed", "zoom": 1.0}


def box_at(i):
    x = 10 + 3 * i
    return [x, 40, x + 24, 64]


def write_clip(path, frames=N):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    assert writer.isOpened()
    for i in range(frames):
        image = np.full((H, W, 3), 25, np.uint8)
        x0, y0, x1, y1 = box_at(i)
        cv2.rectangle(image, (x0, y0), (x1, y1), (220, 220, 220), -1)
        writer.write(image)
    writer.release()


def write_replay(path, score=0.9, frames=N, empty=False):
    data = {"frames": {} if empty else
            {str(i): [{"bbox": box_at(i), "class_id": 0, "score": score}] for i in range(frames)}}
    Path(path).write_text(json.dumps(data))


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


class PipelineCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.clip, self.replay = self.tmp / "clip.mp4", self.tmp / "replay.json"
        write_clip(self.clip)
        write_replay(self.replay)

    def run_cli(self, name="out", extra=(), replay=True, clip=None):
        argv = [str(clip or self.clip), "--output", str(self.tmp / name), *extra]
        if replay:
            argv += ["--detections", str(self.replay)]
        with contextlib.redirect_stdout(io.StringIO()):
            return cli.main(argv)

    def out(self, name="out"):
        return self.tmp / name

    def manifest(self, name="out"):
        return json.loads((self.out(name) / "run_manifest.json").read_text())

    def metrics(self, name="out"):
        return json.loads((self.out(name) / "metrics.json").read_text())


class ArtifactTests(PipelineCase):
    def test_cli_writes_the_full_artifact_set_with_consistent_contents(self):
        assert self.run_cli(extra=["--input-note", "synthetic rectangle clip"]) == 0
        out = self.out()
        assert sorted(p.name for p in out.iterdir()) == sorted(
            [*cli.ARTIFACTS, "run_manifest.json"])
        rows = read_csv(out / "tracks.csv")
        assert list(rows[0]) == list(cli.TRACK_COLUMNS) and len(rows) == N
        assert {r["track_id"] for r in rows} == {"t1"}
        assert rows[0]["lifecycle"] == "tentative" and rows[-1]["lifecycle"] == "confirmed"
        times = [float(r["timestamp_s"]) for r in rows]
        assert all(b > a for a, b in zip(times, times[1:]))
        assert abs(float(rows[-1]["vx_px_s"]) - 30.0) < 5          # 3 px per 0.1 s
        predictions = [PredictionRecord.from_dict(json.loads(line))
                       for line in (out / "predictions.jsonl").read_text().splitlines()]
        assert {p.model_name for p in predictions} == set(cli.FORECAST_MODELS)
        assert {p.horizon_s for p in predictions} == {0.1, 0.25, 0.5, 1.0, 2.0}
        confirmed = [r for r in rows if r["lifecycle"] != "tentative"]
        assert len(predictions) == len(confirmed) * 5 * len(cli.FORECAST_MODELS)     # tentative tracks get none
        assert all(p.issue_time_s <= p.target_time_s for p in predictions)
        capture = cv2.VideoCapture(str(out / "overlay.mp4"))
        assert (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(capture.get(cv2.CAP_PROP_FRAME_COUNT))) == (W, H, N)
        capture.release()
        benchmark = json.loads((out / "benchmark.json").read_text())
        assert {"decode", "preprocess", "inference", "postprocess", "track", "forecast", "total"} \
            <= set(benchmark["stages"])
        assert benchmark["warmup_frames_excluded"] == 1 and benchmark["samples_after_warmup"] == N - 1
        assert benchmark["frames_dropped"] == 0 and "not a benchmark" in benchmark["scope"]
        assert benchmark["stages"]["track"]["samples"] == N - 1 == benchmark["stages"]["forecast"]["samples"]

    def test_manifest_records_provenance_and_artifact_hashes(self):
        assert self.run_cli(extra=["--input-note", "synthetic rectangle clip"]) == 0
        manifest = self.manifest()
        for key in ("code_revision", "source_fingerprint", "command", "detector", "tracker_config_sha256",
                    "environment", "dependency_lock_sha256", "timestamp", "artifact_sha256", "input",
                    "calibration", "dataset", "random_seed", "seed_reason", "warmup_frames"):
            assert key in manifest, key
        assert manifest["input"]["sha256"] == hashlib.sha256(self.clip.read_bytes()).hexdigest()
        assert manifest["input"]["note"] == "synthetic rectangle clip"
        assert manifest["detector"]["kind"] == "replay"
        assert manifest["detector"]["detections_sha256"] == hashlib.sha256(self.replay.read_bytes()).hexdigest()
        assert manifest["timestamp"]["source"] in ("decode_pts", "inferred_fps")
        assert manifest["calibration"]["calibration_id"] is None and "pixel-only" in manifest["calibration"]["reason"]
        for name, digest in manifest["artifact_sha256"].items():
            assert hashlib.sha256((self.out() / name).read_bytes()).hexdigest() == digest, name
        assert set(manifest["artifact_sha256"]) == set(cli.ARTIFACTS)
        assert "-m" in manifest["command"] and "aeris" in manifest["command"]

    def test_tracker_config_is_selectable_and_recorded(self):
        assert self.run_cli("default") == 0
        default = json.loads((self.out("default") / "run_manifest.json").read_text())
        assert default["tracker_config"] == "baseline.json"
        assert default["tracker_config_sha256"] == hashlib.sha256(
            (ROOT / "configs/tracker/baseline.json").read_bytes()).hexdigest()
        chosen = ROOT / "configs/tracker/manoeuvre-a3000-s0-full.json"
        assert self.run_cli("chosen", extra=["--tracker-config", str(chosen)]) == 0
        manifest = json.loads((self.out("chosen") / "run_manifest.json").read_text())
        assert manifest["tracker_config"] == chosen.name
        assert manifest["tracker_config_sha256"] == hashlib.sha256(chosen.read_bytes()).hexdigest()
        i = manifest["command"].index("--tracker-config")
        assert manifest["command"][i + 1] == str(chosen)
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text(json.dumps({**json.loads(chosen.read_text()), "gate_mode": "loose"}))
            assert self.run_cli("bad", extra=["--tracker-config", str(bad)]) == 2
        assert not self.out("bad").exists()

    def test_replay_runs_are_deterministic(self):
        assert self.run_cli("a") == 0 and self.run_cli("b") == 0
        for name in ("tracks.csv", "predictions.jsonl", "metrics.json"):
            assert (self.out("a") / name).read_bytes() == (self.out("b") / name).read_bytes(), name
        a, b = self.manifest("a"), self.manifest("b")
        for name in ("tracks.csv", "predictions.jsonl", "metrics.json"):
            assert a["artifact_sha256"][name] == b["artifact_sha256"][name]
        assert a["source_fingerprint"] == b["source_fingerprint"]

    def test_stride_max_frames_and_no_render_options(self):
        assert self.run_cli("s", ["--stride", "3", "--max-frames", "20", "--no-render"]) == 0
        metrics = self.metrics("s")
        assert metrics["counts"]["frames_processed"] == 7 and metrics["counts"]["frames_skipped"] == {"stride": 13}
        assert not (self.out("s") / "overlay.mp4").exists()
        manifest = self.manifest("s")
        assert "overlay.mp4" not in manifest["artifact_sha256"] and manifest["stride"] == 3
        benchmark = json.loads((self.out("s") / "benchmark.json").read_text())
        assert benchmark["render_note"] == "render disabled"

    def test_min_score_filters_detections_before_tracking(self):
        write_replay(self.replay, score=0.2)
        assert self.run_cli("low") == 0
        assert read_csv(self.out("low") / "tracks.csv") == []
        assert self.metrics("low")["counts"]["detections_at_or_above_min_score"] == 0
        assert self.run_cli("lowered", ["--min-score", "0.1"]) == 0
        assert len(read_csv(self.out("lowered") / "tracks.csv")) == N

    def test_module_entry_point_runs(self):
        env = {**os.environ, "PYTHONPATH": str(ROOT / "python")}
        done = subprocess.run([sys.executable, "-m", "aeris", str(self.clip), "--output", str(self.out("m")),
                               "--detections", str(self.replay), "--no-render"],
                              capture_output=True, text=True, env=env, cwd=self.tmp)
        assert done.returncode == 0, done.stderr
        assert json.loads(done.stdout)["run_manifest"].endswith("run_manifest.json")


class AvailabilityTests(PipelineCase):
    def test_unlabeled_input_gives_null_truth_metrics_with_reasons(self):
        assert self.run_cli() == 0
        truth = self.metrics()["truth_dependent"]
        for section, keys in (("detection", ("ap50", "ap50_95")), ("tracking", ("hota", "idf1")),
                              ("forecast", ("models",))):
            assert truth[section]["available"] is False and "no ground truth" in truth[section]["reason"]
            assert all(truth[section][k] is None for k in keys), section
        manifest = self.manifest()
        assert manifest["accuracy"] is None and "no labels" in manifest["accuracy_reason"]
        assert self.metrics()["counts"]["detections_at_or_above_min_score"] == N       # counts stay available
        assert "no accuracy is claimed" in self.metrics()["note"]

    def test_forecasts_are_scored_against_later_detections_without_labels(self):
        assert self.run_cli() == 0
        fvd = self.metrics()["forecast_vs_detections"]
        assert fvd["available"] and "not ground truth" in fvd["reference"]
        cv = fvd["models"]["constant_velocity"]["1.0"]       # target moves +30 px/s, straight right
        assert cv["scored"] > 10 and cv["median_error_px"] < 3
        assert cv["direction_scored"] == cv["scored"] and cv["median_direction_error_deg"] < 2

    def test_forecast_arrow_needs_a_resolved_heading(self):
        assert self.run_cli() == 0                                  # moving target: arrows drawn
        shown = self.metrics()["forecast_vs_detections_displayed"]["models"]["constant_velocity"]["1.0"]
        assert shown["issued_at_measurement"] > 10 and shown["median_direction_error_deg"] < 2
        still = {"frames": {str(i): [{"bbox": box_at(0), "class_id": 0, "score": 0.9}] for i in range(N)}}
        self.replay.write_text(json.dumps(still))
        assert self.run_cli("still") == 0                           # stationary target: heading undefined
        hidden = self.metrics("still")["forecast_vs_detections_displayed"]["models"]["constant_velocity"]
        assert all(h["issued_at_measurement"] == 0 for h in hidden.values())
        assert self.metrics("still")["counts"]["predictions_unavailable"] < self.metrics("still")["counts"]["predictions"]

    def test_low_score_detections_extend_tracks_without_counting_as_detections(self):
        low = [i for i in range(N) if i >= 5 and i % 2 == 0]       # every other frame after confirmation
        data = {"frames": {str(i): [{"bbox": box_at(i), "class_id": 0, "score": 0.15 if i in low else 0.9}]
                           for i in range(N)}}
        self.replay.write_text(json.dumps(data))
        assert self.run_cli() == 0
        counts = self.metrics()["counts"]
        assert counts["low_score_track_updates"] == len(low)
        assert counts["detections_at_or_above_min_score"] == N - len(low)
        rows = read_csv(self.out() / "tracks.csv")
        assert len({r["track_id"] for r in rows}) == 1 and all(r["observed"] == "1" for r in rows)
        assert self.manifest()["low_score_pass"] is True

    def test_low_score_pass_off_drops_sub_threshold_detections_before_tracking(self):
        low = [i for i in range(N) if i >= 5 and i % 2 == 0]
        data = {"frames": {str(i): [{"bbox": box_at(i), "class_id": 0, "score": 0.15 if i in low else 0.9}]
                           for i in range(N)}}
        self.replay.write_text(json.dumps(data))
        with mock.patch.object(pipeline, "probe_pts", return_value=None):
            manifest = cli.run_pipeline(self.clip, self.out("off"), stages=cli.ReplayStages(self.replay),
                                        render=False, command=["test"], low_score_pass=False)
        counts = self.metrics("off")["counts"]
        assert counts["low_score_track_updates"] == 0 and manifest["low_score_pass"] is False
        observed = {int(r["frame_index"]): r["observed"] for r in read_csv(self.out("off") / "tracks.csv")}
        assert all(observed[i] == "0" for i in low) and observed[N - 1] == "1"     # coasts through the gaps

    def test_camera_motion_removes_pan_induced_arrows(self):
        """Camera pans 4 px/frame past an aircraft fixed in the scene: image-space forecasts see
        motion and draw arrows; scene-relative forecasts see none and draw no arrow."""
        noise = (np.random.default_rng(1).random((H, W + 4 * N)) * 255).astype(np.uint8)
        scene = cv2.cvtColor(cv2.normalize(cv2.GaussianBlur(noise, (0, 0), 2), None, 0, 255,
                                           cv2.NORM_MINMAX), cv2.COLOR_GRAY2BGR)
        clip, frames = self.tmp / "pan.avi", {}
        writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
        for i in range(N):
            image = scene[:, 4 * i:4 * i + W].copy()
            x0 = 140 - 4 * i                                   # fixed scene x = 140
            cv2.rectangle(image, (x0, 40), (x0 + 20, 56), (255, 255, 255), -1)
            writer.write(image)
            frames[str(i)] = [{"bbox": [x0, 40, x0 + 20, 56], "class_id": 0, "score": 0.9}]
        writer.release()
        self.replay.write_text(json.dumps({"frames": frames}))
        assert self.run_cli("fixed", ["--camera-motion", "none"], clip=clip) == 0
        assert self.run_cli("pan", clip=clip) == 0                     # estimate is the default
        fixed, pan = self.metrics("fixed"), self.metrics("pan")
        def shown(m):
            return m["forecast_vs_detections_displayed"]["models"]["constant_velocity"]["1.0"]["issued_at_measurement"]
        assert shown(fixed) > 5 and shown(pan) == 0
        assert pan["camera_motion"]["failed"] == 0 and pan["camera_motion"]["estimated"] == N - 1
        assert fixed["forecast_vs_detections_scene"]["available"] is False
        assert pan["forecast_vs_detections_scene"]["models"]["constant_velocity"]["1.0"]["scored"] > 5
        assert self.manifest("pan")["camera_motion"] == "estimate"
        assert pan["forecast_vs_detections_displayed_frame"] == "reference frame (scene-relative)"
        assert fixed["forecast_vs_detections_displayed_frame"] == "original frame"
        for name in ("predictions.jsonl", "tracks.csv"):              # written outputs stay image-space
            assert (self.out("fixed") / name).read_bytes() == (self.out("pan") / name).read_bytes(), name

    def test_live_loop_gates_the_arrow_like_the_batch_overlay(self):
        """AERIS-026 live path: a stationary target gets no arrow, a steadily moving one does."""
        from aeris.data.contracts import BBox, DetectionRecord
        from aeris.web import live

        def run(step_px):
            class Stages:
                def preprocess(self, frame):
                    return frame
                def infer(self, prepared):
                    return None
                def postprocess(self, raw, prepared, *, width, height, sequence_id, frame_index, timestamp_s):
                    x = 40 + step_px * frame_index
                    return [DetectionRecord(1, sequence_id, frame_index, timestamp_s, width, height,
                                            BBox(x, 50, x + 24, 62), 0, "aeris-aircraft-v1", 0.9, "test", {})]
            drawn, original = [], live.draw_states

            def spy(frame, items):
                drawn.extend(f for _, f in items if f is not None)
                return original(frame, items)
            clip = self.tmp / f"live-{step_px}.avi"
            writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (W, H))
            for _ in range(20):
                writer.write(np.zeros((H, W, 3), np.uint8))
            writer.release()
            with mock.patch.object(live, "draw_states", spy):
                session = live.LiveSession(str(clip), Stages(), paced=True, label="gate")
                session.start()
                session._thread.join(timeout=10)
            return len(drawn)
        assert run(0) == 0                                          # stationary: heading undefined
        assert run(3) > 5                                           # 60 px/s at 20 fps: arrows drawn

    def test_scene_relative_arrow_is_mapped_back_to_the_current_frame(self):
        """Camera pans 4 px/frame while the aircraft moves +3 px/frame in the scene (30 px/s at 10 fps):
        in the image it drifts -1 px/frame. The drawn 1 s arrow must show the scene motion in
        current-frame pixels (about +30 px), not image drift (-10 px) or reference-frame positions."""
        noise = (np.random.default_rng(2).random((H, W + 4 * N)) * 255).astype(np.uint8)
        scene = cv2.cvtColor(cv2.normalize(cv2.GaussianBlur(noise, (0, 0), 2), None, 0, 255,
                                           cv2.NORM_MINMAX), cv2.COLOR_GRAY2BGR)
        clip, frames = self.tmp / "pan-move.avi", {}
        writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
        for i in range(N):
            image = scene[:, 4 * i:4 * i + W].copy()
            x0 = 100 - i                                       # scene x = 100 + 3i, camera x = 4i
            cv2.rectangle(image, (x0, 40), (x0 + 20, 56), (255, 255, 255), -1)
            writer.write(image)
            frames[str(i)] = [{"bbox": [x0, 40, x0 + 20, 56], "class_id": 0, "score": 0.9}]
        writer.release()
        self.replay.write_text(json.dumps({"frames": frames}))
        captured, original, boxes = {}, cli.render_overlay, []
        original_update = cli.CameraMotion.update

        def spy(video, out, drawn, fps):
            captured.update(drawn)
            return original(video, out, drawn, fps)

        def update_spy(motion, frame_bgr, boxes_arg=()):
            boxes.append(len(list(boxes_arg)))
            return original_update(motion, frame_bgr, boxes_arg)
        with mock.patch.object(cli, "render_overlay", spy), mock.patch.object(cli.CameraMotion, "update", update_spy):
            assert self.run_cli("pan-move", clip=clip) == 0                  # estimate is the default
        assert len(boxes) == N and min(boxes) >= 1                          # detections are masked out
        arrows = [(f[0] - s.center[0], f[1] - s.center[1]) for items in captured.values()
                  for s, f in items if f is not None]
        assert len(arrows) > 5
        dx, dy = float(np.median([a[0] for a in arrows])), float(np.median([a[1] for a in arrows]))
        assert abs(dx - 30.0) < 6 and abs(dy) < 6, (dx, dy)

    def test_missing_calibration_nulls_angular_columns_with_a_reason(self):
        assert self.run_cli() == 0
        rows = read_csv(self.out() / "tracks.csv")
        assert all(r["angular_width_deg"] == "" and r["angular_height_deg"] == "" and
                   r["angular_rate_deg_s"] == "" for r in rows)
        assert all("calibration unavailable" in r["availability_reason"] for r in rows)
        assert any(r["image_heading_deg"] != "" for r in rows)       # pixel-only outputs still exist
        assert float(rows[-1]["image_heading_deg"]) < 1 or float(rows[-1]["image_heading_deg"]) > 359

    def test_calibration_enables_angular_outputs(self):
        calibration = self.tmp / "camera.json"
        calibration.write_text(json.dumps(CALIBRATION))
        assert self.run_cli("cal", ["--calibration", str(calibration)]) == 0
        rows = read_csv(self.out("cal") / "tracks.csv")
        assert all(r["angular_width_deg"] != "" for r in rows if r["x_min_px"] != "")
        assert float(rows[-1]["angular_width_deg"]) > 5          # 24 px at f=200 is about 6.9 degrees
        assert any(r["angular_rate_deg_s"] != "" for r in rows)
        assert self.manifest("cal")["calibration"]["calibration_id"] == "synthetic-pinhole"

    def test_no_detections_still_produces_every_artifact(self):
        write_replay(self.replay, empty=True)
        assert self.run_cli("none") == 0
        assert read_csv(self.out("none") / "tracks.csv") == []
        assert (self.out("none") / "predictions.jsonl").read_text() == ""
        counts = self.metrics("none")["counts"]
        assert counts["detections_total"] == 0 and counts["tracks_created"] == 0 and counts["predictions"] == 0
        assert (self.out("none") / "overlay.mp4").exists() and self.manifest("none")["artifact_sha256"]

    def test_labeled_input_computes_metrics_through_the_evaluators(self):
        with mock.patch.object(pipeline, "probe_pts", return_value=None):     # inferred i/fps timestamps
            annotations = []
            for i in range(N):
                frame = FrameRecord(1, "clip", i, i / FPS, "provided", W, H, f"{i}.png", "0" * 64, None,
                                    "complete", {"camera_id": "synthetic"})
                x0, y0, x1, y1 = box_at(i)
                obj = GroundTruthObject(f"a{i}", 0, BBox(x0, y0, x1, y1), "A", False, False, False, {})
                annotations.append(AnnotationRecord(1, frame, "aeris-aircraft-v1", (obj,)))
            manifest = cli.run_pipeline(self.clip, self.out("lab"), stages=cli.ReplayStages(self.replay),
                                        sequence_id="clip", annotations=annotations, render=False,
                                        command=["test"], dataset_info={"dataset_id": "synthetic"})
        truth = self.metrics("lab")["truth_dependent"]
        assert truth["detection"]["available"] and truth["detection"]["ap50"] > 0.9
        assert truth["tracking"]["available"] and truth["tracking"]["idf1"] > 0.5
        assert truth["forecast"]["errors"]["eligible_keys"] > 0
        assert manifest["timestamp"]["source"] == "inferred_fps"
        assert "labels supplied" in manifest["accuracy_reason"] and manifest["dataset"]["dataset_id"] == "synthetic"

    def test_labels_must_cover_processed_frames_with_matching_dimensions(self):
        frame = FrameRecord(1, "other-sequence", 0, 0.0, "provided", W, H, "0.png", "0" * 64,
                            None, "complete", {"camera_id": "unknown"})
        annotation = AnnotationRecord(1, frame, "aeris-aircraft-v1", ())
        with self.assertRaisesRegex(cli.CliError, "annotation missing"):
            cli.run_pipeline(self.clip, self.out("wrong-sequence"), stages=cli.ReplayStages(self.replay),
                             annotations=[annotation], render=False, command=["test"])
        assert not self.out("wrong-sequence").exists()
        annotations = [AnnotationRecord(1, FrameRecord(
            1, "clip", i, i / FPS, "provided", W + 1 if i == 0 else W, H, f"{i}.png",
            "0" * 64, None, "complete", {"camera_id": "unknown"}),
            "aeris-aircraft-v1", ()) for i in range(N)]
        with self.assertRaisesRegex(cli.CliError, "dimensions differ"):
            cli.run_pipeline(self.clip, self.out("wrong-size"), stages=cli.ReplayStages(self.replay),
                             annotations=annotations, render=False, command=["test"])
        assert not self.out("wrong-size").exists()


class FailurePathTests(PipelineCase):
    def assert_clean_failure(self, argv, message, name="bad"):
        with contextlib.redirect_stderr(io.StringIO()):
            code = cli.main(argv)
        assert code == 2, message
        assert not self.out(name).exists()
        assert not [p for p in self.tmp.iterdir() if ".partial-" in p.name], "partial output left behind"

    def test_empty_garbage_and_missing_inputs_fail_cleanly(self):
        empty, garbage = self.tmp / "empty.mp4", self.tmp / "garbage.mp4"
        empty.write_bytes(b"")
        garbage.write_bytes(b"not a video" * 100)
        for clip in (empty, garbage, self.tmp / "missing.mp4"):
            self.assert_clean_failure([str(clip), "--output", str(self.out("bad")), "--detections",
                                       str(self.replay)], str(clip))

    def test_existing_output_directory_is_never_overwritten(self):
        (self.out("keep")).mkdir()
        (self.out("keep") / "marker.txt").write_text("precious")
        assert self.run_cli("keep") == 2
        assert (self.out("keep") / "marker.txt").read_text() == "precious"

    def test_an_existing_empty_directory_is_also_refused(self):
        self.out("empty").mkdir()
        assert self.run_cli("empty") == 2
        assert list(self.out("empty").iterdir()) == []

    def test_wrong_size_overlay_is_rejected(self):
        def wrong_overlay(video, out, drawn, fps):
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 10, (W // 2, H // 2))
            for _ in range(3):
                writer.write(np.zeros((H // 2, W // 2, 3), np.uint8))
            writer.release()
            return 0.0
        with mock.patch.object(cli, "render_overlay", wrong_overlay):
            assert self.run_cli("wrong") == 2
        assert not self.out("wrong").exists()
        assert not [p for p in self.tmp.iterdir() if ".partial-" in p.name]

    def test_forecasters_are_fed_raw_detection_centers_not_filtered_state(self):
        seen = []
        original = cli.TrajectoryForecaster.update

        def spy(self, track_id, timestamp_s, center, sigma_px=1.0):
            seen.append((timestamp_s, center))
            return original(self, track_id, timestamp_s, center, sigma_px)
        with mock.patch.object(cli.TrajectoryForecaster, "update", spy):
            assert self.run_cli("spy", ["--camera-motion", "none"]) == 0   # image-space forecasters only
        expected = {(box_at(i)[0] + box_at(i)[2]) / 2 for i in range(N)}
        assert {c[0] for _, c in seen} <= expected and len(seen) == 2 * N      # two models x N observed frames

    def test_forecasters_use_assigned_measurement_without_nearest_center_guess(self):
        from dataclasses import replace

        original_tracker = cli.KalmanTracker
        observed = []
        original_update = cli.TrajectoryForecaster.update

        class MarkedTracker(original_tracker):
            def step(self, timestamp_s, detections, **kwargs):
                return [replace(s, measurement_center_px=(s.measurement_center_px[0] + 7,
                                                           s.measurement_center_px[1]))
                        if s.observed else s for s in super().step(timestamp_s, detections, **kwargs)]

        def spy(self, track_id, timestamp_s, center, sigma_px=1.0):
            observed.append((timestamp_s, center))
            return original_update(self, track_id, timestamp_s, center, sigma_px)

        with mock.patch.object(cli, "KalmanTracker", MarkedTracker), \
                mock.patch.object(cli.TrajectoryForecaster, "update", spy):
            assert self.run_cli("assigned", ["--camera-motion", "none"]) == 0   # image-space forecasters only
        raw = {(box_at(i)[0] + box_at(i)[2]) / 2 for i in range(N)}
        assert len(observed) == 2 * N
        assert {center[0] for _, center in observed} == {value + 7 for value in raw}

    def test_bad_options_and_inputs_fail_cleanly(self):
        base = [str(self.clip), "--output", str(self.out("bad"))]
        for extra in (["--detections", str(self.tmp / "nope.json")], ["--min-score", "2"],
                      ["--min-score", "nan"],
                      ["--stride", "0"], ["--max-frames", "0"], ["--calibration", str(self.tmp / "nope.json")]):
            self.assert_clean_failure([*base, *extra, *([] if "--detections" in extra else
                                                         ["--detections", str(self.replay)])], str(extra))
        (self.tmp / "bad.json").write_text("{not json")
        self.assert_clean_failure([*base, "--detections", str(self.tmp / "bad.json")], "bad replay")
        K = dict(CALIBRATION, K=[[0, 0, 80], [0, 200, 60], [0, 0, 1]])
        (self.tmp / "badcal.json").write_text(json.dumps(K))
        self.assert_clean_failure([*base, "--detections", str(self.replay), "--calibration",
                                   str(self.tmp / "badcal.json")], "bad calibration")

    def test_failure_while_writing_leaves_no_partial_directory(self):
        with mock.patch.object(cli, "render_overlay", side_effect=cli.CliError("render exploded")):
            assert self.run_cli("boom") == 2
        assert not self.out("boom").exists()
        assert not [p for p in self.tmp.iterdir() if ".partial-" in p.name]

    def test_detector_failure_is_reported_with_the_frame(self):
        class Boom:
            name = "boom"

            def preprocess(self, image):
                return None

            def infer(self, prepared):
                raise RuntimeError("inference exploded")

            def postprocess(self, *a, **k):
                return []
        with self.assertRaisesRegex(cli.CliError, "frame 0"):
            cli.run_pipeline(self.clip, self.out("boom"), stages=Boom(), render=False, command=["t"])
        assert not self.out("boom").exists()

    def test_missing_model_is_a_clear_error_not_a_traceback(self):
        config = json.loads((ROOT / "configs/detector/baseline.json").read_text())
        config["model"]["path"] = "models/onnx/does_not_exist.onnx"
        broken = self.tmp / "config.json"
        broken.write_text(json.dumps(config))
        with contextlib.redirect_stderr(io.StringIO()):
            code = cli.main([str(self.clip), "--output", str(self.out("bad")), "--model-config", str(broken)])
        assert code == 2
        assert not self.out("bad").exists()


class RealModelTests(unittest.TestCase):
    """Local-only: the real YOLOX baseline on a clip derived from one photograph (not real video)."""
    image = ROOT / "data/raw/g1-bounded-sample/images/1143132.jpg"
    config = json.loads((ROOT / "configs/detector/baseline.json").read_text())
    model = ROOT / config["model"]["path"]

    @unittest.skipUnless(image.is_file() and model.is_file(), "local weights/FGVC image absent")
    def test_real_baseline_through_the_cli_on_a_photo_derived_clip(self):
        photo = cv2.imread(str(self.image))
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "pan.mp4"
            writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 400))
            for i in range(8):
                writer.write(photo[100:500, 4 * i:4 * i + 640].copy())
            writer.release()
            out = Path(tmp) / "run"
            with contextlib.redirect_stdout(io.StringIO()):
                code = cli.main([str(clip), "--output", str(out), "--sequence-id", "fgvc-1143132-photo-pan",
                                 "--input-note", "derived from one still photograph; not real video"])
            assert code == 0
            manifest = json.loads((out / "run_manifest.json").read_text())
            assert manifest["detector"]["kind"] == "baseline"
            assert manifest["detector"]["model_sha256"] == self.config["model"]["sha256"]
            assert manifest["detector"]["execution_provider"] == ["CPUExecutionProvider"]
            assert manifest["input"]["note"].endswith("not real video") and manifest["accuracy"] is None
            rows = read_csv(out / "tracks.csv")
            assert len(rows) >= 5 and rows[-1]["lifecycle"] == "confirmed"
            truth = json.loads((out / "metrics.json").read_text())["truth_dependent"]
            assert truth["detection"]["available"] is False
            assert (out / "overlay.mp4").stat().st_size > 0
            assert json.loads((out / "benchmark.json").read_text())["stages"]["inference"]["samples"] == 7


if __name__ == "__main__":
    unittest.main()
