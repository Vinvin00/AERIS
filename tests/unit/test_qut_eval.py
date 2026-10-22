"""QUT evaluation harness: label loading, association counts and split rules (no source media)."""
from collections import defaultdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import qut_eval as q  # noqa: E402

PLAN_DEV = {"S1", "S3", "S5", "S7", "M1", "M3", "M5", "M7"}
PLAN_HOLDOUT = {"S2", "S4", "S6", "M2", "M4", "M6", "M8"}


def det(x, y, score=0.9):
    return SimpleNamespace(bbox=SimpleNamespace(x_min=x - 1, x_max=x + 1, y_min=y - 1, y_max=y + 1),
                           score=score)


def runtime(observed, confirmed_ids=(), detections=None, predictions=None, frames=4):
    return {"frames": [{"frame_index": i, "timestamp_s": i * 0.04} for i in range(frames)],
            "detections": defaultdict(list, detections or {}),
            "observed": defaultdict(list, observed), "confirmed_ids": set(confirmed_ids),
            "predictions": defaultdict(list, predictions or {})}


class ManifestTests(unittest.TestCase):
    def test_manifest_pins_all_fifteen_sequences_with_the_plan_split(self):
        manifest = q.load_manifest()
        split = {s["id"]: s["split"] for s in manifest["sequences"]}
        self.assertEqual({k for k, v in split.items() if v == "dev"}, PLAN_DEV)
        self.assertEqual({k for k, v in split.items() if v == "holdout"}, PLAN_HOLDOUT)
        self.assertEqual(manifest["revision"], "0bb9ece0c70d56a56f0334dfdd867b0d7fce15f9")
        for s in manifest["sequences"]:
            self.assertEqual(len(s["video"]["sha256"]), 64)
            self.assertEqual(len(s["target_labels"]["git_blob_sha1"]), 40)
            self.assertEqual(len(s["target_labels"]["sha256"]), 64)
            self.assertEqual(bool(s["other_labels"]), s["id"].startswith("M"))
            self.assertLess(s["start_frame"], s["end_frame"])
            self.assertIn(s["date"], ("2020-06-23", "2020-06-24"))
        total = sum(f["bytes"] for f in q.all_files(manifest))
        self.assertLessEqual(total, q.BUDGET_BYTES)
        self.assertFalse(any(".DS_Store" in f["remote"] or "/._" in f["remote"] or f["remote"].endswith(".zip")
                             for f in q.all_files(manifest)))

    def test_check_file_rejects_size_and_digest_mismatch(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "f.csv"
            path.write_bytes(b"abc")
            good = {"bytes": 3, "git_blob_sha1": q.git_blob_sha1(b"abc")}
            q.check_file(path, good)
            with self.assertRaisesRegex(q.EvalError, "size"):
                q.check_file(path, {**good, "bytes": 4})
            with self.assertRaisesRegex(q.EvalError, "blob"):
                q.check_file(path, {**good, "git_blob_sha1": "0" * 40})
            with self.assertRaisesRegex(q.EvalError, "SHA-256"):
                q.check_file(path, {**good, "sha256": "0" * 64})
            with self.assertRaisesRegex(q.EvalError, "missing"):
                q.check_file(Path(d) / "absent", good)

    def test_git_blob_digest_matches_git(self):
        self.assertEqual(q.git_blob_sha1(b"hello\n"), "ce013625030ba8dba906f756967f9e9ca394464a")

    def test_holdout_needs_a_written_selection_and_split_membership_is_enforced(self):
        manifest = q.load_manifest()
        with self.assertRaisesRegex(q.EvalError, "selection-record"):
            q.select_sequences(manifest, "holdout", None, None)
        with self.assertRaisesRegex(q.EvalError, "selection-record"):
            q.select_sequences(manifest, "holdout", None, Path("/nonexistent/selection.md"))
        with TemporaryDirectory() as d:
            record = Path(d) / "selection.md"
            record.write_text("chosen: x")
            with self.assertRaisesRegex(q.EvalError, "docs/decisions"):
                q.select_sequences(manifest, "holdout", None, record)
        record = ROOT / "docs/decisions/PLAN-011-qut-eval-tracker-guided-detection.md"
        ids = {s["id"] for s in q.select_sequences(manifest, "holdout", None, record)}
        self.assertEqual(ids, PLAN_HOLDOUT)
        with self.assertRaisesRegex(q.EvalError, "not in the dev split"):
            q.select_sequences(manifest, "dev", ["S2"], None)
        self.assertEqual([s["id"] for s in q.select_sequences(manifest, "dev", ["S5"], None)], ["S5"])

    def test_manifest_rejects_duplicates_and_unknown_split(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "m.json"
            path.write_text(json.dumps({"sequences": [{"id": "S1", "split": "dev"}, {"id": "S1", "split": "dev"}]}))
            with self.assertRaisesRegex(q.EvalError, "duplicate"):
                q.load_manifest(path)
            path.write_text(json.dumps({"sequences": [{"id": "S1", "split": "test"}]}))
            with self.assertRaisesRegex(q.EvalError, "split"):
                q.load_manifest(path)


class LabelTests(unittest.TestCase):
    def write(self, d, text):
        path = Path(d) / "labels.csv"
        path.write_text(text)
        return path

    def test_target_labels_map_global_frames_and_reject_out_of_bounds(self):
        bounds = dict(first_global_frame=1000, frame_count=4, width=40, height=30)
        with TemporaryDirectory() as d:
            path = self.write(d, ",frame,u,v,range\n0,1002,12,15,120\n1,1003,13,16,119\n")
            self.assertEqual(q.read_centroids(path, **bounds), ({2: (12.0, 15.0), 3: (13.0, 16.0)}, 0))
            path = self.write(d, ",frame,u,v,range\n0,1004,12,15,120\n")
            with self.assertRaisesRegex(q.EvalError, "outside video"):
                q.read_centroids(path, **bounds)
            path = self.write(d, ",frame,u,v,range\n0,1002,40,15,120\n")
            with self.assertRaisesRegex(q.EvalError, "coordinates"):
                q.read_centroids(path, **bounds)
            path = self.write(d, ",frame,u,v,range\n0,1003,1,1,1\n1,1002,1,1,1\n")
            with self.assertRaisesRegex(q.EvalError, "order"):
                q.read_centroids(path, **bounds)
            path = self.write(d, ",frame,x,y\n0,1002,1,1\n")
            with self.assertRaisesRegex(q.EvalError, "columns"):
                q.read_centroids(path, **bounds)

    def test_out_of_image_target_rows_are_collected_only_when_asked(self):
        bounds = dict(first_global_frame=1000, frame_count=4, width=40, height=30)
        with TemporaryDirectory() as d:
            path = self.write(d, ",frame,u,v\n0,1001,12,15\n1,1002,12,-19.6\n2,1003,90,15\n")
            with self.assertRaisesRegex(q.EvalError, "coordinates"):
                q.read_centroids(path, **bounds)
            off = []
            self.assertEqual(q.read_centroids(path, **bounds, off_image=off), ({1: (12.0, 15.0)}, 0))
            self.assertEqual(off, [2, 3])
            path = self.write(d, ",frame,u,v\n0,1001,nan,15\n")
            with self.assertRaisesRegex(q.EvalError, "finite"):
                q.read_centroids(path, **bounds, off_image=[])

    def test_other_aircraft_rows_outside_video_are_dropped_and_counted(self):
        bounds = dict(first_global_frame=1000, frame_count=4, width=40, height=30)
        with TemporaryDirectory() as d:
            path = self.write(d, ",frame,u,v\n0,999,1,1\n1,1001,2,2\n2,1004,3,3\n")
            self.assertEqual(q.read_centroids(path, **bounds, drop_outside=True), ({1: (2.0, 2.0)}, 2))

    def test_frame_count_and_alignment_rule(self):
        q.check_frame_count({"id": "S0", "decoded_frames": 4}, 4)
        with self.assertRaisesRegex(q.EvalError, "pins"):
            q.check_frame_count({"id": "S0", "decoded_frames": 4}, 5)
        q.check_alignment({"id": "S0"}, {1: (0, 0), 3: (0, 0)}, 4)
        with self.assertRaisesRegex(q.EvalError, "last decoded"):
            q.check_alignment({"id": "S0"}, {1: (0, 0), 2: (0, 0)}, 4)

    def test_manifest_alignment_offsets_are_the_measured_ones(self):
        offsets = {s["id"]: s["first_decoded_global_frame"] - s["start_frame"]
                   for s in q.load_manifest()["sequences"]}
        self.assertEqual({k: v for k, v in offsets.items() if v}, {"S4": -1})
        inferred = {s["id"] for s in q.load_manifest()["sequences"] if s["alignment"].startswith("inferred")}
        self.assertEqual(inferred, {"S1", "S2", "S4"})


class EvaluationTests(unittest.TestCase):
    def test_fragments_and_switches(self):
        self.assertEqual(q.fragments_and_switches([]), (0, 0))
        self.assertEqual(q.fragments_and_switches(["a", "a", "a"]), (1, 0))
        self.assertEqual(q.fragments_and_switches(["a", "b", "a", "a", "c"]), (3, 3))

    def test_track_recall_counts_only_unique_confirmed_observations(self):
        target = {0: (10.0, 10.0), 1: (12.0, 10.0), 2: (14.0, 10.0), 3: (16.0, 10.0)}
        observed = {0: [("t1", "tentative", (10.0, 10.0))],
                    1: [("t1", "confirmed", (12.0, 10.0))],
                    2: [("t1", "confirmed", (14.0, 10.0)), ("t2", "confirmed", (15.0, 11.0))],  # ambiguous
                    3: [("t2", "confirmed", (16.0, 10.0))]}
        detections = {0: [det(10, 10)], 1: [det(12, 10, score=0.1)], 2: [det(14, 10), det(200, 200)]}
        result = q.evaluate(runtime(observed, {"t1", "t2"}, detections), target, [])
        self.assertEqual(result["labelled_target_frames"], 4)
        self.assertEqual(result["target_detection_hits"], 2)       # frame 1 is below track score
        self.assertEqual(result["target_track_hits"], 2)           # frames 1 and 3
        self.assertEqual((result["target_fragments"], result["target_id_switches"]), (2, 1))
        self.assertEqual(result["non_target_confirmed_tracks"], 0)

    def test_guard_measures_off_target_impurity_swaps_and_non_target(self):
        target = {0: (10.0, 10.0), 1: (10.0, 10.0)}
        other = {0: (300.0, 300.0), 1: (300.0, 300.0)}
        observed = {
            0: [("t1", "confirmed", (10.0, 10.0)), ("t2", "confirmed", (305.0, 300.0)),
                ("t3", "confirmed", (500.0, 500.0)), ("t5", "tentative", (700.0, 700.0))],
            # t1 drifts onto the other aircraft: impurity, identity swap and no off-target count
            1: [("t1", "confirmed", (300.0, 302.0)), ("t2", "lost", (900.0, 900.0)),
                ("t4", "confirmed", (11.0, 10.0))]}
        result = q.evaluate(runtime(observed, {"t1", "t2", "t3", "t4", "t6"}), target, [other])
        self.assertEqual(result["off_target_observations"], 1)        # t3 only; tentative and lost ignored
        self.assertEqual(result["target_track_impurity"], 1)          # t1 at frame 1
        self.assertEqual(result["identity_swaps"], 1)                 # t1 matched the other aircraft
        self.assertEqual(result["non_target_confirmed_tracks"], 2)    # t3 and t6 (never observed)
        self.assertEqual((result["target_fragments"], result["target_id_switches"]), (2, 1))
        self.assertEqual(result["ambiguous_swap_frames"], 0)

    def test_ambiguous_frames_count_label_pairs_within_twice_the_gate_without_observations(self):
        target = {0: (10.0, 10.0), 1: (10.0, 10.0), 2: (10.0, 10.0)}
        other = {0: (90.0, 10.0), 1: (120.0, 10.0), 2: (10.0, 40.0)}   # 80 px, 110 px, 30 px apart
        result = q.evaluate(runtime({}, set()), target, [other])
        self.assertEqual(result["ambiguous_swap_frames"], 2)

    def test_impurity_counts_only_confirmed_observations(self):
        target = {0: (10.0, 10.0), 1: (10.0, 10.0)}
        observed = {0: [("t1", "confirmed", (10.0, 10.0))], 1: [("t1", "lost", (300.0, 10.0))]}
        self.assertEqual(q.evaluate(runtime(observed, {"t1"}), target, [])["target_track_impurity"], 0)
        observed[1] = [("t1", "confirmed", (300.0, 10.0))]
        self.assertEqual(q.evaluate(runtime(observed, {"t1"}), target, [])["target_track_impurity"], 1)

    def test_swap_needs_observation_outside_target_gate_and_close_frames_are_ambiguous(self):
        target = {0: (10.0, 10.0), 1: (10.0, 10.0)}
        other = {0: (40.0, 10.0), 1: (400.0, 10.0)}     # frame 0: aircraft 30 px apart
        observed = {0: [("t1", "confirmed", (38.0, 10.0))],  # near both labels: not a swap
                    1: [("t1", "confirmed", (10.0, 10.0))]}
        result = q.evaluate(runtime(observed, {"t1"}), target, [other])
        self.assertEqual((result["identity_swaps"], result["ambiguous_swap_frames"]), (0, 1))

    def test_forecasts_scored_only_for_target_track_against_interpolated_future_label(self):
        target = {0: (10.0, 20.0), 1: (11.0, 20.0), 2: (12.0, 20.0), 3: (13.0, 20.0)}
        good = SimpleNamespace(track_id="t1", center=(12.0, 20.0), target_time_s=0.1,
                               model_name="constant_velocity", horizon_s=0.1)
        other = SimpleNamespace(track_id="t9", center=(0.0, 0.0), target_time_s=0.1,
                                model_name="constant_velocity", horizon_s=0.1)
        rt = runtime({0: [("t1", "confirmed", (10.0, 20.0))]}, {"t1"}, predictions={0: [good, other]})
        summary = q.summarise(q.evaluate(rt, target, []))
        self.assertEqual(summary["forecast_centroid_error"]["constant_velocity@0.1"]["support"], 1)
        self.assertAlmostEqual(summary["forecast_centroid_error"]["constant_velocity@0.1"]["median_px"], 0.5)
        self.assertEqual(list(q.evaluate(rt, target, [])["forecast_errors_px"]), ["0|constant_velocity@0.1"])

    def test_pool_is_frame_weighted_and_keeps_sequence_keys(self):
        base = {k: 0 for k in q.COUNT_KEYS}
        a = {**base, "labelled_target_frames": 2, "target_track_hits": 1, "forecast_errors_px": {"0|cv@0.1": 1.0}}
        b = {**base, "labelled_target_frames": 6, "target_track_hits": 5,
             "forecast_errors_px": {"0|cv@0.1": 3.0, "1|cv@0.1": 5.0}}
        pooled = q.pool({"S1": a, "S3": b})
        self.assertEqual(pooled["target_track_recall"], 0.75)
        self.assertEqual(pooled["forecast_centroid_error"]["cv@0.1"],
                         {"support": 3, "median_px": 3.0, "p90_px": 5.0, "availability_reason": None})
        self.assertIsNone(q.describe([])["median_px"])


def pooled(**over):
    base = {"target_track_recall": 0.5, "target_fragments": 10, "identity_swaps": 1,
            "off_target_observations": 0, "target_track_impurity": 40, "non_target_confirmed_tracks": 8}
    return {**base, **over}


class RuleTests(unittest.TestCase):
    def test_bound_handles_zero_baseline(self):
        self.assertEqual(q.bound(0), 5)
        self.assertEqual(q.bound(40), 50)

    def test_rule_accepts_equal_and_rejects_each_guard(self):
        self.assertEqual(q.passes_rule(pooled(), pooled()), (True, []))
        self.assertTrue(q.passes_rule(pooled(off_target_observations=5, target_track_impurity=50), pooled())[0])
        for over, reason in ((dict(target_track_recall=0.49), "recall"), (dict(target_fragments=11), "fragments"),
                             (dict(identity_swaps=2), "swaps"), (dict(off_target_observations=6), "off_target"),
                             (dict(target_track_impurity=51), "impurity"),
                             (dict(non_target_confirmed_tracks=14), "non_target")):
            ok, failed = q.passes_rule(pooled(**over), pooled())
            self.assertFalse(ok, over)
            self.assertIn(reason, " ".join(failed))

    def test_rank_prefers_fewer_fragments_then_recall_then_off_target_then_order(self):
        cands = [pooled(target_fragments=5), pooled(target_fragments=4, target_track_recall=0.4),
                 pooled(target_fragments=4, target_track_recall=0.6, off_target_observations=3),
                 pooled(target_fragments=4, target_track_recall=0.6, off_target_observations=1)]
        best = min(range(len(cands)), key=lambda i: q.rank_key(cands[i], i))
        self.assertEqual(best, 3)
        self.assertLess(q.rank_key(pooled(), 0), q.rank_key(pooled(), 1))

    def test_compare_reports_common_forecast_subset_and_regressions(self):
        seq = {"S1": {"target_track_recall": 0.5}, "S3": {"target_track_recall": 0.5}}
        a = {"name": "a", "split": "dev", "pooled": pooled(), "sequences": seq}
        b = {"name": "b", "split": "dev", "pooled": pooled(target_fragments=3),
             "sequences": {"S1": {"target_track_recall": 0.4}, "S3": {"target_track_recall": 0.9}}}
        out = q.compare(a, b, {"S1:0|cv@0.1": 2.0, "S1:1|cv@0.1": 9.0}, {"S1:0|cv@0.1": 1.0, "S3:4|cv@0.1": 1.0})
        self.assertTrue(out["passes_plan011_rule"])
        self.assertEqual(out["sequences_with_lower_track_recall"], ["S1"])
        common = out["forecast_common_subset"]
        self.assertEqual((common["common"], common["only_baseline"], common["only_candidate"]), (1, 1, 1))
        self.assertEqual(common["baseline"]["cv@0.1"]["median_px"], 2.0)


class CandidateListTests(unittest.TestCase):
    def write(self, d, candidates, pan=None):
        import hashlib
        config = ROOT / "configs/tracker/baseline.json"
        digest = hashlib.sha256(config.read_bytes()).hexdigest()
        candidates = [{"tracker_config": "configs/tracker/baseline.json", "tracker_config_sha256": digest, **c}
                      for c in candidates]
        path = Path(d) / "c.json"
        path.write_text(json.dumps({"synthetic_pan": pan or {"amplitude_px": 1.0, "seed": 1},
                                    "candidates": candidates}))
        return path

    def test_only_listed_variants_run_and_the_list_hash_is_returned(self):
        with TemporaryDirectory() as d:
            path = self.write(d, [{"name": "a", "task": "T", "profile": "fast"}])
            candidate, pan, digest = q.load_candidate("a", path)
            self.assertEqual((candidate["profile"], pan["seed"], len(digest)), ("fast", 1, 64))
            with self.assertRaisesRegex(q.EvalError, "not on the candidate list"):
                q.load_candidate("b", path)
            path = self.write(d, [{"name": "a", "task": "T", "tracker_config_sha256": "0" * 64}])
            with self.assertRaisesRegex(q.EvalError, "changed since it was listed"):
                q.load_candidate("a", path)
            path = self.write(d, [{"name": "a", "task": "T"}, {"name": "a", "task": "T"}])
            with self.assertRaisesRegex(q.EvalError, "duplicate"):
                q.load_candidate("a", path)
            path = self.write(d, [{"name": str(i), "task": "T"} for i in range(13)])
            with self.assertRaisesRegex(q.EvalError, "12"):
                q.load_candidate("0", path)

    def test_repository_list_holds_the_032_baselines(self):
        fast, pan, _ = q.load_candidate("baseline")
        self.assertEqual((fast["profile"], fast["tracker_config"]), ("fast", "configs/tracker/baseline.json"))
        self.assertEqual(pan, {"amplitude_px": 120.0, "seed": 11})


class GuidedHarnessTests(unittest.TestCase):
    def test_scratch_results_get_their_own_raw_file(self):
        recorded = q.raw_path_for(ROOT / "benchmarks/qut_eval/a-dev.json")
        self.assertEqual(recorded, q.RAW / "a-dev.json")
        scratch = q.raw_path_for(Path("/tmp/x/a-dev.json"))
        self.assertNotEqual(scratch, recorded)
        self.assertTrue(scratch.name.endswith("-a-dev.json"))

    def test_track_passes_crops_as_extend_only_and_crop_hits_are_reported_beside_frame_hits(self):
        from aeris.data.contracts import BBox, DetectionRecord

        def rec(i, x, score=0.9):
            return DetectionRecord(1, "s", i, i * 0.04, 100, 100, BBox(x - 2.0, 48.0, x + 2.0, 52.0), 0,
                                   "aeris-aircraft-v1", score, "t", {})
        frames = [{"frame_index": i, "timestamp_s": i * 0.04} for i in range(8)]
        detected = {"frames": frames, "decoded_frames": 8, "detect_ms": {"p50": 1.0, "p95": 1.0}, "cache": "hit",
                    "detections": [rec(i, 10.0 + i) for i in range(8) if i not in (5, 6)]}
        seen = []

        def guided(image, states, found, *, sequence_id, frame_index, timestamp_s):
            seen.append((image, frame_index))
            live = [s for s in states if s.lifecycle in ("confirmed", "lost")]
            return ([rec(frame_index, 10.0 + frame_index)] if live and not found else []), len(live)
        import json as _json
        cfg = _json.loads((ROOT / "configs/tracker/baseline.json").read_text())
        rt = q.track(detected, "s", cfg, guided, iter(range(8)))
        self.assertEqual([i for _, i in seen], list(range(8)))
        self.assertEqual([img for img, _ in seen], list(range(8)))          # one decoded frame per step
        self.assertEqual(rt["guided_crops"]["extend_only_matches"], 2)
        observed = sorted(i for i, items in rt["observed"].items() if items)
        self.assertEqual(observed, list(range(8)))
        target = {i: (10.0 + i, 50.0) for i in range(8)}
        result = q.evaluate(rt, target, [])
        self.assertEqual((result["target_detection_hits"], result["target_detection_hits_with_crops"]), (6, 8))
        plain = q.track(detected, "s", cfg)
        self.assertIsNone(plain["guided_crops"])
        self.assertEqual(q.evaluate(plain, target, [])["target_detection_hits_with_crops"], 6)


class SplitAndPanTests(unittest.TestCase):
    def test_overlap_check_finds_same_date_shared_frames_only(self):
        manifest = {"sequences": [
            {"id": "A", "split": "dev", "date": "d1", "start_frame": 0, "end_frame": 10},
            {"id": "B", "split": "holdout", "date": "d1", "start_frame": 8, "end_frame": 20},
            {"id": "C", "split": "holdout", "date": "d2", "start_frame": 0, "end_frame": 10}]}
        self.assertEqual(q.split_overlaps(manifest),
                         [{"dev": "A", "holdout": "B", "date": "d1", "global_frames": [8, 10], "frames": 3}])

    def test_real_manifest_overlap_is_the_known_m3_m4_one(self):
        overlaps = q.split_overlaps(q.load_manifest())
        self.assertEqual([(o["dev"], o["holdout"], o["frames"]) for o in overlaps], [("M3", "M4", 5)])

    def test_pan_is_seeded_bounded_and_shifts_labels_identically(self):
        a = [q.pan_offset(i, 60, 3) for i in range(200)]
        self.assertEqual(a, [q.pan_offset(i, 60, 3) for i in range(200)])
        self.assertNotEqual(a, [q.pan_offset(i, 60, 4) for i in range(200)])
        self.assertTrue(all(abs(x) <= 60 and abs(y) <= 60 for x, y in a))
        self.assertGreater(max(abs(x) for x, _ in a), 5)
        offsets = {0: (5.0, -2.0), 1: (-20.0, 0.0)}
        self.assertEqual(q.shift_labels({0: (10.0, 10.0), 1: (10.0, 10.0)}, offsets, 40, 30),
                         ({0: (15.0, 8.0)}, 1))

    def test_panned_stages_translate_the_frame_before_detection(self):
        import numpy as np
        seen = []
        inner = SimpleNamespace(name="x", preprocess=lambda img: seen.append(img) or img)
        stages = q.PannedStages(inner, {0: (2.0, 1.0)})
        image = np.zeros((4, 6, 3), np.uint8)
        image[0, 0] = 255
        stages.preprocess(image)
        self.assertEqual(seen[0][1, 2].tolist(), [255, 255, 255])
        self.assertEqual(seen[0][0, 0].tolist(), [114, 114, 114])

    def test_holdout_rejects_synthetic_pan(self):
        with self.assertRaisesRegex(q.EvalError, "development-only"):
            q.select_sequences(q.load_manifest(), "holdout", None, ROOT / "docs/decisions/PLAN-011-qut-eval-tracker-guided-detection.md", pan=True)

    def test_subset_runs_get_their_own_file_and_holdout_results_are_never_overwritten(self):
        with TemporaryDirectory() as d:
            args = SimpleNamespace(name="a", split="dev", pan=False, sequences=["S5", "S1"], output_dir=Path(d))
            self.assertEqual(q.output_path(args).name, "a-dev-S1-S5.json")
            args.sequences = None
            self.assertEqual(q.output_path(args).name, "a-dev.json")
            args.split = "holdout"
            (Path(d) / "a-holdout.json").write_text("{}")
            with self.assertRaisesRegex(q.EvalError, "every holdout run is kept"):
                q.output_path(args)

    def test_compare_refuses_raw_errors_that_do_not_belong_to_the_report(self):
        with TemporaryDirectory(dir=ROOT / "data") as d:
            path = Path(d) / "raw.json"
            data = json.dumps({"S1:0|cv@0.1": 1.0}, sort_keys=True).encode()
            path.write_bytes(data)
            import hashlib
            report = {"name": "a", "raw_forecast_errors": {"path": str(path.relative_to(ROOT)),
                                                           "sha256": hashlib.sha256(data).hexdigest()}}
            self.assertEqual(q.load_raw(report), {"S1:0|cv@0.1": 1.0})
            path.write_text(json.dumps({"S5:0|cv@0.1": 2.0}))
            with self.assertRaisesRegex(q.EvalError, "does not belong"):
                q.load_raw(report)


if __name__ == "__main__":
    unittest.main()
