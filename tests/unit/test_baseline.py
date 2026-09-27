"""Measured-baseline tests (AERIS-007).

Unit tests use the SYNTHETIC AOT-format files in tests/fixtures/baseline/ and hand-built
contract records; their expected values are worked out in the comments. They check the
AOT class policy, the label-only selection, threshold selection and the frozen gate, not
detector quality.

The committed-artifact tests check benchmarks/baseline_manifest.json and
configs/evaluation/regression.json against each other and the frozen definition.

Tests marked REAL need the downloaded AOT frames and the baseline run outputs (Git-ignored)
and skip, with the reason, when they are absent; AERIS_REQUIRE_REAL_BASELINE=1 turns those
skips into failures.
"""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import urllib.error

from aeris.data import aot
from aeris.data.contracts import (AnnotationRecord, BBox, DetectionRecord, FrameRecord,
                                  GroundTruthObject)
from aeris.evaluation import baseline as bl

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "baseline"
FIX = {part: FIXTURES / part / "groundtruth.csv" for part in aot.PARTS}
A, B, C, D, E = ("a" * 32, "b" * 32, "c" * 32, "d" * 32, "e" * 32)
STRICT = os.environ.get("AERIS_REQUIRE_REAL_BASELINE") == "1"


def real(ready: bool, reason: str):
    """Skip (or, under AERIS_REQUIRE_REAL_BASELINE=1, fail) when REAL inputs are missing."""
    if ready:
        return
    if STRICT:
        raise AssertionError(f"REAL evidence required but unavailable: {reason}")
    raise unittest.SkipTest(f"REAL: {reason}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frames_of(part: str, flights) -> dict:
    return aot.read_frames(FIX[part], part, flights)


# ------------------------------------------------------------------ AOT adapter

class CategoryTests(unittest.TestCase):
    def test_categories_and_alias(self):
        self.assertEqual(aot.object_category("Airplane12"), "Airplane")
        self.assertEqual(aot.object_category("Helicopter1"), "Helicopter")
        self.assertEqual(aot.object_category("BIrd4"), "Bird")  # the dataset's one-off spelling
        with self.assertRaises(aot.AotError):
            aot.object_category("Kite1")

    def test_nominal_rule(self):
        # Exactly 6x6 or 16x16, or a 6/16 side with the other no larger than 16 (edge cut).
        for w, h in ((6.0, 6.0), (16.0, 16.0), (16.0, 10.0), (4.0, 6.0)):
            self.assertTrue(aot.is_nominal(w, h), (w, h))
        # Larger than 16 px, or no side of exactly 6/16: a measured extent.
        for w, h in ((6.0, 20.0), (17.0, 17.0), (14.6, 15.0), (10.0, 12.0)):
            self.assertFalse(aot.is_nominal(w, h), (w, h))


class ReaderTests(unittest.TestCase):
    def test_both_column_orders_parse_the_same(self):
        # part2 puts `id` second, as the real part2 does.
        rows = list(aot.iter_rows(FIX["part2"]))
        self.assertEqual(len(rows), 82)  # flights B and E, 41 frames each, at most one object per frame
        line, flight, frame, time_ns, img, item = rows[10]
        self.assertEqual((flight, frame, img), (B, 10, f"{time_ns}{B}.png"))
        self.assertEqual(item.object_id, "Airplane1")
        self.assertEqual(item.box, (1000.0, 1000.0, 1010.0, 1012.0))
        self.assertEqual(item.range_m, 2000.0)

    def write(self, text: str) -> Path:
        handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
        handle.write(text)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return Path(handle.name)

    def row(self, **change):
        base = {"": "0", "time": "1000", "flight_id": A, "img_name": f"1000{A}.png", "frame": "0",
                "id": "Airplane1", "range_distance_m": "10", "is_above_horizon": "1",
                "size_width": "2448", "size_height": "2048", "gt_left": "1", "gt_top": "2",
                "gt_right": "30", "gt_bottom": "40"}
        base.update(change)
        return ",".join(aot.COLUMNS) + "\n" + ",".join(base[k] for k in aot.COLUMNS) + "\n"

    def test_refusals(self):
        cases = {
            "header": "a,b\n1,2\n",
            "box outside": self.row(gt_right="2449"),
            "empty box": self.row(gt_right="1"),
            "size": self.row(size_width="1920"),
            "name": self.row(img_name="other.png"),
            "box without id": self.row(id=""),
            "nan box": self.row(gt_left="nan"),
            "category": self.row(id="Kite1"),
            "frame not integer": self.row(frame="1.5"),
        }
        for name, text in cases.items():
            with self.subTest(name), self.assertRaises(aot.AotError):
                list(aot.iter_rows(self.write(text)))

    def test_nan_range_is_none(self):
        (_, _, _, _, _, item), = aot.iter_rows(self.write(self.row(range_distance_m="nan")))
        self.assertIsNone(item.range_m)

    def test_duplicate_object_and_two_timestamps(self):
        text = self.row() + self.row().split("\n", 1)[1]
        with self.assertRaises(aot.AotError):
            aot.read_frames(self.write(text), "part1", [A])
        second = self.row(time="1001", img_name=f"1001{A}.png", id="Airplane2").split("\n", 1)[1]
        with self.assertRaises(aot.AotError):
            aot.read_frames(self.write(self.row() + second), "part1", [A])


class SequenceIndexTests(unittest.TestCase):
    def test_cameras_make_separate_sequences(self):
        index = aot.sequence_index(FIX)
        # A is in part1 (camera0) and part3 (camera1): two sequences, one flight.
        self.assertEqual(index[f"{A}-camera0"]["part"], "part1")
        self.assertEqual(index[f"{A}-camera1"]["part"], "part3")
        self.assertEqual(index[f"{B}-camera0"]["part"], "part2")
        self.assertEqual(len(index), 6)
        self.assertEqual({v["frames"] for v in index.values()}, {41})
        # camera1 of A starts 40 ms after camera0.
        self.assertEqual(index[f"{A}-camera1"]["first_time_ns"] - index[f"{A}-camera0"]["first_time_ns"],
                         40_000_000)

    def test_same_flight_and_camera_in_two_parts_is_refused(self):
        with self.assertRaises(aot.AotError):
            aot.sequence_index({"part1": FIX["part1"], "part2": FIX["part1"]})


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.frames = frames_of("part1", [A, C, D])
        self.first = aot.sequence_index(FIX)[f"{A}-camera0"]["first_time_ns"]

    def record(self, flight, frame, first=None):
        return aot.annotation(self.frames[(flight, frame)],
                              first_time_ns=self.first if first is None else first,
                              image_path="x.png", image_sha256="0" * 64)

    def test_class_policy(self):
        # A frame 10: Airplane1 40x20 measured -> scored class 0.
        (obj,) = self.record(A, 10).objects
        self.assertEqual((obj.class_id, obj.ignore, obj.track_id), (0, False, "Airplane1"))
        self.assertEqual(obj.bbox, BBox(100.0, 200.0, 140.0, 220.0))
        # A frame 20: nominal 16x16 helicopter and a 6x6 unknown object -> both ignore regions.
        objs = self.record(A, 20).objects
        self.assertEqual([(o.annotation_id, o.ignore) for o in objs],
                         [("Helicopter2", True), ("Airborne3", True)])
        # A frame 30: only a bird -> left out; the frame is an annotated negative.
        rec = self.record(A, 30)
        self.assertEqual(rec.objects, ())
        self.assertTrue(rec.is_negative)

    def test_drone_flock_and_alias(self):
        first = aot.sequence_index(FIX)[f"{C}-camera0"]["first_time_ns"]
        objs = self.record(C, 10, first).objects
        self.assertEqual([(o.annotation_id, o.ignore) for o in objs],
                         [("Airplane1", False), ("Drone2", True)])
        self.assertEqual(self.record(C, 20, first).objects, ())  # Flock3 and BIrd4 left out

    def test_frame_fields(self):
        rec = self.record(A, 10)
        f = rec.frame
        self.assertEqual((f.sequence_id, f.frame_index, f.camera_id, f.annotation_status),
                         (f"{A}-camera0", 10, "camera0", "complete"))
        self.assertEqual((f.timestamp_source, f.width, f.height), ("provided", 2448, 2048))
        self.assertAlmostEqual(f.timestamp_s, 1.0)  # frame 10 is 10 x 100 ms after frame 0
        self.assertEqual(rec.class_mapping_version, "aeris-aircraft-v1")
        self.assertEqual(AnnotationRecord.from_json(rec.to_json()), rec)
        with self.assertRaises(aot.AotError):
            self.record(A, 10, first=self.first + 10**10)

    def test_edge_box_is_scored(self):
        # D frame 0: a 48x48 helicopter touching the bottom-right corner is a measured extent.
        first = aot.sequence_index(FIX)[f"{D}-camera0"]["first_time_ns"]
        (obj,) = self.record(D, 0, first).objects
        self.assertFalse(obj.ignore)
        self.assertEqual(obj.bbox, BBox(2400.0, 2000.0, 2448.0, 2048.0))

    def test_frame_kinds(self):
        self.assertEqual(aot.frame_kind(self.frames[(A, 10)]), "scored")
        self.assertEqual(aot.frame_kind(self.frames[(A, 20)]), "other")
        self.assertEqual(aot.frame_kind(self.frames[(A, 30)]), "other")
        self.assertEqual(aot.frame_kind(self.frames[(A, 0)]), "empty")


class FakeResponse(io.BytesIO):
    def __init__(self, data, etag):
        super().__init__(data)
        self.headers = {"ETag": f'"{etag}"'}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FetchTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.data = b"\x89PNG fake bytes"
        self.md5 = hashlib.md5(self.data).hexdigest()
        self.sha = hashlib.sha256(self.data).hexdigest()

    def fetch(self, etag, **kw):
        with mock.patch.object(aot.urllib.request, "urlopen",
                               return_value=FakeResponse(self.data, etag)) as opened:
            result = aot.fetch("https://x/y.png", self.dir / "y.png", limit=1000,
                               sleep=lambda s: None, **kw)
        return result, opened

    def test_good_download(self):
        result, _ = self.fetch(self.md5)
        self.assertEqual((result["sha256"], result["md5"], result["reused"]), (self.sha, self.md5, False))
        self.assertEqual((self.dir / "y.png").read_bytes(), self.data)

    def test_refusals(self):
        for name, etag, kw in (("etag mismatch", "0" * 32, {}),
                               ("multipart etag", self.md5[:30] + "-2", {}),
                               ("sha pin", self.md5, {"expected_sha256": "1" * 64})):
            with self.subTest(name), self.assertRaises(aot.AotError):
                self.fetch(etag, **kw)
        self.assertFalse((self.dir / "y.png").exists())

    def test_size_cap(self):
        with mock.patch.object(aot.urllib.request, "urlopen",
                               return_value=FakeResponse(self.data, self.md5)):
            with self.assertRaises(aot.AotError):
                aot.fetch("https://x/y.png", self.dir / "y.png", limit=4)

    def test_reuse_pinned_file_without_request(self):
        (self.dir / "y.png").write_bytes(self.data)
        result, opened = self.fetch(self.md5, expected_sha256=self.sha)
        self.assertTrue(result["reused"])
        opened.assert_not_called()

    def test_retries_server_errors_only(self):
        calls = []

        def flaky(request, timeout):
            calls.append(1)
            if len(calls) == 1:
                raise urllib.error.HTTPError("u", 503, "busy", {}, None)
            return FakeResponse(self.data, self.md5)
        with mock.patch.object(aot.urllib.request, "urlopen", side_effect=flaky):
            aot.fetch("https://x/y.png", self.dir / "y.png", limit=1000, sleep=lambda s: None)
        self.assertEqual(len(calls), 2)
        with mock.patch.object(aot.urllib.request, "urlopen",
                               side_effect=urllib.error.HTTPError("u", 404, "no", {}, None)) as opened:
            with self.assertRaises(urllib.error.HTTPError):
                aot.fetch("https://x/z.png", self.dir / "z.png", limit=1000, sleep=lambda s: None)
        self.assertEqual(opened.call_count, 1)


# ------------------------------------------------------------------ selection

def fixture_config() -> dict:
    config = bl.load_json(bl.CONFIG_PATH)
    config = copy.deepcopy(config)
    for part in aot.PARTS:
        config["source"]["groundtruth"][part]["sha256"] = sha256(FIX[part])
    config["selection"]["sequences_per_split"] = {"development": 3, "validation": 1, "holdout": 1}
    config["selection"]["per_sequence_caps"] = {"scored": 1, "empty": 1, "other": 1}
    return config


class SplitTests(unittest.TestCase):
    def test_allocation(self):
        days = [f"2019-01-{d:02d}" for d in range(1, 11)]
        split = bl.split_days(days, "s:")
        counts = {s: sum(v == s for v in split.values()) for s in bl.SPLITS}
        # n = 10: holdout n//4 = 2, validation n//2 - n//4 = 3, development 5.
        self.assertEqual(counts, {"holdout": 2, "validation": 3, "development": 5})
        self.assertEqual(split, bl.split_days(reversed(days), "s:"))  # order-independent
        self.assertNotEqual(split, bl.split_days(days, "other salt:"))
        with self.assertRaises(bl.BaselineError):
            bl.split_days(days[:3], "s:")

    def test_flight_day_uses_earliest_camera(self):
        index = {"f-camera0": {"flight_id": "f", "first_time_ns": 86_400 * 10**9 - 1},
                 "f-camera1": {"flight_id": "f", "first_time_ns": 86_400 * 10**9 + 5}}
        # camera1 starts after midnight UTC; the flight's day is camera0's.
        self.assertEqual(bl.flight_days(index), {"f": "1970-01-01"})

    def test_round_robin_over_days(self):
        index = {f"q{i}": {"flight_id": f"f{i}", "first_time_ns": 0} for i in range(6)}
        day_of = {f"f{i}": ("d1" if i < 4 else "d2") for i in range(6)}
        days = {"d1": "development", "d2": "development"}
        chosen = bl.choose_sequences(index, day_of, days, "development", 4, "s:")
        # Four picks alternate days while both have sequences: two from d2, two from d1.
        self.assertEqual(sorted(day_of[index[q]["flight_id"]] for q in chosen), ["d1", "d1", "d2", "d2"])
        self.assertEqual(chosen, bl.choose_sequences(index, day_of, days, "development", 4, "s:"))
        self.assertEqual(bl.choose_sequences(index, day_of, days, "validation", 4, "s:"), [])


class SelectTests(unittest.TestCase):
    def test_select_fixture(self):
        config = fixture_config()
        selection = bl.select(config, FIXTURES)
        seqs = selection["sequences"]
        # Five flights on five days: holdout 1 day, validation 1, development 3 days.
        self.assertEqual({s: len(v) for s, v in seqs.items()},
                         {"development": 3, "validation": 1, "holdout": 1})
        flights = {s: {q.rsplit("-", 1)[0] for q in v} for s, v in seqs.items()}
        self.assertFalse(flights["development"] & flights["validation"])
        self.assertFalse(flights["development"] & flights["holdout"])
        for split, entries in selection["frames"].items():
            for entry in entries:
                self.assertEqual(entry["frame"] % 10, 0)  # stride 10
                self.assertIn(entry["sequence_id"], seqs[split])
            kinds = [e["kind"] for e in entries]
            for kind in ("scored", "empty", "other"):  # cap 1 per kind per sequence
                self.assertLessEqual(kinds.count(kind), len(seqs[split]))
        self.assertEqual(selection, bl.select(config, FIXTURES))  # deterministic

    def test_pinned_groundtruth_is_checked(self):
        config = fixture_config()
        config["source"]["groundtruth"]["part2"]["sha256"] = "0" * 64
        with self.assertRaises(bl.BaselineError):
            bl.select(config, FIXTURES)

    def test_choose_frames_caps_and_stride(self):
        frames = list(frames_of("part1", [A]).values())
        chosen = bl.choose_frames(frames, [f"{A}-camera0"], 10, {"scored": 5, "empty": 1, "other": 5}, "s:")
        # Candidates 0,10,20,30,40: scored {10}, other {20,30}, empty {0,40} capped to one.
        numbers = [f.frame for f in chosen]
        self.assertEqual(numbers, sorted(numbers))
        self.assertEqual(sorted(n for n in numbers if n in (10, 20, 30)), [10, 20, 30])
        self.assertEqual(len([n for n in numbers if n in (0, 40)]), 1)

    def test_entry_round_trip(self):
        frame = frames_of("part1", [A])[(A, 20)]
        entry = bl.frame_entry(frame, "development")
        self.assertEqual(bl.entry_frame(json.loads(json.dumps(entry))), frame)
        self.assertEqual(entry["sequence_id"], f"{A}-camera0")


# ------------------------------------------------------------------ evaluation and gate

def frame(seq="s", index=0):
    return FrameRecord(1, seq, index, 0.0, "provided", 100, 100, "x.png", "0" * 64, "camera0",
                       "complete", {})


def truth(seq, index, *boxes, ignore=()):
    objs = tuple(GroundTruthObject(f"o{i}", 0, BBox(*b), f"o{i}", i in ignore, False, False, {})
                 for i, b in enumerate(boxes))
    return AnnotationRecord(1, frame(seq, index), "aeris-aircraft-v1", objs)


def det(seq, index, box, score):
    return DetectionRecord(1, seq, index, 0.0, 100, 100, BBox(*box), 0, "aeris-aircraft-v1",
                           score, "test", {})


class ThresholdTests(unittest.TestCase):
    def test_max_f1_ties_to_higher(self):
        # Two objects. TPs at 0.9 and 0.3; one FP at 0.5.
        t = [truth("s", 0, (0, 0, 10, 10)), truth("s", 1, (20, 20, 30, 30))]
        d = [det("s", 0, (0, 0, 10, 10), 0.9), det("s", 1, (20, 20, 30, 30), 0.3),
             det("s", 1, (60, 60, 70, 70), 0.5)]
        best, table = bl.select_threshold(t, d, grid=(0.2, 0.4, 0.6))
        # 0.2: P 2/3, R 1 -> F1 0.8. 0.4: P 1/2, R 1/2 -> 0.5. 0.6: P 1, R 1/2 -> 0.667.
        self.assertEqual(best, 0.2)
        self.assertAlmostEqual(table[0]["f1"], 0.8)
        # Without the FP, 0.2 and... F1 at 0.2 = 1.0 only; ties need equal F1:
        d2 = [det("s", 0, (0, 0, 10, 10), 0.9), det("s", 1, (20, 20, 30, 30), 0.9)]
        best2, _ = bl.select_threshold(t, d2, grid=(0.2, 0.4, 0.6))
        self.assertEqual(best2, 0.6)  # F1 = 1 at every grid value: highest wins

    def test_no_true_positive_gives_none(self):
        t = [truth("s", 0, (0, 0, 10, 10))]
        d = [det("s", 0, (50, 50, 60, 60), 0.9)]
        self.assertIsNone(bl.select_threshold(t, d, grid=(0.5,))[0])


def fake_report(**kw):
    report = {"ap50": 0.5, "ap_50_95": 0.3, "precision": 0.8, "recall": 0.6,
              "true_positives": 60, "false_positives": 15, "false_negatives": 40,
              "support": {"ground_truth": 100},
              "negative_frames": {"count": 50, "false_positives_per_negative_frame": 0.1},
              "width_bins": {name: {"support": s, "recall": r} for (name, _), s, r in zip(
                  bl.WIDTH_BINS, (10, 40, 30, 20), (0.0, 0.5, 0.7, 0.9))}}
    report.update(kw)
    return report


class GateTests(unittest.TestCase):
    rule = {"ap": 0.001}
    support = {"objects": 30, "negative_frames": 30}

    def gates(self, deltas=None):
        report = fake_report()
        values = bl.gated_metrics(report)
        deltas = deltas or {}
        return [bl.tolerance(m, values, deltas.get(m, 0.0), self.rule, report, self.support)
                for m in values], values

    def test_tolerances_and_support(self):
        gates, _ = self.gates({"ap50": 0.002})
        by = {g["metric"]: g for g in gates}
        self.assertEqual(by["ap50"]["tolerance_abs"], 0.004)  # 2 x repeat delta > 0.001 floor
        self.assertEqual(by["ap_50_95"]["tolerance_abs"], 0.001)  # floor
        self.assertAlmostEqual(by["recall"]["tolerance_abs"], 1 / 100)
        self.assertAlmostEqual(by["precision"]["tolerance_abs"], 1 / 75)  # 60 TP + 15 FP
        self.assertAlmostEqual(by["false_positives_per_negative_frame"]["tolerance_abs"], 1 / 50)
        self.assertEqual(by["false_positives_per_negative_frame"]["direction"], "lower_is_better")
        # Bins: support 10 and 20 are below 30 -> not gated; 40 and 30 gated at 1/support.
        self.assertFalse(by["width_bin_recall [0,16)"]["gated"])
        self.assertFalse(by["width_bin_recall [64,inf)"]["gated"])
        self.assertAlmostEqual(by["width_bin_recall [16,32)"]["tolerance_abs"], 1 / 40)
        self.assertAlmostEqual(by["width_bin_recall [32,64)"]["tolerance_abs"], 1 / 30)

    def test_gate_check(self):
        gates, values = self.gates()
        gate = {"gates": gates}
        self.assertEqual(bl.gate_check(gate, values)["status"], "pass")
        worse = dict(values, ap50=values["ap50"] - 0.0011)  # just beyond the 0.001 floor
        result = bl.gate_check(gate, worse)
        self.assertEqual((result["status"], result["failures"][0]["metric"]), ("fail", "ap50"))
        within = dict(values, ap50=values["ap50"] - 0.0009)
        self.assertEqual(bl.gate_check(gate, within)["status"], "pass")
        more_fp = dict(values, false_positives_per_negative_frame=0.1 + 0.03)  # > 1/50
        self.assertEqual(bl.gate_check(gate, more_fp)["status"], "fail")
        better = dict(values, false_positives_per_negative_frame=0.0, recall=0.9)
        self.assertEqual(bl.gate_check(gate, better)["status"], "pass")
        missing = dict(values, recall=None)
        self.assertEqual(bl.gate_check(gate, missing)["status"], "blocked")
        # An ungated bin never fails or blocks, whatever its value.
        self.assertEqual(bl.gate_check(gate, dict(values, **{"width_bin_recall [0,16)": None}))["status"], "pass")


class DiagnosticTests(unittest.TestCase):
    def test_missed_with_other_class(self):
        t = [truth("s", 0, (0, 0, 10, 10), (50, 50, 60, 60), (80, 80, 90, 90), ignore=(2,))]
        d = [det("s", 0, (0, 0, 10, 10), 0.9), det("s", 0, (50, 50, 60, 60), 0.1)]
        unmapped = [{"sequence_id": "s", "frame_index": 0, "coco_class": "kite", "score": 0.6,
                     "bbox": [50, 50, 61, 60]},
                    {"sequence_id": "s", "frame_index": 0, "coco_class": "bird", "score": 0.2,
                     "bbox": [50, 50, 60, 60]}]
        result = bl.missed_with_other_class(t, d, unmapped, threshold=0.5)
        # Object 0 found; object 1's only aircraft box is below 0.5 -> missed; the kite
        # candidate (0.6, IoU 10/11) covers it; the bird (0.2) is below threshold; object 2 ignored.
        self.assertEqual(result["missed"], 1)
        self.assertEqual(result["covered_by_unmapped_class"], {"kite": 1})

    def test_compare_runs(self):
        a = [det("s", 0, (0, 0, 10, 10), 0.9), det("s", 1, (1, 1, 5, 5), 0.2)]
        b = [det("s", 0, (0, 0, 10.5, 10), 0.85)]
        result = bl.compare_runs(a, b)
        self.assertEqual(result["frames_with_count_mismatch"], 1)
        self.assertAlmostEqual(result["max_abs_score_difference"], 0.05)
        self.assertAlmostEqual(result["max_abs_box_difference_px"], 0.5)

    def test_restrict_by_category(self):
        rec = AnnotationRecord(1, frame(), "aeris-aircraft-v1", (
            GroundTruthObject("Airplane1", 0, BBox(0, 0, 10, 10), "Airplane1", False, False, False, {}),
            GroundTruthObject("Helicopter2", 0, BBox(20, 20, 30, 30), "Helicopter2", False, False, False, {})))
        (only,) = bl._restrict([rec], {"Airplane"})
        self.assertEqual([o.ignore for o in only.objects], [False, True])


class DigestTests(unittest.TestCase):
    def test_canonical_digest(self):
        self.assertEqual(bl.digest({"b": 1, "a": [1, 2]}), bl.digest({"a": [1, 2], "b": 1}))
        with self.assertRaises(ValueError):
            bl.digest({"x": float("nan")})
