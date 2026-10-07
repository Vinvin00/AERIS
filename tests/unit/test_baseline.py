"""Baseline manifest, metric-report and regression-schema tests (AERIS-007, offline half).

Nothing here measures real detection quality. The manifest describes a smoke sample of
three partially annotated FGVC images; the real baseline and frozen thresholds are the
deferred AERIS-022. Digest checks skip when the Git-ignored weights/images are absent.
"""
import copy
import hashlib
import json
import re
import unittest
from pathlib import Path

import numpy as np

from aeris.data.contracts import (
    AnnotationRecord, BBox, DetectionRecord, FrameRecord, GroundTruthObject,
)
from aeris.evaluation.metrics import detection_metrics

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = json.loads((ROOT / "benchmarks" / "baseline_manifest.json").read_text())
SCHEMA = json.loads((ROOT / "configs" / "evaluation" / "regression_schema.json").read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# -- tiny validator for the schema subset used by regression_schema.json -------------------
def _types(spec):
    value = spec["type"]
    return value if isinstance(value, list) else [value]


def _is(value, name):
    return {"object": isinstance(value, dict), "array": isinstance(value, list),
            "string": isinstance(value, str), "null": value is None,
            "boolean": isinstance(value, bool),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool)}[name]


def schema_errors(value, spec, path="$"):
    errors = []
    if "type" in spec and not any(_is(value, t) for t in _types(spec)):
        return [f"{path}: expected {spec['type']}, got {type(value).__name__}"]
    if "enum" in spec and value not in spec["enum"]:
        errors.append(f"{path}: {value!r} not in {spec['enum']}")
    if isinstance(value, str) and "pattern" in spec and not re.search(spec["pattern"], value):
        errors.append(f"{path}: does not match {spec['pattern']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and "minimum" in spec \
            and value < spec["minimum"]:
        errors.append(f"{path}: below minimum {spec['minimum']}")
    if isinstance(value, dict):
        errors += [f"{path}: missing {k}" for k in spec.get("required", []) if k not in value]
        for key, sub in spec.get("properties", {}).items():
            if key in value:
                errors += schema_errors(value[key], sub, f"{path}.{key}")
    if isinstance(value, list):
        if len(value) < spec.get("minItems", 0):
            errors.append(f"{path}: fewer than {spec['minItems']} items")
        for i, item in enumerate(value):
            if "items" in spec:
                errors += schema_errors(item, spec["items"], f"{path}[{i}]")
    return errors


def rule_errors(config):
    """Cross-field rules R1-R6 of the schema (the contract AERIS-015 must enforce)."""
    errors, frozen = [], config["status"] == "frozen"
    names = [m["name"] for m in config["metrics"]]
    if len(set(names)) != len(names):
        errors.append("R6: duplicate metric names")
    if config["dataset"]["holdout_opened"] is not False:
        errors.append("R5: holdout must not be opened")
    for m in config["metrics"]:
        rep = m["repeatability"]
        if not frozen:
            if any(v is not None for v in (m["threshold"], m["tolerance"], m["rationale"], rep["runs"],
                                            rep["observed_std"])):
                errors.append(f"R1: draft metric {m['name']} carries values")
            continue
        if m["threshold"] is None or m["tolerance"] is None or not m["rationale"]:
            errors.append(f"R2: frozen metric {m['name']} lacks threshold, tolerance or rationale")
        if rep["runs"] is None or rep["observed_std"] is None or not rep["seeds"] \
                or len(rep["seeds"]) != rep["runs"]:
            errors.append(f"R2: frozen metric {m['name']} lacks repeatability evidence")
        elif m["tolerance"] is not None and m["tolerance"] < rep["observed_std"]:
            errors.append(f"R3: tolerance of {m['name']} is below its measured noise")
    if frozen:
        f, d = config["freeze"], config["dataset"]
        if not f["frozen_at"] or not f["frozen_by_commit"] or f["before_candidate_comparison"] is not True:
            errors.append("R4: freeze record incomplete")
        if not d["development_split_sha256"] or not d["holdout_split_sha256"] \
                or d["development_split_sha256"] == d["holdout_split_sha256"]:
            errors.append("R4: development and holdout splits must be set and distinct")
    return errors


def validate(config):
    return schema_errors(config, SCHEMA) + rule_errors(config) if not schema_errors(config, SCHEMA) \
        else schema_errors(config, SCHEMA)


def metric(name="ap50", **kw):
    base = {"name": name, "direction": "higher_is_better", "support_key": "complete_frame_count",
            "min_support": 30, "threshold": None, "tolerance": None, "rationale": None,
            "repeatability": {"runs": None, "seeds": None, "observed_std": None}}
    base.update(kw)
    return base


def draft():
    return {"schema_version": 1, "protocol_id": "example", "status": "draft",
            "baseline_manifest_sha256": "0" * 64,
            "dataset": {"development_split_sha256": None, "holdout_split_sha256": None, "holdout_opened": False},
            "metrics": [metric()], "unavailable_policy": "fail_closed",
            "freeze": {"frozen_at": None, "frozen_by_commit": None, "before_candidate_comparison": None}}


def frozen():
    """Structural example only; numbers are placeholders for rule testing, not results."""
    cfg = draft()
    cfg["status"] = "frozen"
    cfg["dataset"].update(development_split_sha256="1" * 64, holdout_split_sha256="2" * 64)
    cfg["metrics"] = [metric(threshold=0.5, tolerance=0.02, rationale="example",
                             repeatability={"runs": 3, "seeds": [1, 2, 3], "observed_std": 0.01})]
    cfg["freeze"] = {"frozen_at": "2026-01-01T00:00:00+00:00", "frozen_by_commit": "abc1234",
                     "before_candidate_comparison": True}
    return cfg


class RegressionSchemaTests(unittest.TestCase):
    def test_schema_is_wellformed_and_lists_the_rules(self):
        assert SCHEMA["schema_version"] == 1 and {r["id"] for r in SCHEMA["rules"]} == {f"R{i}" for i in range(1, 7)}
        assert "no threshold value exists" in SCHEMA["$comment"]

    def test_draft_without_values_and_a_complete_frozen_example_are_valid(self):
        assert validate(draft()) == [] and validate(frozen()) == []

    def test_each_rule_rejects_its_violation(self):
        def broken(mutate, base=frozen):
            cfg = copy.deepcopy(base())
            mutate(cfg)
            return validate(cfg)
        cases = {
            "R1": (lambda c: c["metrics"][0].update(threshold=0.9), draft),
            "R2a": (lambda c: c["metrics"][0].update(rationale=""), frozen),
            "R2b": (lambda c: c["metrics"][0]["repeatability"].update(seeds=[1]), frozen),
            "R2c": (lambda c: c["metrics"][0]["repeatability"].update(observed_std=None), frozen),
            "R3": (lambda c: c["metrics"][0].update(tolerance=0.001), frozen),
            "R4a": (lambda c: c["freeze"].update(before_candidate_comparison=False), frozen),
            "R4b": (lambda c: c["freeze"].update(frozen_by_commit=None), frozen),
            "R4c": (lambda c: c["dataset"].update(holdout_split_sha256="1" * 64), frozen),
            "R5": (lambda c: c["dataset"].update(holdout_opened=True), frozen),
            "R6": (lambda c: c["metrics"].append(metric(threshold=0.5, tolerance=0.02, rationale="x",
                   repeatability={"runs": 2, "seeds": [1, 2], "observed_std": 0.0})), frozen),
        }
        for label, (mutate, base) in cases.items():
            assert broken(mutate, base), label

    def test_schema_violations_are_reported(self):
        for mutate in (lambda c: c.pop("freeze"), lambda c: c.update(status="final"),
                       lambda c: c.update(baseline_manifest_sha256="xyz"),
                       lambda c: c["metrics"][0].update(direction="sideways"),
                       lambda c: c["metrics"][0].update(min_support=0),
                       lambda c: c.update(unavailable_policy="pass_open"),
                       lambda c: c.update(metrics=[])):
            cfg = draft()
            mutate(cfg)
            assert schema_errors(cfg, SCHEMA), cfg

    def test_no_threshold_values_are_committed_without_a_real_baseline(self):
        committed = ROOT / "configs" / "evaluation" / "regression.json"
        if committed.exists():           # AERIS-022 may add it; it must then satisfy this contract
            assert validate(json.loads(committed.read_text())) == []
        for path in (ROOT / "configs" / "evaluation").glob("*.json"):
            assert path.name in {"regression_schema.json", "regression.json"}, path.name


class ManifestTests(unittest.TestCase):
    def test_manifest_states_it_is_a_smoke_record_not_a_baseline(self):
        assert MANIFEST["status"] == "smoke_not_a_baseline"
        assert "not a measured baseline" in MANIFEST["status_reason"] and "AERIS-022" in MANIFEST["status_reason"]
        split = MANIFEST["dataset"]["splits"]
        assert split["development"] is None and split["holdout"] is None and split["reason"]
        assert "NOT selected" in MANIFEST["evaluator"]["score_threshold_note"]
        assert MANIFEST["random_seed"] is None and MANIFEST["seed_reason"]

    def test_required_reproducibility_fields_are_present(self):
        for key in ("code_revision", "model", "configs", "evaluator", "environment", "dependency_locks",
                    "dataset", "smoke_measurement", "reproduce"):
            assert MANIFEST[key], key
        env = MANIFEST["environment"]
        assert env["execution_provider"] == ["CPUExecutionProvider"] and env["precision"] == "float32"
        assert {"numpy", "opencv-python-headless", "onnxruntime"} <= set(env["packages"])
        assert re.fullmatch(r"[0-9a-f]{40}", MANIFEST["code_revision"])

    def test_digests_match_the_files_they_describe(self):
        for path, digest in MANIFEST["configs"].items():
            assert sha(ROOT / path) == digest, path
        assert sha(ROOT / MANIFEST["evaluator"]["module"]) == MANIFEST["evaluator"]["sha256"]
        for path, digest in MANIFEST["dependency_locks"].items():
            assert sha(ROOT / path) == digest, path
        detector = json.loads((ROOT / "configs/detector/baseline.json").read_text())
        for key in ("sha256", "size_bytes", "name", "source_url"):
            assert MANIFEST["model"][key] == detector["model"][key], key

    def test_pins_in_the_manifest_match_the_requirement_files(self):
        text = (ROOT / "requirements/detection-macos-arm64-py314.txt").read_text()
        assert f"onnxruntime=={MANIFEST['environment']['packages']['onnxruntime']}" in text
        evaluation = (ROOT / "requirements/evaluation-macos-arm64-py314.txt").read_text()
        assert f"pycocotools=={MANIFEST['evaluator']['pycocotools']}" in evaluation
        assert f"trackeval=={MANIFEST['evaluator']['trackeval']}" in evaluation

    def test_local_artifact_digests_when_present(self):
        model = ROOT / MANIFEST["model"]["path"]
        if model.is_file():
            assert sha(model) == MANIFEST["model"]["sha256"]
        data = ROOT / "data" / "processed" / "g1-bounded-sample"
        if (data / "dataset.json").is_file():
            assert sha(data / "dataset.json") == MANIFEST["dataset"]["dataset_json_sha256"]
            assert sha(data / "annotations.jsonl") == MANIFEST["dataset"]["annotations_jsonl_sha256"]
        for image in MANIFEST["dataset"]["images"]:
            path = ROOT / "data" / "raw" / "g1-bounded-sample" / "images" / image["file"]
            if path.is_file():
                assert sha(path) == image["sha256"] and path.stat().st_size == image["size_bytes"], image["file"]

    def test_smoke_measurement_is_honestly_unavailable(self):
        smoke, images = MANIFEST["smoke_measurement"], MANIFEST["dataset"]["images"]
        assert smoke["metric_report_available"] is False and smoke["reason"] == "no_complete_annotations"
        assert smoke["ap50"] is None and smoke["ap50_95"] is None
        assert all(i["annotation_status"] == "partial" for i in images)
        assert smoke["skipped_frames"] == {"partial": len(images), "unlabeled": 0}
        assert smoke["detections_total"] == sum(i["detections_at_score_floor"] for i in images)

    def test_no_performance_numbers_appear_in_the_manifest(self):
        # "environment.precision" is the inference dtype (a CONTRACTS run-manifest field), not a metric
        text = json.dumps({k: v for k, v in MANIFEST.items() if k != "environment"}).lower()
        for banned in ('"map"', '"hota"', '"idf1"', '"recall"', '"precision"', '"f1"', "latency"):
            assert banned not in text, banned


# -- metric report coverage (AC2): AP, size recall, negatives, supports, unavailable reasons --
def frame(i, status):
    return FrameRecord(1, "s", i, None, "unavailable", 640, 480, f"{i}.png", "0" * 64, None, status,
                       {"camera_id": "synthetic", "timestamp_s": "still"})


def ann(i, boxes, status="complete"):
    objs = tuple(GroundTruthObject(f"o{i}-{j}", 0, b, None, False, False, False, {"track_id": "none"})
                 for j, b in enumerate(boxes))
    return AnnotationRecord(1, frame(i, status), "aeris-aircraft-v1", objs)


def det(i, box, score=0.9):
    return DetectionRecord(1, "s", i, None, 640, 480, box, 0, "aeris-aircraft-v1", score, "synthetic",
                           {"timestamp_s": "still"})


class MetricReportTests(unittest.TestCase):
    def test_complete_labels_give_ap_size_recall_negatives_and_supports(self):
        tiny, small, large = BBox(10, 10, 20, 20), BBox(100, 100, 125, 125), BBox(200, 200, 300, 280)
        annotations = [ann(0, [tiny, large]), ann(1, [small]), ann(2, []), ann(3, [])]
        detections = [det(0, large), det(1, small), det(2, BBox(5, 5, 50, 50))]   # tiny missed, 1 negative error
        report = detection_metrics(annotations, detections, score_threshold=0.5)
        assert report["available"] and report["ap50"] is not None and report["ap50_95"] is not None
        bins = {tuple(b["width_px"]): b for b in report["width_bins"]}
        assert bins[(0, 16)]["support"] == 1 and bins[(0, 16)]["recall"] == 0.0
        assert bins[(16, 32)]["support"] == 1 and bins[(16, 32)]["recall"] == 1.0
        assert bins[(64, None)]["support"] == 1 and bins[(64, None)]["recall"] == 1.0
        assert report["negative_frame_count"] == 2 and report["negative_frame_errors"] == 1
        assert report["complete_frame_count"] == 4 and report["counts"] == {"tp": 2, "fp": 1, "fn": 1}
        assert "false_positives_per_complete_frame" in report and report["skipped"] == {"partial": 0, "unlabeled": 0}

    def test_partial_labels_make_every_truth_metric_unavailable_with_a_reason(self):
        annotations = [ann(i, [BBox(10, 10, 100, 100)], "partial") for i in range(3)]
        report = detection_metrics(annotations, [det(0, BBox(10, 10, 100, 100))], score_threshold=0.5)
        assert report["available"] is False and report["reason"] == "no_complete_annotations"
        assert report["ap50"] is None and report["ap50_95"] is None and report["skipped"]["partial"] == 3
        assert "width_bins" not in report and "counts" not in report

    def test_mixed_complete_and_partial_frames_score_only_the_complete_ones(self):
        annotations = [ann(0, [BBox(10, 10, 100, 100)]), ann(1, [BBox(10, 10, 100, 100)], "partial")]
        report = detection_metrics(annotations, [det(0, BBox(10, 10, 100, 100)), det(1, BBox(300, 300, 400, 400))],
                                   score_threshold=0.5)
        assert report["complete_frame_count"] == 1 and report["skipped"]["partial"] == 1
        assert report["counts"]["fp"] == 0          # the box in the partial frame is not a false positive

    def test_real_sample_reproduces_the_recorded_smoke_result_when_local_data_exists(self):
        data = ROOT / "data" / "processed" / "g1-bounded-sample"
        model = ROOT / MANIFEST["model"]["path"]
        if not (data / "dataset.json").is_file() or not model.is_file():
            self.skipTest("local sample/weights absent")
        import cv2
        from aeris.data.validate import read_annotations
        from aeris.detection import baseline as b
        config = b.load_config()
        runner = b.OnnxRuntimeRunner(b.verified_model_path(config))
        pairs, problems = read_annotations(data)
        assert not problems
        annotations, detections, counts = [a for _, a in pairs], [], {}
        for a in annotations:
            image = cv2.imread(str(ROOT / "data/raw/g1-bounded-sample/images" / a.frame.image_path),
                               cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            found = b.detect(image, runner, config, sequence_id=a.frame.sequence_id,
                             frame_index=a.frame.frame_index)
            detections += found
            counts[a.frame.image_path] = len(found)
        report = detection_metrics(annotations, detections, score_threshold=0.5)
        smoke = MANIFEST["smoke_measurement"]
        assert report["available"] == smoke["metric_report_available"] and report["reason"] == smoke["reason"]
        assert {i["file"]: i["detections_at_score_floor"] for i in MANIFEST["dataset"]["images"]} == counts
        assert np.isfinite(len(detections))


if __name__ == "__main__":
    unittest.main()
