"""Loader/QA/conversion/contact-sheet tests on owned synthetic fixtures only."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from aeris.data.contracts import ContractError
from aeris.data.fgvc import convert_fgvc
from aeris.data.validate import decode, validate_dataset
from aeris.data.visualize import contact_sheet

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/data"


class DatasetTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.temp)
        self.canonical = self.temp / "canonical"
        shutil.copytree(FIXTURES / "canonical", self.canonical)

    def lines(self):
        return [json.loads(line) for line in (self.canonical / "annotations.jsonl").read_text().splitlines()]

    def write_lines(self, records):
        (self.canonical / "annotations.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records))

    def index(self):
        return json.loads((self.canonical / "dataset.json").read_text())

    def write_index(self, index):
        (self.canonical / "dataset.json").write_text(json.dumps(index))

    def assertProblem(self, text, dataset=None):
        problems = validate_dataset(dataset or self.canonical)["problems"]
        self.assertTrue(any(text in problem for problem in problems),
                        f"expected a problem containing {text!r}, got {problems}")

    def assertControlledCliProblem(self, text):
        result = subprocess.run(
            [sys.executable, "-m", "aeris.data.validate", str(self.canonical)],
            capture_output=True, text=True, check=False, cwd=self.temp,
            env={**os.environ, "PYTHONPATH": str(ROOT / "python")})
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertTrue(any(text in problem for problem in json.loads(result.stdout)["problems"]),
                        result.stdout)


class CanonicalFixtureTests(DatasetTestCase):
    def test_fixture_is_clean_with_exact_statistics(self):
        report = validate_dataset(self.canonical)
        self.assertEqual(report["problems"], [])
        self.assertEqual(report["duplicate_images_within_split"], 0)
        development, holdout = report["stats"]["development"], report["stats"]["holdout"]
        self.assertEqual((development["frames"], development["negative_frames"],
                          development["objects"], development["ignored_objects"],
                          development["tracks"], development["objects_without_track"]),
                         (4, 1, 4, 1, 1, 1))
        self.assertEqual(development["width_bins"],
                         {"[0,16)": 0, "[16,32)": 0, "[32,64)": 0, "[64,inf)": 3})
        self.assertEqual(holdout["status"], {"complete": 1, "partial": 1, "unlabeled": 1})
        self.assertEqual((holdout["negative_frames"], holdout["tracks"], holdout["occluded_objects"]),
                         (0, 2, 1))
        self.assertEqual(holdout["width_bins"],
                         {"[0,16)": 1, "[16,32)": 0, "[32,64)": 2, "[64,inf)": 0})

    def test_missing_corrupt_undecodable_and_resized_frames(self):
        images = self.canonical / "images"
        (images / "clip-a_0000.png").unlink()
        self.assertProblem("annotations.jsonl:1: missing image clip-a_0000.png")
        data = bytearray((images / "clip-a_0001.png").read_bytes())
        data[-20] ^= 0xFF
        (images / "clip-a_0001.png").write_bytes(bytes(data))
        self.assertProblem("annotations.jsonl:2: checksum mismatch")
        garbage = b"not an image"
        (images / "clip-a_0002.png").write_bytes(garbage)
        records = self.lines()
        records[2]["frame"]["image_checksum"] = hashlib.sha256(garbage).hexdigest()
        self.write_lines(records)
        self.assertProblem("annotations.jsonl:3: cannot decode")
        ok, encoded = cv2.imencode(".png", np.zeros((10, 12, 3), np.uint8))
        (images / "clip-a_0004.png").write_bytes(encoded.tobytes())
        records[3]["frame"]["image_checksum"] = hashlib.sha256(encoded.tobytes()).hexdigest()
        self.write_lines(records)
        self.assertProblem("annotations.jsonl:4: decoded size 12x10 differs from recorded 128x96")

    def test_malformed_boxes_and_lines_are_reported_with_line_numbers(self):
        records = self.lines()
        records[0]["objects"][0]["bbox"]["x_max"] = 129
        records[1]["objects"][0]["bbox"]["x_min"] = records[1]["objects"][0]["bbox"]["x_max"]
        self.write_lines(records)
        self.assertProblem("annotations.jsonl:1: bbox: half-open extents exceed original frame")
        self.assertProblem("annotations.jsonl:2: bbox: require x_min < x_max")
        text = (self.canonical / "annotations.jsonl").read_text()
        (self.canonical / "annotations.jsonl").write_text(
            text.replace('"x_min": 24', '"x_min": NaN', 1) + "\n")
        report = validate_dataset(self.canonical)
        self.assertTrue(any("annotations.jsonl:3: JSON: nonfinite" in p for p in report["problems"]))
        self.assertTrue(any("annotations.jsonl:8: JSON" in p for p in report["problems"]))

    def test_duplicate_objects_and_track_identity_errors(self):
        records = self.lines()
        duplicate = dict(records[0]["objects"][0], annotation_id="copy", track_id=None,
                         availability_reason={"track_id": "copy"})
        records[0]["objects"].append(duplicate)
        self.write_lines(records)
        self.assertProblem("annotations.jsonl:1: duplicate object (same class and bbox) copy")
        records = self.lines()
        records[4]["objects"][1]["track_id"] = "t1"
        self.write_lines(records)
        self.assertProblem("annotations.jsonl:5: track_id: duplicate identity in frame")
        records = self.lines()
        records[1]["objects"][0]["class_id"] = 1
        self.write_lines(records)
        self.assertProblem("sequence clip-a: track_id: class changes within sequence")

    def test_sequence_order_timing_and_membership(self):
        records = self.lines()
        records[1], records[2] = records[2], records[1]
        self.write_lines(records)
        self.assertProblem("sequence clip-a: frame_index: must strictly increase")
        records = self.lines()
        records[0]["class_mapping_version"] = "other-v9"
        records[4]["frame"]["sequence_id"] = "clip-z"
        self.write_lines(records)
        self.assertProblem("annotations.jsonl:1: class_mapping_version other-v9 differs")
        self.assertProblem("annotations.jsonl:5: sequence clip-z is not listed")
        self.write_lines([record for record in self.lines() if record["frame"]["sequence_id"] != "clip-b"])
        self.assertProblem("dataset.json: sequence clip-b has no frames")

    def test_split_leakage_by_group_bytes_and_pixels(self):
        index = self.index()
        index["sequences"][1]["group_id"] = "session-1"
        self.write_index(index)
        self.assertProblem("split leakage: group session-1 is in splits ['development', 'holdout']")

        self.setUp()
        records = self.lines()
        clip_a = records[0]["frame"]
        records[4]["frame"].update(image_path=clip_a["image_path"], image_checksum=clip_a["image_checksum"],
                                   width=clip_a["width"], height=clip_a["height"])
        records[4]["objects"] = []
        self.write_lines(records)
        self.assertProblem("annotations.jsonl:5: split leakage: same encoded bytes as annotations.jsonl:1")

        self.setUp()
        pixels = cv2.imread(str(self.canonical / "images/clip-a_0000.png"))
        ok, encoded = cv2.imencode(".png", pixels, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        (self.canonical / "images/copy.png").write_bytes(encoded.tobytes())
        records = self.lines()
        records[4]["frame"].update(image_path="copy.png", width=128, height=96,
                                   image_checksum=hashlib.sha256(encoded.tobytes()).hexdigest())
        records[4]["objects"] = []
        self.write_lines(records)
        report = validate_dataset(self.canonical)
        self.assertTrue(any("same decoded pixels as annotations.jsonl:1" in p for p in report["problems"]))
        self.assertFalse(any("same encoded bytes" in p for p in report["problems"]))

    def test_image_paths_cannot_escape_image_root(self):
        records = self.lines()
        records[0]["frame"]["image_path"] = "../dataset.json"
        records[1]["frame"]["image_path"] = "/etc/hosts"
        self.write_lines(records)
        self.assertProblem("annotations.jsonl:1: image_path ../dataset.json escapes image_root")
        self.assertProblem("annotations.jsonl:2: image_path /etc/hosts escapes image_root")

    def test_index_errors(self):
        original = self.index()
        index = copy.deepcopy(original)
        index["sequences"].append(dict(index["sequences"][0]))
        self.write_index(index)
        self.assertProblem("duplicate sequence_id clip-a")
        for change, text in ((lambda i: i.pop("group_basis"), "expected exactly keys"),
                             (lambda i: i.__setitem__("schema_version", True), "schema_version"),
                             (lambda i: i.__setitem__("unannotated_attributes", ["color"]), "unannotated"),
                             (lambda i: i.__setitem__("sequences", {}), "sequences must be an array")):
            index = copy.deepcopy(original)
            change(index)
            self.write_index(index)
            with self.subTest(text=text):
                self.assertProblem(text)
        (self.canonical / "dataset.json").write_text('{"schema_version":1,"schema_version":1}')
        self.assertProblem("duplicate key")

    def test_cli_exit_codes(self):
        command = [sys.executable, "-m", "aeris.data.validate", str(self.canonical)]
        environment = {**os.environ, "PYTHONPATH": str(ROOT / "python")}
        clean = subprocess.run(command, capture_output=True, text=True, env=environment, check=False)
        self.assertEqual(clean.returncode, 0, clean.stderr)
        self.assertEqual(json.loads(clean.stdout)["problems"], [])
        (self.canonical / "images/clip-a_0000.png").unlink()
        broken = subprocess.run(command, capture_output=True, text=True, env=environment, check=False)
        self.assertEqual(broken.returncode, 1)

    def test_empty_images_decoder_errors_and_bad_paths_are_controlled(self):
        (self.canonical / "images/clip-a_0000.png").write_bytes(b"")
        # A second failure must still be reported after the empty image.
        (self.canonical / "images/clip-a_0001.png").unlink()
        self.assertProblem("annotations.jsonl:1: cannot decode")
        self.assertProblem("annotations.jsonl:2: missing image")
        self.assertControlledCliProblem("annotations.jsonl:1: cannot decode")
        self.assertIsNone(decode(b""))
        with patch("aeris.data.validate.cv2.imdecode", side_effect=cv2.error("bad data")):
            self.assertIsNone(decode(b"encoded but broken"))
            self.assertProblem("annotations.jsonl:3: cannot decode")
        records = self.lines()
        records[2]["frame"]["image_path"] = "bad\x00.png"
        self.write_lines(records)
        self.assertProblem("annotations.jsonl:3: image_path 'bad\\x00.png': invalid path")
        self.assertControlledCliProblem("annotations.jsonl:3: image_path")
        index = self.index()
        index["image_root"] = "bad\x00root"
        self.write_index(index)
        self.assertProblem("dataset.json: image_root: invalid path")
        self.assertControlledCliProblem("dataset.json: image_root: invalid path")

    def test_attribute_elements_are_checked_without_hashing(self):
        original = self.index()
        for value in ([{}], [[]], [True], [None], ["occluded", {}]):
            with self.subTest(attributes=value):
                self.write_index(dict(original, unannotated_attributes=value))
                self.assertProblem("dataset.json: unannotated_attributes")
                self.assertControlledCliProblem("dataset.json: unannotated_attributes")

    def test_supported_crop_schema_and_original_frame_bounds(self):
        original = self.index()
        cases = [
            ([], "transforms must be an object"),
            ({"unknown": {}}, "unsupported transforms"),
            ({"model_input_crop": []}, "expected exactly bottom_px and reason"),
            ({"model_input_crop": {"bottom_px": 20}}, "expected exactly bottom_px and reason"),
            ({"model_input_crop": {"bottom_px": 20, "reason": " "}}, "reason must be nonempty"),
        ]
        for value in (True, -1, 0.5, "20", None, {}, 1e400):
            cases.append(({"model_input_crop": {"bottom_px": value, "reason": "test"}},
                          "bottom_px must be a nonnegative integer"))
        for value in (96, 1000):
            cases.append(({"model_input_crop": {"bottom_px": value, "reason": "test"}},
                          "annotations.jsonl:1: transforms.model_input_crop.bottom_px"))
        for transforms, message in cases:
            with self.subTest(transforms=transforms):
                # Use a valid JSON number overflowing float, not a forbidden Infinity token.
                text = json.dumps(dict(original, transforms=transforms)).replace("Infinity", "1e400")
                (self.canonical / "dataset.json").write_text(text)
                self.assertProblem(message)
                self.assertControlledCliProblem(message)
                with self.assertRaises(ContractError):
                    contact_sheet(self.canonical, self.temp / "invalid.png")
        for value in (0, 95):
            self.write_index(dict(original, transforms={
                "model_input_crop": {"bottom_px": value, "reason": "supported crop"}}))
            self.assertEqual(validate_dataset(self.canonical)["problems"], [])
            contact_sheet(self.canonical, self.temp / "valid.png")

    def test_versioned_class_registry_is_not_relative_to_cwd(self):
        result = subprocess.run(
            [sys.executable, "-m", "aeris.data.validate", str(self.canonical)],
            capture_output=True, text=True, check=False, cwd=self.temp,
            env={**os.environ, "PYTHONPATH": str(ROOT / "python")})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["problems"], [])
        records = self.lines()
        for record in records:
            for item in record["objects"]:
                item["class_id"] = 987
        self.write_lines(records)
        self.assertProblem("annotations.jsonl:1: object a0: class_id 987 is not in")
        self.assertControlledCliProblem("class_id 987 is not in")
        records[0]["class_mapping_version"] = "nonsense"
        self.write_lines(records)
        self.assertProblem("annotations.jsonl:1: unknown class_mapping_version nonsense")
        index = self.index()
        index["class_mapping_version"] = "nonsense"
        for record in records:
            record["class_mapping_version"] = "nonsense"
        self.write_lines(records)
        self.write_index(index)
        self.assertProblem("dataset.json: unknown class_mapping_version nonsense")
        self.assertControlledCliProblem("dataset.json: unknown class_mapping_version nonsense")


class FgvcConversionTests(DatasetTestCase):
    def setUp(self):
        super().setUp()
        self.source = self.temp / "fgvc_like"
        shutil.copytree(FIXTURES / "fgvc_like", self.source)
        self.out = self.temp / "converted"

    def converted(self):
        return {json.loads(line)["frame"]["sequence_id"]: json.loads(line)
                for line in (self.out / "annotations.jsonl").read_text().splitlines()}

    def test_coordinates_flags_grouping_and_clean_validation(self):
        report = convert_fgvc(self.source, self.out)
        self.assertEqual(report["flags"], {"zero_coordinate_clamped": ["0000002"],
                                           "box_reaches_banner": ["0000003"]})
        self.assertEqual(report["splits"], {"train": 2, "val": 1, "test": 1})
        boxes = {key: record["objects"][0]["bbox"] for key, record in self.converted().items()}
        self.assertEqual(boxes["fgvc-0000001"], {"x_min": 10, "y_min": 20, "x_max": 70, "y_max": 45})
        self.assertEqual(boxes["fgvc-0000002"], {"x_min": 0, "y_min": 14, "x_max": 60, "y_max": 40})
        record = self.converted()["fgvc-0000003"]
        self.assertEqual((record["frame"]["annotation_status"], record["frame"]["timestamp_s"],
                          record["objects"][0]["track_id"]), ("partial", None, None))
        index = json.loads((self.out / "dataset.json").read_text())
        membership = {entry["sequence_id"]: (entry["group_id"], entry["split"]) for entry in index["sequences"]}
        self.assertEqual(membership, {"fgvc-0000001": ("fgvc-0000001", "train"),
                                      "fgvc-0000002": ("fgvc-0000002", "train"),
                                      "fgvc-0000003": ("fgvc-0000003", "val"),
                                      "fgvc-0000004": ("fgvc-0000004", "test")})
        self.assertEqual(index["transforms"]["model_input_crop"]["bottom_px"], 20)
        self.assertEqual(index["unannotated_attributes"], ["occluded", "truncated"])
        result = validate_dataset(self.out)
        self.assertEqual(result["problems"], [])
        self.assertEqual(result["stats"]["train"]["status"]["partial"], 2)
        self.assertEqual(sum(split["negative_frames"] for split in result["stats"].values()), 0)

    def test_conversion_rejects_bad_sources_without_writing(self):
        cases = (
            ("images_test.txt", "0000001\n", "listed in splits train and test"),
            ("images_box.txt", "0000001 11 21 121 45\n", "outside the 120x90 image"),
            ("images_box.txt", "0000001 11 21 70\n", "images_box.txt:1: expected 5 fields"),
            ("images_box.txt", "0000001 5 5 x 9\n", "must be integers"),
            ("images_box.txt", "0000001 50 21 40 45\n", "bbox: require x_min < x_max"),
        )
        for name, text, message in cases:
            with self.subTest(message=message):
                shutil.rmtree(self.source)
                shutil.copytree(FIXTURES / "fgvc_like", self.source)
                path = self.source / name
                lines = path.read_text().splitlines(keepends=True)
                path.write_text(text + "".join(lines[1:]) if name == "images_box.txt"
                                else "".join(lines) + text)
                with self.assertRaisesRegex(ContractError, message):
                    convert_fgvc(self.source, self.out)
                self.assertFalse(self.out.exists())
        shutil.rmtree(self.source)
        shutil.copytree(FIXTURES / "fgvc_like", self.source)
        (self.source / "images/0000004.jpg").unlink()
        with self.assertRaisesRegex(ContractError, "0000004: missing image"):
            convert_fgvc(self.source, self.out)

    def test_symlinked_source_and_output_roots_resolve_to_real_images(self):
        alias = self.temp / "alias"
        alias.symlink_to(self.temp, target_is_directory=True)
        convert_fgvc(alias / "fgvc_like", alias / "converted")
        self.assertEqual(validate_dataset(alias / "converted")["problems"], [])
        index = json.loads((self.out / "dataset.json").read_text())
        self.assertEqual((self.out / index["image_root"]).resolve(), (self.source / "images").resolve())
        # The rendered tile must contain source pixels, not a missing-image placeholder.
        output = self.temp / "sheet.png"
        contact_sheet(alias / "converted", output, columns=1, tile_width=120, max_frames=1)
        sheet = cv2.imread(str(output))
        source = cv2.imread(str(self.source / "images/0000001.jpg"))
        np.testing.assert_array_equal(sheet[2:7, 2:122], source[:5])

    def test_invalid_source_ids_and_symlink_escape_write_nothing(self):
        original_box = (self.source / "images_box.txt").read_text()
        original_train = (self.source / "images_train.txt").read_text()
        for invalid in ("../escape", "/tmp/escape", "sub/0000001", "sub\\0000001", "bad\x00id", "123", "١٢٣٤٥٦٧"):
            with self.subTest(image_id=repr(invalid)):
                (self.source / "images_box.txt").write_text(original_box.replace("0000001", invalid))
                (self.source / "images_train.txt").write_text(original_train.replace("0000001", invalid))
                with self.assertRaisesRegex(ContractError, "images_box.txt:1: invalid image ID"):
                    convert_fgvc(self.source, self.out)
                self.assertFalse(self.out.exists())
        (self.source / "images_box.txt").write_text(original_box)
        (self.source / "images_train.txt").write_text(original_train)
        image = self.source / "images/0000001.jpg"
        outside = self.source / "escape.jpg"
        image.rename(outside)
        image.symlink_to(outside)
        with self.assertRaisesRegex(ContractError, "0000001: image_path 0000001.jpg escapes image_root"):
            convert_fgvc(self.source, self.out)
        self.assertFalse(self.out.exists())

    def test_empty_image_and_bad_roots_have_controlled_conversion_failure(self):
        (self.source / "images/0000001.jpg").write_bytes(b"")
        with self.assertRaisesRegex(ContractError, "0000001: cannot decode"):
            convert_fgvc(self.source, self.out)
        self.assertFalse(self.out.exists())
        result = subprocess.run([sys.executable, "-m", "aeris.data.fgvc", str(self.source), str(self.out)],
                                capture_output=True, text=True, check=False,
                                env={**os.environ, "PYTHONPATH": str(ROOT / "python")})
        self.assertEqual(result.returncode, 1)
        self.assertIn("0000001: cannot decode", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        for source, output in (("bad\x00source", self.out), (self.source, "bad\x00output")):
            with self.assertRaisesRegex(ContractError, "invalid source/output path"):
                convert_fgvc(source, output)
        self.assertFalse(self.out.exists())


class ContactSheetTests(DatasetTestCase):
    def test_sheet_draws_every_frame_and_survives_missing_images(self):
        output = self.temp / "sheet.png"
        summary = contact_sheet(self.canonical, output, columns=4, tile_width=200)
        self.assertEqual((summary["frames_drawn"], summary["frames_total"]), (7, 7))
        sheet = cv2.imread(str(output))
        self.assertEqual(sheet.shape[1], 4 * (200 + 4))
        self.assertEqual(list(summary["size"]), [sheet.shape[1], sheet.shape[0]])
        (self.canonical / "images/clip-b_0000.png").unlink()
        self.assertEqual(contact_sheet(self.canonical, output, columns=1, tile_width=160,
                                      max_frames=5)["frames_drawn"], 5)
        missing_sheet = cv2.imread(str(output))
        # All five tiles have 120 image + 30 caption + 4 border pixels in height.
        missing = missing_sheet[4 * 154 + 2:4 * 154 + 122, 2:162]
        np.testing.assert_array_equal(missing[:10, :10], np.full((10, 10, 3), 60, np.uint8))
        # The red placeholder message crosses the middle; the frame caption stays present.
        self.assertTrue(np.any(missing[45:65, :, 2] > 150))
        self.assertTrue(np.any(missing_sheet[4 * 154 + 122:4 * 154 + 152, 2:162] < 100))
        self.assertFalse(np.array_equal(missing_sheet[2:122, 2:162], missing))

    def test_sheet_draws_truth_boxes_at_original_coordinates(self):
        output = self.temp / "sheet.png"
        contact_sheet(self.canonical, output, columns=1, tile_width=128, max_frames=1)
        sheet = cv2.imread(str(output))
        tile = sheet[2:98, 2:130]  # 2 px border; scale 1.0 for the 128x96 frame
        green = (tile[:, :, 1] > 150) & (tile[:, :, 0] < 60) & (tile[:, :, 2] < 60)
        ys, xs = np.nonzero(green[29:])  # skip the label text above the box
        self.assertEqual((xs.min(), xs.max(), ys.min() + 29, ys.max() + 29), (4, 73, 30, 49))


if __name__ == "__main__":
    unittest.main()
