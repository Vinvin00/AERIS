"""Offline adversarial contract tests; no real perception quality is asserted."""
from dataclasses import replace
import json
import unittest

from aeris.data.contracts import (
    AnnotationRecord, BBox, ContractError, DetectionRecord, FrameRecord,
    GroundTruthObject, PredictionRecord, validate_sequence,
)


def frame(**changes):
    values = dict(schema_version=1, sequence_id="clip-1", frame_index=0,
                  timestamp_s=0.0, timestamp_source="decode_pts", width=640, height=480,
                  image_path="frames/000.png", image_checksum="a" * 64, camera_id=None,
                  annotation_status="complete", availability_reason={"camera_id": "not recorded"})
    values.update(changes)
    return FrameRecord(**values)


def truth(**changes):
    values = dict(annotation_id="a1", class_id=0, bbox=BBox(0, 0, 640, 480),
                  track_id=None, ignore=False, occluded=False, truncated=False,
                  availability_reason={"track_id": "still image has no identity annotation"})
    values.update(changes)
    return GroundTruthObject(**values)


def detection(**changes):
    values = dict(schema_version=1, sequence_id="clip-1", frame_index=0, timestamp_s=0,
                  width=640, height=480, bbox=BBox(10, 20, 30, 40), class_id=0,
                  class_mapping_version="aircraft-v1", score=0.75,
                  model_provenance="fixture-only:no-trained-model", availability_reason={})
    values.update(changes)
    return DetectionRecord(**values)


def prediction(**changes):
    values = dict(schema_version=1, sequence_id="clip-1", track_id="t1", issue_time_s=2,
                  horizon_s=0.1, target_time_s=2.1, center=(650.0, -10.0),
                  covariance=((4.0, 1.0), (1.0, 9.0)), model_name="constant-velocity",
                  availability_reason={})
    values.update(changes)
    return PredictionRecord(**values)


class ContractTests(unittest.TestCase):
    def test_all_record_roundtrips(self):
        records = [frame(), truth(), detection(), prediction(), BBox(0.5, 1.5, 10, 20),
                   AnnotationRecord(1, frame(), "aircraft-v1", (truth(),))]
        for record in records:
            with self.subTest(record=type(record).__name__):
                decoded = type(record).from_json(record.to_json())
                self.assertEqual(decoded, record)
                self.assertEqual(decoded.to_dict(), record.to_dict())

    def test_negative_partial_and_unlabeled_are_distinct(self):
        for status in ("complete", "partial", "unlabeled"):
            record = AnnotationRecord(1, frame(annotation_status=status), "v1", ())
            decoded = AnnotationRecord.from_json(record.to_json())
            self.assertEqual(decoded.frame.annotation_status, status)
            self.assertEqual(decoded.is_negative, status == "complete")
        with self.assertRaisesRegex(ContractError, "unlabeled"):
            AnnotationRecord(1, frame(annotation_status="unlabeled"), "v1", (truth(),))

    def test_still_image_null_time_and_track_identity(self):
        image = frame(timestamp_s=None, timestamp_source="unavailable",
                      availability_reason={"camera_id": "unknown", "timestamp_s": "still image"})
        record = AnnotationRecord(1, image, "v1", (truth(),))
        decoded = AnnotationRecord.from_json(record.to_json())
        self.assertIsNone(decoded.frame.timestamp_s)
        self.assertIsNone(decoded.objects[0].track_id)
        det = detection(timestamp_s=None, availability_reason={"timestamp_s": "still image"})
        self.assertEqual(DetectionRecord.from_json(det.to_json()), det)
        validate_sequence([image, replace(image, frame_index=1)])

    def test_missing_stale_and_empty_availability_reasons(self):
        for reasons in ({}, {"camera_id": ""}, {"camera_id": "unknown", "timestamp_s": "x"}):
            with self.subTest(reasons=reasons), self.assertRaisesRegex(ContractError, "availability_reason"):
                frame(availability_reason=reasons)
        with self.assertRaisesRegex(ContractError, "availability_reason"):
            truth(availability_reason={})
        with self.assertRaisesRegex(ContractError, "availability_reason"):
            detection(timestamp_s=None)

    def test_timestamp_source_consistency_and_inferred_flag(self):
        for changes in ({"timestamp_s": None}, {"timestamp_source": "unavailable"},
                        {"timestamp_source": "invented"}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ContractError, "timestamp_source"):
                frame(**changes)
        inferred = frame(timestamp_source="inferred_fps")
        self.assertEqual(FrameRecord.from_json(inferred.to_json()).timestamp_source, "inferred_fps")

    def test_numeric_types_finiteness_and_ids(self):
        for field in ("schema_version", "frame_index", "width", "height"):
            for value in (True, 1.5, -1, "1"):
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ContractError, field):
                    frame(**{field: value})
        for value in (-1, True, float("nan"), float("inf"), "0", 10**400):
            with self.subTest(value=value), self.assertRaisesRegex(ContractError, "timestamp_s"):
                frame(timestamp_s=value)
        for field in ("sequence_id", "camera_id"):
            for value in ("", " ", "bad id", 7, False):
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ContractError, field):
                    frame(**{field: value})
        for value in (True, -1, "aircraft"):
            with self.assertRaisesRegex(ContractError, "class_id"):
                truth(class_id=value)
        with self.assertRaisesRegex(ContractError, "schema_version"):
            frame(schema_version=2)
        for field in ("width", "height"):
            with self.assertRaisesRegex(ContractError, field):
                frame(**{field: 0})

    def test_frame_metadata_rejects_malformed_values(self):
        for field, value in (("image_path", ""), ("image_checksum", "abc"),
                             ("annotation_status", "negative")):
            with self.assertRaisesRegex(ContractError, field):
                frame(**{field: value})

    def test_bbox_half_open_edges_and_malformed_geometry(self):
        BBox(0, 0, 640, 480).validate_bounds(640, 480)
        for coordinates in ((0, 0, 0, 1), (2, 0, 1, 1), (-1, 0, 1, 1),
                            (0, 1, 1, 1), (0, 0, float("nan"), 1),
                            (0, 0, 1, float("inf")), (False, 0, 1, 1)):
            with self.subTest(coordinates=coordinates), self.assertRaisesRegex(ContractError, "bbox"):
                BBox(*coordinates)
        with self.assertRaisesRegex(ContractError, "original frame"):
            detection(bbox=BBox(0, 0, 641, 10))
        with self.assertRaisesRegex(ContractError, "original frame"):
            AnnotationRecord(1, frame(), "v1", (truth(bbox=BBox(0, 0, 10, 481)),))
        with self.assertRaisesRegex(ContractError, "bbox"):
            truth(bbox=[0, 0, 1, 1])

    def test_duplicate_annotation_and_track_ids(self):
        item = truth(track_id="t1", availability_reason={})
        for other in (item, replace(item, annotation_id="a2")):
            with self.assertRaisesRegex(ContractError, "duplicate"):
                AnnotationRecord(1, frame(), "v1", (item, other))
        AnnotationRecord(1, frame(), "v1", (truth(), truth(annotation_id="a2")))

    def test_truth_flags_must_be_boolean(self):
        for flag in ("ignore", "occluded", "truncated"):
            with self.assertRaisesRegex(ContractError, flag):
                truth(**{flag: 1})

    def test_runtime_label_boundary(self):
        for extra in ("ignore", "track_id", "objects"):
            values = detection().to_dict()
            values[extra] = False
            with self.assertRaisesRegex(ContractError, "unknown fields"):
                DetectionRecord.from_dict(values)
        values = truth().to_dict()
        values["score"] = 0.9
        with self.assertRaisesRegex(ContractError, "unknown fields"):
            GroundTruthObject.from_dict(values)

    def test_detection_score_and_frame_consistency(self):
        for score in (True, -0.1, 1.1, float("nan"), float("inf")):
            with self.assertRaisesRegex(ContractError, "score"):
                detection(score=score)
        for score in (0, 1):
            detection(score=score).validate_frame(frame())
        for name, value in (("sequence_id", "other"), ("frame_index", 1),
                            ("timestamp_s", 0.1), ("width", 641), ("height", 481)):
            with self.assertRaisesRegex(ContractError, name):
                detection().validate_frame(frame(**{name: value}))

    def test_sequence_strict_order_gaps_and_local_identities(self):
        validate_sequence([])
        validate_sequence([frame(), frame(frame_index=4, timestamp_s=0.23)])
        for second in (frame(), frame(frame_index=1),
                       frame(frame_index=1, timestamp_s=0.1, sequence_id="other")):
            with self.assertRaises(ContractError):
                validate_sequence([frame(), second])
        untimed = frame(frame_index=1, timestamp_s=None, timestamp_source="unavailable",
                        availability_reason={"camera_id": "unknown", "timestamp_s": "still"})
        with self.assertRaisesRegex(ContractError, "mixed timed"):
            validate_sequence([frame(), untimed])
        item = truth(track_id="local-1", availability_reason={})
        first = AnnotationRecord(1, frame(), "v1", (item,))
        second = AnnotationRecord(1, frame(frame_index=2, timestamp_s=1), "v1", (item,))
        validate_sequence([first, second])
        validate_sequence([AnnotationRecord(1, frame(sequence_id="other"), "v1", (item,))])
        with self.assertRaisesRegex(ContractError, "class changes"):
            validate_sequence([first, replace(second, objects=(replace(item, class_id=1),))])
        with self.assertRaisesRegex(ContractError, "class_mapping_version"):
            validate_sequence([first, replace(second, class_mapping_version="v2")])

    def test_prediction_unknown_values_and_immutable_nested_values(self):
        record = prediction(center=None, covariance=None,
                            availability_reason={"center": "lost track", "covariance": "lost track"})
        self.assertEqual(PredictionRecord.from_json(record.to_json()), record)
        prediction(covariance=None, availability_reason={"covariance": "not estimated"})
        with self.assertRaisesRegex(ContractError, "covariance"):
            prediction(center=None, availability_reason={"center": "unknown"})
        matrix = [[1, 0], [0, 1]]
        record = prediction(covariance=matrix)
        matrix[0][0] = float("nan")
        self.assertEqual(record.covariance[0][0], 1)
        reasons = {"camera_id": "unknown"}
        record = frame(availability_reason=reasons)
        reasons["camera_id"] = ""
        self.assertEqual(record.availability_reason["camera_id"], "unknown")
        with self.assertRaises(TypeError):
            record.availability_reason["camera_id"] = "changed"

    def test_prediction_time_geometry_and_covariance_errors(self):
        for changes in ({"horizon_s": 0}, {"horizon_s": True}, {"target_time_s": 2.2},
                        {"issue_time_s": -1}, {"center": [1]}, {"center": [True, 1]},
                        {"center": [float("inf"), 1]}, {"covariance": [[1]]},
                        {"covariance": [[1, 2], [0, 1]]},
                        {"covariance": [[-1, 0], [0, 1]]},
                        {"covariance": [[1, 2], [2, 1]]},
                        {"covariance": [[0, 1e-15], [1e-15, 0]]},
                        {"covariance": [[float("nan"), 0], [0, 1]]}):
            with self.subTest(changes=changes), self.assertRaises(ContractError):
                prediction(**changes)
        for matrix in (((0, 0), (0, 0)), ((1, 1), (1, 1)), ((1e300, 0), (0, 1e300))):
            prediction(covariance=matrix)
        prediction(target_time_s=2.1 + 1e-10)
        with self.assertRaisesRegex(ContractError, "target_time_s"):
            prediction(issue_time_s=1e20, target_time_s=1e20)

    def test_strict_json_unknown_missing_duplicate_and_nonfinite(self):
        valid = frame().to_json()
        for payload in ('{"schema_version":1,"schema_version":1}',
                        valid.replace('"timestamp_s":0.0', '"timestamp_s":NaN'),
                        valid.replace('"timestamp_s":0.0', '"timestamp_s":Infinity'),
                        valid.replace('"timestamp_s":0.0', '"timestamp_s":1e999'),
                        '[]', '{}', '{broken', valid[:-1] + ',"extra":1}'):
            with self.subTest(payload=payload), self.assertRaises(ContractError):
                FrameRecord.from_json(payload)
        nested = AnnotationRecord(1, frame(), "v1", (truth(),)).to_json()
        nested = nested.replace('"ignore":false', '"ignore":false,"ignore":true')
        with self.assertRaisesRegex(ContractError, "duplicate key"):
            AnnotationRecord.from_json(nested)
        self.assertIsInstance(json.loads(valid), dict)


if __name__ == "__main__":
    unittest.main()
