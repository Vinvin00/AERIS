"""Canonical dataset loading, QA and statistics for contract-v1 records.

Layout (see docs/data_qa.md):
  DATASET/dataset.json       index: sequences with group and split, image root, transforms
  DATASET/annotations.jsonl  one AnnotationRecord per line; each sequence in frame order
Images resolve under ``image_root`` and are decoded in stored orientation (EXIF ignored).
Validation reports every problem found instead of stopping at the first one.

Usage: python -m aeris.data.validate DATASET [--output report.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from aeris.data.contracts import (
    SCHEMA_VERSION, AnnotationRecord, ContractError, loads, validate_sequence,
)

WIDTH_BINS = (("[0,16)", 16), ("[16,32)", 32), ("[32,64)", 64), ("[64,inf)", float("inf")))
INDEX_KEYS = {"schema_version", "dataset_id", "class_mapping_version", "image_root", "group_basis",
              "unannotated_attributes", "transforms", "sequences"}
SEQUENCE_KEYS = {"sequence_id", "group_id", "split"}
ANNOTATABLE_ATTRIBUTES = {"ignore", "occluded", "truncated"}
# Explicit package registry: never depend on a checkout's working directory.
CLASS_MAPPINGS = {"aeris-aircraft-v1": {0: "aircraft"}}
DECODE_FLAGS = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decode(data: bytes) -> np.ndarray | None:
    """Decode stored pixels, returning None for corrupt/empty encoded images."""
    if not data:
        return None
    try:
        return cv2.imdecode(np.frombuffer(data, np.uint8), DECODE_FLAGS)
    except cv2.error:
        return None


def image_root_path(root: Path, value: str) -> Path:
    """Resolve a declared image root or raise a field-specific data error."""
    try:
        return (root / value).resolve()
    except (OSError, ValueError, RuntimeError) as error:
        raise ContractError(f"dataset.json: image_root: invalid path: {error}") from error


def image_path(root: Path, value: str) -> Path:
    """Resolve a relative image path, including symlinks, within the image root."""
    try:
        path = (root / value).resolve()
    except (OSError, ValueError, RuntimeError) as error:
        raise ContractError(f"image_path {value!r}: invalid path: {error}") from error
    if Path(value).is_absolute() or not path.is_relative_to(root):
        raise ContractError(f"image_path {value} escapes image_root")
    return path


def _transform_problems(transforms) -> list[str]:
    where = "dataset.json: transforms"
    if not isinstance(transforms, dict):
        return [f"{where} must be an object"]
    problems = []
    unknown = set(transforms) - {"model_input_crop"}
    if unknown:
        problems.append(f"{where}: unsupported transforms {sorted(unknown)}")
    if "model_input_crop" in transforms:
        where += ".model_input_crop"
        crop = transforms["model_input_crop"]
        if not isinstance(crop, dict) or set(crop) != {"bottom_px", "reason"}:
            problems.append(f"{where}: expected exactly bottom_px and reason")
        else:
            if type(crop["bottom_px"]) is not int or crop["bottom_px"] < 0:
                problems.append(f"{where}.bottom_px must be a nonnegative integer")
            if not isinstance(crop["reason"], str) or not crop["reason"].strip():
                problems.append(f"{where}.reason must be nonempty text")
    return problems


def crop_frame_problems(index, records) -> list[str]:
    """Check a structurally validated crop leaves pixels in every affected frame."""
    crop = index["transforms"].get("model_input_crop")
    if crop is None:
        return []
    return [f"annotations.jsonl:{number}: transforms.model_input_crop.bottom_px "
            f"{crop['bottom_px']} must be less than original frame height {record.frame.height}"
            for number, record in records if crop["bottom_px"] >= record.frame.height]


def _identifier(value) -> bool:
    return isinstance(value, str) and bool(value) and not any(c.isspace() for c in value)


def read_index(root: Path) -> tuple[dict | None, dict[str, dict], list[str]]:
    """Return (index, well-formed sequences by ID, problems); index is None if unusable."""
    try:
        index = loads((root / "dataset.json").read_text())
    except (OSError, ValueError, UnicodeDecodeError, ContractError) as error:
        return None, {}, [f"dataset.json: {error}"]
    if not isinstance(index, dict) or set(index) != INDEX_KEYS:
        return None, {}, [f"dataset.json: expected exactly keys {sorted(INDEX_KEYS)}"]
    problems = []
    if type(index["schema_version"]) is not int or index["schema_version"] != SCHEMA_VERSION:
        problems.append(f"dataset.json: schema_version must be integer {SCHEMA_VERSION}")
    for key in ("dataset_id", "class_mapping_version"):
        if not _identifier(index[key]):
            problems.append(f"dataset.json: {key} must be an identifier")
    if isinstance(index["class_mapping_version"], str) and index["class_mapping_version"] not in CLASS_MAPPINGS:
        problems.append(f"dataset.json: unknown class_mapping_version {index['class_mapping_version']}")
    for key in ("image_root", "group_basis"):
        if not isinstance(index[key], str) or not index[key].strip():
            problems.append(f"dataset.json: {key} must be nonempty text")
    if isinstance(index["image_root"], str):
        try:
            image_root_path(root, index["image_root"])
        except ContractError as error:
            problems.append(str(error))
    attributes = index["unannotated_attributes"]
    if (not isinstance(attributes, list)
            or not all(isinstance(item, str) and item in ANNOTATABLE_ATTRIBUTES for item in attributes)):
        problems.append(f"dataset.json: unannotated_attributes must list items of {sorted(ANNOTATABLE_ATTRIBUTES)}")
    problems += _transform_problems(index["transforms"])
    if not isinstance(index["sequences"], list):
        return None, {}, problems + ["dataset.json: sequences must be an array"]
    if problems:
        return None, {}, problems

    sequences, group_splits = {}, {}
    for number, entry in enumerate(index["sequences"]):
        where = f"dataset.json: sequences[{number}]"
        if not isinstance(entry, dict) or set(entry) != SEQUENCE_KEYS or not all(
                _identifier(entry[key]) for key in SEQUENCE_KEYS):
            problems.append(f"{where}: expected identifiers {sorted(SEQUENCE_KEYS)}")
        elif entry["sequence_id"] in sequences:
            problems.append(f"{where}: duplicate sequence_id {entry['sequence_id']}")
        else:
            sequences[entry["sequence_id"]] = entry
            group_splits.setdefault(entry["group_id"], set()).add(entry["split"])
    for group, splits in sorted(group_splits.items()):
        if len(splits) > 1:
            problems.append(f"split leakage: group {group} is in splits {sorted(splits)}")
    return index, sequences, problems


def read_annotations(root: Path) -> tuple[list[tuple[int, AnnotationRecord]], list[str]]:
    """Return (line number, record) pairs and one problem per unparseable line."""
    try:
        lines = (root / "annotations.jsonl").read_text().splitlines()
    except (OSError, ValueError, UnicodeDecodeError) as error:
        return [], [f"annotations.jsonl: {error}"]
    records, problems = [], []
    for number, line in enumerate(lines, 1):
        try:
            records.append((number, AnnotationRecord.from_json(line)))
        except ContractError as error:
            problems.append(f"annotations.jsonl:{number}: {error}")
    return records, problems


def _record_problems(index, sequences, records) -> list[str]:
    problems, by_sequence = [], {}
    for number, record in records:
        where, frame = f"annotations.jsonl:{number}", record.frame
        if frame.sequence_id not in sequences:
            problems.append(f"{where}: sequence {frame.sequence_id} is not listed in dataset.json")
        if record.class_mapping_version != index["class_mapping_version"]:
            problems.append(f"{where}: class_mapping_version {record.class_mapping_version} "
                            f"differs from dataset {index['class_mapping_version']}")
        mapping = CLASS_MAPPINGS.get(record.class_mapping_version)
        if mapping is None:
            problems.append(f"{where}: unknown class_mapping_version {record.class_mapping_version}")
        seen = set()
        for item in record.objects:
            if mapping is not None and item.class_id not in mapping:
                problems.append(f"{where}: object {item.annotation_id}: class_id {item.class_id} "
                                f"is not in class_mapping_version {record.class_mapping_version}")
            if (item.class_id, item.bbox) in seen:
                problems.append(f"{where}: duplicate object (same class and bbox) {item.annotation_id}")
            seen.add((item.class_id, item.bbox))
        by_sequence.setdefault(frame.sequence_id, []).append(record)
    for sequence_id, sequence_records in by_sequence.items():
        try:
            validate_sequence(sequence_records)
        except ContractError as error:
            problems.append(f"sequence {sequence_id}: {error}")
    problems += [f"dataset.json: sequence {sid} has no frames" for sid in sequences if sid not in by_sequence]
    return problems


def _image_problems(root, index, sequences, records) -> tuple[list[str], int]:
    """Check files, checksums, decoding and dimensions; find cross-split duplicates."""
    try:
        image_root = image_root_path(root, index["image_root"])
    except ContractError as error:
        return [str(error)], 0
    problems, owners, within_split = [], {}, 0
    for number, record in records:
        where, frame = f"annotations.jsonl:{number}", record.frame
        try:
            path = image_path(image_root, frame.image_path)
        except ContractError as error:
            problems.append(f"{where}: {error}")
            continue
        try:
            data = path.read_bytes()
        except OSError:
            problems.append(f"{where}: missing image {frame.image_path}")
            continue
        digest = sha256(data)
        if digest != frame.image_checksum:
            problems.append(f"{where}: checksum mismatch for {frame.image_path} (corrupt or changed file)")
        image = decode(data)
        if image is None:
            problems.append(f"{where}: cannot decode {frame.image_path}")
            continue
        if image.shape[:2] != (frame.height, frame.width):
            problems.append(f"{where}: decoded size {image.shape[1]}x{image.shape[0]} differs from "
                            f"recorded {frame.width}x{frame.height}")
        split = sequences.get(frame.sequence_id, {}).get("split")
        if split is None:
            continue
        pixels = sha256(repr(image.shape).encode() + image.tobytes())
        for kind, key in (("encoded bytes", "b:" + digest), ("decoded pixels", "p:" + pixels)):
            if key not in owners:
                owners[key] = (split, where)
            elif owners[key][0] != split:
                problems.append(f"{where}: split leakage: same {kind} as {owners[key][1]} "
                                f"(split {owners[key][0]} vs {split})")
            elif kind == "encoded bytes":
                within_split += 1
    return problems, within_split


def statistics(records, sequences) -> dict:
    """Per-split counts. Width bins count non-ignored objects by original-pixel width."""
    splits = {}
    for _, record in records:
        frame = record.frame
        entry = sequences.get(frame.sequence_id, {})
        split = splits.setdefault(entry.get("split", "(unlisted)"), {
            "sequences": set(), "groups": set(), "frames": 0, "timed_frames": 0,
            "status": {"complete": 0, "partial": 0, "unlabeled": 0}, "negative_frames": 0,
            "objects": 0, "ignored_objects": 0, "occluded_objects": 0, "truncated_objects": 0,
            "tracks": set(), "objects_without_track": 0,
            "width_bins": {name: 0 for name, _ in WIDTH_BINS}})
        split["sequences"].add(frame.sequence_id)
        split["groups"].add(entry.get("group_id", "(unlisted)"))
        split["frames"] += 1
        split["timed_frames"] += frame.timestamp_s is not None
        split["status"][frame.annotation_status] += 1
        split["negative_frames"] += record.is_negative
        for item in record.objects:
            split["objects"] += 1
            split["occluded_objects"] += item.occluded
            split["truncated_objects"] += item.truncated
            if item.track_id is None:
                split["objects_without_track"] += 1
            else:
                split["tracks"].add((frame.sequence_id, item.track_id))
            if item.ignore:
                split["ignored_objects"] += 1
                continue
            width = item.bbox.x_max - item.bbox.x_min
            split["width_bins"][next(name for name, upper in WIDTH_BINS if width < upper)] += 1
    for split in splits.values():
        for key in ("sequences", "groups", "tracks"):
            split[key] = len(split[key])
    return dict(sorted(splits.items()))


def validate_dataset(dataset) -> dict:
    """Return {"dataset", "problems", "duplicate_images_within_split", "stats"}."""
    root = Path(dataset)
    index, sequences, problems = read_index(root)
    if index is None:
        return {"dataset": str(root), "problems": problems,
                "duplicate_images_within_split": None, "stats": None}
    records, parse_problems = read_annotations(root)
    image_problems, within_split = _image_problems(root, index, sequences, records)
    problems += (parse_problems + _record_problems(index, sequences, records)
                 + crop_frame_problems(index, records) + image_problems)
    return {"dataset": str(root), "problems": problems,
            "duplicate_images_within_split": within_split,
            "stats": statistics(records, sequences)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate a canonical AERIS dataset.")
    parser.add_argument("dataset")
    parser.add_argument("--output", help="write the JSON report here instead of stdout")
    args = parser.parse_args(argv)
    report = validate_dataset(args.dataset)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    else:
        sys.stdout.write(text)
    return 1 if report["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
