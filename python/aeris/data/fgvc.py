"""Convert FGVC-Aircraft 2013b annotations into a canonical contract-v1 dataset.

Adopted rule (ADR 002): assume 1-based inclusive boxes, so x_min = xmin - 1 and
x_max = xmax (same for y), giving 0-based half-open original-frame pixels. Original
image/annotation pairs have not established endpoint inclusion. Zero coordinates
that this assumption forbids are clamped to 0 and flagged, never silently fixed.
The 20-pixel bottom banner is recorded as a model-input crop; truth is not cropped.
FGVC has no groups, so each image is its own group and keeps its official split.

Usage: python -m aeris.data.fgvc FGVC_DATA_DIR OUT_DIR
FGVC_DATA_DIR holds images_box.txt, images_{train,val,test}.txt and images/<id>.jpg.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

from aeris.data.contracts import (
    AnnotationRecord, BBox, ContractError, FrameRecord, GroundTruthObject,
)
from aeris.data.validate import decode, image_path, sha256

BANNER_PX = 20
SPLITS = ("train", "val", "test")
CLASS_MAPPING_VERSION = "aeris-aircraft-v1"
AIRCRAFT = 0
RULE = ("1-based inclusive -> 0-based half-open: x_min=xmin-1, y_min=ymin-1, x_max=xmax, "
        "y_max=ymax; zero coordinates clamped to 0 and flagged")


def _table(path: Path, columns: int, problems: list[str]) -> dict[str, list[str]]:
    rows = {}
    try:
        lines = path.read_text().splitlines()
    except (OSError, UnicodeDecodeError) as error:
        problems.append(f"{path.name}: {error}")
        return rows
    for number, line in enumerate(lines, 1):
        tokens = line.split()
        if len(tokens) != columns:
            problems.append(f"{path.name}:{number}: expected {columns} fields")
        elif re.fullmatch(r"[0-9]{7}", tokens[0]) is None:
            problems.append(f"{path.name}:{number}: invalid image ID {tokens[0]!r}; expected seven ASCII digits")
        elif tokens[0] in rows:
            problems.append(f"{path.name}:{number}: duplicate image {tokens[0]}")
        else:
            rows[tokens[0]] = tokens[1:]
    return rows


def _record(image_id: str, data: bytes, width: int, height: int, bbox: BBox) -> AnnotationRecord:
    frame = FrameRecord(
        schema_version=1, sequence_id=f"fgvc-{image_id}", frame_index=0, timestamp_s=None,
        timestamp_source="unavailable", width=width, height=height,
        image_path=f"{image_id}.jpg", image_checksum=sha256(data), camera_id=None,
        annotation_status="partial",
        availability_reason={"timestamp_s": "FGVC still photograph",
                             "camera_id": "FGVC does not record the camera"})
    main = GroundTruthObject(
        annotation_id="main", class_id=AIRCRAFT, bbox=bbox, track_id=None, ignore=False,
        occluded=False, truncated=False,
        availability_reason={"track_id": "FGVC has no identity labels"})
    return AnnotationRecord(1, frame, CLASS_MAPPING_VERSION, (main,))


def convert_fgvc(data_dir, out_dir, dataset_id: str = "fgvc-aircraft-2013b",
                 splits: tuple[str, ...] = SPLITS) -> dict:
    """Write dataset.json, annotations.jsonl and conversion_report.json into out_dir.

    ``splits`` selects which official splits to convert; all three split files are
    still read so every box must belong to exactly one split. Raises ContractError
    listing every problem; nothing is written on failure.
    """
    try:
        data_dir, out_dir = Path(data_dir).resolve(), Path(out_dir).resolve()
        images_root = (data_dir / "images").resolve()
    except (OSError, ValueError, RuntimeError) as error:
        raise ContractError(f"FGVC conversion: invalid source/output path: {error}") from error
    if not isinstance(dataset_id, str) or not dataset_id or any(c.isspace() for c in dataset_id):
        raise ContractError("FGVC conversion: dataset_id must be an identifier")
    if (not splits or any(split not in SPLITS for split in splits)
            or len(set(splits)) != len(splits)):
        raise ContractError("FGVC conversion: splits must be distinct names from train, val, test")
    problems: list[str] = []
    boxes = _table(data_dir / "images_box.txt", 5, problems)
    split_of: dict[str, str] = {}
    # Read every official split so membership is checked even when converting a subset.
    for split in SPLITS:
        for image_id in _table(data_dir / f"images_{split}.txt", 1, problems):
            if image_id in split_of:
                problems.append(f"{image_id}: listed in splits {split_of[image_id]} and {split}")
            else:
                split_of[image_id] = split
    for image_id in sorted(set(boxes) ^ set(split_of)):
        problems.append(f"{image_id}: needs exactly one box and one split")

    records, flags = [], {"zero_coordinate_clamped": [], "box_reaches_banner": []}
    for image_id in sorted(i for i in set(boxes) & set(split_of) if split_of[i] in splits):
        try:
            path = image_path(images_root, f"{image_id}.jpg")
            data = path.read_bytes()
        except ContractError as error:
            problems.append(f"{image_id}: {error}")
            continue
        except OSError:
            problems.append(f"{image_id}: missing image images/{image_id}.jpg")
            continue
        image = decode(data)
        if image is None:
            problems.append(f"{image_id}: cannot decode images/{image_id}.jpg")
            continue
        height, width = image.shape[:2]
        if height <= BANNER_PX:
            problems.append(f"{image_id}: image height must exceed {BANNER_PX}-pixel banner crop")
            continue
        try:
            xmin, ymin, xmax, ymax = (int(token) for token in boxes[image_id])
        except ValueError:
            problems.append(f"{image_id}: box coordinates must be integers")
            continue
        if min(xmin, ymin) < 0 or xmax > width or ymax > height:
            problems.append(f"{image_id}: box {boxes[image_id]} outside the {width}x{height} image")
            continue
        try:
            bbox = BBox(max(xmin - 1, 0), max(ymin - 1, 0), xmax, ymax)
        except ContractError as error:
            problems.append(f"{image_id}: {error}")
            continue
        if 0 in (xmin, ymin):
            flags["zero_coordinate_clamped"].append(image_id)
        if ymax > height - BANNER_PX:
            flags["box_reaches_banner"].append(image_id)
        records.append(_record(image_id, data, width, height, bbox))
    if problems:
        raise ContractError("FGVC conversion failed:\n" + "\n".join(problems))

    index = {
        "schema_version": 1, "dataset_id": dataset_id,
        "class_mapping_version": CLASS_MAPPING_VERSION,
        "image_root": os.path.relpath(images_root, out_dir),
        "group_basis": "one group per image: FGVC files carry no photographer, event or airframe "
                       "key, and the official split is image-level",
        "unannotated_attributes": ["occluded", "truncated"],
        "transforms": {"model_input_crop": {
            "bottom_px": BANNER_PX,
            "reason": "FGVC copyright banner; ground truth stays in original-frame coordinates"}},
        "sequences": [{"sequence_id": record.frame.sequence_id,
                       "group_id": record.frame.sequence_id,
                       "split": split_of[record.frame.sequence_id.removeprefix("fgvc-")]}
                      for record in records],
    }
    report = {"source": dataset_id, "rule": RULE, "images": len(records),
              "splits": {split: sum(1 for value in split_of.values() if value == split)
                         for split in splits},
              "flags": flags}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset.json").write_text(json.dumps(index, indent=2) + "\n")
    (out_dir / "annotations.jsonl").write_text("".join(r.to_json() + "\n" for r in records))
    (out_dir / "conversion_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Convert FGVC-Aircraft annotations to AERIS v1.")
    parser.add_argument("data_dir")
    parser.add_argument("out_dir")
    args = parser.parse_args(argv)
    try:
        report = convert_fgvc(args.data_dir, args.out_dir)
    except ContractError as error:
        print(error, file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
