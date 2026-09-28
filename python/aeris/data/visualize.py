"""Annotation contact sheet for human QA of canonical datasets. Draws truth only.

Green boxes are annotated objects, yellow are ignore regions, and the red line marks a
recorded model-input crop. A tile says NEGATIVE only for complete frames without objects.

Usage: python -m aeris.data.visualize DATASET --output sheet.png [--columns 4]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

from aeris.data.contracts import ContractError
from aeris.data.validate import (
    crop_frame_problems, decode, image_path, image_root_path, read_annotations, read_index,
)

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN, YELLOW, RED, BLACK = (0, 170, 0), (0, 210, 255), (0, 0, 230), (0, 0, 0)


def _tile(image_root: Path, record, split: str, crop_px: int | None, width: int) -> np.ndarray:
    frame = record.frame
    scale = width / frame.width
    height = max(1, round(frame.height * scale))
    try:
        path = image_path(image_root, frame.image_path)
        image = decode(path.read_bytes())
    except (ContractError, OSError):
        image = None
    if image is None or image.shape[:2] != (frame.height, frame.width):
        canvas = np.full((height, width, 3), 60, np.uint8)
        cv2.putText(canvas, "MISSING OR BAD IMAGE", (4, height // 2), FONT, 0.4, RED, 1, cv2.LINE_AA)
    else:
        canvas = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    if crop_px:
        y = round((frame.height - crop_px) * scale)
        cv2.line(canvas, (0, y), (width - 1, y), RED, 1)
    for item in record.objects:
        box = item.bbox
        top_left = (math.floor(box.x_min * scale), math.floor(box.y_min * scale))
        bottom_right = (max(top_left[0], math.ceil(box.x_max * scale) - 1),
                        max(top_left[1], math.ceil(box.y_max * scale) - 1))
        color = YELLOW if item.ignore else GREEN
        cv2.rectangle(canvas, top_left, bottom_right, color, 1)
        tags = [item.track_id or item.annotation_id] + [
            name for name in ("ignore", "occluded", "truncated") if getattr(item, name)]
        cv2.putText(canvas, " ".join(tags), (top_left[0], max(9, top_left[1] - 3)), FONT, 0.32,
                    color, 1, cv2.LINE_AA)
    status = "NEGATIVE" if record.is_negative else frame.annotation_status
    time = "t=n/a" if frame.timestamp_s is None else f"t={frame.timestamp_s:.3f}s"
    caption = np.full((30, width, 3), 255, np.uint8)
    for row, text in enumerate((f"{frame.sequence_id} #{frame.frame_index} [{split}]",
                                f"{status} {time} objects={len(record.objects)}")):
        cv2.putText(caption, text, (3, 12 + 13 * row), FONT, 0.35, BLACK, 1, cv2.LINE_AA)
    return np.vstack([canvas, caption])


def contact_sheet(dataset, output, columns: int = 4, tile_width: int = 240,
                  max_frames: int = 48) -> dict:
    """Draw up to max_frames records in file order into a PNG grid; return a summary."""
    root = Path(dataset)
    if any(type(value) is not int or value <= 0 for value in (columns, tile_width, max_frames)):
        raise ContractError("columns, tile_width and max_frames must be positive integers")
    index, sequences, problems = read_index(root)
    if index is None:
        raise ContractError("; ".join(problems))
    records, skipped = read_annotations(root)
    if not records:
        raise ContractError("no parseable annotation records to draw")
    crop_problems = crop_frame_problems(index, records)
    if crop_problems:
        raise ContractError("; ".join(crop_problems))
    image_root = image_root_path(root, index["image_root"])
    crop_px = index["transforms"].get("model_input_crop", {}).get("bottom_px")
    tiles = [_tile(image_root, record,
                   sequences.get(record.frame.sequence_id, {}).get("split", "unlisted"),
                   crop_px, tile_width)
             for _, record in records[:max_frames]]
    tile_height = max(tile.shape[0] for tile in tiles)
    padded = [cv2.copyMakeBorder(tile, 2, 2 + tile_height - tile.shape[0], 2, 2,
                                 cv2.BORDER_CONSTANT, value=(200, 200, 200)) for tile in tiles]
    blank = np.full_like(padded[0], 200)
    padded += [blank] * (-len(padded) % columns)
    sheet = np.vstack([np.hstack(padded[start:start + columns])
                       for start in range(0, len(padded), columns)])
    try:
        written = cv2.imwrite(str(output), sheet)
    except cv2.error as error:  # e.g. no encoder for the output extension
        raise OSError(f"could not write {output}: {error}") from error
    if not written:
        raise OSError(f"could not write {output}")
    return {"output": str(output), "frames_drawn": len(tiles), "frames_total": len(records),
            "unparseable_lines": len(skipped), "size": [sheet.shape[1], sheet.shape[0]]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Draw an annotation contact sheet.")
    parser.add_argument("dataset")
    parser.add_argument("--output", required=True)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--tile-width", type=int, default=240)
    parser.add_argument("--max-frames", type=int, default=48)
    args = parser.parse_args(argv)
    try:
        summary = contact_sheet(args.dataset, args.output, args.columns, args.tile_width,
                                args.max_frames)
    except (ContractError, OSError) as error:
        print(error, file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
