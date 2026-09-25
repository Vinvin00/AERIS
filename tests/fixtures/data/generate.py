"""Regenerate the owned synthetic data fixtures. Not perception evidence.

Run from the repository root: .venv/bin/python tests/fixtures/data/generate.py
Writes canonical/ (contract-v1 dataset) and fgvc_like/ (FGVC 2013b file layout).
Drawing is deterministic; encoded bytes may differ across OpenCV versions, so the
canonical checksums are recomputed here whenever images are regenerated.
"""
from pathlib import Path
import hashlib
import json

import cv2
import numpy as np

from aeris.data.contracts import AnnotationRecord, BBox, FrameRecord, GroundTruthObject

HERE = Path(__file__).resolve().parent
DARK = (70, 70, 70)


def sky(width, height):
    rows = np.linspace(240, 180, height, dtype=np.float32)[:, None]
    image = np.empty((height, width, 3), np.uint8)
    for channel, gain in enumerate((1.0, 0.88, 0.65)):  # BGR
        image[:, :, channel] = np.broadcast_to(rows * gain, (height, width))
    return image


def aircraft(image, box):
    """Fuselage, wings and tail inside a half-open integer box."""
    x0, y0, x1, y1 = box
    cx, cy, half_w, half_h = (x0 + x1 - 1) // 2, (y0 + y1 - 1) // 2, (x1 - x0) // 2, (y1 - y0) // 2
    cv2.ellipse(image, (cx, cy), (max(1, half_w - 1), max(1, half_h // 3)), 0, 0, 360, DARK, -1)
    wing = np.array([[cx - 1, y0], [cx + max(1, half_w // 4), cy], [cx - 1, y1 - 1]], np.int32)
    cv2.fillPoly(image, [wing], DARK)
    cv2.line(image, (x0, cy), (x0, max(y0, cy - half_h // 2)), DARK, 1)


def write_canonical():
    root = HERE / "canonical"
    (root / "images").mkdir(parents=True, exist_ok=True)
    # (sequence, frame_index, t, source, size, camera, status, objects, drawn-but-unlabeled)
    # objects: (annotation_id, track_id, box, ignore, occluded)
    frames = [
        ("clip-a", 0, 0.0, "decode_pts", (128, 96), "cam-1", "complete",
         [("a0", "t1", (4, 30, 74, 50), False, False)], []),
        ("clip-a", 1, 0.1, "decode_pts", (128, 96), "cam-1", "complete",
         [("a0", "t1", (14, 30, 84, 50), False, False)], []),
        ("clip-a", 2, 0.2, "decode_pts", (128, 96), "cam-1", "complete",
         [("a0", "t1", (24, 30, 94, 50), False, False),
          ("a1", None, (100, 10, 112, 16), True, False)], []),
        ("clip-a", 4, 0.4, "decode_pts", (128, 96), "cam-1", "complete", [], []),
        ("clip-b", 0, 0.0, "inferred_fps", (160, 120), None, "complete",
         [("b0", "t1", (20, 20, 30, 26), False, False),
          ("b1", "t2", (80, 60, 120, 76), False, False)], []),
        ("clip-b", 1, 1 / 15, "inferred_fps", (160, 120), None, "partial",
         [("b1", "t2", (84, 60, 124, 76), False, True)], [(22, 20, 32, 26)]),
        ("clip-b", 2, 2 / 15, "inferred_fps", (160, 120), None, "unlabeled",
         [], [(24, 20, 34, 26), (88, 60, 128, 76)]),
    ]
    lines = []
    for sequence, index, time, source, (width, height), camera, status, objects, extra in frames:
        image = sky(width, height)
        for _, _, box, ignore, occluded in objects:
            if ignore:
                cv2.circle(image, ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2), 2, DARK, -1)
            else:
                aircraft(image, box)
                if occluded:
                    cv2.ellipse(image, (box[2] - 8, (box[1] + box[3]) // 2), (10, 9), 0, 0, 360,
                                (250, 250, 250), -1)
        for box in extra:
            aircraft(image, box)
        name = f"{sequence}_{index:04d}.png"
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        (root / "images" / name).write_bytes(encoded.tobytes())
        reasons = {} if camera else {"camera_id": "fixture camera not identified"}
        frame = FrameRecord(1, sequence, index, time, source, width, height, name,
                            hashlib.sha256(encoded.tobytes()).hexdigest(), camera, status, reasons)
        truths = tuple(GroundTruthObject(
            annotation_id, 0, BBox(*box), track, ignore, occluded, False,
            {} if track else {"track_id": "ambiguous distant object, no identity"})
            for annotation_id, track, box, ignore, occluded in objects)
        lines.append(AnnotationRecord(1, frame, "aeris-aircraft-v1", truths).to_json())
    (root / "annotations.jsonl").write_text("\n".join(lines) + "\n")
    index = {
        "schema_version": 1, "dataset_id": "fixture-canonical",
        "class_mapping_version": "aeris-aircraft-v1", "image_root": "images",
        "group_basis": "one capture session per sequence (synthetic)",
        "unannotated_attributes": ["truncated"], "transforms": {},
        "sequences": [{"sequence_id": "clip-a", "group_id": "session-1", "split": "development"},
                      {"sequence_id": "clip-b", "group_id": "session-2", "split": "holdout"}]}
    (root / "dataset.json").write_text(json.dumps(index, indent=2) + "\n")


def write_fgvc_like():
    root = HERE / "fgvc_like"
    (root / "images").mkdir(parents=True, exist_ok=True)
    # FGVC 1-based inclusive boxes; 0000002 has the forbidden zero, 0000003 reaches the banner.
    boxes = {"0000001": (11, 21, 70, 45), "0000002": (0, 15, 60, 40),
             "0000003": (30, 40, 110, 80), "0000004": (5, 5, 50, 30)}
    splits = {"train": ["0000001", "0000002"], "val": ["0000003"], "test": ["0000004"]}
    width, height = 120, 90
    for image_id, (xmin, ymin, xmax, ymax) in boxes.items():
        image = sky(width, height)
        aircraft(image, (max(xmin - 1, 0), max(ymin - 1, 0), xmax, ymax))
        image[height - 20:] = (40, 40, 40)
        cv2.putText(image, "(c) fixture", (4, height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (255, 255, 255), 1, cv2.LINE_AA)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        assert ok
        (root / "images" / f"{image_id}.jpg").write_bytes(encoded.tobytes())
    (root / "images_box.txt").write_text(
        "".join(f"{key} {' '.join(map(str, box))}\n" for key, box in boxes.items()))
    for split, ids in splits.items():
        (root / f"images_{split}.txt").write_text("".join(f"{key}\n" for key in ids))


if __name__ == "__main__":
    write_canonical()
    write_fgvc_like()
