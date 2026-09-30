"""Small deterministic analytical fixtures. Synthetic engineering checks only.

``input`` contains observations and permitted camera calibration. ``truth`` is an
evaluator-only output: callers must not pass it to a detector or tracker. The
generator has no perception dependency and uses only Python's standard library.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

from aeris.data.contracts import (
    AnnotationRecord, BBox, FrameRecord, GroundTruthObject, validate_sequence,
)

SEED = 17
WIDTH, HEIGHT = 64, 48
TIMES_S = (0.0, 0.1, 0.25, 0.5, 0.9)
CLASS_MAP = "aeris-aircraft-v1"


def _json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _write(root: Path, name: str, data: bytes, hashes: dict[str, str]) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    hashes[name] = hashlib.sha256(data).hexdigest()


def _image(centers: tuple[tuple[float, float], ...], rng: random.Random) -> bytes:
    """Portable binary PPM with a seeded sky texture and visible 4x4 sprites."""
    pixels = bytearray()
    tint = rng.randrange(8)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            marked = any(abs(x + 0.5 - cx) <= 2 and abs(y + 0.5 - cy) <= 2
                         for cx, cy in centers)
            pixels.extend((24, 27, 32) if marked else (160 + tint, 190 + y // 4, 230))
    return f"P6\n{WIDTH} {HEIGHT}\n255\n".encode() + pixels


def _object(track: str, center: tuple[float, float]) -> GroundTruthObject:
    x, y = center
    return GroundTruthObject(track, 0, BBox(x - 2, y - 2, x + 2, y + 2),
                             track, False, False, False, {})


def generate(root: Path) -> dict[str, str]:
    """Regenerate the complete fixture bundle and return per-file SHA-256 hashes.

    Existing generated files under input/truth are replaced. No model or simulator
    state is consumed. The manifest is excluded from its own digest map.
    """
    root = Path(root)
    rng = random.Random(SEED)
    hashes: dict[str, str] = {}
    camera = {
        "camera_id": "analytic-pinhole-1", "width": WIDTH, "height": HEIGHT,
        "K": [[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]],
        "distortion_model": "none", "distortion_coefficients": [],
        "calibration_id": "fixture-pinhole-v1", "quality": "analytic-exact",
        "lens": "synthetic-fixed", "zoom": 1.0,
    }
    _write(root, "input/camera.json", _json(camera), hashes)

    frame_lines: list[bytes] = []
    annotation_lines: list[bytes] = []
    image_records: list[AnnotationRecord] = []
    for index, t in enumerate(TIMES_S):
        # Analytic image-plane velocities are +20 and -20 px/s. Both tracks
        # meet at (20,24) at t=0.5; t=0.25 hides A and t=0.9 is empty.
        centers = () if index == 4 else tuple(
            center for track, center in (("A", (10 + 20 * t, 20 + 8 * t)),
                                         ("B", (30 - 20 * t, 28 - 8 * t)))
            if not (index == 2 and track == "A")
        )
        objects = () if index == 4 else tuple(
            _object(track, center)
            for track, center in (("A", (10 + 20 * t, 20 + 8 * t)),
                                  ("B", (30 - 20 * t, 28 - 8 * t)))
            if not (index == 2 and track == "A")
        )
        image_name = f"input/images/motion-{index:03d}.ppm"
        _write(root, image_name, _image(centers, rng), hashes)
        frame = FrameRecord(1, "analytic-motion", index, t, "provided", WIDTH, HEIGHT,
                            f"images/motion-{index:03d}.ppm", hashes[image_name],
                            camera["camera_id"], "complete", {})
        annotation = AnnotationRecord(1, frame, CLASS_MAP, objects)
        image_records.append(annotation)
        frame_lines.append(_json(frame.to_dict()))
        annotation_lines.append(_json(annotation.to_dict()))
    validate_sequence(image_records)
    _write(root, "input/frames.jsonl", b"".join(frame_lines), hashes)
    _write(root, "truth/annotations.jsonl", b"".join(annotation_lines), hashes)

    # Separate three-frame pinhole projection. World positions are deliberately
    # stored only with evaluator truth. u=fx X/Z+cx, v=fy Y/Z+cy, no distortion.
    projection_frames: list[bytes] = []
    projection_truth: list[bytes] = []
    world_points = ((0.0, 0.0, 10.0), (1.0, 0.0, 10.0), (-1.0, 0.0, 10.0))
    for index, (x, y, z) in enumerate(world_points):
        u, v = 100.0 * x / z + 32.0, 100.0 * y / z + 24.0
        image_name = f"input/images/projection-{index:03d}.ppm"
        _write(root, image_name, _image(((u, v),), rng), hashes)
        frame = FrameRecord(1, "analytic-projection", index, float(index), "provided",
                            WIDTH, HEIGHT, f"images/projection-{index:03d}.ppm",
                            hashes[image_name], camera["camera_id"], "complete", {})
        projection_frames.append(_json(frame.to_dict()))
        projection_truth.append(_json({
            "frame_index": index, "timestamp_s": float(index), "track_id": "P",
            "world_xyz_m": [x, y, z], "projected_center_xy_px": [u, v],
            "bbox_xyxy_px": [u - 2, v - 2, u + 2, v + 2],
        }))
    _write(root, "input/projection_frames.jsonl", b"".join(projection_frames), hashes)
    _write(root, "truth/projection.jsonl", b"".join(projection_truth), hashes)
    manifest = {
        "schema_version": 1, "fixture_id": "analytic-v1", "seed": SEED,
        "scope": "synthetic engineering fixture; not a perception benchmark",
        "units": {"time": "s", "image": "original-frame half-open px", "world": "m"},
        "source": "analytical equations in aeris.simulation.fixtures.generate",
        "rights": "repository-generated synthetic data", "split": "development",
        "truth_policy": "input/ is runtime-visible; truth/ is evaluator-only",
        "sha256": hashes,
    }
    (root / "manifest.json").write_bytes(_json(manifest))
    return hashes
