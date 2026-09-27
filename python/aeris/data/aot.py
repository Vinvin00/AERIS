"""Airborne Object Tracking (AOT) ground truth as contract-v1 records (AERIS-007).

Source: Amazon Prime Air's AOT training set, public S3 bucket
`airborne-obj-detection-challenge-training`, licensed under the Community Data License
Agreement - Permissive 1.0. Frames are 8-bit grayscale PNG, 2448 x 2048, about 10 fps,
filmed from aircraft. Parts 1 and 2 hold camera 0 and part 3 camera 1 (each part's
`groundtruth.json` names its camera); 716 flights appear in part 3 as well as in part 1 or
2. A sequence is therefore a flight and a camera (`<flight_id>-camera0`). Nothing from the dataset is committed; frames are fetched by key and
checked against the S3 ETag (the object's MD5) and, once recorded, their SHA-256.

`groundtruth.csv` has one row per frame and object (object columns empty for frames
without objects). Boxes are `gt_left/top/right/bottom` in original-frame pixels; they are
used unchanged as contract half-open `xyxy`.

Class policy (mapping `aeris-aircraft-v1`, one class `0 aircraft`):

- `Airplane`, `Helicopter`: aircraft. A box with a measured extent is scored. Most small
  targets carry a *nominal* marker box instead, exactly 6 x 6 or 16 x 16 px (a side
  of 6 or 16 with the other side no larger than 16 also counts, for markers cut by the
  frame edge). A nominal box is not an extent, so the object becomes an `ignore`
  region: a detection on it is neither a true nor a false positive.
- `Airborne` (unidentified) and `Drone`: `ignore` regions.
- `Bird`, `Flock` (and the one-off spelling `BIrd`): not aircraft and left out, so a
  detection on one is a false positive.

AOT labels no occlusion or truncation; those flags are written as `false` and mean
"not labelled". Frames are treated as `complete`: the dataset labels every airborne
object annotators found, planned or not (see docs/baseline.md for the limits of that).
Timestamps are the dataset's capture times in seconds from the sequence's first frame
(`timestamp_source = "provided"`).
"""
from __future__ import annotations

import csv
import hashlib
import math
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from aeris.data.contracts import (SCHEMA_VERSION, AnnotationRecord, BBox, FrameRecord,
                                  GroundTruthObject)

DATASET_ID = "aot-training-2021"
BUCKET_URL = "https://airborne-obj-detection-challenge-training.s3.amazonaws.com"
PARTS = ("part1", "part2", "part3")
PART_CAMERA = {"part1": "camera0", "part2": "camera0", "part3": "camera1"}
WIDTH, HEIGHT = 2448, 2048
CLASS_MAPPING_VERSION = "aeris-aircraft-v1"
AIRCRAFT_CLASS = 0
AIRCRAFT = frozenset({"Airplane", "Helicopter"})
UNKNOWN = frozenset({"Airborne", "Drone"})
NOT_AIRCRAFT = frozenset({"Bird", "Flock"})
ALIASES = {"BIrd": "Bird"}
NOMINAL_SIDES_PX = (6.0, 16.0)
NOMINAL_MAX_SIDE_PX = 16.0
COLUMNS = ("", "time", "flight_id", "img_name", "frame", "id", "range_distance_m",
           "is_above_horizon", "size_width", "size_height", "gt_left", "gt_top",
           "gt_right", "gt_bottom")
USER_AGENT = "AERIS-aot-baseline/0.1 (research benchmark retrieval)"


class AotError(ValueError):
    """AOT ground truth or a fetched file does not meet the expected format."""


@dataclass(frozen=True)
class AotObject:
    object_id: str
    category: str
    box: tuple[float, float, float, float]
    range_m: float | None

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]

    @property
    def nominal(self) -> bool:
        return is_nominal(self.width, self.height)

    @property
    def scored(self) -> bool:
        """An aircraft with a measured extent (the only objects that count as truth)."""
        return self.category in AIRCRAFT and not self.nominal


@dataclass
class AotFrame:
    part: str
    flight_id: str
    frame: int
    time_ns: int
    img_name: str
    objects: list[AotObject] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.part}/Images/{self.flight_id}/{self.img_name}"

    @property
    def camera(self) -> str:
        return PART_CAMERA[self.part]

    @property
    def sequence_id(self) -> str:
        return sequence_id(self.flight_id, self.part)


def sequence_id(flight_id: str, part: str) -> str:
    return f"{flight_id}-{PART_CAMERA[part]}"


def object_category(object_id: str) -> str:
    """`Airplane3` -> `Airplane`; unknown categories are refused."""
    name = re.sub(r"\d+$", "", object_id)
    name = ALIASES.get(name, name)
    if name not in AIRCRAFT | UNKNOWN | NOT_AIRCRAFT:
        raise AotError(f"unknown AOT object category in id {object_id!r}")
    return name


def is_nominal(width: float, height: float) -> bool:
    return (max(width, height) <= NOMINAL_MAX_SIDE_PX
            and (width in NOMINAL_SIDES_PX or height in NOMINAL_SIDES_PX))


def _float(value: str, name: str, line: int) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise AotError(f"line {line}: {name} is not a number: {value!r}") from exc
    if not math.isfinite(number):
        raise AotError(f"line {line}: {name} is not finite")
    return number


def iter_rows(path: Path | str):
    """Validated rows: (line, flight_id, frame, time_ns, img_name, AotObject | None)."""
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        # The parts order their columns differently (part2 puts `id` second), so columns are
        # read by name; the set of names must still match exactly.
        if header is None or sorted(header) != sorted(COLUMNS) or len(set(header)) != len(header):
            raise AotError(f"{path}: unexpected header {header!r}")
        at = {name: header.index(name) for name in COLUMNS}
        for line, raw in enumerate(reader, start=2):
            if len(raw) != len(COLUMNS):
                raise AotError(f"line {line}: expected {len(COLUMNS)} columns")
            row = [raw[at[name]] for name in COLUMNS]
            _, time_s, flight_id, img_name, frame_s, object_id, range_s, _, w_s, h_s = row[:10]
            try:
                time_ns, frame = int(time_s), int(frame_s)
            except ValueError as exc:
                raise AotError(f"line {line}: time/frame must be integers") from exc
            if (w_s, h_s) != (str(WIDTH), str(HEIGHT)):
                raise AotError(f"line {line}: frame size {w_s}x{h_s} is not {WIDTH}x{HEIGHT}")
            if img_name != f"{time_s}{flight_id}.png" or not re.fullmatch("[0-9a-f]{32}", flight_id):
                raise AotError(f"line {line}: image name {img_name!r} does not match time and flight")
            item = None
            if object_id:
                box = tuple(_float(v, n, line) for v, n in zip(row[10:14], COLUMNS[10:14]))
                if not (0 <= box[0] < box[2] <= WIDTH and 0 <= box[1] < box[3] <= HEIGHT):
                    raise AotError(f"line {line}: box {box} outside the frame or empty")
                range_m = None
                if range_s and range_s.lower() != "nan":
                    range_m = _float(range_s, "range_distance_m", line)
                item = AotObject(object_id, object_category(object_id), box, range_m)
            elif any(row[10:14]):
                raise AotError(f"line {line}: box without an object id")
            yield line, flight_id, frame, time_ns, img_name, item


def sequence_index(paths: dict[str, Path | str]) -> dict[str, dict]:
    """sequence_id -> {flight_id, part, camera, first_time_ns, frames} over every part.

    A flight may appear once per camera; the same flight and camera in two parts is refused.
    """
    sequences: dict[str, dict] = {}
    for part, path in paths.items():
        for line, flight_id, frame, time_ns, _, _ in iter_rows(path):
            key = sequence_id(flight_id, part)
            entry = sequences.get(key)
            if entry is None:
                entry = sequences[key] = {"flight_id": flight_id, "part": part,
                                          "camera": PART_CAMERA[part], "first_time_ns": time_ns,
                                          "frames": set()}
            elif entry["part"] != part:
                raise AotError(f"line {line}: {key} appears in {entry['part']} and {part}")
            entry["first_time_ns"] = min(entry["first_time_ns"], time_ns)
            entry["frames"].add(frame)
    return {k: {**v, "frames": len(v["frames"])} for k, v in sequences.items()}


def read_frames(path: Path | str, part: str, flight_ids) -> dict[tuple[str, int], AotFrame]:
    """Every frame (with its objects) of the given flights in one part's ground truth."""
    wanted = set(flight_ids)
    frames: dict[tuple[str, int], AotFrame] = {}
    for line, flight_id, frame, time_ns, img_name, item in iter_rows(path):
        if flight_id not in wanted:
            continue
        entry = frames.get((flight_id, frame))
        if entry is None:
            entry = frames[(flight_id, frame)] = AotFrame(part, flight_id, frame, time_ns, img_name)
        elif (entry.time_ns, entry.img_name) != (time_ns, img_name):
            raise AotError(f"line {line}: frame {flight_id}/{frame} has two timestamps")
        if item is not None:
            if any(o.object_id == item.object_id for o in entry.objects):
                raise AotError(f"line {line}: object {item.object_id} twice in one frame")
            entry.objects.append(item)
    return frames


def annotation(frame: AotFrame, *, first_time_ns: int, image_path: str,
               image_sha256: str) -> AnnotationRecord:
    """Contract-v1 annotation for one AOT frame under the class policy above."""
    if frame.time_ns < first_time_ns:
        raise AotError(f"{frame.flight_id}/{frame.frame}: time before the sequence's first frame")
    record = FrameRecord(
        schema_version=SCHEMA_VERSION, sequence_id=frame.sequence_id, frame_index=frame.frame,
        timestamp_s=(frame.time_ns - first_time_ns) / 1e9, timestamp_source="provided",
        width=WIDTH, height=HEIGHT, image_path=image_path, image_checksum=image_sha256,
        camera_id=frame.camera, annotation_status="complete", availability_reason={})
    objects = [GroundTruthObject(
        annotation_id=item.object_id, class_id=AIRCRAFT_CLASS, bbox=BBox(*item.box),
        track_id=item.object_id, ignore=not item.scored, occluded=False, truncated=False,
        availability_reason={})
        for item in frame.objects if item.category not in NOT_AIRCRAFT]
    return AnnotationRecord(SCHEMA_VERSION, record, CLASS_MAPPING_VERSION, tuple(objects))


def frame_kind(frame: AotFrame) -> str:
    """`scored` (a measured-extent aircraft), `empty` (no object at all) or `other`."""
    if any(item.scored for item in frame.objects):
        return "scored"
    return "other" if frame.objects else "empty"


def image_url(key: str) -> str:
    return f"{BUCKET_URL}/{key}"


def fetch(url: str, dest: Path, *, expected_sha256: str | None = None, limit: int,
          attempts: int = 4, sleep=time.sleep) -> dict:
    """Download `url` to `dest`, checking the S3 ETag (MD5) and a pinned SHA-256.

    An existing `dest` whose SHA-256 matches the pin is reused without a request.
    Returns {"sha256", "md5", "size", "etag", "reused"}.
    """
    if expected_sha256 and dest.is_file():
        data = dest.read_bytes()
        if hashlib.sha256(data).hexdigest() == expected_sha256:
            return {"sha256": expected_sha256, "md5": hashlib.md5(data).hexdigest(),
                    "size": len(data), "etag": None, "reused": True}
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                etag = (response.headers.get("ETag") or "").strip('"')
                data = response.read(limit + 1)
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            status = getattr(exc, "code", None)
            if (status is not None and status < 500) or attempt == attempts - 1:
                raise
            sleep(2 ** attempt)
    if len(data) > limit:
        raise AotError(f"{url}: larger than the {limit}-byte cap")
    md5 = hashlib.md5(data).hexdigest()
    if not re.fullmatch("[0-9a-f]{32}", etag) or md5 != etag:
        raise AotError(f"{url}: MD5 {md5} does not match the single-part ETag {etag!r}")
    sha256 = hashlib.sha256(data).hexdigest()
    if expected_sha256 and sha256 != expected_sha256:
        raise AotError(f"{url}: SHA-256 {sha256} != recorded {expected_sha256}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    partial.write_bytes(data)
    partial.replace(dest)
    return {"sha256": sha256, "md5": md5, "size": len(data), "etag": etag, "reused": False}
