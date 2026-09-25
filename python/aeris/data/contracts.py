"""Version 1 canonical records. Coordinates are original-frame, half-open pixels.

These records contain metadata and geometry, not decoded pixels or evaluation results.
JSON parsing rejects unknown fields, duplicate keys, NaN and Infinity. Nullable fields
must have a field-specific entry in ``availability_reason``.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
import json
import math
import re
from types import MappingProxyType
from typing import Any, ClassVar

SCHEMA_VERSION = 1
PREDICTION_HORIZONS_S = (0.1, 0.25, 0.5, 1.0, 2.0)


class ContractError(ValueError):
    """A record violates the versioned data contract."""


def _identifier(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value or any(c.isspace() for c in value):
        raise ContractError(f"{name}: expected nonempty identifier without whitespace")


def _text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name}: expected nonempty text")


def _integer(value: Any, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ContractError(f"{name}: expected integer >= {minimum} (not bool)")


def _number(value: Any, name: str, minimum: float | None = None) -> None:
    if type(value) not in (int, float):
        raise ContractError(f"{name}: expected finite number (not bool)")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or (minimum is not None and value < minimum):
        raise ContractError(f"{name}: expected finite number" +
                            (f" >= {minimum}" if minimum is not None else ""))


def _enum(value: Any, name: str, choices: tuple[str, ...]) -> None:
    if not isinstance(value, str) or value not in choices:
        raise ContractError(f"{name}: expected one of {choices}")


def _availability(record: Any, nullable_fields: tuple[str, ...]) -> None:
    reasons = record.availability_reason
    if not isinstance(reasons, Mapping):
        raise ContractError("availability_reason: expected mapping of field names to reasons")
    expected = {name for name in nullable_fields if getattr(record, name) is None}
    if set(reasons) != expected:
        raise ContractError(f"availability_reason: requires exactly null fields {sorted(expected)}")
    for name, reason in reasons.items():
        _text(reason, f"availability_reason.{name}")
    object.__setattr__(record, "availability_reason", MappingProxyType(dict(reasons)))


def _encode(value: Any) -> Any:
    if isinstance(value, Record):
        return {field.name: _encode(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    return value


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"JSON: duplicate key {key!r}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ContractError(f"JSON: nonfinite constant {value} is forbidden")


def loads(value: str) -> Any:
    """Parse JSON, rejecting duplicate keys and NaN/Infinity constants."""
    try:
        return json.loads(value, object_pairs_hook=_pairs, parse_constant=_constant)
    except (json.JSONDecodeError, TypeError) as error:
        raise ContractError(f"JSON: {error}") from error


class Record:
    """Shared strict serialization; subclasses validate direct construction too."""

    _nested: ClassVar[dict[str, type[Record]]] = {}

    def to_dict(self) -> dict[str, Any]:
        return _encode(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]):
        if not isinstance(value, Mapping):
            raise ContractError(f"{cls.__name__}: expected object")
        required = {field.name for field in fields(cls)}
        if set(value) != required:
            missing = sorted(required - set(value))
            extra = sorted(str(key) for key in set(value) - required)
            raise ContractError(f"{cls.__name__}: missing fields {missing}; unknown fields {extra}")
        arguments = dict(value)
        for name, record_type in cls._nested.items():
            arguments[name] = record_type.from_dict(arguments[name])
        if cls is AnnotationRecord:
            objects = arguments["objects"]
            if not isinstance(objects, (list, tuple)):
                raise ContractError("objects: expected array")
            arguments["objects"] = tuple(GroundTruthObject.from_dict(item) for item in objects)
        return cls(**arguments)

    @classmethod
    def from_json(cls, value: str):
        return cls.from_dict(loads(value))

    def _version(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ContractError(f"schema_version: only integer {SCHEMA_VERSION} is supported")


@dataclass(frozen=True)
class BBox(Record):
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        for name in ("x_min", "y_min", "x_max", "y_max"):
            _number(getattr(self, name), f"bbox.{name}", 0)
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ContractError("bbox: require x_min < x_max and y_min < y_max")

    def validate_bounds(self, width: int, height: int) -> None:
        _integer(width, "width", 1)
        _integer(height, "height", 1)
        if self.x_max > width or self.y_max > height:
            raise ContractError("bbox: half-open extents exceed original frame width/height")


def _bbox(value: Any, width: int | None = None, height: int | None = None) -> None:
    if not isinstance(value, BBox):
        raise ContractError("bbox: expected BBox")
    if width is not None:
        value.validate_bounds(width, height)


@dataclass(frozen=True)
class FrameRecord(Record):
    schema_version: int
    sequence_id: str
    frame_index: int
    timestamp_s: float | None
    timestamp_source: str
    width: int
    height: int
    image_path: str
    image_checksum: str
    camera_id: str | None
    annotation_status: str
    availability_reason: Mapping[str, str]

    def __post_init__(self) -> None:
        self._version()
        _identifier(self.sequence_id, "sequence_id")
        _integer(self.frame_index, "frame_index")
        _integer(self.width, "width", 1)
        _integer(self.height, "height", 1)
        _text(self.image_path, "image_path")
        if not isinstance(self.image_checksum, str) or not re.fullmatch("[0-9a-f]{64}", self.image_checksum):
            raise ContractError("image_checksum: expected lowercase SHA-256 hex digest")
        if self.camera_id is not None:
            _identifier(self.camera_id, "camera_id")
        _enum(self.annotation_status, "annotation_status", ("complete", "partial", "unlabeled"))
        _enum(self.timestamp_source, "timestamp_source", ("decode_pts", "inferred_fps", "provided", "unavailable"))
        if self.timestamp_s is not None:
            _number(self.timestamp_s, "timestamp_s", 0)
        if (self.timestamp_s is None) != (self.timestamp_source == "unavailable"):
            raise ContractError("timestamp_source: unavailable if and only if timestamp_s is null")
        _availability(self, ("timestamp_s", "camera_id"))


@dataclass(frozen=True)
class GroundTruthObject(Record):
    annotation_id: str
    class_id: int
    bbox: BBox
    track_id: str | None
    ignore: bool
    occluded: bool
    truncated: bool
    availability_reason: Mapping[str, str]
    _nested: ClassVar = {"bbox": BBox}

    def __post_init__(self) -> None:
        _identifier(self.annotation_id, "annotation_id")
        _integer(self.class_id, "class_id")
        _bbox(self.bbox)
        if self.track_id is not None:
            _identifier(self.track_id, "track_id")
        for name in ("ignore", "occluded", "truncated"):
            if type(getattr(self, name)) is not bool:
                raise ContractError(f"{name}: expected bool")
        _availability(self, ("track_id",))


@dataclass(frozen=True)
class AnnotationRecord(Record):
    schema_version: int
    frame: FrameRecord
    class_mapping_version: str
    objects: tuple[GroundTruthObject, ...]
    _nested: ClassVar = {"frame": FrameRecord}

    def __post_init__(self) -> None:
        self._version()
        if not isinstance(self.frame, FrameRecord):
            raise ContractError("frame: expected FrameRecord")
        _identifier(self.class_mapping_version, "class_mapping_version")
        if not isinstance(self.objects, (list, tuple)):
            raise ContractError("objects: expected array")
        object.__setattr__(self, "objects", tuple(self.objects))
        if self.frame.annotation_status == "unlabeled" and self.objects:
            raise ContractError("objects: unlabeled frame cannot contain ground truth")
        annotation_ids, track_ids = set(), set()
        for item in self.objects:
            if not isinstance(item, GroundTruthObject):
                raise ContractError("objects: expected GroundTruthObject entries")
            item.bbox.validate_bounds(self.frame.width, self.frame.height)
            if item.annotation_id in annotation_ids:
                raise ContractError("annotation_id: duplicate identity in frame")
            annotation_ids.add(item.annotation_id)
            if item.track_id is not None:
                if item.track_id in track_ids:
                    raise ContractError("track_id: duplicate identity in frame")
                track_ids.add(item.track_id)

    @property
    def is_negative(self) -> bool:
        return self.frame.annotation_status == "complete" and not self.objects


@dataclass(frozen=True)
class DetectionRecord(Record):
    schema_version: int
    sequence_id: str
    frame_index: int
    timestamp_s: float | None
    width: int
    height: int
    bbox: BBox
    class_id: int
    class_mapping_version: str
    score: float
    model_provenance: str
    availability_reason: Mapping[str, str]
    _nested: ClassVar = {"bbox": BBox}

    def __post_init__(self) -> None:
        self._version()
        _identifier(self.sequence_id, "sequence_id")
        _integer(self.frame_index, "frame_index")
        _integer(self.width, "width", 1)
        _integer(self.height, "height", 1)
        if self.timestamp_s is not None:
            _number(self.timestamp_s, "timestamp_s", 0)
        _bbox(self.bbox, self.width, self.height)
        _integer(self.class_id, "class_id")
        _identifier(self.class_mapping_version, "class_mapping_version")
        _number(self.score, "score", 0)
        if self.score > 1:
            raise ContractError("score: must be in [0,1]")
        _text(self.model_provenance, "model_provenance")
        _availability(self, ("timestamp_s",))

    def validate_frame(self, frame: FrameRecord) -> None:
        if not isinstance(frame, FrameRecord):
            raise ContractError("frame: expected FrameRecord")
        for name in ("sequence_id", "frame_index", "timestamp_s", "width", "height"):
            if getattr(self, name) != getattr(frame, name):
                raise ContractError(f"detection.{name}: does not match frame")


@dataclass(frozen=True)
class PredictionRecord(Record):
    schema_version: int
    sequence_id: str
    track_id: str
    issue_time_s: float
    horizon_s: float
    target_time_s: float
    center: tuple[float, float] | None
    covariance: tuple[tuple[float, float], tuple[float, float]] | None
    model_name: str
    availability_reason: Mapping[str, str]

    def __post_init__(self) -> None:
        self._version()
        _identifier(self.sequence_id, "sequence_id")
        _identifier(self.track_id, "track_id")
        _text(self.model_name, "model_name")
        for name in ("issue_time_s", "horizon_s", "target_time_s"):
            _number(getattr(self, name), name, 0)
        if self.horizon_s not in PREDICTION_HORIZONS_S:
            raise ContractError(f"horizon_s: expected one of {PREDICTION_HORIZONS_S}")
        expected = self.issue_time_s + self.horizon_s
        if (self.target_time_s <= self.issue_time_s or not math.isfinite(expected)
                or not math.isclose(self.target_time_s, expected, rel_tol=0, abs_tol=1e-9)):
            raise ContractError("target_time_s: must equal issue_time_s + horizon_s and increase")
        if self.center is not None:
            if not isinstance(self.center, (list, tuple)) or len(self.center) != 2:
                raise ContractError("center: expected [x,y] in original-frame pixels")
            for number in self.center:
                _number(number, "center")
            object.__setattr__(self, "center", tuple(self.center))
        if self.covariance is not None:
            if self.center is None:
                raise ContractError("covariance: unavailable when center is null")
            matrix = self.covariance
            if (not isinstance(matrix, (list, tuple)) or len(matrix) != 2 or
                    any(not isinstance(row, (list, tuple)) or len(row) != 2 for row in matrix)):
                raise ContractError("covariance: expected 2x2 matrix in px^2")
            for row in matrix:
                for number in row:
                    _number(number, "covariance")
            a, b = matrix[0]
            c, d = matrix[1]
            if a < 0 or d < 0 or b != c:
                raise ContractError("covariance: expected symmetric positive semidefinite matrix")
            scale = max(a, abs(b), d) or 1
            if (a / scale) * (d / scale) - (b / scale) ** 2 < -1e-12:
                raise ContractError("covariance: expected positive semidefinite matrix")
            object.__setattr__(self, "covariance", tuple(tuple(row) for row in matrix))
        _availability(self, ("center", "covariance"))


def validate_sequence(records: Iterable[FrameRecord | AnnotationRecord]) -> None:
    """Validate one ordered sequence; never sort away timing or identity errors.

    Gaps in frame indices are allowed. A still-image sequence may have all-null
    timestamps, but mixed timed/untimed records are rejected. Track IDs are local
    to this call's sequence and must retain their annotated class.
    """
    previous = None
    mapping_version = None
    track_classes: dict[str, int] = {}
    for record in records:
        frame = record.frame if isinstance(record, AnnotationRecord) else record
        if not isinstance(frame, FrameRecord):
            raise ContractError("sequence: expected FrameRecord or AnnotationRecord")
        if previous is not None:
            if frame.sequence_id != previous.sequence_id:
                raise ContractError("sequence_id: mixed sequences")
            if frame.frame_index <= previous.frame_index:
                raise ContractError("frame_index: must strictly increase within a sequence")
            if (frame.timestamp_s is None) != (previous.timestamp_s is None):
                raise ContractError("timestamp_s: mixed timed and untimed frames")
            if frame.timestamp_s is not None and frame.timestamp_s <= previous.timestamp_s:
                raise ContractError("timestamp_s: must strictly increase within a sequence")
        if isinstance(record, AnnotationRecord):
            if mapping_version is not None and record.class_mapping_version != mapping_version:
                raise ContractError("class_mapping_version: changes within sequence")
            mapping_version = record.class_mapping_version
            for item in record.objects:
                if item.track_id is not None:
                    if item.track_id in track_classes and track_classes[item.track_id] != item.class_id:
                        raise ContractError("track_id: class changes within sequence")
                    track_classes[item.track_id] = item.class_id
        previous = frame
