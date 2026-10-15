"""Fixed-camera pixels to local ENU bearings, without range or 3-D target position.

The ENU frame is a site survey frame named by ``reference_id``: x east, y north,
z up, all positions in metres relative to that frame's surveyed origin. Camera
coordinates are x right, y down, z forward. A pose is an operator assertion of
a fixed mount, not an automatic stability measurement.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math

import numpy as np

from aeris.estimation.camera import CameraModel, FrameContext, GeometryError


SCHEMA_VERSION = 1
UNCERTAINTY_REASON = "angular uncertainty unavailable: no measured calibration and pose error model"


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise GeometryError(f"{name}: expected finite number")
    return float(value)


def _text(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GeometryError(f"{name}: expected nonempty text")
    return value


def _vector(value, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise GeometryError(f"{name}: expected three numbers")
    return tuple(_number(v, name) for v in value)


@dataclass(frozen=True)
class FixedCameraPose:
    """Surveyed camera origin and right-handed camera-to-ENU rotation.

    ``fixed_mount`` is an explicit operator assertion for this pose epoch. It does
    not detect later camera movement; callers must set FrameContext.camera_moved
    when they have evidence of movement and stop using a stale pose.
    """

    schema_version: int
    pose_id: str
    camera_id: str
    calibration_id: str
    reference_id: str
    position_enu_m: tuple[float, float, float]
    rotation_camera_to_enu: tuple[tuple[float, float, float], ...]
    surveyed_at_utc: str
    fixed_mount: bool

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise GeometryError(f"pose schema_version: expected {SCHEMA_VERSION}")
        for name in ("pose_id", "camera_id", "calibration_id", "reference_id"):
            _text(getattr(self, name), name)
        object.__setattr__(self, "position_enu_m", _vector(self.position_enu_m, "position_enu_m"))
        try:
            rotation = np.asarray(self.rotation_camera_to_enu, dtype=float)
        except (TypeError, ValueError) as exc:
            raise GeometryError("rotation_camera_to_enu: expected finite 3x3 matrix") from exc
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise GeometryError("rotation_camera_to_enu: expected finite 3x3 matrix")
        if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-6) or \
                not math.isclose(float(np.linalg.det(rotation)), 1.0, rel_tol=0, abs_tol=1e-6):
            raise GeometryError("rotation_camera_to_enu: expected right-handed orthonormal rotation")
        object.__setattr__(self, "rotation_camera_to_enu",
                           tuple(tuple(float(v) for v in row) for row in rotation))
        _text(self.surveyed_at_utc, "surveyed_at_utc")
        try:
            survey_time = datetime.fromisoformat(self.surveyed_at_utc.replace("Z", "+00:00"))
        except ValueError as exc:
            raise GeometryError("surveyed_at_utc: expected ISO 8601 UTC time") from exc
        if survey_time.tzinfo is None or survey_time.utcoffset() != timezone.utc.utcoffset(survey_time):
            raise GeometryError("surveyed_at_utc: expected UTC time")
        if type(self.fixed_mount) is not bool:
            raise GeometryError("fixed_mount: expected boolean operator assertion")

    @classmethod
    def from_dict(cls, data: dict) -> "FixedCameraPose":
        if not isinstance(data, dict):
            raise GeometryError("pose: expected object")
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class WorldBearingRecord:
    """Observed or causal forecast *direction* from one fixed camera, never range."""

    schema_version: int
    sequence_id: str
    track_id: str
    kind: str  # observation | forecast
    issue_time_s: float
    horizon_s: float
    target_time_s: float
    reference_id: str | None
    pose_id: str | None
    calibration_id: str | None
    origin_enu_m: tuple[float, float, float] | None
    direction_enu: tuple[float, float, float] | None
    azimuth_deg: float | None
    elevation_deg: float | None
    angular_uncertainty_rad: float | None
    uncertainty_reason: str
    availability_reason: str | None

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise GeometryError(f"bearing schema_version: expected {SCHEMA_VERSION}")
        _text(self.sequence_id, "sequence_id")
        _text(self.track_id, "track_id")
        if self.kind not in ("observation", "forecast"):
            raise GeometryError("kind: expected observation or forecast")
        issue, horizon, target = (_number(self.issue_time_s, "issue_time_s"),
                                  _number(self.horizon_s, "horizon_s"),
                                  _number(self.target_time_s, "target_time_s"))
        if issue < 0 or horizon < 0 or target < 0 or \
                not math.isclose(target, issue + horizon, rel_tol=0, abs_tol=1e-9):
            raise GeometryError("bearing times: require nonnegative target = issue + horizon")
        if (self.kind == "observation") != (horizon == 0):
            raise GeometryError("observation requires zero horizon; forecast requires positive horizon")
        if self.angular_uncertainty_rad is not None or self.uncertainty_reason != UNCERTAINTY_REASON:
            raise GeometryError("angular uncertainty must be null with the stated reason")
        present = self.direction_enu is not None
        geometry = (self.reference_id, self.pose_id, self.calibration_id, self.origin_enu_m,
                    self.direction_enu, self.azimuth_deg, self.elevation_deg)
        if present:
            if any(value is None for value in geometry) or self.availability_reason is not None:
                raise GeometryError("available bearing requires full geometry and no availability reason")
            for name in ("reference_id", "pose_id", "calibration_id"):
                _text(getattr(self, name), name)
            origin = _vector(self.origin_enu_m, "origin_enu_m")
            direction = _vector(self.direction_enu, "direction_enu")
            if not math.isclose(math.sqrt(sum(v * v for v in direction)), 1.0, rel_tol=0, abs_tol=1e-6):
                raise GeometryError("direction_enu: expected unit vector")
            azimuth = _number(self.azimuth_deg, "azimuth_deg")
            elevation = _number(self.elevation_deg, "elevation_deg")
            expected_az = math.degrees(math.atan2(direction[0], direction[1])) % 360.0
            expected_el = math.degrees(math.atan2(direction[2], math.hypot(*direction[:2])))
            if not 0 <= azimuth < 360 or not -90 <= elevation <= 90 or \
                    min(abs(azimuth - expected_az), abs(azimuth - expected_az - 360),
                        abs(azimuth - expected_az + 360)) > 1e-6 or abs(elevation - expected_el) > 1e-6:
                raise GeometryError("azimuth/elevation do not match direction_enu")
            object.__setattr__(self, "origin_enu_m", origin)
            object.__setattr__(self, "direction_enu", direction)
        elif any(value is not None for value in geometry) or not self.availability_reason:
            raise GeometryError("unavailable bearing requires null geometry and an availability reason")

    @classmethod
    def from_dict(cls, data: dict) -> "WorldBearingRecord":
        if not isinstance(data, dict):
            raise GeometryError("bearing: expected object")
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)


def world_bearing_from_pixel(*, camera: CameraModel, pose: FixedCameraPose | None,
                             pixel: tuple[float, float], sequence_id: str, track_id: str,
                             issue_time_s: float, horizon_s: float = 0.0,
                             frame_size: tuple[int, int] | None = None,
                             context: FrameContext | None = None,
                             camera_stable: bool | None = None) -> WorldBearingRecord:
    """Convert one original-frame center to a surveyed-site ENU direction.

    A forecast pixel must come from a causal predictor; this function never reads
    future observations. It only changes coordinates at the supplied target time.
    The caller must supply current zoom and an explicit per-frame stability
    assessment. ``FrameContext.camera_moved=False`` alone is insufficient because
    that is its default, not evidence. An operator's fixed-mount assertion cannot
    certify a later unobserved movement.
    """
    if not isinstance(camera, CameraModel):
        raise GeometryError("camera: expected CameraModel")
    if pose is not None and not isinstance(pose, FixedCameraPose):
        raise GeometryError("pose: expected FixedCameraPose or None")
    if not isinstance(pixel, (tuple, list)) or len(pixel) != 2:
        raise GeometryError("pixel: expected two original-frame coordinates")
    x, y = _number(pixel[0], "pixel x"), _number(pixel[1], "pixel y")
    kind = "observation" if horizon_s == 0 else "forecast"
    common = dict(schema_version=SCHEMA_VERSION, sequence_id=sequence_id, track_id=track_id,
                  kind=kind, issue_time_s=issue_time_s, horizon_s=horizon_s,
                  target_time_s=_number(issue_time_s, "issue_time_s") + _number(horizon_s, "horizon_s"),
                  angular_uncertainty_rad=None, uncertainty_reason=UNCERTAINTY_REASON)

    def unavailable(reason: str) -> WorldBearingRecord:
        return WorldBearingRecord(**common, reference_id=None, pose_id=None, calibration_id=None,
                                  origin_enu_m=None, direction_enu=None, azimuth_deg=None,
                                  elevation_deg=None, availability_reason=reason)

    cal = camera.calibration
    if cal is None:
        return unavailable("camera calibration unavailable")
    if pose is None:
        return unavailable("surveyed camera pose unavailable")
    if pose.camera_id != cal.camera_id or pose.calibration_id != cal.calibration_id:
        return unavailable("camera pose does not match calibration identity")
    if not pose.fixed_mount:
        return unavailable("fixed-mount assertion unavailable")
    if type(camera_stable) is not bool or not camera_stable:
        return unavailable("per-frame camera stability unassessed or movement detected")
    if frame_size is None or tuple(frame_size) != (cal.width, cal.height):
        return unavailable("frame resolution does not match calibration")
    if context is None or context.zoom is None:
        return unavailable("current zoom or camera stability assessment unavailable")
    if type(context.camera_moved) is not bool or context.camera_moved:
        return unavailable("camera movement detected or stability assessment invalid")
    if not math.isclose(_number(context.zoom, "zoom"), cal.zoom, rel_tol=1e-9, abs_tol=0):
        return unavailable("current zoom does not match calibration")
    ray = camera.pixel_to_ray(x, y, context=context)
    if ray.direction is None:
        return unavailable(ray.availability_reason.get("direction", "camera ray unavailable"))
    direction = np.asarray(pose.rotation_camera_to_enu) @ np.asarray(ray.direction)
    direction /= np.linalg.norm(direction)
    east, north, up = (float(v) for v in direction)
    return WorldBearingRecord(**common, reference_id=pose.reference_id, pose_id=pose.pose_id,
                              calibration_id=cal.calibration_id, origin_enu_m=pose.position_enu_m,
                              direction_enu=(east, north, up),
                              azimuth_deg=math.degrees(math.atan2(east, north)) % 360.0,
                              elevation_deg=math.degrees(math.atan2(up, math.hypot(east, north))),
                              availability_reason=None)
