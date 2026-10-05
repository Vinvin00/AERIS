"""Fixed-camera pixel/ray geometry with graceful pixel-only mode.

Conventions (docs/CONTRACTS.md): image origin top-left, x right, y down, original-frame
pixels; camera ray ``(x, y, z)`` with z forward, x right, y down, unit length. Distortion
is removed *before* the ray is formed. Angular extent is the angle between edge rays,
not a physical size. Angular rate is camera-relative rad/s. Image heading is
``atan2(-vy, vx)`` degrees counterclockwise from image right, in ``[0, 360)``; it is not
aircraft yaw. Nothing here estimates range or approach. Unknown outputs are ``None`` with
an ``availability_reason`` entry, never NaN.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import cv2
import numpy as np

from aeris.data.contracts import BBox

MODELS = ("none", "brown_conrady")   # brown_conrady: OpenCV k1,k2,p1,p2[,k3]
_ITER = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12)


class GeometryError(ValueError):
    """Calibration, transform or input violates the geometry contract."""


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise GeometryError(f"{name}: expected finite number")
    return float(value)


@dataclass(frozen=True)
class Calibration:
    camera_id: str
    width: int
    height: int
    K: tuple[tuple[float, float, float], ...]
    distortion_model: str
    distortion_coefficients: tuple[float, ...]
    calibration_id: str
    quality: str
    lens: str
    zoom: float

    def __post_init__(self):
        for name in ("camera_id", "calibration_id", "quality", "lens"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise GeometryError(f"{name}: expected nonempty text")
        for name in ("width", "height"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise GeometryError(f"{name}: expected positive integer")
        k = np.array(self.K, dtype=float)
        if k.shape != (3, 3) or not np.all(np.isfinite(k)):
            raise GeometryError("K: expected finite 3x3 matrix")
        if k[0, 0] <= 0 or k[1, 1] <= 0 or abs(k[0, 1]) > 1e-9 * k[0, 0] or tuple(k[2]) != (0, 0, 1) \
                or k[1, 0] != 0:
            raise GeometryError("K: expected [[fx,0,cx],[0,fy,cy],[0,0,1]] with positive focal lengths")
        if not (0 <= k[0, 2] <= self.width and 0 <= k[1, 2] <= self.height):
            raise GeometryError("K: principal point outside the image")
        if self.distortion_model not in MODELS:
            raise GeometryError(f"distortion_model: expected one of {MODELS}")
        coeffs = [_finite(c, "distortion coefficient") for c in self.distortion_coefficients]
        if self.distortion_model == "none" and coeffs:
            raise GeometryError("distortion_coefficients: must be empty for model 'none'")
        if self.distortion_model == "brown_conrady" and len(coeffs) not in (4, 5):
            raise GeometryError("brown_conrady needs 4 or 5 coefficients (k1,k2,p1,p2[,k3])")
        if _finite(self.zoom, "zoom") <= 0:
            raise GeometryError("zoom: must be positive")
        object.__setattr__(self, "K", tuple(tuple(float(v) for v in row) for row in k))
        object.__setattr__(self, "distortion_coefficients", tuple(coeffs))

    @classmethod
    def from_dict(cls, data: dict) -> "Calibration":
        keys = ("camera_id", "width", "height", "K", "distortion_model", "distortion_coefficients",
                "calibration_id", "quality", "lens", "zoom")
        missing = [k for k in keys if k not in data]
        if missing:
            raise GeometryError(f"calibration missing keys: {missing}")
        return cls(**{k: data[k] for k in keys})

    @property
    def matrix(self) -> np.ndarray:
        return np.array(self.K, dtype=float)

    @property
    def dist(self) -> np.ndarray | None:
        return np.array(self.distortion_coefficients, dtype=float) if self.distortion_coefficients else None


@dataclass(frozen=True)
class ImageTransform:
    """Axis-aligned crop/resize/pad applied to the original frame: ``p = (o - crop) * scale + pad``."""
    crop_x: float = 0.0
    crop_y: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    pad_x: float = 0.0
    pad_y: float = 0.0

    def __post_init__(self):
        for name in ("crop_x", "crop_y", "pad_x", "pad_y"):
            _finite(getattr(self, name), name)
        for name in ("scale_x", "scale_y"):
            if _finite(getattr(self, name), name) <= 0:
                raise GeometryError(f"{name}: must be positive")

    @classmethod
    def letterbox(cls, ratio: float) -> "ImageTransform":
        """The AERIS-006 detector input: uniform scale, padding only at the bottom/right."""
        return cls(scale_x=ratio, scale_y=ratio)

    def to_original(self, u: float, v: float) -> tuple[float, float]:
        return ((u - self.pad_x) / self.scale_x + self.crop_x, (v - self.pad_y) / self.scale_y + self.crop_y)

    def from_original(self, x: float, y: float) -> tuple[float, float]:
        return ((x - self.crop_x) * self.scale_x + self.pad_x, (y - self.crop_y) * self.scale_y + self.pad_y)


IDENTITY = ImageTransform()


@dataclass(frozen=True)
class FrameContext:
    """Facts about how a frame was captured. ``None`` zoom means 'assumed unchanged'."""
    zoom: float | None = None
    camera_moved: bool = False


@dataclass(frozen=True)
class Ray:
    direction: tuple[float, float, float] | None
    availability_reason: dict = field(default_factory=dict)


@dataclass(frozen=True)
class AngularExtent:
    width_rad: float | None
    height_rad: float | None
    availability_reason: dict = field(default_factory=dict)

    @property
    def width_deg(self):
        return None if self.width_rad is None else math.degrees(self.width_rad)

    @property
    def height_deg(self):
        return None if self.height_rad is None else math.degrees(self.height_rad)


@dataclass(frozen=True)
class AngularRate:
    rate_rad_s: float | None
    azimuth_rate_rad_s: float | None
    elevation_rate_rad_s: float | None
    availability_reason: dict = field(default_factory=dict)

    @property
    def rate_deg_s(self):
        return None if self.rate_rad_s is None else math.degrees(self.rate_rad_s)


@dataclass(frozen=True)
class ImageHeading:
    heading_deg: float | None
    speed_px_s: float
    convention: str = "atan2(-vy, vx), degrees counterclockwise from image right, [0,360); not aircraft yaw"
    availability_reason: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SizeChange:
    label: str            # expanding | shrinking | stable | uncertain
    log_rate_per_s: float | None
    standard_error: float | None
    samples: int
    basis: str | None     # "angular_width" | "pixel_width"
    interpretation: str = "apparent size change in the image only; not range, closing speed or approach"
    availability_reason: dict = field(default_factory=dict)


class CameraModel:
    """Geometry for one fixed camera. ``calibration=None`` is the supported pixel-only mode."""

    def __init__(self, calibration: Calibration | None = None, *, min_speed_px_s: float = 1.0,
                 size_stable_log_rate_per_s: float = 0.1, size_min_samples: int = 4,
                 size_min_span_s: float = 0.3):
        if calibration is not None and not isinstance(calibration, Calibration):
            raise GeometryError("calibration must be a Calibration or None")
        for name, value in (("min_speed_px_s", min_speed_px_s),
                            ("size_stable_log_rate_per_s", size_stable_log_rate_per_s),
                            ("size_min_span_s", size_min_span_s)):
            if _finite(value, name) <= 0:
                raise GeometryError(f"{name}: must be positive")
        if size_min_samples < 3:
            raise GeometryError("size_min_samples must be >= 3")
        self.calibration = calibration
        self.min_speed_px_s = min_speed_px_s
        self.size_stable_log_rate_per_s = size_stable_log_rate_per_s
        self.size_min_samples = size_min_samples
        self.size_min_span_s = size_min_span_s

    # -- validity ------------------------------------------------------------
    def _unsupported(self, context: FrameContext | None) -> dict:
        """Reasons angular claims are not supported; empty means supported."""
        if self.calibration is None:
            return {"angular": "calibration unavailable (pixel-only mode)"}
        if context is not None:
            if context.camera_moved:
                return {"angular": "camera moved; the calibrated fixed-camera ray model no longer applies"}
            if context.zoom is not None and not math.isclose(context.zoom, self.calibration.zoom,
                                                              rel_tol=1e-9, abs_tol=0):
                return {"angular": f"zoom {context.zoom} differs from calibrated zoom {self.calibration.zoom}"}
        return {}

    # -- rays ----------------------------------------------------------------
    def pixel_to_ray(self, u: float, v: float, transform: ImageTransform = IDENTITY,
                     context: FrameContext | None = None) -> Ray:
        """Unit camera ray for a pixel in the (possibly cropped/resized) image ``transform`` describes."""
        reason = self._unsupported(context)
        if reason:
            return Ray(None, reason)
        x, y = transform.to_original(_finite(u, "u"), _finite(v, "v"))
        cal = self.calibration
        if not (0 <= x <= cal.width and 0 <= y <= cal.height):
            return Ray(None, {"direction": "pixel lies outside the calibrated original frame"})
        point = np.array([[[x, y]]], dtype=np.float64)
        if cal.dist is None:
            normalized = (point[0, 0] - cal.matrix[:2, 2]) / np.array([cal.matrix[0, 0], cal.matrix[1, 1]])
        else:
            normalized = cv2.undistortPoints(point, cal.matrix, cal.dist, None, np.eye(3), np.eye(3), _ITER)[0, 0]
        ray = np.array([normalized[0], normalized[1], 1.0])
        ray /= np.linalg.norm(ray)
        return Ray(tuple(float(c) for c in ray))

    def ray_to_pixel(self, ray: tuple[float, float, float], transform: ImageTransform = IDENTITY
                     ) -> tuple[float, float] | None:
        """Project a ray (z > 0) back to a pixel in ``transform``'s image, applying distortion."""
        if self.calibration is None:
            raise GeometryError("calibration unavailable")
        r = np.array(ray, dtype=float)
        if r.shape != (3,) or not np.all(np.isfinite(r)) or r[2] <= 0:
            raise GeometryError("ray must be finite with positive z")
        cal = self.calibration
        pixel, _ = cv2.projectPoints(r.reshape(1, 1, 3), np.zeros(3), np.zeros(3), cal.matrix,
                                     cal.dist if cal.dist is not None else np.zeros(5))
        return transform.from_original(float(pixel[0, 0, 0]), float(pixel[0, 0, 1]))

    @staticmethod
    def _angle(a, b) -> float:
        a, b = np.array(a), np.array(b)
        return float(math.atan2(np.linalg.norm(np.cross(a, b)), float(a @ b)))

    # -- extent and rate -----------------------------------------------------
    def bbox_angular_extent(self, bbox: BBox, transform: ImageTransform = IDENTITY,
                            context: FrameContext | None = None) -> AngularExtent:
        """Angle between mid-edge rays (left/right and top/bottom). Not a physical dimension."""
        cx, cy = (bbox.x_min + bbox.x_max) / 2, (bbox.y_min + bbox.y_max) / 2
        rays = [self.pixel_to_ray(*p, transform, context) for p in
                ((bbox.x_min, cy), (bbox.x_max, cy), (cx, bbox.y_min), (cx, bbox.y_max))]
        bad = next((r for r in rays if r.direction is None), None)
        if bad is not None:
            return AngularExtent(None, None, bad.availability_reason)
        return AngularExtent(self._angle(rays[0].direction, rays[1].direction),
                             self._angle(rays[2].direction, rays[3].direction))

    def angular_rate(self, p0: tuple[float, float], t0: float, p1: tuple[float, float], t1: float,
                     transform: ImageTransform = IDENTITY, context0: FrameContext | None = None,
                     context1: FrameContext | None = None) -> AngularRate:
        """Camera-relative angular speed of a point between two timed pixels (needs t1 > t0)."""
        t0, t1 = _finite(t0, "t0"), _finite(t1, "t1")
        if t1 <= t0:
            raise GeometryError("t1 must be greater than t0")
        a, b = self.pixel_to_ray(*p0, transform, context0), self.pixel_to_ray(*p1, transform, context1)
        for ray in (a, b):
            if ray.direction is None:
                return AngularRate(None, None, None, ray.availability_reason)
        dt = t1 - t0

        def azel(d):
            return math.atan2(d[0], d[2]), math.atan2(-d[1], math.hypot(d[0], d[2]))
        (az0, el0), (az1, el1) = azel(a.direction), azel(b.direction)
        return AngularRate(self._angle(a.direction, b.direction) / dt, (az1 - az0) / dt, (el1 - el0) / dt)

    # -- heading and size change --------------------------------------------
    def image_heading(self, vx_px_s: float, vy_px_s: float) -> ImageHeading:
        vx, vy = _finite(vx_px_s, "vx"), _finite(vy_px_s, "vy")
        speed = math.hypot(vx, vy)
        if speed < self.min_speed_px_s:
            return ImageHeading(None, speed, availability_reason={
                "heading_deg": f"speed {speed:.3g} px/s below {self.min_speed_px_s} px/s; heading undefined"})
        return ImageHeading(math.degrees(math.atan2(-vy, vx)) % 360.0, speed)

    def apparent_size_change(self, samples: list[tuple[float, float]], *, basis: str = "angular_width",
                             contexts: list[FrameContext] | None = None) -> SizeChange:
        """Label the trend of ``ln(size)`` over time. ``samples`` are ``(t_s, size)`` with size in
        radians (basis ``angular_width``) or pixels (``pixel_width``). Never states range or approach."""
        if basis not in ("angular_width", "pixel_width"):
            raise GeometryError("basis must be angular_width or pixel_width")
        reasons = self._unsupported(None) if basis == "angular_width" else {}
        zooms = {c.zoom for c in contexts or [] if c.zoom is not None}
        for context in contexts or []:
            if not reasons:
                reasons = self._unsupported(context) if basis == "angular_width" else (
                    {"size": "camera moved during the window"} if context.camera_moved else {})
        if not reasons and len(zooms) > 1:
            reasons = {"size": f"zoom changed within the window: {sorted(zooms)}"}
        if not reasons and basis == "pixel_width" and self.calibration is not None \
                and any(not math.isclose(z, self.calibration.zoom, rel_tol=1e-9) for z in zooms):
            reasons = {"size": "zoom differs from the calibrated zoom"}
        if reasons:
            return SizeChange("uncertain", None, None, len(samples), None, availability_reason=reasons)
        ordered = sorted((_finite(t, "t"), _finite(s, "size")) for t, s in samples)
        if any(s <= 0 for _, s in ordered) or len({t for t, _ in ordered}) != len(ordered):
            raise GeometryError("sizes must be positive and timestamps distinct")
        n = len(ordered)
        if n < self.size_min_samples or ordered[-1][0] - ordered[0][0] < self.size_min_span_s:
            return SizeChange("uncertain", None, None, n, basis, availability_reason={
                "label": f"need >= {self.size_min_samples} samples spanning >= {self.size_min_span_s} s"})
        t = np.array([o[0] for o in ordered])
        y = np.log([o[1] for o in ordered])
        tc = t - t.mean()
        sxx = float(tc @ tc)
        slope = float(tc @ (y - y.mean()) / sxx)
        resid = y - y.mean() - slope * tc
        se = math.sqrt(float(resid @ resid) / (n - 2) / sxx)
        thr = self.size_stable_log_rate_per_s
        if abs(slope) - 2 * se > thr:
            label = "expanding" if slope > 0 else "shrinking"
        elif abs(slope) + 2 * se < thr:
            label = "stable"
        else:
            label = "uncertain"
        return SizeChange(label, slope, se, n, basis)
