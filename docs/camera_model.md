# Camera geometry (AERIS-011)

`python/aeris/estimation/camera.py`: pixel-to-ray, angular extent and rate, image heading
and apparent size change for a **fixed camera**, with a pixel-only mode when no valid
calibration exists. Nothing here estimates range, closing speed, approach or aircraft
orientation. It is tested only on analytic fixtures and a synthetic distorted camera; no
real calibration exists, so no real-camera accuracy is claimed.

## Calibration and conventions

`Calibration` holds `K = [[fx,0,cx],[0,fy,cy],[0,0,1]]` (positive focal lengths, zero skew,
principal point inside the image), the original resolution, a distortion model, a
calibration ID, quality, lens and zoom. Models: `none` and `brown_conrady` (OpenCV
`k1,k2,p1,p2[,k3]`); anything else (fisheye, rational) is rejected, not guessed. Image
origin is top-left, x right, y down, in original-frame pixels. Rays are unit
`(x, y, z)` with z forward, x right, y down. Distortion is removed first (iterative
`undistortPoints`, tolerance 1e-12, up to 100 iterations) and the ray is `normalize([x, y, 1])`.

A calibration whose distortion polynomial is not invertible over the frame (undistort then
re-project must return each of a 21x13 pixel grid, corners included, within 0.01 px) is rejected
at construction with `GeometryError`; `pixel_to_ray` repeats the re-projection check per call and
returns no ray with a reason if the solve did not converge. This closes a review finding where
extreme coefficients such as (-0.5, 0.3, 0, 0, -0.1) silently gave rays up to ~378 px off.

## Crop, resize and padding

`ImageTransform` is the explicit axis-aligned map `processed = (original - crop) * scale +
pad`; `to_original` and `from_original` are exact inverses and `ImageTransform.letterbox(r)`
reproduces the AERIS-006 detector input (a test checks the same numbers as the detector
tests). Pass the transform with any pixel that is not in original-frame coordinates. Rays
and extents are identical whichever processed image a pixel came from.

## Outputs

| Function | Result | Convention |
|---|---|---|
| `pixel_to_ray` / `ray_to_pixel` | unit ray / pixel (distortion applied back); `ray_to_pixel` is `None` when the pixel falls outside the calibrated frame | round trip tested |
| `bbox_angular_extent` | width/height in rad (deg properties) | angle between mid-edge rays; **not** a physical size |
| `angular_rate` | rad/s, plus azimuth/elevation rates | camera-relative; needs `t1 > t0`; image y down = negative elevation |
| `image_heading` | degrees in `[0, 360)` | `atan2(-vy, vx)` counterclockwise from image right; `None` below 1 px/s; **not aircraft yaw** |
| `apparent_size_change` | `expanding` / `shrinking` / `stable` / `uncertain` with log-rate and standard error | trend of `ln(size)` over time; says nothing about range or approach |

Unknown values are `None` with an `availability_reason`; NaN/inf inputs raise
`GeometryError`.

## Suppression rules (AC2)

- No calibration (pixel-only mode): rays, extents, rates and angular size change are `None`
  or `uncertain` with a reason. Heading and pixel-basis size change still work.
- `FrameContext(camera_moved=True)` or a zoom different from the calibrated zoom: angular
  claims are suppressed; a size-change window containing such a frame, or two different
  zooms, is `uncertain` for both bases.
- `zoom=None` means "assumed unchanged"; that is an assumption the caller owns, not a check.
- A pixel outside the calibrated frame has no ray.
- Size change needs at least 4 samples spanning at least 0.3 s; the label requires the
  slope to clear a 0.1 /s threshold by twice its standard error (stable: fully inside it),
  otherwise `uncertain`. These thresholds are untuned engineering defaults.

## Limits

Single fixed camera only (no pose, motion compensation or zoom tracking), no fisheye or
rational distortion, no calibration quality metric beyond the stored text, and no real
calibration to test against. Perspective effects of target pose on apparent size are not
modelled, which is why size change is never converted to approach or range. Tested on macOS
arm64, Python 3.14, OpenCV 5.0.0.
