# AERIS-030 — Live fixed-camera bearing binding

Status: implementation for independent review. Date: 2026-10-15.

The loopback server accepts operator-provided calibration, surveyed fixed-camera
pose and current zoom at startup. They are read locally, not sent from the
browser. Missing files leave live pixel output available and ENU bearings null.

The live loop converts the tracker's assigned original-frame detection center
for the observed bearing. Its scene-relative forecast is mapped back to the
current original frame before bearing conversion. Observation time is the
source frame time: paced files use decoded PTS relative to their first frame,
with a marked frame/FPS fallback when PTS are unavailable. Forecast target
time is issue time plus the causal horizon.
Both kinds of bearing appear separately in the panel with availability reasons.

`CameraMotion.update` returning false means the background could not be
assessed, not that the camera was stationary. The first frame is also
unassessed. A measured transform moving an original-frame corner by more
than 1 px counts as movement. Either event after the first frame invalidates
the fixed pose for the remainder of that session. The threshold is an untuned
suppression rule, not a validated mount-motion detector. A featureless sky
may therefore produce no physical bearings. No range or target 3-D position
is produced, and no real-camera angular accuracy or uncertainty is claimed.

Tests use a synthetic camera, pose, paced file and controlled motion-estimator
outcomes. A physical camera, surveyed reference points and independent
angular-error measurement remain the next evidence requirement.
