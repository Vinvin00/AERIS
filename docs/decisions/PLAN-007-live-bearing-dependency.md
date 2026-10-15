# PLAN-007 — Fixed-camera bearing live task dependency

Status: independently reviewed. Date: 2026-10-15.

AERIS-030 originally depended on AERIS-028, a scene-relative image forecast under
camera pan. A world-referenced bearing from a surveyed *fixed* camera cannot use
that correction to infer the camera's physical orientation after movement. It
must suppress physical bearings when movement is detected or cannot be assessed.
The accepted AERIS-010 forecaster supplies causal original-frame pixel predictions;
the accepted AERIS-011 camera model supplies calibrated rays; AERIS-029 supplies
the surveyed-site conversion. Those are the direct prerequisites of AERIS-030.

Replace the AERIS-028 dependency with AERIS-010/011/029. Keep AERIS-030's
acceptance criteria and checks. AERIS-030 must inspect the actual live loop,
convert its current-frame original pixels, and test that featureless or failed
camera-motion assessment suppresses physical bearings. It must not treat
scene-relative motion compensation as an updated camera pose. Add the live UI
file to required artifacts because the displayed availability is part of the
task. This planning change neither accepts AERIS-023 through AERIS-028 nor
changes their criteria, evidence or status. No physical camera accuracy is claimed.

This supersedes the final sentence of the AERIS-029 decision's next-step section,
which said AERIS-030 waited for the scene-relative forecast prerequisites. The
existing live module already contains unaccepted AERIS-028 work; AERIS-030's
implementer and reviewer must inspect that current code path and verify the
current-frame pixel conversion and fail-closed stability behavior end to end.

Read-only planning review confirmed DAG lint (36 tasks, 25 checks), unchanged
AC1–AC3 and check list, and the scientific distinction between scene-relative
display motion and fixed-camera surveyed rays. It flagged the existing live
reference-frame forecast path and `CameraMotion.update` failure as mandatory
integration-review points; a failed estimate is unknown stability, never proof
of a stationary camera.
