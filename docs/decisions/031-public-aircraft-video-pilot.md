# AERIS-031 — bounded public aircraft-video pilot

Date: 2026-10-15. Status: implementation for independent review.

QUT S5 is a small rights-cleared fixed-camera aircraft clip with approximate
target-centroid labels. Its published terms are BSD-3-Clause. We pin the source
revision and measured file hashes, keep media and labels out of Git, and enforce
the 2 MB task budget. The accepted detector, tracker and forecasters are composed
in a separate script so the pilot does not depend on unaccepted AERIS-012.

The video-frame offset is established from the published global frame bounds,
the decoded frame count, and the last label. Evaluator-only labels never enter
the inference path. A fixed unique-center gate provides narrow descriptive
pixel-error summaries with explicit support. No threshold is tuned or frozen.

This task adds one real-video observation, not an aircraft performance baseline.
The source has no verified exhaustive boxes, negative frames, all-aircraft track
IDs, camera intrinsics or surveyed fixed pose. AP, HOTA/IDF1, real ENU bearing,
predicted range, live accuracy and intervals remain unavailable. Deferred
AERIS-021/022 remain separate evidence tasks.
