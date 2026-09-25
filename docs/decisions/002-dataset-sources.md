# ADR 002 — Dataset sources and split unit

Status: proposed for AERIS-002 review. Date: 2026-09-24.

Decision: use FGVC-Aircraft 2013b only as a still-image detection smoke check.
It has partial labels, no timestamps, no identities, no annotated negatives and no
boxes under 64 px wide. The check scores only main-aircraft recall and localization; unmatched
detections are unscored, and AP, precision and false positives per frame are
unavailable. The annotations archive is pinned by its measured SHA-256 because the
published MD5 files predate the archives. Do not use DUT Anti-UAV for aircraft claims: it covers the
UAV domain and has no explicit dataset license. The real benchmark is an owned,
fixed-camera video pilot. Its split unit is the capture session (3 development,
1 validation, 2 sealed holdout), and derivatives inherit the session's split.

Reason: no audited public source provides identity-annotated manned-aircraft video
with complete negatives and tiny targets under clear terms. Session-level splitting
is the only rule that prevents adjacent-frame leakage.

Planned capabilities are recorded as `planned_evaluations`. A source becomes
`eligible` only once it is acquired and every artifact has a measured SHA-256.

Consequences: AERIS-003 implements the FGVC conversion (1-based inclusive to 0-based
half-open, clamp and flag 130 zero coordinates, banner as a recorded input crop) and
the duplicate/near-duplicate checks. Real tracking and forecast evidence is blocked
until the owner decides on footage rights and the annotation budget.
