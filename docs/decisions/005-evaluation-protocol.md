# ADR 005 — Standard evaluator and eligibility policy

Status: proposed for AERIS-005 review. Date: 2026-09-30.

Use pycocotools 2.0.11 and TrackEval 1.3.0 rather than assigning their metric names
to local approximations. Exclude partial and unlabeled frames from AP/FP counts.
Require complete timed identity sequences for tracking. Evaluate forecast models on
identical eligible keys, with bounded evaluator-only interpolation and explicit
misses. Keep width-bin recall separate from COCO area metrics. These choices prevent
unlabeled truth, box spam and missing predictions from silently improving scores.

The adapter tests use synthetic records only. Real-video metric claims await rights-
cleared complete labels and identities. This protocol is intentionally conservative:
partially annotated FGVC does not yield COCO AP or tracking scores.
