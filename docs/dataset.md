# Dataset feasibility audit (AERIS-002)

Audited 2026-09-24. Machine-readable record: `configs/data/source_manifest.json`,
checked by `tests/unit/test_manifest.py`. No images, video or full archives were
acquired. One small metadata file was inspected (see "Measured facts").

## Verdict

- **Usable once acquired:** FGVC-Aircraft 2013b, as a *still-image detection smoke
  check* only. Because only the main aircraft is boxed, the check can measure
  main-aircraft recall and localization, but not AP, precision or false positives
  per frame.
- **Not usable for aircraft claims:** DUT Anti-UAV (UAV domain, unclear data terms).
- **Real video benchmark:** does not exist. Tracking, forecasting, negatives and
  tiny-target evidence all depend on the owned video pilot planned below.

## Sources

| | FGVC-Aircraft 2013b | DUT Anti-UAV | AERIS owned pilot |
|---|---|---|---|
| Role | still-image smoke baseline | out-of-domain reference | primary video benchmark |
| Terms | [non-commercial research only](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/); photographers keep copyright | "[released publicly for academic research](https://arxiv.org/abs/2205.10851)"; no dataset license found | owner-held; undecided |
| Modality | RGB stills, ~1–2 MP | RGB stills + 20 video sequences, moving camera | RGB video, fixed tripod/zoom |
| Classes → AERIS | main aircraft (100 variants) → `0 aircraft` | UAV → not mapped | aircraft → `0 aircraft` |
| Boxes | one main aircraft/image, not exhaustive | per image / one target per sequence | all visible aircraft |
| Track IDs / PTS | no / no | single-object only / unverified | yes / yes (planned) |
| Annotated negatives | none | unverified | yes (planned) |
| Status | not acquired | not acquired | planned |

AERIS class mapping version `aeris-aircraft-v1` has a single class, `0 aircraft`.
FGVC's variant/family/manufacturer labels are not used for detection.

DUT rights note: the repository's Apache-2.0 license covers the repository, and
arXiv's CC BY-NC-SA 4.0 covers the paper. Neither grants rights to the media.
Since DUT shows drones rather than manned aircraft, it cannot stand in for aircraft
evidence. Using it later needs written confirmation of permitted use.

## Measured facts (FGVC annotations-only archive)

To avoid copying published numbers blindly, the 407,927-byte annotations archive was
downloaded to a temporary directory outside the repository, inspected and deleted.
No images were fetched.

- 10,000 images, 100 variants, 70 families, 30 manufacturers. Splits
  3,334/3,333/3,333; no image appears in two splits.
- **Count discrepancy:** the landing page and the bundled README say 10,200/102/41.
  The files match the paper (10,000/100/30).
- **Checksum discrepancy:** the published MD5 `e3799900…6427e` does not match the
  served file's MD5 `6939ddc4…f999` (same result on two downloads). The byte size
  matches. Both `.md5` files were last modified at 17:28:19 GMT on 2014-01-17,
  *before* the annotations archive (17:42:51) and the full archive (17:55:16), so
  the published MD5s probably describe earlier builds. The annotations archive is
  pinned by the SHA-256 in the manifest. The full 2,753,340,328-byte archive was not
  downloaded; if its MD5 also mismatches, AERIS-003 logs a discrepancy and pins the
  measured SHA-256 only after the owner approves or the maintainers confirm.
- **Box convention discrepancy:** the README says the top-left pixel is (1,1),
  yet 130 of 10,000 boxes have `xmin` or `ymin` = 0.
- Box width (`xmax-xmin+1`) ranges from 94 to 1596 px, median 1004. **All 10,000
  boxes fall in the `[64,inf)` bin**, so FGVC provides no tiny-object support.

### Conversion rule for AERIS-003 (proposal, to be tested there)

Treat FGVC boxes as 1-based inclusive: `x_min = xmin - 1`, `x_max = xmax`, and the
same for y. That gives 0-based half-open original-frame pixels. The 130 zero-valued
coordinates must be clamped to 0 *and* flagged in the loader's QA report; do not
correct them silently. The 20-px bottom banner is a model-input crop recorded as a
transform. Ground truth stays in original-frame coordinates. Any box reaching into
the banner must be reported. FGVC frames are `partial` (main aircraft only) and
have null timestamps and null track IDs.

**FGVC ignore policy** (from `metric_policy` in the manifest): only the main
aircraft is labeled, so a detection that doesn't match its box is left unscored,
neither a true nor a false positive. Valid measures are main-aircraft recall at a
stated IoU threshold and localization error. AP, precision, false positives per
frame and negative-frame errors are unavailable on FGVC.

## Split strategy (AC2)

Rules, encoded in `split_policy`:

1. **Unit is the source group**: a capture session for the pilot, or a tracking
   sequence where no session exists. Whole groups go to one split before any model
   result is seen. Frames, clips and every derivative inherit their group's split.
2. **Pilot allocation:** at least 6 sessions: 3 development, 1 validation,
   2 holdout. The holdout is sealed until release evaluation (AERIS-018) and never
   drives thresholds or model selection.
3. **Exact duplicates:** compare the SHA-256 of encoded bytes *and* of decoded RGB
   pixels across splits. Resolve any match by moving or removing the later group,
   and log it.
4. **Near-duplicates:** perceptual-hash candidate pairs across splits are reviewed
   manually. The Hamming threshold stays null until AERIS-003 measures the actual
   distance distribution, so no threshold is guessed here.
5. **Negatives:** only `complete` frames with zero objects count. Each split needs
   negatives from at least one session, and the count per split is reported.
   `partial`/`unlabeled` frames never count. FGVC contributes none.
6. **Tiny objects:** report per-split, per-bin support using the contract width bins
   in original pixels. A bin with zero holdout support reports metrics as
   *unavailable*, never as zero or pass. FGVC contributes nothing below 64 px.
7. **FGVC:** its official random split is image-level, not photographer- or
   event-isolated (the files carry no group key). It is used only as a smoke check,
   never for the development regression set or the holdout.
8. **Pretraining overlap:** unknown until a detector is chosen (AERIS-006), so it is
   null with a reason for every source. Pilot footage has no overlap only if the owner
   captures it after the detector weights are fixed. Licensed footage may already be
   public.
9. **Intervals:** 2 holdout sessions (≈4 clips) are too few resampling units for
   informative sequence-bootstrap intervals. Holdout results are per-sequence and
   descriptive. AERIS-005 sets the minimum sequence count for reporting intervals
   before any holdout result; below it, intervals are null with a reason.

`planned_evaluations` lists what a source could support once acquired.
`eligible_evaluations` stays empty until a source is `acquired` and every
artifact has a measured SHA-256. The validator enforces this. DUT has no planned
AERIS evaluation.

## Real-video acquisition and annotation plan (AC3)

**Missing labels today:** track IDs across frames, decode PTS, per-frame complete
status, annotated negatives, tiny targets (<32 px), occlusion/truncation flags,
simultaneous multi-aircraft identities, camera ID and optional calibration.

**Scope and caps:** up to 12 clips × ~20 s (≈7,200 frames at 30 fps), at least 6
independent sessions (different days/locations), ≤1 GB retained media, ≤24 h
annotation plus review. Seek negatives, tiny/distant targets, entries/exits,
occlusion, lighting changes and genuine simultaneous aircraft. Regimes not captured
are reported as gaps.

**Effort (engineering judgement, not measured):** selection and provenance 2–4 h,
annotation 8–14 h, independent review 4–6 h. This assumes keyframe interpolation
with a per-frame review. Capture travel and waiting for rare events are excluded
and may exceed the caps.

**Procedure:** preserve original container PTS and the SHA-256 of each file and
decoded frame at ingest. Annotate every frame for all visible in-scope aircraft
with track IDs, visibility and ignore flags. An independent reviewer checks every
identity change and ambiguous frame, plus a random sample of ordinary frames. The
sample fraction is fixed before review starts. Stop at a cap and report.

**Constraints:** use only owner-captured or explicitly licensed footage. Shoot from
a fixed tripod at fixed zoom. Use lawful public vantage points, follow airport
rules, and do not identify people. No media or annotations go into Git; commit only
manifests, hashes and retrieval notes.

**Decisions needed from the project owner:** who captures or licenses the footage
and under what terms; locations and dates for ≥6 sessions; the annotation tool
(CVAT is a candidate) and its export mapping to contract v1; and approval of the
24 h annotation/review budget. Without the pilot, G3+ real tracking and forecast
claims stay blocked.

## Limitations

- FGVC's published MD5 could not be reconciled (probably stale); the full archive is
  unmeasured.
- The pilot's size supports descriptive per-sequence results, not generalization
  claims or tight intervals.
- DUT facts come from the paper and README only; formats, sizes and negatives are
  unverified.
- The real sample images (FGVC paper figure, DUT montage) were not visually
  inspected in this task. G1's real-visual-sample evidence belongs to AERIS-003.
- Effort estimates are judgement and will be re-estimated after the first session
  is annotated.
