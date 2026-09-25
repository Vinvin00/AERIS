# Dataset loading and QA (AERIS-003)

Tools for loading canonical contract-v1 datasets, rejecting bad data, computing
statistics, converting FGVC annotations and drawing annotation contact sheets.
Automated regressions use **owned synthetic fixtures**. A separate bounded sample
of three original FGVC images and labels was ingested and inspected on 2026-09-25
(details below). Neither fixture success nor this convenience sample measures
perception performance or establishes full dataset fitness.

## Canonical dataset layout

```
DATASET/dataset.json       index (strict JSON, exactly these keys)
DATASET/annotations.jsonl  one AnnotationRecord (contract v1) per line
```

`dataset.json` keys:

| Key | Meaning |
|---|---|
| `schema_version` | integer `1` |
| `dataset_id`, `class_mapping_version` | identifiers; every record must use the supported class map `aeris-aircraft-v1` (`0: aircraft`) |
| `image_root` | directory for `image_path`, relative to DATASET unless absolute |
| `group_basis` | how groups were formed, e.g. capture session or one per FGVC image |
| `unannotated_attributes` | subset of `ignore`/`occluded`/`truncated` that the source did not label; their `false` values mean "not annotated" and must not be scored |
| `transforms` | empty object or `model_input_crop` with exactly `bottom_px` (nonnegative integer smaller than every original frame height) and `reason` (nonempty text); truth is never cropped |
| `sequences` | `[{sequence_id, group_id, split}]`; the split unit is the group |

Within a sequence, records must follow frame order. Different sequences can be
interleaved. Images are decoded with `IMREAD_COLOR | IMREAD_IGNORE_ORIENTATION`, so
EXIF rotation never changes the original-frame coordinate system.
The supported class registry is explicit package code, independent of the current
working directory. Unknown mapping versions, class IDs, or transforms are rejected.
Crop geometry is in original integer pixels; bool, fractional and nonfinite values
are rejected. An empty transform object means no crop.

## Commands

```sh
PYTHONPATH=python .venv/bin/python -m aeris.data.validate DATASET [--output report.json]
PYTHONPATH=python .venv/bin/python -m aeris.data.fgvc FGVC_DATA_DIR OUT_DIR
PYTHONPATH=python .venv/bin/python -m aeris.data.visualize DATASET --output sheet.png
.venv/bin/python tests/fixtures/data/generate.py     # regenerate fixtures
```

`validate` prints a JSON report and exits 1 when there are problems. Setting
`PYTHONPATH` is needed on this Mac because macOS sometimes marks the editable-install
`.pth` file `hidden`, and Python 3.14 skips hidden `.pth` files. After
`pip install -e .` it works without `PYTHONPATH` until the flag returns.

## What is rejected (AC1)

Validation reports every problem, each with a file and line number where possible.

| Problem | Check |
|---|---|
| Missing frame | image file absent under `image_root` |
| Corrupt frame | SHA-256 of bytes ≠ `image_checksum`; empty bytes or OpenCV cannot decode; decoded size ≠ recorded width×height |
| Path escape or malformed path | absolute `image_path`, symlink or other path resolving outside `image_root`, null bytes or failed path resolution |
| Malformed boxes and lines | contract v1 parsing: out-of-frame or zero-area boxes, NaN/Infinity, duplicate keys, unknown or missing fields, blank lines, JSON the parser cannot handle (nesting past the recursion limit, integers past Python's digit limit) |
| Duplicate objects | same `annotation_id` or `track_id` twice in a frame (contract); same class and identical box twice in a frame |
| Track IDs | duplicate identity within a frame; class change of a track within a sequence |
| Sequence errors | non-increasing frame index or time, mixed timed/untimed frames, class-map change, sequence not listed, listed sequence without frames |
| Split leakage | a group in more than one split; identical encoded bytes in two splits; identical decoded pixels in two splits, which catches re-encodes such as PNG compression changes |
| Index errors | wrong or missing keys, duplicate `sequence_id`, bool as version, malformed or unknown attributes, unsupported class-map versions/transforms, invalid crop structure/range |

Exact duplicates within one split are counted in `duplicate_images_within_split`
but are not errors. **Near-duplicate** detection (for example a perceptual hash) is
not implemented. ADR 002 requires its threshold to be set from the hash-distance
distribution of real acquired data, and none exists yet.

## Statistics

Statistics are reported per split: sequences, groups, frames, timed frames,
annotation status counts, negative frames, and objects. Negative frames are
`complete` with zero objects. Object counts include ignored, occluded and
truncated objects, distinct `(sequence, track)` pairs, and objects without a track.
Width bins use contract bins over **non-ignored** objects:
`x_max - x_min` in original pixels. The counts cover all statuses; evaluation
support must still be restricted to eligible frames.

## FGVC conversion (AC2)

This implements the adopted ADR 002 conversion assumption: treat FGVC boxes as
1-based inclusive. Original endpoint inclusion remains unproven; visual inspection
at contact-sheet resolution cannot establish one-pixel semantics. The exact rule is:
`x_min = xmin - 1`, `y_min = ymin - 1`, `x_max = xmax`, `y_max = ymax`. The result is
0-based half-open original-frame pixels.

- Zero coordinates, which the stated convention forbids, are clamped to 0 and
  listed in `conversion_report.json` under `flags.zero_coordinate_clamped`.
- Boxes reaching the 20-px banner are listed under `flags.box_reaches_banner`.
- The banner is recorded as `transforms.model_input_crop` and is never applied
  to the truth boxes.
- Boxes outside the image, non-integer values, missing images, an image in two
  splits, and images without both a box and a split all abort the conversion.
  Nothing is written when it aborts.
- Source image IDs must be seven ASCII digits. Paths or traversal IDs are rejected
  before image access, and resolved image symlinks must stay within the declared
  image directory. Source and output roots are physically resolved before storing
  their relative path, so symlinked directory aliases work. Images remain external
  to the converted metadata directory; moving only metadata can break those paths.
- **Grouping is preserved explicitly.** FGVC carries no photographer, event or
  airframe key, so each image becomes its own group and keeps its official split.
  `group_basis` states this. The image-level random split therefore stays a smoke
  check only, as ADR 002 says.
- Frames are `partial` with null timestamp, camera and track ID, each with a
  reason. `occluded` and `truncated` are listed in `unannotated_attributes`,
  because contract v1 has no null for those flags.

## Fixtures and inspected contact sheets (AC3)

`tests/fixtures/data/` is generated by `generate.py`: synthetic sky gradients with
drawn aircraft shapes.

- `canonical/` has two sequences:
  - `clip-a`: session-1 → development. It has 128×96 frames with PTS 0, 0.1, 0.2
    and 0.4 s. Frame 3 is absent, so there is an index gap. It includes one ignore
    region and one complete negative frame.
  - `clip-b`: session-2 → holdout. It has 160×120 frames with FPS-inferred times.
    Frames: complete with a 10-px and a 40-px aircraft; partial with an occluded
    aircraft and an unannotated one; and unlabeled.
- `fgvc_like/` has the FGVC 2013b layout: 4 JPEGs of 120×90 with a 20-px banner.
  `0000002` contains the forbidden zero coordinate, and `0000003` reaches the
  banner.

Measured with `validate` (2026-09-25):

| Split | frames | status c/p/u | negatives | objects (ignored) | tracks | width bins [0,16)/[16,32)/[32,64)/[64,∞) |
|---|---|---|---|---|---|---|
| development | 4 | 4/0/0 | 1 | 4 (1) | 1 | 0/0/0/3 |
| holdout | 3 | 1/1/1 | 0 | 3 (0) | 2 | 1/0/2/0 |
| fgvc train/val/test | 2/1/1 | all partial | 0 | 1 per image | 0 | 32–64 px: 2/0/1; ≥64 px: 0/1/0 |

The FGVC conversion reported `zero_coordinate_clamped: [0000002]` and
`box_reaches_banner: [0000003]`. Both converted and canonical fixtures validate
with no problems.

**Visual inspection** (by eye, of both sheets drawn by `visualize`):

- Green boxes enclose the drawn aircraft, and the yellow box sits on the ignore blob.
- The complete empty frame is captioned NEGATIVE.
- The partial frame shows an unboxed aircraft beside the occluded, boxed `t2`.
- The unlabeled frame shows aircraft with no boxes.
- The FGVC tiles show the red crop line at the banner edge, and `0000003`'s box
  visibly crosses it.

`test_sheet_draws_truth_boxes_at_original_coordinates` also checks box pixels at
scale 1.0. The sheets are generated outputs and are not committed. Regenerate them
with the commands above.

**Linux re-inspection (2026-09-25).** The cloud coordinator session regenerated both
sheets from the committed fixtures on CPython 3.11.15 with OpenCV 5.0.0 (tile width 320)
and viewed them. Every observation above held. The 1-px red crop line is hard to see at
preview scale, so a pixel check was also run: it found the line at the expected row in
all four FGVC tiles (sheet y = 189, original y = 70), and a 3× zoom showed it on the
banner edge. `validate` reproduced the statistics table and the conversion flags exactly.

## Bounded original-sample QA

**Evidence availability.** This section is the original macOS session's record. Its
evidence files (`orchestration/runs/research/…`, `data/raw/g1-bounded-sample/…`) are
Git-ignored and stayed on that machine. The 2026-09-25 cloud session that re-verified
AERIS-001 to 003 did not have them, so it could not re-check the claims below. Treat
them as unverified until that archive is restored or the sample is fetched again
within the recorded budget.

The coordinator fetched exactly the first 8 MiB of the
[official FGVC archive](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz)
using HTTP Range (`bytes=0-8388607`, HTTP 206), plus 407,927 bytes of annotations.
The task-local budget is recorded at
`orchestration/runs/research/g1-bounded-sample/budget.md`. Streaming extraction kept
only the first three complete JPEG members, with their original box and split rows.
This was a convenience sample, not a selected or representative benchmark.

The metadata SHA-256 matched the AERIS-002 pin:
`50e4419bd330b68a2e85dbcb32339d31120190efe6944682e87c9b431f5f202a`.
The archive prefix SHA-256 was
`c713c8721b6d8eb4899ac773e7383be6b3ddc48a13fc0743b7aecba752ba9b3c`.
The full archive was **not acquired or verified**; its checksum is unknown here.
Image hashes, prefix provenance and sampling policy are recorded in the ignored
`data/raw/g1-bounded-sample/sample_provenance.json`. Research-only local images
and generated contact sheets are not committed or redistributed.

The converter retained two train images, no validation images and one test image.
The validator returned no structural problems and no within-split duplicates.
All three records are partial, untimed, without track identities, and their
annotated widths fall in `[64,infinity)`. Conversion reported no zero-coordinate
clamps and `box_reaches_banner: [1885942]`. These are QA counts, not detection scores.

Coordinator and repair agent inspected the generated contact sheet:

| Image ID | Original split | Visual observation |
|---|---|---|
| `1143132` | train | Side-on exterior aircraft with a tight visible green box |
| `1627560` | train | Airborne exterior aircraft with a tight visible green box |
| `1885942` | test | Passenger cabin interior with a nearly full-frame box overlapping the banner |

**Image `1885942` is semantically unsuitable for the intended exterior-aircraft
detection task despite its structurally valid source annotation.** It remains
recorded as source evidence pending explicit dataset curation. It was neither
reclassified as a negative nor silently removed. This is direct evidence that
schema validation cannot establish label fitness.

Evidence: `orchestration/runs/research/g1-bounded-sample/validation.json`,
`inspection.md` and `contact-sheet.png`. Preserve these private run artifacts
with the source provenance between sessions. No training, accuracy metrics,
tracking identities or video performance were evaluated.

Before this ingestion, the coordinator also inspected the published montage in
[Figure 1 of the authors' paper](https://arxiv.org/pdf/1306.5151v1), as recorded in
`orchestration/runs/research/real-sample-inspection.md`. That earlier activity
was inspection of a published figure, **not original image ingestion**. The
montage shows varied viewpoints, lighting, background clutter and secondary
aircraft, supporting caution about annotation completeness and target scale.

## Environment

pyproject declares `numpy>=2,<3` and `opencv-python-headless>=5,<6`. Two
environments have run the tooling:

- **macOS arm64:** the original local `.venv`, CPython 3.14.6 on Apple M4 Pro, with
  numpy 2.5.3 and opencv-python-headless 5.0.0.93. Hash-pinned in
  `requirements/data-macos-arm64-py314.txt`.
- **Linux x86_64 (cloud container, 2026-09-25):** CPython 3.11.15 with numpy 2.4.6
  and opencv-python-headless 5.0.0.93. Hash-pinned in
  `requirements/data-linux-x86_64-py311.txt`. A hash-checked install of that lock
  into a fresh venv passed the full unittest suite. Running `generate.py` there
  reproduced every committed fixture file byte for byte.

CI (GitHub Actions, Linux Python 3.11/3.12) resolves the pyproject ranges rather
than a lock. Python 3.12 has no lock.

```sh
python3.11 -m venv .venv
.venv/bin/pip install --only-binary=:all: --require-hashes -r requirements/data-linux-x86_64-py311.txt
```

## Limitations

- Automated tests use generated fixtures; three real original image/label pairs
  provide bounded ingestion/visual QA only. Full-release acquisition and fitness,
  endpoint inclusion, tiny-target coverage and real video evidence remain unproven.
- There is no near-duplicate check. Locks exist only for macOS arm64 CPython 3.14
  and Linux x86_64 CPython 3.11.
- The FGVC `occluded`/`truncated` flags are `false` placeholders. Consumers must
  honour `unannotated_attributes`.
- The FGVC conversion was exercised on generated fixtures and the three-image
  original sample, not the full 10,000-image release. The cabin-interior sample
  requires an explicit curation decision before use in an exterior-aircraft benchmark.
