# ADR 003 — Canonical dataset layout and data tooling dependencies

Status: proposed for AERIS-003 review. Date: 2026-09-25.

Decision: a dataset is a directory with a strict `dataset.json` index and an
`annotations.jsonl` file holding one contract-v1 AnnotationRecord per line. Group and
split live in the index per sequence, not in frame records. Contract v1 has no such
fields, and keeping them in one place means one edit moves a whole group. Attributes a
source never labels are listed in `unannotated_attributes`, because contract v1
flags cannot be null. Images are decoded with EXIF orientation ignored. Leakage is
checked on groups, encoded bytes and decoded pixels.

Data tooling depends on numpy and opencv-python-headless (the GUI-free build;
Apache-2.0), declared as ranges in pyproject and hash-locked for the local macOS
arm64 Python 3.14 environment.

Reason: this is the smallest format that keeps provenance, grouping and truth
separate from pixels. It reuses the contract parser, so malformed records fail the
same way everywhere.

Consequences: later tasks read datasets through `aeris.data.validate.read_index` and
`read_annotations`, or validate first. Near-duplicate thresholds wait for real data.
Heavy ML dependencies (AERIS-006) may still need a Python 3.11/3.12 environment.
