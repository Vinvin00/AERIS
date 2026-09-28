"""Shared evaluation helpers: evaluator pins, availability, width bins and intervals."""
from __future__ import annotations

import base64
import contextlib
import hashlib
import importlib.metadata
import io
import json
import math
import random

import numpy as np

PYCOCOTOOLS_VERSION = "2.0.11"
TRACKEVAL_COMMIT = "12c8791b303e0a0b50f753af204249e622d0281a"
TRACKEVAL_SOURCE = f"git+https://github.com/JonathonLuiten/TrackEval@{TRACKEVAL_COMMIT}"
# Contract width bins in original pixels (docs/CONTRACTS.md); not COCO area ranges.
WIDTH_BINS = (("[0,16)", 16.0), ("[16,32)", 32.0), ("[32,64)", 64.0), ("[64,inf)", math.inf))
# docs/dataset.md defers this number to AERIS-005: fewer resampling units than this and
# sequence-bootstrap intervals are reported as null with a reason.
MIN_SEQUENCES_FOR_INTERVALS = 10


class EvaluationError(ValueError):
    """Inputs cannot be evaluated as supplied (mismatched frames, ambiguous keys...)."""


class EvaluatorVersionError(RuntimeError):
    """An installed evaluator differs from the pinned version."""


def evaluator_versions() -> dict:
    """Installed pycocotools version and TrackEval VCS commit (None if not a VCS install)."""
    trackeval = importlib.metadata.distribution("trackeval")
    direct = trackeval.read_text("direct_url.json")
    commit = json.loads(direct).get("vcs_info", {}).get("commit_id") if direct else None
    return {"pycocotools": importlib.metadata.version("pycocotools"),
            "trackeval": {"version": trackeval.version, "commit": commit,
                          "source": TRACKEVAL_SOURCE}}


def record_mismatches(distribution) -> list[str]:
    """Installed Python files whose SHA-256 differs from the wheel RECORD pip wrote."""
    changed = []
    for entry in distribution.files or ():
        if entry.hash is None or entry.hash.mode != "sha256" or not str(entry).endswith(".py"):
            continue
        digest = hashlib.sha256(entry.locate().read_bytes()).digest()
        if base64.urlsafe_b64encode(digest).rstrip(b"=").decode() != entry.hash.value:
            changed.append(str(entry))
    return changed


def require_pinned_evaluators() -> dict:
    """Check versions, the pip-recorded TrackEval commit and its installed-file hashes."""
    versions = evaluator_versions()
    problems = []
    if versions["pycocotools"] != PYCOCOTOOLS_VERSION:
        problems.append(f"pycocotools {versions['pycocotools']} != pinned {PYCOCOTOOLS_VERSION}")
    if versions["trackeval"]["commit"] != TRACKEVAL_COMMIT:
        problems.append(f"trackeval commit {versions['trackeval']['commit']} != pinned {TRACKEVAL_COMMIT}")
    changed = record_mismatches(importlib.metadata.distribution("trackeval"))
    if changed:
        problems.append(f"trackeval files differ from the installed RECORD: {changed[:5]}")
    if problems:
        raise EvaluatorVersionError("; ".join(problems))
    return versions


@contextlib.contextmanager
def quiet():
    """Capture evaluator console output (pycocotools and TrackEval print progress)."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


@contextlib.contextmanager
def numpy_scalar_aliases():
    """Restore np.float/np.int (the builtins NumPy aliased before 1.24) while TrackEval runs.

    TrackEval at the pinned commit still references them. Only missing names are added,
    and they are removed again afterwards; not thread-safe.
    """
    added = {name: kind for name, kind in (("float", float), ("int", int))
             if name not in np.__dict__}
    np.__dict__.update(added)
    try:
        yield
    finally:
        for name in added:
            np.__dict__.pop(name, None)


def width_bin(width: float) -> str:
    return next(name for name, upper in WIDTH_BINS if width < upper)


def ratio(numerator: float, denominator: float, name: str, reasons: dict, why: str):
    """numerator / denominator, or None with reasons[name] = why when the denominator is 0."""
    if denominator == 0:
        reasons[name] = why
        return None
    return numerator / denominator


def sequence_bootstrap(groups: dict[str, list], statistic, *, level: float = 0.95,
                       resamples: int = 1000, seed: int = 0,
                       min_sequences: int = MIN_SEQUENCES_FOR_INTERVALS) -> dict:
    """Percentile interval by resampling whole sequences (not correlated frames).

    groups maps sequence ID to that sequence's values; statistic takes a flat list.
    Returns {"low", "high", "level", "sequences", "resamples", "availability_reason"}.
    """
    names = sorted(name for name, values in groups.items() if values)
    result = {"low": None, "high": None, "level": level, "sequences": len(names),
              "resamples": resamples, "availability_reason": {}}
    if len(names) < min_sequences:
        reason = (f"{len(names)} sequences < minimum {min_sequences} for sequence-bootstrap "
                  "intervals")
        result["availability_reason"] = {"low": reason, "high": reason}
        return result
    rng = random.Random(seed)
    estimates = sorted(
        statistic([value for _ in names for value in groups[names[rng.randrange(len(names))]]])
        for _ in range(resamples))
    tail = (1 - level) / 2
    result["low"] = estimates[int(math.floor(tail * (resamples - 1)))]
    result["high"] = estimates[int(math.ceil((1 - tail) * (resamples - 1)))]
    return result
