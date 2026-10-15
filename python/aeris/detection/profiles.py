"""Selectable detector profiles: which weights, whether to zoom into tiles, and when.

Speed numbers are wall-clock model time on one development machine over seven unlabeled
smoke clips (2026-10-15); they are not benchmarks. Detection behaviour per profile on those
clips is in the notes below. No profile has been validated against labelled ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROFILE = "fast"            # the AERIS-006 baseline; CLI and benchmarks use this
WEB_DEFAULT_PROFILE = "fast-zoom"   # always zooms: a large aircraft cannot suppress the zoom


@dataclass(frozen=True)
class Profile:
    label: str
    config: str         # file under configs/detector/
    tiles: int
    gate: float | None  # None = always tile (when tiles > 1); else cascade gate score
    note: str


PROFILES = {
    "fast": Profile("Fast", "baseline.json", 1, None,
        "Small model, whole frame only. About 11 ms/frame. Misses small, distant aircraft "
        "(0 of 60 frames on the distant-landing smoke clip)."),
    "fast-zoom": Profile("Fast + zoom", "baseline.json", 2, None,
        "Small model, always zoomed into 2x2 crops. About 50 ms/frame. Finds the distant landing "
        "in every frame of the smoke clip, but flags more birds as aircraft (1 false confirmed "
        "track on the bird flock). Synthetic test with a large aircraft also in frame: the distant "
        "one was found in 19-30 of 30 frames (Accurate: 30 of 30), so the small model is marginal there."),
    "balanced": Profile("Balanced", "yolox-m.json", 1, None,
        "Large model, whole frame only. About 80 ms/frame. Quieter on birds; finds the distant "
        "landing in half its frames."),
    "accurate": Profile("Accurate", "yolox-m.json", 2, None,
        "Large model, always zoomed in. About 300-350 ms/frame (~3 fps). Best result on the smoke clips: "
        "finds the distant landing in every frame, fewest bird boxes among the profiles that find it, "
        "no false confirmed tracks."),
    "accurate-cascade": Profile("Accurate, zoom when needed", "yolox-m.json", 2, 0.5,
        "Large model; zooms only when the whole frame finds nothing confident. About 75 ms/frame on "
        "clear aircraft, 300-350 ms otherwise. Same detections as Accurate on the smoke clips, but a "
        "confident large aircraft suppresses the zoom: in a synthetic test the small aircraft beside "
        "one was found in 9-20 of 30 frames, vs 30 of 30 for Accurate. Do not use when a missed "
        "small aircraft matters."),
}


def config_path(name: str) -> Path:
    return ROOT / "configs" / "detector" / PROFILES[name].config


def weights_path(name: str) -> Path:
    import json
    return ROOT / json.loads(config_path(name).read_text())["model"]["path"]


def build(name: str, config_file: Path | None = None):
    """Verified model + runner + stages for a profile. ``config_file`` overrides the
    profile's detector config (tiling and cascade still come from the profile)."""
    from aeris.detection import baseline
    from aeris.video import pipeline
    profile = PROFILES[name]
    config = baseline.load_config(config_file or config_path(name))
    runner = baseline.OnnxRuntimeRunner(baseline.verified_model_path(config))
    stages = pipeline.BaselineStages(config, runner, tiles=profile.tiles, gate=profile.gate)
    stages.profile = name
    return stages
