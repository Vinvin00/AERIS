"""Rights-aware, on-demand catalog of the real smoke-test videos.

Media stays outside Git. A source is accepted only after its published SHA-1 matches;
the excerpt is then generated locally, without labels or future observations.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil
import subprocess
import urllib.request


ROOT = Path(__file__).resolve().parents[3]
LOCAL_RUNS = ROOT / "benchmarks/runs/real-media-smoke-2026-10-13"
MAX_SOURCE_BYTES = 35_000_000


@dataclass(frozen=True)
class Sample:
    slug: str
    title: str
    description: str
    page: str
    url: str
    author: str
    license: str
    sha1: str
    source_name: str
    start_s: float
    duration_s: float
    local_input: str
    local_overlay: str
    observation: str
    width: int | None = None
    height: int | None = None

    @property
    def source_key(self) -> str:
        return self.sha1

    def existing_input(self) -> Path | None:
        path = LOCAL_RUNS / self.local_input
        return path if path.is_file() else None

    def existing_overlay(self) -> Path | None:
        path = LOCAL_RUNS / self.local_overlay
        return path if path.is_file() else None


SAMPLES = (
    Sample("formation", "Four aircraft · formation", "Multiple nearby aircraft",
           "https://commons.wikimedia.org/wiki/File:USAF_Blue_Angels_in_close_formation.webm",
           "https://upload.wikimedia.org/wikipedia/commons/0/03/USAF_Blue_Angels_in_close_formation.webm",
           "Lucastphotography", "CC BY 4.0", "2e80a41b8e1e468ec2eba3bf9a1c98a1cf2abcb5",
           "blue-angels-formation.webm", 0, 10, "source/blue-angels-10fps.mp4",
           "blue-angels-run/overlay.mp4", "Four tracks in the local smoke test."),
    Sample("c5-low-quality", "C-5 · original 320×240", "Distant, hazy low-resolution approach",
           "https://commons.wikimedia.org/wiki/File:C-5_Galaxy.ogv",
           "https://upload.wikimedia.org/wikipedia/commons/c/cf/C-5_Galaxy.ogv",
           "US Air Force", "Public domain (US federal government work)",
           "f01b93a352ee572503b53650100684db13686161", "c5-galaxy.ogv", 39, 10,
           "source/c5-39to49s-320x240-10fps.mp4", "c5-low-quality-run/overlay.mp4",
           "Detected in all 100 sampled frames; two track IDs."),
    Sample("hkia-late", "Distant landing · later", "Small aircraft against a sunset",
           "https://commons.wikimedia.org/wiki/File:Airplane_Landing_at_South_Runway,_HKIA_20210215.webm",
           "https://upload.wikimedia.org/wikipedia/commons/3/34/Airplane_Landing_at_South_Runway%2C_HKIA_20210215.webm",
           "IamCristYe", "CC BY-SA 4.0", "1179c85d4747cf4085643ee4eeadfb6a6e246ad9",
           "hkia-landing.webm", 10, 6, "source/hkia-10to16s-1920x1080.mp4",
           "hkia-run/overlay.mp4", "54 detections in 60 sampled frames."),
    Sample("a380", "Hazy A380 takeoff", "Single aircraft in haze",
           "https://commons.wikimedia.org/wiki/File:A380_Taking_off_from_YYZ.webm",
           "https://upload.wikimedia.org/wikipedia/commons/e/e2/A380_Taking_off_from_YYZ.webm",
           "LeoTor", "CC BY 4.0", "f5d7ae357b304c821e448745b2642bd465dc3164",
           "a380-yyz.webm", 10, 6, "source/a380-yyz-10to16s-960x540.mp4",
           "a380-run/overlay.mp4", "Detected in all 60 sampled frames.", 960, 540),
    Sample("hkia-early", "Distant landing · earlier", "Small aircraft missed by the default detector",
           "https://commons.wikimedia.org/wiki/File:Airplane_Landing_at_South_Runway,_HKIA_20210215.webm",
           "https://upload.wikimedia.org/wikipedia/commons/3/34/Airplane_Landing_at_South_Runway%2C_HKIA_20210215.webm",
           "IamCristYe", "CC BY-SA 4.0", "1179c85d4747cf4085643ee4eeadfb6a6e246ad9",
           "hkia-landing.webm", 3, 6, "source/hkia-3to9s-10fps.mp4",
           "hkia-early-run/overlay.mp4", "No detections above the default threshold in 60 frames."),
    Sample("gulls-ferry", "Gulls · ferry", "Bird-only scene with one brief false positive",
           "https://commons.wikimedia.org/wiki/File:Birds_flying_behind_a_ferry.webm",
           "https://upload.wikimedia.org/wikipedia/commons/0/00/Birds_flying_behind_a_ferry.webm",
           "Subhashish Panigrahi", "CC BY-SA 3.0", "97a5ba75026e2b8b616490435dcd07ad6021842d",
           "birds-behind-ferry.webm", 5, 10, "source/birds-5to15s-10fps.mp4",
           "birds-ferry-run/overlay.mp4", "One gull briefly boxed as aircraft."),
    Sample("bird-flock", "Bird flock", "Bird-only scene with several brief false positives",
           "https://commons.wikimedia.org/wiki/File:Graceful_Birds_in_the_Sky-Amazing_Bird_Sound-User_FREE_Videos.webm",
           "https://upload.wikimedia.org/wikipedia/commons/2/29/Graceful_Birds_in_the_Sky-Amazing_Bird_Sound-User_FREE_Videos.webm",
           "Earthwild TV", "CC BY 3.0", "3c9c85d2ac26b5c081ebc549232f1671cbd9da8a",
           "graceful-birds.webm", 0, 13, "source/graceful-birds-10fps.mp4",
           "graceful-birds-run/overlay.mp4", "21 airplane detections on birds; no confirmed tracks."),
)

BY_SLUG = {sample.slug: sample for sample in SAMPLES}


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_sample(sample: Sample, cache: Path, *, interrupted=lambda: False) -> Path:
    """Return a local excerpt, fetching the original only when needed.

    Existing workspace excerpts are used for this checkout. On another machine, the
    original is fetched from Commons, checked, then excerpted by ffmpeg in the cache.
    """
    existing = sample.existing_input()
    if existing is not None:
        return existing
    cache.mkdir(parents=True, exist_ok=True)
    excerpt = cache / f"{sample.slug}-10fps.mp4"
    if excerpt.is_file():
        return excerpt
    source = cache / f"{sample.source_key}-{sample.source_name}"
    if source.is_file() and sha1_file(source) != sample.sha1:
        source.unlink()
    if not source.is_file():
        partial = source.with_suffix(source.suffix + ".part")
        try:
            request = urllib.request.Request(sample.url, headers={"User-Agent": "AERIS-demo/1.0"})
            with urllib.request.urlopen(request, timeout=30) as response, partial.open("wb") as out:
                total = 0
                while chunk := response.read(1 << 20):
                    if interrupted():
                        raise RuntimeError("Sample download cancelled")
                    total += len(chunk)
                    if total > MAX_SOURCE_BYTES:
                        raise RuntimeError("Source exceeds the 35 MB demo download limit")
                    out.write(chunk)
            if sha1_file(partial) != sample.sha1:
                raise RuntimeError("Downloaded video checksum does not match Wikimedia Commons")
            partial.replace(source)
        finally:
            partial.unlink(missing_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is needed to prepare this sample")
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", str(sample.start_s),
               "-t", str(sample.duration_s), "-i", str(source), "-vf", "fps=10", "-an"]
    if sample.width and sample.height:
        command[-2] = f"fps=10,scale={sample.width}:{sample.height}"
    temporary = excerpt.with_suffix(".partial.mp4")
    try:
        subprocess.run([*command, "-y", str(temporary)], check=True, timeout=90,
                       capture_output=True, text=True)
        if interrupted():
            raise RuntimeError("Sample preparation cancelled")
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError("ffmpeg produced no video")
        temporary.replace(excerpt)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed: {exc.stderr[-500:]}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return excerpt
