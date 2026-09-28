"""Fetch the real video samples for AERIS-008 with verified provenance.

Entries come from `tests/fixtures/video/real_samples.json`. For each Wikimedia Commons
file the Commons API is asked for the licence, author, original URL and SHA-1. The file
is kept only if its licence code is allowed, it fits the size cap, and its bytes match
both the SHA-1 Commons records and the SHA-256 pinned in the manifest. Videos are never
committed; they land in `data/raw/video/` (Git-ignored).

    python -m aeris.video.samples fetch [--dest DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from aeris.detection.samples import SampleError, commons_info, http_get, licence_allowed

ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = ROOT / "tests" / "fixtures" / "video" / "real_samples.json"


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    for entry in manifest["entries"]:
        if not entry.get("title", "").startswith("File:"):
            raise SampleError(f"{entry.get('id')}: Commons entries need a File: title")
        for key, size in (("sha1", 40), ("sha256", 64)):
            value = entry.get(key) or ""
            if len(value) != size or any(c not in "0123456789abcdef" for c in value):
                raise SampleError(f"{entry['id']}: {key} must be a lowercase hex digest")
    return manifest


def sample_path(entry: dict, dest: Path | None = None) -> Path:
    suffix = Path(entry["title"]).suffix.lower() or ".video"
    return (dest or ROOT / "data" / "raw" / "video") / f"{entry['id']}{suffix}"


def verified(path: Path, entry: dict) -> bool:
    """True when the stored file matches the pinned SHA-256."""
    return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]


def fetch_entry(entry: dict, manifest: dict, dest: Path | None = None, get=http_get) -> dict:
    """Download (unless already verified), check licence and hashes, and store one file."""
    path = sample_path(entry, dest)
    if verified(path, entry):
        return {"id": entry["id"], "path": str(path), "sha256": entry["sha256"], "reused": True}
    info = commons_info(entry["title"], get)
    if not info["license_code"] or not licence_allowed(info["license_code"], manifest["allowed_licenses"]):
        raise SampleError(f"{entry['id']}: licence {info['license_code']!r} is not allowed")
    if info["license_code"] != entry["license_code"]:
        raise SampleError(f"{entry['id']}: Commons now reports licence {info['license_code']!r}, "
                          f"not the recorded {entry['license_code']!r}")
    if info["sha1"] != entry["sha1"]:
        raise SampleError(f"{entry['id']}: Commons SHA-1 {info['sha1']} != recorded {entry['sha1']}")
    data = get(info["url"], manifest["max_file_bytes"])
    if hashlib.sha1(data).hexdigest() != entry["sha1"]:
        raise SampleError(f"{entry['id']}: bytes do not match the recorded SHA-1")
    sha256 = hashlib.sha256(data).hexdigest()
    if sha256 != entry["sha256"]:
        raise SampleError(f"{entry['id']}: SHA-256 {sha256} != pinned {entry['sha256']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_bytes(data)
    partial.replace(path)
    return {"id": entry["id"], "path": str(path), "sha256": sha256, "reused": False,
            "license_code": info["license_code"], "author": info["author"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="download and verify the real video samples")
    fetch.add_argument("--dest", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest()
        results = [fetch_entry(entry, manifest, args.dest) for entry in manifest["entries"]]
    except (SampleError, OSError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
