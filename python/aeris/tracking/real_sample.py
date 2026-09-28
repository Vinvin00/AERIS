"""Fetch AERIS-009's pinned real AOT evidence frames (AC3), reusing `aeris.data.aot.fetch`
for its S3 ETag and SHA-256 checks. Frames are Git-ignored; only their pinned hashes in
`tests/fixtures/tracking/real_samples.json` are committed (AGENTS.md: no dataset
contents in Git). An already-present file whose SHA-256 matches the pin is reused
without a request.

Usage: python -m aeris.tracking.real_sample fetch
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aeris.data import aot

ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = ROOT / "tests" / "fixtures" / "tracking" / "real_samples.json"
FRAME_LIMIT = 10 * 1024 * 1024  # observed frames are ~2.3 MB; refuse anything wildly larger


def load_manifest(path: Path | str = MANIFEST_PATH) -> dict:
    return json.loads(Path(path).read_text())


def image_root(manifest: dict, root: Path = ROOT) -> Path:
    return root / "data" / "raw" / "aot" / "images" / manifest["part"] / "Images" / manifest["flight_id"]


def fetch_all(manifest: dict, root: Path = ROOT) -> list[dict]:
    """Fetch (or reuse, by pinned SHA-256) every frame the manifest names."""
    dest_dir = image_root(manifest, root)
    results = []
    for entry in manifest["frames"]:
        key = f"{manifest['part']}/Images/{manifest['flight_id']}/{entry['img_name']}"
        result = aot.fetch(aot.image_url(key), dest_dir / entry["img_name"],
                           expected_sha256=entry["sha256"], limit=FRAME_LIMIT)
        results.append({"frame": entry["frame"], **result})
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="fetch (or verify) the pinned real evidence frames")
    parser.parse_args(argv)
    manifest = load_manifest()
    results = fetch_all(manifest)
    print(json.dumps({"flight_id": manifest["flight_id"], "frames": len(results),
                      "fetched": sum(not r["reused"] for r in results),
                      "reused": sum(r["reused"] for r in results)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
