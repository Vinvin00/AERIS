"""Fetch the detector's real sample images with verified provenance (AERIS-006).

Images are never committed. Entries come from `configs/detector/sample_images.json`:

- `wikimedia_commons`: the Commons API is asked for the file's licence, author, size and
  SHA-1 at fetch time, and for a rendition `fetch_width` pixels wide. A file is kept only
  if its licence code is allowed and the downloaded bytes are within the size cap. When
  Commons serves the original (it is no wider than `fetch_width`), the bytes must match
  the SHA-1 it records. Otherwise the file is Commons' own scaled rendition of that
  original: its SHA-256 is recorded with the original's SHA-1 and the rendition URL, and
  later runs pin it by SHA-256.
- `pinned_url`: the bytes must match the recorded SHA-256.

The verified record for every kept file (source, author, licence, hashes, dimensions) is
written to `manifest.json` beside the images. Refusals are recorded with a reason.

    python -m aeris.detection.samples fetch [--only ID ...] [--dest DIR]
"""
from __future__ import annotations

import argparse
import datetime
import fnmatch
import hashlib
import html
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = ROOT / "configs" / "detector" / "sample_images.json"
DEFAULT_DEST = ROOT / "data" / "raw" / "detector_samples"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = os.environ.get(
    "AERIS_HTTP_USER_AGENT", "AERIS-detector-sample-fetch/0.1 (one-off research download)")
MAX_API_BYTES = 1_000_000


class SampleError(ValueError):
    """An entry cannot be fetched or does not meet the recorded requirements."""


def http_get(url: str, limit: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise SampleError(f"{url}: larger than the {limit}-byte cap")
    return data


def load_manifest(path: Path | str = MANIFEST_PATH) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = [entry["id"] for entry in manifest["entries"]]
    if len(ids) != len(set(ids)):
        raise SampleError("manifest: duplicate entry id")
    for entry in manifest["entries"]:
        if entry["source"] == "pinned_url":
            if not re.fullmatch("[0-9a-f]{64}", entry.get("sha256") or ""):
                raise SampleError(f"{entry['id']}: pinned_url entries need a SHA-256")
        elif entry["source"] == "wikimedia_commons":
            if not entry.get("title", "").startswith("File:"):
                raise SampleError(f"{entry['id']}: Commons entries need a File: title")
        else:
            raise SampleError(f"{entry['id']}: unknown source {entry['source']!r}")
    return manifest


def licence_allowed(code: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(code, pattern) for pattern in patterns)


def _plain(value: str | None) -> str | None:
    if value is None:
        return None
    text = html.unescape(re.sub(r"<[^>]+>", " ", value))
    return " ".join(text.split()) or None


def commons_info(title: str, get=http_get, width: int | None = None) -> dict:
    """URL, SHA-1, size, MIME type, licence metadata and (optionally) a scaled rendition."""
    params = {
        "action": "query", "format": "json", "formatversion": "2", "prop": "imageinfo",
        "titles": title, "iiprop": "url|sha1|size|mime|extmetadata",
        "iiextmetadatafilter": "License|LicenseShortName|LicenseUrl|Artist|Credit|"
                               "AttributionRequired|UsageTerms"}
    if width is not None:
        params["iiurlwidth"] = str(width)
    query = urllib.parse.urlencode(params)
    try:
        pages = json.loads(get(f"{COMMONS_API}?{query}", MAX_API_BYTES))["query"]["pages"]
        info = pages[0]["imageinfo"][0]
        meta = {key: value.get("value") for key, value in info.get("extmetadata", {}).items()}
        return {"title": pages[0]["title"], "url": info["url"], "page_url": info["descriptionurl"],
                "sha1": info["sha1"], "size_bytes": info["size"], "mime": info["mime"],
                "original_width": info.get("width"), "original_height": info.get("height"),
                "rendition_url": info.get("thumburl"),
                "license_code": (meta.get("License") or "").strip().lower() or None,
                "license_name": _plain(meta.get("LicenseShortName")),
                "license_url": _plain(meta.get("LicenseUrl")),
                "author": _plain(meta.get("Artist")), "credit": _plain(meta.get("Credit")),
                "attribution_required": _plain(meta.get("AttributionRequired")),
                "usage_terms": _plain(meta.get("UsageTerms"))}
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise SampleError(f"{title}: unexpected Commons API response ({exc})") from exc


def _image_facts(data: bytes) -> dict:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise SampleError("bytes do not decode as an image")
    return {"width": int(image.shape[1]), "height": int(image.shape[0]),
            "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def fetch_entry(entry: dict, manifest: dict, dest: Path, get=http_get) -> dict:
    """Download, verify and store one entry; the verified provenance record."""
    cap = manifest["max_file_bytes"]
    if entry["source"] == "pinned_url":
        data = get(entry["url"], cap)
        facts = _image_facts(data)
        if facts["sha256"] != entry["sha256"]:
            raise SampleError(f"{entry['id']}: SHA-256 {facts['sha256']} != recorded {entry['sha256']}")
        record = {"source": "pinned_url", "url": entry["url"], "license_note": entry["license_note"]}
    else:
        info = commons_info(entry["title"], get, manifest.get("fetch_width"))
        if not info["license_code"] or not licence_allowed(info["license_code"],
                                                           manifest["allowed_licenses"]):
            raise SampleError(f"{entry['id']}: licence {info['license_code']!r} "
                              f"({info['license_name']}) is not allowed")
        rendition = info["rendition_url"] and info["rendition_url"] != info["url"]
        if not rendition and info["size_bytes"] > cap:
            raise SampleError(f"{entry['id']}: {info['size_bytes']} bytes exceeds the cap")
        data = get(info["rendition_url"] if rendition else info["url"], cap)
        if not rendition and hashlib.sha1(data).hexdigest() != info["sha1"]:
            raise SampleError(f"{entry['id']}: bytes do not match the SHA-1 Commons records")
        facts = _image_facts(data)
        if entry.get("sha256") and facts["sha256"] != entry["sha256"]:
            raise SampleError(f"{entry['id']}: SHA-256 {facts['sha256']} != recorded {entry['sha256']}")
        record = {"source": "wikimedia_commons", **info,
                  "fetched": "commons_rendition" if rendition else "original",
                  "rendition_of_sha1": info["sha1"] if rendition else None}
    fetched_url = record.get("rendition_url") if record.get("fetched") == "commons_rendition" else record["url"]
    suffix = Path(urllib.parse.urlparse(fetched_url).path).suffix.lower() or ".img"
    name = f"{entry['id']}{suffix}"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / name).write_bytes(data)
    return {"id": entry["id"], "file": name, "role": entry["role"], **facts, **record}


def _still_valid(dest: Path, record: dict) -> bool:
    path = dest / record["file"]
    return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]


def fetch_all(manifest: dict, dest: Path, get=http_get, only=None, now=None) -> dict:
    """Fetch every (or the selected) entry; write and return the local manifest.

    Records from earlier runs are kept for entries not fetched this time, provided the
    stored file still has the recorded SHA-256.
    """
    stamp = (now or datetime.datetime.now(datetime.timezone.utc)).isoformat(timespec="seconds")
    previous = {}
    if (dest / "manifest.json").is_file():
        previous = {r["id"]: r for r in json.loads((dest / "manifest.json").read_text())["fetched"]}
    known = {entry["id"] for entry in manifest["entries"]}
    fetched, refused = [], []
    for entry in manifest["entries"]:
        if only and entry["id"] not in only:
            continue
        previous.pop(entry["id"], None)
        try:
            fetched.append({**fetch_entry(entry, manifest, dest, get), "retrieved_at": stamp})
        except (SampleError, OSError) as exc:
            refused.append({"id": entry["id"], "reason": str(exc)})
    fetched += [r for key, r in previous.items() if key in known and _still_valid(dest, r)]
    order = [entry["id"] for entry in manifest["entries"]]
    fetched.sort(key=lambda r: order.index(r["id"]))
    local = {"schema_version": 1, "manifest_id": manifest["manifest_id"], "last_run_at": stamp,
             "fetched": fetched, "refused": refused}
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "manifest.json").write_text(json.dumps(local, indent=2, sort_keys=True) + "\n",
                                        encoding="utf-8")
    return local


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="download and verify sample images")
    fetch.add_argument("--only", nargs="+", help="entry ids to fetch")
    fetch.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    fetch.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
    except (OSError, ValueError, KeyError) as exc:
        print(f"manifest error: {exc}", file=sys.stderr)
        return 2
    local = fetch_all(manifest, args.dest, only=set(args.only or ()))
    for item in local["fetched"]:
        print(f"ok      {item['id']}  {item['width']}x{item['height']}  {item['sha256']}")
    for item in local["refused"]:
        print(f"refused {item['id']}: {item['reason']}")
    return 0 if not local["refused"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
