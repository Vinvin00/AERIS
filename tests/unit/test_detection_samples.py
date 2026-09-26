"""Sample-image fetch tests (AERIS-006), offline with a fake HTTP layer.

They check licence gating, hash verification, size caps and provenance records. No
network access and no real images.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import email.message
import unittest
from unittest import mock
import urllib.error
import urllib.parse

import cv2
import numpy as np

from aeris.detection import samples

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = samples.load_manifest()
IMAGE = cv2.imencode(".png", np.full((6, 9, 3), 200, np.uint8))[1].tobytes()


def commons_response(title, data, licence="cc-by-sa-3.0", size=None, sha1=None, rendition=None):
    meta = {"License": {"value": licence}, "LicenseShortName": {"value": "CC BY-SA 3.0"},
            "LicenseUrl": {"value": "https://creativecommons.org/licenses/by-sa/3.0"},
            "Artist": {"value": "<a href=\"//commons.wikimedia.org/wiki/User:J\">Jane &amp; Doe</a>"},
            "AttributionRequired": {"value": "true"}}
    name = title.removeprefix("File:").replace(" ", "_")
    url = f"https://upload.wikimedia.org/a/ab/{name}"
    return {"query": {"pages": [{"title": title, "imageinfo": [{
        "url": url, "thumburl": rendition or url, "width": 4000, "height": 3000,
        "descriptionurl": f"https://commons.wikimedia.org/wiki/File:{name}",
        "sha1": sha1 or hashlib.sha1(data).hexdigest(), "size": size or len(data),
        "mime": "image/png", "extmetadata": meta}]}]}}


class FakeWeb:
    def __init__(self, files=None, api=None):
        self.files, self.api, self.calls = files or {}, api or {}, []

    def __call__(self, url, limit):
        self.calls.append(url)
        if url.startswith(samples.COMMONS_API):
            title = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["titles"][0]
            return json.dumps(self.api[title]).encode()
        data = self.files[url]
        if len(data) > limit:
            raise samples.SampleError("over cap")
        return data


def manifest(*entries, cap=25_000_000):
    return {"manifest_id": "test", "allowed_licenses": MANIFEST["allowed_licenses"],
            "max_file_bytes": cap, "entries": list(entries)}


COMMONS = {"id": "c1", "source": "wikimedia_commons", "title": "File:Plane test.png", "role": "test"}


def web_for(entry, data=IMAGE, **response):
    api = commons_response(entry["title"], data, **response)
    info = api["query"]["pages"][0]["imageinfo"][0]
    return FakeWeb({info["thumburl"]: data, info["url"]: data}, {entry["title"]: api})


class Response:
    def __init__(self, data):
        self.data = data

    def read(self, n):
        return self.data[:n]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def http_error(code, retry_after=None):
    headers = email.message.Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError("https://example.org/x", code, "error", headers, None)


class HttpTests(unittest.TestCase):
    def get(self, outcomes):
        waits = []
        with mock.patch.object(urllib.request, "urlopen", side_effect=outcomes):
            data = samples.http_get("https://example.org/x", 100, sleep=waits.append, clock=lambda: 0.0)
        return data, [w for w in waits if w > 0]

    def test_retries_rate_limits_after_the_requested_delay(self):
        data, waits = self.get([http_error(429, "7"), http_error(503), Response(b"ok")])
        self.assertEqual(data, b"ok")
        self.assertIn(7.0, waits)  # Retry-After honoured
        self.assertIn(5.0, waits)  # default when absent

    def test_other_errors_and_exhausted_retries_raise(self):
        with self.assertRaises(urllib.error.HTTPError):
            self.get([http_error(404)])
        with self.assertRaises(urllib.error.HTTPError):
            self.get([http_error(429, "1")] * (samples.MAX_RETRIES + 1))
        _, waits = self.get([http_error(429, "3600"), Response(b"ok")])
        self.assertIn(samples.MAX_RETRY_WAIT_S, waits)  # waits are capped

    def test_size_cap(self):
        with self.assertRaises(samples.SampleError):
            self.get([Response(b"x" * 101)])


class ManifestTests(unittest.TestCase):
    def test_repository_manifest(self):
        entries = MANIFEST["entries"]
        commons = [e for e in entries if e["source"] == "wikimedia_commons"]
        self.assertLessEqual(len(commons), 12)
        self.assertLessEqual(MANIFEST["max_file_bytes"], 4_000_000)
        self.assertEqual(MANIFEST["fetch_width"], 2048)
        config = json.loads((ROOT / "configs/detector/baseline.json").read_text())
        budget = next(i for i in config["download_budget"]["items"] if "sample images" in i["item"])
        self.assertGreaterEqual(budget["download_bytes"], len(entries) * MANIFEST["max_file_bytes"])
        (dog,) = [e for e in entries if e["id"] == "yolox-dog"]
        self.assertIn(config["model"]["code"]["commit"], dog["url"])

    def test_bad_manifests(self):
        good = json.loads((ROOT / "configs/detector/sample_images.json").read_text())
        for edit in (lambda m: m["entries"].append(dict(m["entries"][0])),
                     lambda m: m["entries"][0].update(sha256=None),
                     lambda m: m["entries"][1].update(title="Plane.jpg"),
                     lambda m: m["entries"][1].update(source="flickr")):
            changed = json.loads(json.dumps(good))
            edit(changed)
            with tempfile.TemporaryDirectory() as tmp, self.assertRaises(samples.SampleError):
                path = Path(tmp) / "m.json"
                path.write_text(json.dumps(changed))
                samples.load_manifest(path)

    def test_licence_patterns(self):
        allowed = MANIFEST["allowed_licenses"]
        for code in ("pd", "cc0", "cc-by-2.0", "cc-by-4.0", "cc-by-sa-3.0", "cc-by-sa-4.0"):
            self.assertTrue(samples.licence_allowed(code, allowed), code)
        for code in ("cc-by-nc-4.0", "cc-by-nd-4.0", "cc-by-nc-sa-2.0", "gfdl", "arr", ""):
            self.assertFalse(samples.licence_allowed(code, allowed), code)


class FetchTests(unittest.TestCase):
    def test_commons_file_is_verified_and_recorded(self):
        web = web_for(COMMONS)
        with tempfile.TemporaryDirectory() as tmp:
            local = samples.fetch_all(manifest(COMMONS), Path(tmp), web)
            (record,) = local["fetched"]
            self.assertEqual((Path(tmp) / record["file"]).read_bytes(), IMAGE)
            written = json.loads((Path(tmp) / "manifest.json").read_text())
        self.assertEqual(written, local)
        self.assertEqual(local["refused"], [])
        self.assertEqual(record["file"], "c1.png")
        self.assertEqual((record["width"], record["height"]), (9, 6))
        self.assertEqual(record["sha256"], hashlib.sha256(IMAGE).hexdigest())
        self.assertEqual(record["author"], "Jane & Doe")
        self.assertEqual((record["license_code"], record["license_name"]), ("cc-by-sa-3.0", "CC BY-SA 3.0"))
        self.assertEqual(record["page_url"], "https://commons.wikimedia.org/wiki/File:Plane_test.png")

    def refused(self, entry, web, cap=25_000_000):
        with tempfile.TemporaryDirectory() as tmp:
            local = samples.fetch_all(manifest(entry, cap=cap), Path(tmp), web)
            files = sorted(p.name for p in Path(tmp).iterdir())
        self.assertEqual(local["fetched"], [])
        self.assertEqual(files, ["manifest.json"])
        return local["refused"][0]["reason"]

    def test_refusals(self):
        self.assertIn("not allowed", self.refused(COMMONS, web_for(COMMONS, licence="cc-by-nc-4.0")))
        self.assertIn("not allowed", self.refused(COMMONS, web_for(COMMONS, licence="")))
        self.assertIn("SHA-1", self.refused(COMMONS, web_for(COMMONS, sha1="0" * 40)))
        self.assertIn("exceeds", self.refused(COMMONS, web_for(COMMONS, size=30_000_000)))
        self.assertIn("over cap", self.refused(COMMONS, web_for(COMMONS, size=1), cap=len(IMAGE) - 1))
        self.assertIn("decode", self.refused(COMMONS, web_for(COMMONS, data=b"not an image")))
        broken = FakeWeb(api={COMMONS["title"]: {"query": {"pages": [{"title": "x", "missing": True}]}}})
        self.assertIn("unexpected Commons API", self.refused(COMMONS, broken))

    def test_pinned_url(self):
        url = "https://raw.githubusercontent.com/o/r/abc/x.png"
        entry = {"id": "p1", "source": "pinned_url", "url": url, "role": "test", "license_note": "n",
                 "sha256": hashlib.sha256(IMAGE).hexdigest()}
        with tempfile.TemporaryDirectory() as tmp:
            (record,) = samples.fetch_all(manifest(entry), Path(tmp), FakeWeb({url: IMAGE}))["fetched"]
        self.assertEqual((record["file"], record["source"]), ("p1.png", "pinned_url"))
        wrong = dict(entry, sha256="0" * 64)
        self.assertIn("SHA-256", self.refused(wrong, FakeWeb({url: IMAGE})))

    def test_selected_runs_keep_earlier_valid_records(self):
        other = dict(COMMONS, id="c2", title="File:Other.png")
        web = web_for(COMMONS)
        web.files.update(web_for(other).files)
        web.api.update(web_for(other).api)
        both = manifest(COMMONS, other)
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            samples.fetch_all(both, dest, web, only={"c1"})
            local = samples.fetch_all(both, dest, web, only={"c2"})
            self.assertEqual([r["id"] for r in local["fetched"]], ["c1", "c2"])
            (dest / "c1.png").write_bytes(b"changed")
            local = samples.fetch_all(both, dest, web, only={"c2"})
            self.assertEqual([r["id"] for r in local["fetched"]], ["c2"])

    def test_rendition_is_recorded_as_derived(self):
        # Commons serves a 2048 px rendition: its bytes differ from the original, so the
        # original's SHA-1 cannot be checked. It is recorded with the rendition's SHA-256.
        rendition = "https://upload.wikimedia.org/a/thumb/ab/Plane_test.png/2048px-Plane_test.png"
        web = web_for(COMMONS, sha1="f" * 40, rendition=rendition, size=5_000_000)
        with tempfile.TemporaryDirectory() as tmp:
            (record,) = samples.fetch_all(dict(manifest(COMMONS), fetch_width=2048), Path(tmp), web)["fetched"]
        self.assertEqual((record["fetched"], record["rendition_of_sha1"]), ("commons_rendition", "f" * 40))
        self.assertEqual(record["sha256"], hashlib.sha256(IMAGE).hexdigest())
        self.assertEqual(record["size_bytes"], len(IMAGE))  # the stored rendition
        self.assertEqual(record["original_size_bytes"], 5_000_000)  # larger than the cap
        self.assertIn("iiurlwidth=2048", web.calls[0])
        self.assertEqual(web.calls[1], rendition)  # the rendition, not the original

    def test_original_served_is_sha1_checked(self):
        # thumburl equal to url means Commons served the original: its SHA-1 must match.
        with tempfile.TemporaryDirectory() as tmp:
            local = samples.fetch_all(manifest(COMMONS), Path(tmp), web_for(COMMONS))
        self.assertEqual(local["fetched"][0]["fetched"], "original")
        self.assertIsNone(local["fetched"][0]["rendition_of_sha1"])
        self.assertIn("SHA-1", self.refused(COMMONS, web_for(COMMONS, sha1="0" * 40)))

    def test_original_with_tracking_query_is_sha1_checked(self):
        # For a file no wider than the requested width, Commons returns the original URL
        # plus utm_* parameters as the "rendition": that is still the original.
        name = COMMONS["title"].removeprefix("File:").replace(" ", "_")
        tracked = f"https://upload.wikimedia.org/a/ab/{name}?utm_source=commons.wikimedia.org"
        with tempfile.TemporaryDirectory() as tmp:
            local = samples.fetch_all(manifest(COMMONS), Path(tmp), web_for(COMMONS, rendition=tracked))
        self.assertEqual(local["fetched"][0]["fetched"], "original")
        self.assertIn("SHA-1", self.refused(COMMONS, web_for(COMMONS, rendition=tracked, sha1="0" * 40)))

    def test_pinned_commons_sha256(self):
        pinned = dict(COMMONS, sha256="0" * 64)
        self.assertIn("SHA-256", self.refused(pinned, web_for(pinned)))

    def test_cli_reports_manifest_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.json"
            self.assertEqual(samples.main(["fetch", "--manifest", str(path), "--dest", tmp]), 2)


if __name__ == "__main__":
    unittest.main()
