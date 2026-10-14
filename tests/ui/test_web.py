"""Focused local web UI checks without downloading media or loading the model."""
from __future__ import annotations

from http.server import ThreadingHTTPServer
import io
import json
from pathlib import Path
import tempfile
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import unittest
from unittest import mock

from aeris.web.server import DemoState, handler_for


class WebUITests(unittest.TestCase):
    def setUp(self):
        self.state = DemoState()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(self.state))
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)

    def request(self, path, data=None, *, origin=True, headers=None):
        headers = dict(headers or {})
        if data is not None and origin:
            headers["Origin"] = self.base
        request = Request(self.base + path, data=data, headers=headers)
        return urlopen(request, timeout=3)

    def test_page_and_sample_catalog(self):
        with self.request("/") as response:
            page = response.read().decode()
        self.assertIn("Analysis console", page)
        self.assertIn("Accuracy stays blank", page)
        with self.request("/api/samples") as response:
            samples = json.load(response)["samples"]
        self.assertEqual(len(samples), 7)
        self.assertTrue(all(s["page"].startswith("https://commons.wikimedia.org/") for s in samples))

    def test_post_requires_local_origin_and_rejects_unknown_sample(self):
        body = json.dumps({"slug": "unknown"}).encode()
        with self.assertRaises(HTTPError) as context:
            self.request("/api/select", body, origin=False)
        self.assertEqual(context.exception.code, 403)
        with self.assertRaises(HTTPError) as context:
            self.request("/api/select", body)
        self.assertEqual(context.exception.code, 400)

    def test_get_rejects_unexpected_host(self):
        with self.assertRaises(HTTPError) as context:
            self.request("/api/state", headers={"Host": "untrusted.example"})
        self.assertEqual(context.exception.code, 403)

    def test_upload_and_byte_range_preview(self):
        body = b"0123456789"
        with self.request("/api/upload", body, headers={"X-Filename": "own%20video.mp4"}) as response:
            state = json.load(response)
        self.assertEqual(state["selected_name"], "own video.mp4")
        self.assertEqual(state["status"], "ready")
        with self.request("/api/video", headers={"Range": "bytes=2-5"}) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.read(), b"2345")
        self.state.selected_video.unlink()

    def test_failed_upload_clears_previous_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            old = Path(directory) / "old.mp4"
            old.write_bytes(b"old")
            self.state.selected_video = old
            self.state.preview_video = old
            self.state.selected_name = "old.mp4"
            self.state.run_dir = Path(directory)
            self.state.status = "complete"
            with self.assertRaises(ValueError):
                self.state.select_upload("new.mp4", io.BytesIO(b"short"), 10)
            snapshot = self.state.snapshot()
            self.assertEqual(snapshot["status"], "failed")
            self.assertIsNone(snapshot["selected_name"])
            self.assertIsNone(snapshot["run_dir"])
            self.assertIsNone(snapshot["video_url"])
            self.assertIn("Could not prepare video", snapshot["message"])

    def test_run_requires_model_and_has_no_stale_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.mp4"
            path.write_bytes(b"video")
            self.state.selected_video = path
            self.state.selected_name = "clip.mp4"
            self.state.status = "ready"
            with mock.patch("aeris.web.server.shutil.which", return_value=None):
                with self.assertRaises(HTTPError) as context:
                    self.request("/api/run", b"{}")
            self.assertEqual(context.exception.code, 400)
            self.assertIsNone(self.state.snapshot()["metrics"])


if __name__ == "__main__":
    unittest.main()
