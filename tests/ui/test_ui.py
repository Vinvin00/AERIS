"""Focused checks for the local desktop sample and result flow."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from aeris.ui.samples import BY_SLUG, SAMPLES, prepare_sample, sha1_file


class RequiredRunnerUITests(unittest.TestCase):
    """Checks discovered by the repository's required unittest gate."""

    def test_sample_catalog_has_sources_and_terms(self):
        self.assertEqual(len(SAMPLES), 7)
        self.assertEqual(len({sample.source_key for sample in SAMPLES}), 6)
        for sample in SAMPLES:
            self.assertTrue(sample.page.startswith("https://commons.wikimedia.org/wiki/File:"))
            self.assertEqual(len(sample.sha1), 40)
            self.assertTrue(sample.license)

    def test_existing_sample_does_not_fetch_media(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            video = directory / "existing.mp4"
            video.write_bytes(b"sample already present")
            sample = BY_SLUG["formation"]
            with mock.patch.object(type(sample), "existing_input", return_value=video):
                self.assertEqual(prepare_sample(sample, directory / "cache"), video)
            self.assertFalse((directory / "cache").exists())

    def test_checksum(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "input"
            path.write_bytes(b"abc")
            self.assertEqual(sha1_file(path), "a9993e364706816aba3e25717850c26c9cd0d89d")

    def test_unlabeled_result_never_claims_accuracy(self):
        from aeris.ui.app import metric_summary

        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            (directory / "metrics.json").write_text(json.dumps({
                "counts": {"frames_processed": 1, "detections_at_or_above_min_score": 0,
                           "tracks_created": 0},
                "truth_dependent": {"detection": {"available": False,
                                                   "reason": "no ground truth supplied"}},
            }))
            self.assertIn("Detection accuracy: unavailable", metric_summary(directory))

    def test_dark_landing_and_compact_workspace_reach_controls(self):
        script = """
from aeris.ui import ensure_qt_plugins_visible
ensure_qt_plugins_visible()
from aeris.ui.app import QApplication, DemoWindow, QPushButton
app = QApplication([])
window = DemoWindow()
window.show()
app.processEvents()
assert window.pages.currentIndex() == 0
explore = next(button for button in window.pages.widget(0).findChildren(QPushButton)
               if button.text().startswith('OPEN SAMPLE LIBRARY'))
assert '#080d11' in window.styleSheet()
explore.click()
assert window.pages.currentIndex() == 1
assert window.sample_combo.count() == 7
assert any('Bird clips expose current false positives' in label.text()
           for label in window.centralWidget().findChildren(type(window.sample_info)))
assert not window.run_button.isEnabled()
for width, height in ((1080, 850), (900, 600)):
    window.resize(width, height)
    app.processEvents()
    scroll = window.pages.widget(1)
    assert scroll.horizontalScrollBar().maximum() == 0
    for control in (window.prepare_button, window.choose_button, window.run_button,
                    window.cancel_button, window.replay_button, window.folder_button):
        scroll.ensureWidgetVisible(control)
        app.processEvents()
        assert control.isVisibleTo(scroll)
window.close()
"""
        env = os.environ.copy()
        env["QT_QPA_PLATFORM"] = "offscreen"
        import PySide6

        env["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(
            Path(PySide6.__file__).parent / "Qt/plugins/platforms"
        )
        python_root = str(Path(__file__).resolve().parents[2] / "python")
        env["PYTHONPATH"] = python_root + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True,
                                text=True, timeout=20, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
