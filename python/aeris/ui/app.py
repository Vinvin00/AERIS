"""Native Qt front end for the existing recorded-video CLI.

All model work runs in a child process. The UI never supplies annotations or replay
detections, and its sample labels describe observed smoke tests rather than accuracy.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
from uuid import uuid4

import cv2
from PySide6.QtCore import QProcess, QProcessEnvironment, QStandardPaths, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QDesktopServices, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFrame, QGroupBox, QHBoxLayout, QLabel, QMainWindow,
    QPlainTextEdit, QPushButton, QStackedWidget, QVBoxLayout, QWidget,
)
from PySide6.QtCore import QUrl

from aeris.ui import ensure_qt_plugins_visible
from aeris.ui.samples import ROOT, SAMPLES, Sample, prepare_sample


class SampleWorker(QThread):
    prepared = Signal(str)
    failed = Signal(str)

    def __init__(self, sample: Sample, cache: Path, parent=None):
        super().__init__(parent)
        self.sample = sample
        self.cache = cache

    def run(self):
        try:
            path = prepare_sample(self.sample, self.cache,
                                  interrupted=self.isInterruptionRequested)
            if not self.isInterruptionRequested():
                self.prepared.emit(str(path))
        except Exception as exc:  # surface download, checksum, and ffmpeg failures
            if not self.isInterruptionRequested():
                self.failed.emit(str(exc))


def metric_summary(directory: Path) -> str:
    """Render counts and null metric availability without inventing accuracy."""
    metrics = json.loads((directory / "metrics.json").read_text())
    counts = metrics["counts"]
    lines = [
        f"Frames processed: {counts['frames_processed']}",
        f"Aircraft-class detections at the run threshold: "
        f"{counts['detections_at_or_above_min_score']}",
        f"Track IDs created: {counts['tracks_created']}",
    ]
    track_file = directory / "tracks.csv"
    if track_file.is_file():
        with track_file.open(newline="") as stream:
            confirmed = {row["track_id"] for row in csv.DictReader(stream)
                         if row["lifecycle"] == "confirmed"}
        lines.append(f"Confirmed track IDs: {len(confirmed)}")
    for name, value in metrics["truth_dependent"].items():
        if not value["available"]:
            lines.append(f"{name.capitalize()} accuracy: unavailable — {value['reason']}")
    return "\n".join(lines)


class DemoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AERIS · recorded video demo")
        self.resize(1080, 850)
        self.selected_video: Path | None = None
        self.output_dir: Path | None = None
        self.worker: SampleWorker | None = None
        self.process: QProcess | None = None
        self.capture: cv2.VideoCapture | None = None
        self._frame_delay_ms = 100
        self._cancelled = False
        self._cache = Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.CacheLocation)) / "samples"

        self.pages = QStackedWidget()
        self.setCentralWidget(self.pages)
        self.pages.addWidget(self._landing_page())
        central = QWidget()
        self.pages.addWidget(central)
        layout = QVBoxLayout(central)
        back = QPushButton("← About AERIS")
        back.clicked.connect(lambda: self.pages.setCurrentIndex(0))
        layout.addWidget(back, alignment=Qt.AlignmentFlag.AlignLeft)
        heading = QLabel("AERIS · See what the current aircraft pipeline does")
        heading.setStyleSheet("font-size: 23px; font-weight: 700; padding: 8px 0")
        layout.addWidget(heading)
        intro = QLabel(
            "Choose a recorded sample or your own video. Run the current detector, tracker, "
            "and image-space forecasts, then inspect the overlay and counts. A box or track "
            "is not a measured accuracy result; without annotations, AP, HOTA, IDF1 and "
            "forecast accuracy remain unavailable. Bird clips expose current false positives."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        samples_box = QGroupBox("Recorded sample collection · downloaded only when needed")
        samples_layout = QVBoxLayout(samples_box)
        row = QHBoxLayout()
        self.sample_combo = QComboBox()
        for sample in SAMPLES:
            self.sample_combo.addItem(sample.title, sample.slug)
        self.sample_combo.currentIndexChanged.connect(self._describe_sample)
        row.addWidget(self.sample_combo, 1)
        self.prepare_button = QPushButton("Use sample")
        self.prepare_button.clicked.connect(self.use_sample)
        row.addWidget(self.prepare_button)
        self.source_button = QPushButton("Source + license")
        self.source_button.clicked.connect(self.open_source)
        row.addWidget(self.source_button)
        samples_layout.addLayout(row)
        self.sample_info = QLabel()
        self.sample_info.setWordWrap(True)
        samples_layout.addWidget(self.sample_info)
        layout.addWidget(samples_box)

        actions = QHBoxLayout()
        self.choose_button = QPushButton("Choose my video…")
        self.choose_button.clicked.connect(self.choose_video)
        actions.addWidget(self.choose_button)
        self.run_button = QPushButton("Run AERIS")
        self.run_button.clicked.connect(self.run_aeris)
        actions.addWidget(self.run_button)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel)
        self.cancel_button.setEnabled(False)
        actions.addWidget(self.cancel_button)
        self.replay_button = QPushButton("Replay overlay")
        self.replay_button.clicked.connect(self.replay)
        actions.addWidget(self.replay_button)
        self.folder_button = QPushButton("Open output folder")
        self.folder_button.clicked.connect(self.open_output)
        actions.addWidget(self.folder_button)
        layout.addLayout(actions)

        self.status = QLabel("Choose a sample or a video to begin.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.video_label = QLabel("Video preview")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumHeight(360)
        self.video_label.setStyleSheet("background: #101820; color: #dce5ea; border-radius: 5px")
        layout.addWidget(self.video_label, 1)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(180)
        layout.addWidget(self.details)
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._next_frame)
        self._describe_sample()
        self._set_busy(False)
        self.pages.setCurrentIndex(0)

    def _landing_page(self) -> QWidget:
        page = QWidget()
        page.setStyleSheet("background: #f5f3ed; color: #1c2935;")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(64, 50, 64, 45)
        layout.setSpacing(20)
        eyebrow = QLabel("AERIS  /  RECORDED VIDEO RESEARCH")
        eyebrow.setStyleSheet("color: #176e72; font-weight: 700; letter-spacing: 2px;")
        layout.addWidget(eyebrow)
        title = QLabel("See what AERIS sees\nin a recorded video.")
        title.setStyleSheet("font-size: 42px; font-weight: 700;")
        layout.addWidget(title)
        intro = QLabel(
            "AERIS looks for aircraft, follows them across frames, and draws short path "
            "forecasts. Pick one of the recorded examples or use your own video, run the "
            "pipeline locally, then inspect its overlay and data."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("font-size: 18px; color: #40515b;")
        layout.addWidget(intro)
        buttons = QHBoxLayout()
        explore = QPushButton("Explore sample videos  →")
        explore.setStyleSheet("background: #176e72; color: white; padding: 13px 18px;"
                              "border-radius: 8px; font-weight: 700;")
        explore.clicked.connect(lambda: self.pages.setCurrentIndex(1))
        buttons.addWidget(explore)
        own = QPushButton("Choose my video")
        own.setStyleSheet("background: white; padding: 13px 18px; border-radius: 8px;")
        own.clicked.connect(self.choose_video)
        buttons.addWidget(own)
        buttons.addStretch()
        layout.addLayout(buttons)
        cards = QHBoxLayout()
        for number, heading, detail in (
            ("01", "Choose", "Seven recorded examples are available. You can also open a local file."),
            ("02", "Run", "The existing detector, tracker and forecaster process the video."),
            ("03", "Inspect", "See the overlay and counts, with accuracy marked unavailable without labels."),
        ):
            card = QFrame()
            card.setObjectName("stepCard")
            card.setStyleSheet("QFrame#stepCard { background: white; border: 1px solid #d9e0de;"
                               "border-radius: 10px; padding: 12px; }")
            content = QVBoxLayout(card)
            content.setAlignment(Qt.AlignmentFlag.AlignTop)
            content.setSpacing(8)
            content.addWidget(QLabel(number))
            heading_label = QLabel(heading)
            heading_label.setStyleSheet("font-size: 19px; font-weight: 700; border: none;")
            content.addWidget(heading_label)
            detail_label = QLabel(detail)
            detail_label.setWordWrap(True)
            detail_label.setStyleSheet("border: none;")
            content.addWidget(detail_label)
            cards.addWidget(card)
        layout.addLayout(cards)
        layout.addStretch()
        note = QLabel("Research demo · unlabeled footage · no validated accuracy claim")
        note.setStyleSheet("color: #65747b;")
        layout.addWidget(note)
        return page

    def _sample(self) -> Sample:
        return next(item for item in SAMPLES if item.slug == self.sample_combo.currentData())

    def _describe_sample(self, *_):
        sample = self._sample()
        self.sample_info.setText(
            f"{sample.description}. Local observation: {sample.observation} "
            f"Source: {sample.author} · {sample.license}. Recorded footage, unlabeled smoke test."
        )

    def _set_busy(self, busy: bool):
        self.sample_combo.setEnabled(not busy)
        self.prepare_button.setEnabled(not busy)
        self.choose_button.setEnabled(not busy)
        self.run_button.setEnabled(not busy and self.selected_video is not None)
        self.cancel_button.setEnabled(busy)

    def use_sample(self):
        if self.worker is not None or (self.process and self.process.state() != QProcess.ProcessState.NotRunning):
            return
        sample = self._sample()
        overlay = sample.existing_overlay()
        self.selected_video = None
        self.output_dir = overlay.parent if overlay else None
        if overlay:
            self._show_output(overlay.parent)
        existing = sample.existing_input()
        if existing:
            self.selected_video = existing
            self.status.setText(f"Ready: {sample.title}. The saved overlay is available; run again to reproduce it.")
            if overlay is None:
                self._play(existing)
            self._set_busy(False)
            return
        self.status.setText(f"Preparing {sample.title} from Wikimedia Commons; verifying checksum…")
        self._set_busy(True)
        self.worker = SampleWorker(sample, self._cache, self)
        self.worker.prepared.connect(self._sample_ready)
        self.worker.failed.connect(self._sample_failed)
        self.worker.finished.connect(self._worker_finished)
        self.worker.start()

    def _sample_ready(self, path: str):
        self.selected_video = Path(path)
        self.status.setText(f"Ready: {self._sample().title}. Press Run AERIS to create a new overlay.")
        if self.output_dir is None:
            self._play(self.selected_video)

    def _sample_failed(self, message: str):
        self.status.setText(f"Sample unavailable: {message}")

    def _worker_finished(self):
        self.worker = None
        self._set_busy(False)

    def choose_video(self):
        name, _ = QFileDialog.getOpenFileName(self, "Choose a recorded video", str(Path.home()),
                                               "Videos (*.mp4 *.mov *.webm *.avi *.ogv);;All files (*)")
        if name:
            self.pages.setCurrentIndex(1)
            self.selected_video = Path(name)
            self.output_dir = None
            self.details.setPlainText("Own video · no annotations supplied. Accuracy metrics will be unavailable.")
            self.status.setText(f"Ready: {self.selected_video.name}")
            self._play(self.selected_video)
            self._set_busy(False)

    def run_aeris(self):
        if self.selected_video is None or not self.selected_video.is_file():
            self.status.setText("Choose a video first.")
            return
        model = ROOT / "models/onnx/yolox_tiny.onnx"
        if not model.is_file():
            self.status.setText("Model weights missing. Follow models/README.md to obtain the pinned local model.")
            return
        if shutil.which("ffprobe") is None:
            self.status.setText("ffprobe is missing. Install ffmpeg before running a video.")
            return
        self.output_dir = ROOT / "benchmarks/runs" / (
            "ui-demo-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
        )
        self._cancelled = False
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(ROOT))
        environment = QProcessEnvironment.systemEnvironment()
        python_path = str(ROOT / "python")
        old = environment.value("PYTHONPATH")
        environment.insert("PYTHONPATH", python_path + (os.pathsep + old if old else ""))
        environment.insert("ORT_DISABLE_TELEMETRY", "1")
        self.process.setProcessEnvironment(environment)
        self.process.readyReadStandardOutput.connect(self._read_process)
        self.process.readyReadStandardError.connect(self._read_process)
        self.process.finished.connect(self._process_finished)
        self.process.errorOccurred.connect(self._process_error)
        args = ["-m", "aeris", str(self.selected_video), "--output", str(self.output_dir),
                "--input-note", "Unlabeled local UI demonstration; no accuracy claim"]
        self.process.start(sys.executable, args)
        self.status.setText("Running AERIS… The interface remains usable; processing may take a moment.")
        self.details.setPlainText("Starting recorded-video pipeline…")
        self._set_busy(True)

    def _process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.status.setText("Could not start AERIS. Check the local Python installation.")
            self.details.appendPlainText(self.process.errorString() if self.process else "Process failed to start.")
            self.process = None
            self._set_busy(False)

    def _read_process(self):
        if self.process is None:
            return
        data = bytes(self.process.readAllStandardOutput()) + bytes(self.process.readAllStandardError())
        if data:
            self.details.appendPlainText(data.decode("utf-8", errors="replace").strip())

    def _process_finished(self, exit_code: int, _status):
        self._read_process()
        if self._cancelled:
            self.status.setText("Run cancelled. No completed result was published.")
        elif exit_code == 0 and self.output_dir and (self.output_dir / "metrics.json").is_file():
            self._show_output(self.output_dir)
            self.status.setText(f"Run complete: {self.output_dir.name}")
        else:
            self.status.setText(f"Run failed (exit {exit_code}). See process output below.")
        self.process = None
        self._set_busy(False)

    def _show_output(self, directory: Path):
        try:
            self.details.setPlainText(metric_summary(directory))
        except (OSError, ValueError, KeyError) as exc:
            self.details.setPlainText(f"Result summary unavailable: {exc}")
        overlay = directory / "overlay.mp4"
        if overlay.is_file():
            self._play(overlay)

    def cancel(self):
        if self.worker is not None:
            self.worker.requestInterruption()
            self.status.setText("Cancelling sample preparation…")
        if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
            self._cancelled = True
            process = self.process
            process.terminate()
            QTimer.singleShot(2000, lambda: process.kill() if process.state()
                              != QProcess.ProcessState.NotRunning else None)

    def _play(self, path: Path):
        self.play_timer.stop()
        if self.capture is not None:
            self.capture.release()
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            self.video_label.setText(f"Cannot preview {path.name}")
            return
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        self._frame_delay_ms = round(1000 / fps) if 1 <= fps <= 120 else 100
        self._next_frame()
        self.play_timer.start(self._frame_delay_ms)

    def _next_frame(self):
        if self.capture is None:
            return
        ok, frame = self.capture.read()
        if not ok:
            self.play_timer.stop()
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(rgb.data, width, height, width * channels, QImage.Format.Format_RGB888).copy()
        self.video_label.setPixmap(QPixmap.fromImage(image).scaled(
            self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def replay(self):
        if self.output_dir and (self.output_dir / "overlay.mp4").is_file():
            self._play(self.output_dir / "overlay.mp4")
        elif self.selected_video:
            self._play(self.selected_video)

    def open_output(self):
        if self.output_dir and self.output_dir.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.output_dir)))
        else:
            self.status.setText("No completed output folder yet.")

    def open_source(self):
        QDesktopServices.openUrl(QUrl(self._sample().page))

    def closeEvent(self, event):
        self.cancel()
        if self.process is not None:
            self.process.kill()
            self.process.waitForFinished(2000)
        self.play_timer.stop()
        if self.capture is not None:
            self.capture.release()
        if self.worker is not None and not self.worker.wait(3000):
            event.ignore()
            self.status.setText("Waiting for sample preparation to stop…")
            return
        super().closeEvent(event)


def main() -> int:
    ensure_qt_plugins_visible()
    app = QApplication.instance() or QApplication(sys.argv)
    window = DemoWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
