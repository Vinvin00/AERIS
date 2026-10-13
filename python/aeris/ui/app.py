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
    QPlainTextEdit, QPushButton, QScrollArea, QStackedWidget, QVBoxLayout, QWidget,
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
        self.setWindowTitle("AERIS // Recorded Video Analysis")
        self.resize(1180, 760)
        self.setMinimumSize(720, 520)
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
        workspace_scroll = QScrollArea()
        workspace_scroll.setWidgetResizable(True)
        workspace_scroll.setFrameShape(QFrame.Shape.NoFrame)
        workspace_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.pages.addWidget(workspace_scroll)
        central = QWidget()
        central.setObjectName("workspace")
        workspace_scroll.setWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(28, 18, 28, 24)
        layout.setSpacing(10)
        top = QHBoxLayout()
        top.addWidget(QLabel("AERIS  /  ANALYSIS CONSOLE"), alignment=Qt.AlignmentFlag.AlignVCenter)
        top.addStretch()
        back = QPushButton("←  OVERVIEW")
        back.clicked.connect(lambda: self.pages.setCurrentIndex(0))
        top.addWidget(back)
        layout.addLayout(top)
        heading = QLabel("RECORDED VIDEO  /  LOCAL ANALYSIS")
        heading.setObjectName("pageHeading")
        layout.addWidget(heading)
        intro = QLabel(
            "Choose a recorded sample or your own file. The pipeline draws detections, tracks "
            "and image-space forecasts. Without annotations, accuracy remains unavailable. "
            "Bird clips expose current false positives."
        )
        intro.setWordWrap(True)
        intro.setObjectName("subtleText")
        layout.addWidget(intro)

        samples_box = QGroupBox("01  /  SOURCE SELECTION")
        samples_layout = QVBoxLayout(samples_box)
        samples_layout.setSpacing(8)
        row = QHBoxLayout()
        self.sample_combo = QComboBox()
        self.sample_combo.setMinimumWidth(240)
        for sample in SAMPLES:
            self.sample_combo.addItem(sample.title, sample.slug)
        self.sample_combo.currentIndexChanged.connect(self._describe_sample)
        row.addWidget(self.sample_combo, 1)
        self.prepare_button = QPushButton("LOAD SAMPLE")
        self.prepare_button.clicked.connect(self.use_sample)
        row.addWidget(self.prepare_button)
        self.source_button = QPushButton("SOURCE / LICENSE")
        self.source_button.clicked.connect(self.open_source)
        row.addWidget(self.source_button)
        samples_layout.addLayout(row)
        self.sample_info = QLabel()
        self.sample_info.setWordWrap(True)
        self.sample_info.setObjectName("sampleInfo")
        samples_layout.addWidget(self.sample_info)
        layout.addWidget(samples_box)

        working = QHBoxLayout()
        working.setSpacing(12)
        feed = QFrame()
        feed.setObjectName("panel")
        feed_layout = QVBoxLayout(feed)
        feed_layout.setContentsMargins(14, 14, 14, 14)
        feed_layout.setSpacing(10)
        feed_layout.addWidget(QLabel("02  /  VISUAL FEED"))
        self.video_label = QLabel("NO VIDEO LOADED")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumHeight(250)
        self.video_label.setObjectName("videoFeed")
        feed_layout.addWidget(self.video_label, 1)
        working.addWidget(feed, 3)

        control = QFrame()
        control.setObjectName("panel")
        control_layout = QVBoxLayout(control)
        control_layout.setContentsMargins(12, 12, 12, 12)
        control_layout.setSpacing(6)
        control_layout.addWidget(QLabel("03  /  RUN CONTROL"))
        self.status = QLabel("Choose a sample or a video to begin.")
        self.status.setWordWrap(True)
        self.status.setObjectName("statusText")
        control_layout.addWidget(self.status)
        control_layout.addSpacing(4)
        self.choose_button = QPushButton("CHOOSE MY VIDEO")
        self.choose_button.clicked.connect(self.choose_video)
        control_layout.addWidget(self.choose_button)
        self.run_button = QPushButton("RUN ANALYSIS  →")
        self.run_button.setObjectName("primaryAction")
        self.run_button.clicked.connect(self.run_aeris)
        control_layout.addWidget(self.run_button)
        self.cancel_button = QPushButton("CANCEL RUN")
        self.cancel_button.clicked.connect(self.cancel)
        self.cancel_button.setEnabled(False)
        control_layout.addWidget(self.cancel_button)
        self.replay_button = QPushButton("REPLAY OVERLAY")
        self.replay_button.clicked.connect(self.replay)
        control_layout.addWidget(self.replay_button)
        self.folder_button = QPushButton("OPEN OUTPUT FOLDER")
        self.folder_button.clicked.connect(self.open_output)
        control_layout.addWidget(self.folder_button)
        control_layout.addStretch()
        working.addWidget(control, 1)
        layout.addLayout(working)

        self.output_heading = QLabel("04  /  RUN OUTPUT")
        layout.addWidget(self.output_heading)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMinimumHeight(130)
        self.details.setMaximumHeight(180)
        layout.addWidget(self.details)
        self.output_heading.setVisible(False)
        self.details.setVisible(False)
        layout.addWidget(QLabel("UNLABELED DEMO  //  COUNTS ARE OUTPUTS, NOT ACCURACY SCORES"))
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._next_frame)
        self._describe_sample()
        self._set_busy(False)
        self.pages.setCurrentIndex(0)
        self._apply_theme()

    def _landing_page(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        page = QWidget()
        page.setObjectName("landing")
        scroll.setWidget(page)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(30, 22, 30, 28)
        layout.setSpacing(18)
        header = QHBoxLayout()
        header.addWidget(QLabel("AERIS  /  SYSTEM OVERVIEW"))
        header.addStretch()
        header.addWidget(QLabel("RECORDED VIDEO  ·  LOCAL EXECUTION"))
        layout.addLayout(header)

        hero = QFrame()
        hero.setObjectName("heroPanel")
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(28, 26, 28, 26)
        hero_layout.setSpacing(26)
        copy = QVBoxLayout()
        copy.setSpacing(13)
        kicker = QLabel("AIRCRAFT PERCEPTION / RESEARCH DEMO")
        kicker.setObjectName("accentLabel")
        copy.addWidget(kicker)
        title = QLabel("AERIS")
        title.setObjectName("heroTitle")
        copy.addWidget(title)
        subtitle = QLabel("RECORDED-VIDEO ANALYSIS")
        subtitle.setObjectName("heroSubtitle")
        copy.addWidget(subtitle)
        intro = QLabel(
            "AERIS looks for aircraft in recorded footage, follows detections across frames, "
            "and draws short image-space forecasts. Select an example or your own video, "
            "run the pipeline locally, and inspect the result."
        )
        intro.setWordWrap(True)
        intro.setObjectName("heroDescription")
        copy.addWidget(intro)
        actions = QHBoxLayout()
        explore = QPushButton("OPEN SAMPLE LIBRARY  →")
        explore.setObjectName("primaryAction")
        explore.clicked.connect(lambda: self.pages.setCurrentIndex(1))
        actions.addWidget(explore)
        own = QPushButton("USE MY VIDEO")
        own.clicked.connect(self.choose_video)
        actions.addWidget(own)
        actions.addStretch()
        copy.addLayout(actions)
        copy.addStretch()
        artifacts = QLabel("OUTPUT  /  OVERLAY.MP4  ·  TRACKS.CSV  ·  METRICS.JSON  ·  MANIFEST")
        artifacts.setObjectName("artifactLabel")
        copy.addWidget(artifacts)
        hero_layout.addLayout(copy, 3)

        flow = QFrame()
        flow.setObjectName("flowPanel")
        flow.setMinimumWidth(230)
        flow_layout = QVBoxLayout(flow)
        flow_layout.setContentsMargins(16, 16, 16, 16)
        flow_layout.setSpacing(13)
        label = QLabel("PIPELINE / 01—03")
        label.setObjectName("accentLabel")
        flow_layout.addWidget(label)
        for number, heading, detail in (
            ("01", "INPUT", "Recorded video"),
            ("02", "PROCESS", "Detect · track · forecast"),
            ("03", "REVIEW", "Overlay · counts · manifest"),
        ):
            stage = QLabel(f"{number}   {heading}\n{detail}")
            stage.setObjectName("flowStage")
            stage.setWordWrap(True)
            flow_layout.addWidget(stage)
        flow_layout.addStretch()
        mode = QLabel("MODE  /  OFFLINE FILE\nTRUTH /  NOT SUPPLIED")
        mode.setObjectName("flowMode")
        flow_layout.addWidget(mode)
        hero_layout.addWidget(flow, 1)
        layout.addWidget(hero, 1)

        foot = QHBoxLayout()
        foot.addWidget(QLabel("DATA STATUS  /  UNLABELED DEMONSTRATION"))
        foot.addStretch()
        foot.addWidget(QLabel("ACCURACY  /  UNAVAILABLE WITHOUT GROUND TRUTH"))
        layout.addLayout(foot)
        return scroll

    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow, QStackedWidget, QScrollArea, QWidget#landing, QWidget#workspace,
            QScrollArea > QWidget > QWidget { background: #080d11; color: #dce8eb; }
            QLabel { color: #dce8eb; background: transparent; font-family: Menlo, monospace;
                font-size: 11px; }
            QLabel#pageHeading { color: #f1f8f8; font-family: Arial, sans-serif;
                font-size: 23px; font-weight: 700; }
            QLabel#subtleText, QLabel#sampleInfo { color: #94a9ad; font-size: 12px; }
            QLabel#statusText { color: #e0f0d5; font-size: 12px; padding: 3px 0; }
            QLabel#accentLabel { color: #b4dbad; letter-spacing: 2px; font-weight: 700; }
            QLabel#heroTitle { color: #f2f7f4; font-family: Arial, sans-serif;
                font-size: 70px; font-weight: 800; letter-spacing: 6px; }
            QLabel#heroSubtitle { color: #b9d0d0; font-size: 15px; letter-spacing: 3px; }
            QLabel#heroDescription { color: #a6bac0; font-family: Arial, sans-serif;
                font-size: 15px; line-height: 1.35; }
            QLabel#flowStage { color: #dbe8e7; background: #101a1f;
                border-left: 2px solid #8bc89a; padding: 12px; }
            QLabel#flowMode, QLabel#artifactLabel { color: #668188; font-size: 10px; }
            QLabel#videoFeed { background: #030608; color: #638087;
                border: 1px solid #294047; font-size: 13px; }
            QFrame#heroPanel, QFrame#panel, QGroupBox { background: #0e171c;
                border: 1px solid #273a40; border-radius: 2px; }
            QFrame#flowPanel { background: #0a1115; border: 1px solid #294047; }
            QGroupBox { margin-top: 12px; padding: 13px 12px 12px 12px;
                color: #afc5c7; font: 700 11px Menlo; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; }
            QPushButton { background: #142127; color: #dce8eb; border: 1px solid #365059;
                border-radius: 2px; padding: 7px 11px; font: 700 11px Menlo; }
            QPushButton:hover { background: #20353b; border-color: #8fc6a0; }
            QPushButton:disabled { background: #11191d; color: #62777d; border-color: #26363c; }
            QPushButton#primaryAction { background: #a9d39e; color: #081109;
                border-color: #c7e5b9; }
            QPushButton#primaryAction:hover { background: #c7e5b9; }
            QPushButton#primaryAction:disabled { background: #182522; color: #6d897d;
                border-color: #2f463f; }
            QComboBox, QPlainTextEdit { background: #080f13; color: #e0ebec;
                border: 1px solid #365059; padding: 8px; font: 12px Menlo; }
            QComboBox QAbstractItemView { background: #0e171c; color: #e0ebec;
                selection-background-color: #24413b; }
            QScrollBar:vertical { background: #0a1115; width: 9px; }
            QScrollBar::handle:vertical { background: #405b61; min-height: 28px; }
        """)

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
        self.output_heading.setVisible(True)
        self.details.setVisible(True)
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
        self.output_heading.setVisible(True)
        self.details.setVisible(True)
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
