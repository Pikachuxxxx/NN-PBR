"""Windows Qt workstation for NN-PBR training and Vulkan preview."""

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, QSize, Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QIcon, QImageReader, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .studio_inspector import LossGraph, ModelData

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ("Albedo", "Normal", "AO", "Roughness", "Metallic")
PHASES = ("Warmup", "BC6 training", "Decoder tuning")
# reference size, feature sizes, mip counts, phase lengths, batch, device
PROFILES = {
    "512 · GPU / short": (
        512,
        "128,64,32,16",
        "6,5,4,3",
        (500, 5000, 200),
        16384,
        "cuda:0",
    ),
    "64 · CPU / smoke": (64, "32,16,8,4", "4,3,2,1", (150, 400, 100), 512, "cpu"),
    "2048 · GPU / full": (
        2048,
        "512,256,128,64",
        "8,7,6,5",
        (5000, 200000, 1000),
        262144,
        "cuda:0",
    ),
}
PROGRESS = re.compile(r"\[phase (\d+) (\d+)/(\d+)\] mse=(\S+)")


def heading(text):
    label = QLabel(text)
    label.setObjectName("heading")
    return label


def integer(form, label, minimum, maximum, value=None):
    field = QSpinBox()
    field.setRange(minimum, maximum)
    field.setValue(minimum if value is None else value)
    form.addRow(label, field)
    return field


class Studio(QMainWindow):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.run = None
        self.export = None
        self.stage = ""
        self.pending = ""
        self.worker = QProcess(self)
        self.worker.setProcessChannelMode(QProcess.MergedChannels)
        self.worker.setWorkingDirectory(str(ROOT))
        env = QProcessEnvironment.systemEnvironment()
        env.insert("OMP_NUM_THREADS", "2")
        env.insert("PYTHONIOENCODING", "utf-8")
        self.worker.setProcessEnvironment(env)
        self.worker.readyReadStandardOutput.connect(self.read_log)
        self.worker.finished.connect(self.finished)
        self.worker.errorOccurred.connect(self.worker_error)

        self.inspector = None
        self.files = None
        self.setWindowTitle("NN-PBR Studio")
        self.resize(980, 600)
        self.setStyleSheet("""
            QWidget {background:#111214; color:#dfe2e8; font:10pt 'Segoe UI';}
            QToolBar {background:#0c0d0f; spacing:7px; padding:4px;}
            QLabel#heading {font-weight:600; color:#f2f4f6; padding:5px 0;}
            QLineEdit,QSpinBox,QDoubleSpinBox,QComboBox,QPlainTextEdit,QTableWidget {
                background:#090a0c; color:#e2e5ea; border:1px solid #40454c; border-radius:3px; padding:3px;}
            QPushButton {background:#24272c;color:white;border:1px solid #484e57;border-radius:3px;padding:5px 9px;}
            QPushButton:hover {background:#464d55;} QPushButton:disabled {color:#9298a0;}
            QPushButton#train {background:#346684;border-color:#4f859f;}
            QProgressBar {background:#16191d;color:white;text-align:center;} QProgressBar::chunk {background:#367594;}
            QHeaderView::section {background:#343940;color:#e2e5ea;}
            QScrollArea {border:none;} QSplitter::handle {background:#1a1d21;}
        """)
        bar = QToolBar()
        bar.setMovable(False)
        self.addToolBar(bar)
        bar.addAction("Open model…", self.open_export)
        bar.addAction("Export package…", self.copy_export)
        model_menu = self.menuBar().addMenu("File")
        model_menu.addAction("Open weights and DDS files…", self.open_files)
        model_menu.addAction("Open analysis image", self.show_analysis)
        self.model_label = QLabel()
        bar.addWidget(self.model_label)

        root = QWidget()
        self.setCentralWidget(root)
        columns = QHBoxLayout(root)
        columns.setContentsMargins(8, 4, 8, 6)
        columns.setSpacing(9)
        self.settings_box = QWidget()
        left = QVBoxLayout(self.settings_box)
        left.setContentsMargins(6, 2, 6, 4)
        left.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(heading("Source maps"))
        top.addStretch()
        folder = QPushButton("Import folder…")
        folder.clicked.connect(self.import_folder)
        top.addWidget(folder)
        left.addLayout(top)
        self.paths = []
        self.thumbs = []
        fields = QGridLayout()
        left.addLayout(fields)
        for i, name in enumerate(CHANNELS):
            edit = QLineEdit()
            edit.setPlaceholderText("Choose texture…")
            edit.editingFinished.connect(self.prepare_analysis)
            edit.setClearButtonEnabled(True)
            self.paths.append(edit)
            tile = QPushButton()
            tile.setFixedSize(30, 30)
            tile.setIconSize(QSize(24, 24))
            tile.setToolTip(f"Choose {name} texture")
            self.thumbs.append(tile)
            tile.clicked.connect(lambda _, e=edit: self.browse_map(e))
            edit.textChanged.connect(lambda _, index=i: self.refresh_map(index))
            fields.addWidget(tile, i, 0)
            fields.addWidget(QLabel(name), i, 1)
            fields.addWidget(edit, i, 2)
        left.addSpacing(5)
        left.addWidget(heading("Training"))
        self.profile = QComboBox()
        self.profile.addItems(PROFILES)
        left.addWidget(self.profile)
        self.advanced_toggle = QCheckBox("Advanced settings")
        left.addWidget(self.advanced_toggle)
        advanced = QWidget()
        form = QFormLayout(advanced)
        form.setContentsMargins(0, 4, 0, 4)
        left.addWidget(advanced)
        advanced.hide()
        self.advanced_toggle.toggled.connect(advanced.setVisible)
        self.device = QComboBox()
        self.device.addItems(["cuda:0", "cpu"])
        form.addRow("Device", self.device)
        self.size = QComboBox()
        self.size.addItems([str(2**i) for i in range(5, 13)])
        form.addRow("Reference size", self.size)
        self.res = QLineEdit()
        self.mips = QLineEdit()
        form.addRow("Feature sizes", self.res)
        form.addRow("Mips per feature", self.mips)
        self.steps = [integer(form, s, 0, 1000000) for s in PHASES]
        self.batch = integer(form, "Batch size", 1, 1048576)
        self.micro = integer(form, "Microbatch", 1, 1048576)
        self.seed = integer(form, "Seed", 0, 2147483647, 7)
        self.log_every = integer(form, "Log interval", 1, 100000, 25)
        self.checkpoint_every = integer(form, "Checkpoint interval", 1, 1000000, 1000)
        self.fused = QCheckBox("Fused Adam")
        self.compile_model = QCheckBox("Compile forward with PyTorch")
        form.addRow(self.fused)
        form.addRow(self.compile_model)
        self.rates = {}
        for flag, label, rate in (
            ("lr-feat-phase1", "Warmup · feature LR", 0.05),
            ("lr-mlp-phase1", "Warmup · decoder LR", 0.001),
            ("lr-feat-phase2", "BC6 · feature LR", 0.01),
            ("lr-mlp-phase2", "BC6 · decoder LR", 0.001),
            ("lr-mlp-phase3", "Final · decoder LR", 0.0005),
        ):
            field = QDoubleSpinBox()
            field.setDecimals(7)
            field.setRange(0.0000001, 1)
            field.setSingleStep(0.0001)
            field.setValue(rate)
            self.rates[flag] = field
            form.addRow(label, field)
        self.profile.currentTextChanged.connect(self.apply_profile)
        self.apply_profile(self.profile.currentText())
        self.train_button = QPushButton("Train material")
        self.train_button.setObjectName("train")
        self.train_button.clicked.connect(self.start_training)
        left.addWidget(self.train_button)
        left.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.settings_box)
        scroll.setMinimumWidth(242)
        scroll.setMaximumWidth(285)
        columns.addWidget(scroll, 1)

        center = QVBoxLayout()
        center.setSpacing(4)
        columns.addLayout(center, 3)
        header = QHBoxLayout()
        header.addWidget(heading("Material preview"))
        header.addStretch()
        self.channel = QComboBox()
        self.channel.addItems(["Albedo", "Normal", "AO", "Roughness", "Metallic"])
        header.addWidget(self.channel)
        self.vk = QPushButton("Launch Vulkan")
        self.vk.setEnabled(False)
        header.addWidget(self.vk)
        self.details = QCheckBox("Weights && mips")
        self.details.setEnabled(False)
        header.addWidget(self.details)
        center.addLayout(header)
        self.canvas = QLabel(
            "Open an exported model or finish training to see its analysis."
        )
        self.canvas.setAlignment(Qt.AlignCenter)
        self.canvas.setStyleSheet("background:#090a0c;color:#a7adb4;")
        self.canvas.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        center.addWidget(self.canvas, 3)
        self.comparison = QWidget()
        grid = QGridLayout(self.comparison)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setRowStretch(1, 1)
        self.comparison.setMinimumHeight(152)
        self.comparison.setMaximumHeight(198)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(1)
        self.tiles = []
        self.tile_labels = []
        for i, title in enumerate(("Source", "Decoded", "Absolute error")):
            label = heading(title)
            pix = QLabel()
            pix.setAlignment(Qt.AlignCenter)
            pix.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
            pix.setMinimumSize(72, 112)
            pix.setStyleSheet("background:#08090b")
            self.tile_labels.append(label)
            self.tiles.append(pix)
            grid.addWidget(label, 0, i)
            grid.addWidget(pix, 1, i)
            grid.setColumnStretch(i, 1)
        center.addWidget(self.comparison, 3)
        self.comparison.hide()
        self.latent_row = QWidget()
        latent_layout = QHBoxLayout(self.latent_row)
        latent_layout.setContentsMargins(0, 1, 0, 1)
        latent_layout.setSpacing(5)
        latent_layout.addWidget(QLabel("Latents"))
        self.latent_tiles = []
        for i in range(4):
            latent_layout.addWidget(QLabel(str(i)))
            pix = QLabel()
            pix.setFixedSize(48, 48)
            pix.setToolTip(f"Decoded BC6H feature texture {i}, mip 0")
            self.latent_tiles.append(pix)
            latent_layout.addWidget(pix)
        latent_layout.addStretch()
        center.addWidget(self.latent_row)
        self.latent_row.hide()
        self.caption = QLabel()
        self.caption.setWordWrap(True)
        center.addWidget(self.caption)
        center.addWidget(heading("Training loss"))
        self.graph = LossGraph()
        center.addWidget(self.graph, 1)
        center.addWidget(
            QLabel(
                "<span style='color:#55b8ef'>Warmup</span>　 <span style='color:#e8a352'>BC6 training</span>　 <span style='color:#63c6a5'>Decoder tuning</span>"
            )
        )
        control = QHBoxLayout()
        self.progress = QProgressBar()
        control.addWidget(self.progress)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_training)
        control.addWidget(self.stop_button)
        center.addLayout(control)
        self.status = QLabel("Ready")
        self.status.setWordWrap(True)
        center.addWidget(self.status)
        log_toggle = QCheckBox("Show training log")
        center.addWidget(log_toggle)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setFixedHeight(84)
        center.addWidget(self.log)
        self.log.hide()
        log_toggle.toggled.connect(self.log.setVisible)
        self.data_panel = QVBoxLayout()
        columns.addLayout(self.data_panel)

        self.render_cache = tempfile.TemporaryDirectory()
        self.result_dir = None
        self.images = QProcess(self)
        self.images.setWorkingDirectory(str(ROOT))
        self.images.setProcessEnvironment(env)
        self.images.finished.connect(self.result_ready)
        self.channel.currentIndexChanged.connect(self.show_result)
        self.renderer = QProcess(self)
        self.renderer.setProcessChannelMode(QProcess.MergedChannels)
        self.renderer.readyReadStandardOutput.connect(self.renderer_output)
        self.renderer.errorOccurred.connect(
            lambda _: self.caption.setText(self.renderer.errorString())
        )
        self.vk.clicked.connect(self.launch_vk)
        self.details.toggled.connect(self.toggle_details)

    def apply_profile(self, name):
        size, res, mips, steps, batch, device = PROFILES[name]
        self.size.setCurrentText(str(size))
        self.res.setText(res)
        self.mips.setText(mips)
        for field, value in zip(self.steps, steps):
            field.setValue(value)
        self.batch.setValue(batch)
        self.micro.setValue(batch)
        self.device.setCurrentText(device)
        self.fused.setChecked(size == 512)

    def refresh_map(self, index):
        field = self.paths[index]
        reader = QImageReader(field.text())
        size = reader.size()
        pix = QPixmap()
        if size.isValid():
            reader.setScaledSize(size.scaled(48, 48, Qt.KeepAspectRatio))
            pix = QPixmap.fromImage(reader.read())
        self.thumbs[index].setIcon(QIcon(pix))
        field.setToolTip(field.text())

    def browse_map(self, edit):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select texture",
            "",
            "Images (*.png *.jpg *.jpeg *.tga *.bmp *.tif *.tiff)",
        )
        if path:
            edit.setText(path)
            self.prepare_analysis()

    def import_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select folder containing named PBR maps"
        )
        if folder:
            self.set_folder(Path(folder))
            self.prepare_analysis()

    def set_folder(self, folder):
        for name, field in zip(CHANNELS, self.paths):
            path = folder / (name.lower() + ".png")
            field.setText(str(path) if path.is_file() else "")

    def start_training(self):
        files = [Path(field.text()) for field in self.paths]
        if not all(f.is_file() for f in files):
            QMessageBox.warning(
                self,
                "Material maps",
                "Select an image for each of the five PBR channels.",
            )
            return
        size = int(self.size.currentText())
        steps = tuple(field.value() for field in self.steps)
        res, mips = self.res.text().strip(), self.mips.text().strip()
        try:
            sizes = [int(x) for x in res.split(",")]
            levels = [int(x) for x in mips.split(",")]
            if (
                len(sizes) != 4
                or len(levels) != 4
                or any(r < 4 or r & (r - 1) or r > size for r in sizes)
                or any(n < 1 or r >> (n - 1) < 4 for r, n in zip(sizes, levels))
            ):
                raise ValueError(
                    "Use four power-of-two feature sizes and mip chains ending at 4×4 or larger."
                )
            if min(steps[:2]) < 1 or self.micro.value() > self.batch.value():
                raise ValueError(
                    "Warmup and BC6 training require updates. Microbatch must fit within the batch."
                )
        except ValueError as error:
            QMessageBox.warning(self, "Training properties", str(error))
            return
        self.settings = (
            size,
            res,
            mips,
            steps,
            self.batch.value(),
            self.device.currentText(),
        )
        parent = ROOT / "runs/studio"
        try:
            parent.mkdir(parents=True, exist_ok=True)
            self.run = Path(
                tempfile.mkdtemp(
                    prefix=datetime.now().strftime("%Y%m%d-%H%M%S-"), dir=parent
                )
            )
        except OSError as error:
            QMessageBox.warning(self, "Training output", str(error))
            return
        self.train_button.setEnabled(False)
        self.settings_box.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.graph.points.clear()
        self.graph.update()
        self.progress.setRange(0, sum(self.settings[3]))
        self.progress.setValue(0)
        self.log.clear()
        self.status.setText("Preparing reference textures")
        self.launch_worker(
            "prepare",
            [
                "-m",
                "nnpbr.prepare",
                "--files",
                *map(str, files),
                "--size",
                str(size),
                "--levels",
                str(min(9, size.bit_length())),
                "--output",
                str(self.run / "reference.pt"),
            ],
        )

    def launch_worker(self, stage, args):
        self.stage = stage
        self.pending = ""
        self.worker.start(sys.executable, ["-u", *args])

    def read_log(self):
        self.pending += bytes(self.worker.readAllStandardOutput()).decode(
            "utf-8", "replace"
        )
        lines = self.pending.split("\n")
        self.pending = lines.pop()
        for line in lines:
            self.log.appendPlainText(line.rstrip())
            match = PROGRESS.search(line)
            if match:
                phase, step, total = map(int, match.groups()[:3])
                self.progress.setValue(sum(self.settings[3][: phase - 1]) + step)
                self.graph.points.append(
                    (sum(self.settings[3][: phase - 1]) + step, float(match[4]), phase)
                )
                self.graph.update()
                self.status.setText(
                    f"{PHASES[phase - 1]}   {step:,} / {total:,}   MSE {match[4]}"
                )

    def worker_error(self, error):
        if error == QProcess.FailedToStart:
            self.status.setText(self.worker.errorString())
            self.train_button.setEnabled(True)
            self.settings_box.setEnabled(True)
            self.stop_button.setEnabled(False)

    def stop_training(self):
        if (
            self.worker.state() != QProcess.NotRunning
            and QMessageBox.question(
                self,
                "Stop training",
                "Stop this job? Completed checkpoints stay in the training folder.",
            )
            == QMessageBox.Yes
        ):
            self.stage = "stopped"
            self.stop_process(self.worker)

    def finished(self, code, exit_status):
        self.read_log()
        if self.pending:
            self.log.appendPlainText(self.pending)
        if code == 0 and self.stage == "prepare":
            size, res, mips, steps, batch, device = self.settings
            args = [
                "-m",
                "nnpbr",
                "--mode",
                "full",
                "--device",
                device,
                "--reference-pt",
                str(self.run / "reference.pt"),
                "--output-dir",
                str(self.run),
                "--latent-res",
                res,
                "--latent-mips",
                mips,
                "--ref-mips",
                str(min(9, size.bit_length())),
                "--batch-size",
                str(batch),
                "--microbatch-size",
                str(self.micro.value()),
                "--log-every",
                str(self.log_every.value()),
                "--seed",
                str(self.seed.value()),
                "--checkpoint-every",
                str(self.checkpoint_every.value()),
                "--phase1-iters",
                str(steps[0]),
                "--phase2-iters",
                str(steps[1]),
                "--phase3-iters",
                str(steps[2]),
            ]
            for flag, field in self.rates.items():
                args.extend(["--" + flag, str(field.value())])
            if self.compile_model.isChecked():
                args.append("--compile-model")
            if self.fused.isChecked() and device != "cpu":
                args.append("--fused-adam")
            self.status.setText("Starting trainer")
            self.launch_worker("train", args)
            return
        self.train_button.setEnabled(True)
        self.settings_box.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self.stage == "stopped":
            self.status.setText(f"Training stopped · {self.run}")
            return
        if code:
            self.status.setText(
                f"{self.stage.capitalize()} failed (exit {code}). See log below."
            )
        else:
            self.status.setText(f"Export ready · {self.run.name}")
            self.load_export(self.run / "export")

    def open_export(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Open NN-PBR export folder", str(ROOT / "models")
        )
        if folder:
            self.load_export(Path(folder))

    def open_files(self):
        weight, _ = QFileDialog.getOpenFileName(
            self, "Select FP16 decoder", "", "Decoder weights (*.bin)"
        )
        if not weight:
            return
        textures = []
        for index in range(4):
            path, _ = QFileDialog.getOpenFileName(
                self,
                f"Select feature texture {index}",
                str(Path(weight).parent),
                "BC6H textures (*.dds)",
            )
            if not path:
                return
            textures.append(Path(path))
        self.export = None
        self.set_model(Path(weight), textures, Path(weight).name)

    def load_export(self, folder):
        if (folder / "export").is_dir():
            folder = folder / "export"
        weights = folder / "decoder_fp16.bin"
        textures = [folder / f"latent_{i:02d}.bc6.dds" for i in range(4)]
        if (
            not all(f.is_file() for f in (weights, *textures))
            or not (folder / "metadata.json").is_file()
        ):
            self.status.setText(
                "Select an NN-PBR export with metadata, decoder weights and four latent textures."
            )
            return
        self.export = folder
        self.set_model(weights, textures, folder.parent.name)
        history = folder.parent / "training_history.json"
        if self.worker.state() == QProcess.NotRunning:
            self.graph.points.clear()
            self.graph.update()
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
        if self.worker.state() == QProcess.NotRunning and history.exists():
            try:
                records = json.loads(history.read_text())
                self.graph.points = [
                    (r["global_step"], r["mse"], r["phase"])
                    for r in records
                    if "mse" in r
                ]
                self.graph.update()
                if self.graph.points:
                    updates = self.graph.points[-1][0]
                    self.progress.setRange(0, updates)
                    self.progress.setValue(updates)
            except (OSError, ValueError, KeyError) as error:
                self.log.appendPlainText(f"Training history: {error}")

    def set_model(self, weights, textures, name):
        self.stop_process(self.renderer)
        self.details.setChecked(False)
        self.stop_process(self.images)
        self.files = (weights, textures, name)
        self.vk.setEnabled(True)
        self.details.setEnabled(True)
        self.model_label.setText(f"   {name}")
        self.result_dir = None
        self.summary = None
        self.comparison.hide()
        self.latent_row.hide()
        self.canvas.show()
        self.channel.setEnabled(self.export is not None)
        if self.export is None:
            self.canvas.setText(
                "Explicit runtime files loaded. Launch Vulkan or inspect weights and mips."
            )
            self.caption.setText("")
            return
        self.prepare_analysis()

    def prepare_analysis(self):
        if self.export is None:
            return
        self.stop_process(self.images)
        self.result_dir = Path(tempfile.mkdtemp(dir=self.render_cache.name))
        self.summary = None
        self.comparison.hide()
        self.latent_row.hide()
        self.canvas.show()
        self.canvas.setText("Analyzing exported material…")
        self.caption.setText("")
        args = ["-m", "nnpbr.studio_analysis", str(self.export), str(self.result_dir)]
        paths = [field.text() for field in self.paths]
        if all(Path(path).is_file() for path in paths):
            args += ["--files", *paths]
        self.images.start(sys.executable, args)

    @staticmethod
    def stop_process(process):
        if process.state() != QProcess.NotRunning:
            pid = process.processId()
            if os.name == "nt" and pid:
                # Windows virtual environments may launch the interpreter as a child.
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            process.kill()
            process.waitForFinished(3000)

    def result_ready(self, code, status):
        if code:
            self.canvas.setText("Material analysis failed. See training log.")
            self.log.appendPlainText(
                bytes(self.images.readAllStandardError()).decode("utf8", "replace")[
                    -1200:
                ]
            )
            return
        self.summary = json.loads((self.result_dir / "summary.json").read_text())
        self.canvas.hide()
        self.comparison.show()
        self.latent_row.show()
        for i, label in enumerate(self.latent_tiles):
            label.setPixmap(
                QPixmap(str(self.result_dir / f"latent-{i}.png")).scaled(
                    label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )
        self.show_result()

    def show_result(self, *args):
        if not getattr(self, "summary", None):
            return
        i = self.channel.currentIndex()
        for label, stem in zip(self.tiles, ("source", "decoded", "error")):
            file = self.result_dir / f"{stem}-{i}.png"
            if file.exists():
                pix = QPixmap(str(file))
                label.setPixmap(
                    pix.scaled(
                        label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                )
            else:
                label.setText("Source maps required")
        stats = self.summary["channels"]
        self.tile_labels[0].setText(self.summary["source"] or "Source")
        self.caption.setText(
            f"{CHANNELS[i]}   ·   MSE {stats[i]['mse']:.5g}   ·   Error scale 0 – {stats[i]['peak']:.4g}"
            if stats
            else "Import the matching source maps for reference and error views."
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "result_dir"):
            QTimer.singleShot(0, self.show_result)

    def renderer_output(self):
        text = bytes(self.renderer.readAllStandardOutput()).decode("utf8", "replace")
        if "Error:" in text:
            self.status.setText(text[text.index("Error:") :].strip())

    def launch_vk(self):
        if self.files:
            self.stop_process(self.renderer)
            weights, textures, _ = self.files
            self.renderer.start(
                str(self.viewer),
                [
                    "--final-view",
                    "--weights",
                    str(weights),
                    "--latents",
                    *map(str, textures),
                ],
            )

    def toggle_details(self, enabled):
        if self.inspector:
            self.inspector.dispose()
            self.inspector.hide()
            self.inspector.deleteLater()
            self.inspector = None
        if enabled and self.files:
            try:
                self.inspector = ModelData(self.files[0], self.files[1], self)
                self.inspector.setMaximumWidth(270)
                self.data_panel.addWidget(self.inspector)
                self.inspector.show()
            except (OSError, ValueError, struct.error) as error:
                self.caption.setText(str(error))
                self.details.setChecked(False)

    def copy_export(self):
        if not self.export:
            self.status.setText(
                "Open an export folder or finish training to export a material package."
            )
            return
        target = QFileDialog.getExistingDirectory(
            self, "Choose destination for the material", str(Path.home())
        )
        if target:
            dest = Path(target) / self.export.parent.name
            try:
                with tempfile.TemporaryDirectory(dir=target) as tmp:
                    staged = Path(tmp) / "material"
                    staged.mkdir()
                    for source in self.export.iterdir():
                        if source.is_file():
                            shutil.copy2(source, staged)
                    if dest.exists():
                        raise FileExistsError(f"Destination already exists: {dest}")
                    staged.rename(dest)
                self.status.setText(f"Exported to {dest}")
            except OSError as error:
                QMessageBox.warning(self, "Export failed", str(error))

    def show_analysis(self):
        path = self.export.parent / "analysis/all_analysis.png" if self.export else None
        if path and path.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        else:
            self.status.setText(
                "Analysis is available after training in this workspace."
            )

    def closeEvent(self, event):
        if self.worker.state() != QProcess.NotRunning:
            if (
                QMessageBox.question(
                    self, "Training active", "Stop training and close?"
                )
                != QMessageBox.Yes
            ):
                event.ignore()
                return
            self.stage = "stopped"
            self.stop_process(self.worker)
        self.details.setChecked(False)
        self.stop_process(self.renderer)
        self.stop_process(self.images)
        self.render_cache.cleanup()
        event.accept()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--viewer", type=Path, default=ROOT / "build/Release/vk_neural_demo.exe"
    )
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument(
        "--maps",
        type=Path,
        help="Preload five source thumbnails from a named map folder",
    )
    parser.add_argument("--weights", type=Path)
    parser.add_argument(
        "--latents", type=Path, nargs=4, metavar=("DDS0", "DDS1", "DDS2", "DDS3")
    )
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("The Qt studio currently targets Windows")
    if bool(args.weights) != bool(args.latents) or (args.weights and args.export_dir):
        parser.error(
            "Use either --export-dir or --weights FILE --latents DDS0 DDS1 DDS2 DDS3"
        )
    app = QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    window = Studio(args.viewer.resolve())
    if args.maps:
        window.set_folder(args.maps.resolve())
    window.show()
    if args.weights:
        QTimer.singleShot(
            0,
            lambda: window.set_model(
                args.weights.resolve(),
                [p.resolve() for p in args.latents],
                args.weights.name,
            ),
        )
    else:
        if args.export_dir:
            QTimer.singleShot(0, lambda: window.load_export(args.export_dir.resolve()))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
