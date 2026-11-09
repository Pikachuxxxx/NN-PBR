"""Qt workstation for NN-PBR training and Vulkan preview."""

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
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
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

from .studio_inspector import LossGraph, ModelData, panel_splitter

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
            QScrollArea {border:none;}
            QSplitter::handle {background:#23272c;}
            QSplitter::handle:hover, QSplitter::handle:pressed {background:#4a7188;}
            QScrollBar {background:#111418; border:0; margin:0;}
            QScrollBar:vertical {width:11px;}
            QScrollBar:horizontal {height:11px;}
            QScrollBar::handle {background:#49515a; border:2px solid #111418; border-radius:5px;}
            QScrollBar::handle:vertical {min-height:26px;}
            QScrollBar::handle:horizontal {min-width:26px;}
            QScrollBar::handle:hover {background:#6c7884;}
            QScrollBar::handle:pressed {background:#648da5;}
            QScrollBar::sub-line, QScrollBar::add-line {height:0; width:0;}
            QScrollBar::add-page, QScrollBar::sub-page {background:transparent;}
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
        layout = QHBoxLayout(root)
        layout.setContentsMargins(8, 4, 8, 6)
        self.columns = panel_splitter(Qt.Horizontal)
        layout.addWidget(self.columns)
        self.settings_box = panel_splitter(Qt.Vertical)
        self.columns.addWidget(self.settings_box)

        def sidebar_panel():
            content = QWidget()
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(content)
            self.settings_box.addWidget(scroll)
            return QVBoxLayout(content)

        left = sidebar_panel()
        left.setContentsMargins(6, 2, 6, 4)
        left.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(heading("Sources"))
        top.addStretch()
        add = QPushButton("Add material…")
        add.clicked.connect(self.add_material)
        top.addWidget(add)
        left.addLayout(top)
        remove = QPushButton("−")
        remove.setToolTip("Remove the highlighted source from this list")
        remove.setFixedWidth(26)
        remove.clicked.connect(self.remove_source)
        top.addWidget(remove)
        self.sources = QListWidget()
        self.sources.setIconSize(QSize(26, 26))
        self.sources.setMinimumHeight(44)
        self.sources.setToolTip(
            "Check materials to train. Highlight a row to inspect or edit its maps."
        )
        self.sources.hide()
        left.addWidget(self.sources)
        self.sources.currentItemChanged.connect(self.display_source)
        self.sources.itemChanged.connect(self.update_train_label)
        self.sources.itemDoubleClicked.connect(self.open_source_result)
        self.updating_source = False
        self.remaining_sources = []
        self.active_source = None
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
            choose = QPushButton("…")
            choose.setFixedWidth(24)
            choose.setToolTip(f"Choose {name} PNG")
            choose.clicked.connect(lambda _, e=edit: self.browse_map(e))
            fields.addWidget(choose, i, 3)
        left = sidebar_panel()
        left.setContentsMargins(6, 2, 6, 4)
        left.setSpacing(4)
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
        self.compile_model.setToolTip(
            "Uses Inductor when available. GPUs without Triton use AOT Autograd with eager kernels."
        )
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
        self.train_button.clicked.connect(self.train_selection)
        left.addWidget(self.train_button)
        left.addStretch()
        self.settings_box.setMinimumWidth(200)
        self.settings_box.setSizes([260, 250])
        self.workspace = panel_splitter(Qt.Vertical)
        self.columns.addWidget(self.workspace)
        self.columns.setStretchFactor(1, 1)
        self.columns.setSizes([244, 730])
        preview_panel = QWidget()
        self.workspace.addWidget(preview_panel)
        center = QVBoxLayout(preview_panel)
        center.setContentsMargins(2, 0, 0, 1)
        center.setSpacing(4)
        header = QHBoxLayout()
        header.addWidget(heading("Material preview"))
        header.addStretch()
        self.channel = QComboBox()
        self.channel.addItems(["Overview", *CHANNELS])
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
        self.overview = QWidget()
        matrix = QGridLayout(self.overview)
        matrix.setContentsMargins(0, 0, 0, 0)
        matrix.setSpacing(2)
        self.overview_source = heading("Source")
        matrix.addWidget(self.overview_source, 0, 1)
        for column, title in enumerate(("Decoded", "Absolute error", "Latents"), 2):
            matrix.addWidget(heading(title), 0, column)
        self.overview_tiles = []
        self.overview_names = []
        for row, name in enumerate(("Albedo", "Normal", "ORM"), 1):
            row_label = QLabel(name)
            self.overview_names.append(row_label)
            matrix.addWidget(row_label, row, 0)
            cells = []
            for column in (1, 2, 3):
                cell = QLabel()
                cell.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
                cell.setMinimumSize(48, 48)
                cell.setAlignment(Qt.AlignCenter)
                cell.setStyleSheet("background:#08090b")
                matrix.addWidget(cell, row, column)
                cells.append(cell)
            self.overview_tiles.append(cells)
            matrix.setRowStretch(row, 1)
        features = QGridLayout()
        features.setContentsMargins(0, 0, 0, 0)
        features.setSpacing(2)
        self.overview_latents = []
        for i in range(4):
            tile = QLabel()
            tile.setAlignment(Qt.AlignCenter)
            tile.setMinimumSize(32, 32)
            tile.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
            tile.setToolTip(f"Feature {i}, decoded mip 0")
            features.addWidget(tile, i // 2, i % 2)
            self.overview_latents.append(tile)
        matrix.addLayout(features, 1, 4, 3, 1)
        for column in range(1, 5):
            matrix.setColumnStretch(column, 1)
        center.addWidget(self.overview, 4)
        self.overview.hide()
        self.comparison = QWidget()
        grid = QGridLayout(self.comparison)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setRowStretch(1, 1)
        self.comparison.setMinimumHeight(152)
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
        loss_panel = QWidget()
        self.workspace.addWidget(loss_panel)
        center = QVBoxLayout(loss_panel)
        center.setContentsMargins(2, 0, 0, 1)
        center.setSpacing(3)
        center.addWidget(heading("Training loss"))
        self.graph = LossGraph()
        self.graph.setMinimumHeight(92)
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
        self.log.setMinimumHeight(40)
        self.workspace.addWidget(self.log)
        self.log.hide()
        log_toggle.toggled.connect(self.log.setVisible)
        self.workspace.setSizes([345, 180, 80])
        self.workspace.splitterMoved.connect(lambda *_: self.show_result())
        self.columns.splitterMoved.connect(lambda *_: self.show_result())

        self.render_cache = tempfile.TemporaryDirectory()
        self.result_dir = None
        self.images = QProcess(self)
        self.images.setWorkingDirectory(str(ROOT))
        self.images.setProcessEnvironment(env)
        self.images.finished.connect(self.result_ready)
        self.channel.currentIndexChanged.connect(self.show_result)
        self.renderer = QProcess(self)
        self.renderer.setWorkingDirectory(str(ROOT))
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

        item = self.sources.currentItem()
        if item and not self.updating_source:
            paths = [f.text() for f in self.paths]
            item.setData(Qt.UserRole, paths)
            item.setIcon(self.thumbs[0].icon())
            item.setData(Qt.UserRole + 1, None)
            item.setText(item.data(Qt.UserRole + 2))

    def add_source(self, name, paths):
        item = QListWidgetItem()
        item.setText(name)
        item.setData(Qt.UserRole, list(paths))
        item.setData(Qt.UserRole + 2, name)
        item.setCheckState(Qt.Checked)
        item.setToolTip("\n".join(paths))
        self.sources.addItem(item)
        self.sources.show()
        self.sources.setCurrentItem(item)
        self.update_train_label()

    def display_source(self, item, previous=None):
        self.updating_source = True
        try:
            paths = item.data(Qt.UserRole) if item else [""] * len(CHANNELS)
            for field, path in zip(self.paths, paths):
                field.setText(path)
            if item:
                item.setIcon(self.thumbs[0].icon())
        finally:
            self.updating_source = False

    def remove_source(self):
        row = self.sources.currentRow()
        if row >= 0:
            self.sources.takeItem(row)
            self.sources.setVisible(bool(self.sources.count()))
            self.update_train_label()

    def open_source_result(self, item):
        result = item.data(Qt.UserRole + 1)
        if result:
            self.load_export(Path(result))

    def checked_sources(self):
        return [
            self.sources.item(i)
            for i in range(self.sources.count())
            if self.sources.item(i).checkState() == Qt.Checked
        ]

    def update_train_label(self, *args):
        count = len(self.checked_sources())
        self.train_button.setText(
            f"Train {count} materials" if count > 1 else "Train material"
        )
        self.train_button.setEnabled(bool(count) or self.sources.count() == 0)

    def train_selection(self):
        self.remaining_sources = self.checked_sources()
        for item in self.remaining_sources:
            if not all(Path(p).is_file() for p in item.data(Qt.UserRole)):
                self.sources.setCurrentItem(item)
                QMessageBox.warning(
                    self, "Material maps", "Select all five maps for this material."
                )
                self.remaining_sources.clear()
                return
        if self.remaining_sources:
            self.start_next_source()
        else:
            self.start_training()

    def start_next_source(self):
        self.active_source = self.remaining_sources.pop(0)
        self.sources.setCurrentItem(self.active_source)
        if not self.start_training():
            self.active_source = None
            self.remaining_sources.clear()
        else:
            self.active_source.setText(
                f"{self.active_source.data(Qt.UserRole + 2)} · Training"
            )

    def finish_source(self, status):
        if self.active_source is None:
            return
        item = self.active_source
        self.active_source = None
        item.setText(f"{item.data(Qt.UserRole + 2)} · {status}")
        if status == "Done":
            item.setData(Qt.UserRole + 1, str(self.run / "export"))
        if self.remaining_sources:
            self.start_next_source()

    def browse_map(self, edit):
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Choose {CHANNELS[self.paths.index(edit)]} texture",
            str(Path(edit.text()).parent) if edit.text() else "",
            "PNG textures (*.png);;Other images (*.jpg *.jpeg *.tga *.bmp *.tif *.tiff)",
        )
        if path:
            edit.setText(path)
            self.prepare_analysis()

    def add_material(self):
        name, accepted = QInputDialog.getText(self, "Add material", "Material name")
        if not accepted or not name.strip():
            return
        # Adopt any loose maps already entered for the first named material.
        paths = (
            [f.text() for f in self.paths]
            if not self.sources.count()
            else [""] * len(CHANNELS)
        )
        self.add_source(name.strip(), paths)
        self.paths[0].setFocus()

    def set_folder(self, folder):
        if (folder / "maps").is_dir():
            folder = folder / "maps"
        paths = [str((folder / (name.lower() + ".png")).resolve()) for name in CHANNELS]
        name = (folder.parent if folder.name == "maps" else folder).name
        if not any(Path(p).is_file() for p in paths):
            QMessageBox.warning(
                self, "Import source", "Choose a folder with named PBR maps."
            )
            return
        paths = [p if Path(p).is_file() else "" for p in paths]
        if not self.sources.count() and any(f.text() for f in self.paths):
            self.add_source(self.material_name(), [f.text() for f in self.paths])
        for i in range(self.sources.count()):
            item = self.sources.item(i)
            if item.data(Qt.UserRole) == paths:
                self.sources.setCurrentItem(item)
                return
        self.add_source(name, paths)

    def material_name(self):
        item = self.sources.currentItem()
        if item:
            return item.data(Qt.UserRole + 2)
        folder = Path(self.paths[0].text()).parent
        return (
            folder.parent if folder.name.lower() == "maps" else folder
        ).name or "material"

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
        output_name = (
            re.sub(r"[^\w .-]+", "_", self.material_name()).strip(" .")[:72]
            or "material"
        )
        parent = ROOT / "runs/studio"
        if (ROOT / "build-info.json").is_file() and os.environ.get("LOCALAPPDATA"):
            parent = Path(os.environ["LOCALAPPDATA"]) / "NN-PBR Studio/runs"
        try:
            parent.mkdir(parents=True, exist_ok=True)
            self.run = Path(
                tempfile.mkdtemp(
                    prefix=f"{output_name}-{datetime.now():%Y%m%d-%H%M%S}-",
                    dir=parent,
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

        return True

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
            self.finish_source("Failed")

    def stop_training(self):
        if (
            self.worker.state() != QProcess.NotRunning
            and QMessageBox.question(
                self,
                "Stop training",
                "Stop training the selection? Completed checkpoints stay in their material folders.",
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
            self.remaining_sources.clear()
            self.finish_source("Stopped")
            self.status.setText(f"Training stopped · {self.run}")
            return
        if code:
            self.status.setText(
                f"{self.stage.capitalize()} failed (exit {code}). See log below."
            )
        else:
            self.status.setText(f"Export ready · {self.run.name}")
            if not self.remaining_sources:
                self.load_export(self.run / "export")
        self.finish_source("Failed" if code else "Done")

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
        self.overview.hide()
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
        self.overview.hide()
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
        overview = self.channel.currentIndex() == 0
        self.overview.setVisible(overview)
        self.comparison.setVisible(not overview)
        self.latent_row.setVisible(not overview)
        if overview:
            self.overview_source.setText(self.summary["source"] or "Source")
            stats = self.summary["channels"]
            for row, (name, idx) in enumerate(
                zip(("Albedo", "Normal", "ORM"), (0, 1, 5))
            ):
                self.overview_names[row].setText(
                    name + (f"\n{stats[idx]['mse']:.4g}" if stats else "")
                )
                for cell, key in zip(
                    self.overview_tiles[row], ("source", "decoded", "error")
                ):
                    pix = QPixmap(str(self.result_dir / f"{key}-{idx}.png"))
                    if pix.isNull():
                        cell.setText("—")
                    else:
                        cell.setPixmap(
                            pix.scaled(
                                cell.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                            )
                        )
                    if key == "error" and stats:
                        cell.setToolTip(
                            f"Absolute error · scale 0 to {stats[idx]['peak']:.4g}"
                        )
            for idx, cell in enumerate(self.overview_latents):
                pix = QPixmap(str(self.result_dir / f"latent-{idx}.png"))
                cell.setPixmap(
                    pix.scaled(cell.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
            self.caption.setText(
                "MSE per row · error images scaled independently"
                if stats
                else "Load matching source maps to compare."
            )
            return
        i = self.channel.currentIndex() - 1
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
                self.inspector.setMinimumWidth(155)
                self.columns.addWidget(self.inspector)
                self.columns.setSizes(
                    [
                        self.settings_box.width(),
                        max(295, self.workspace.width() - 210),
                        210,
                    ]
                )
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
        "--viewer",
        type=Path,
        default=ROOT
        / (
            "build/Release/vk_neural_demo.exe"
            if os.name == "nt"
            else "build/vk_neural_demo"
        ),
    )
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument(
        "--maps",
        type=Path,
        action="append",
        default=[],
        help="Add a source material folder (repeat for multiple materials)",
    )
    parser.add_argument("--weights", type=Path)
    parser.add_argument(
        "--latents", type=Path, nargs=4, metavar=("DDS0", "DDS1", "DDS2", "DDS3")
    )
    args = parser.parse_args()
    if bool(args.weights) != bool(args.latents) or (args.weights and args.export_dir):
        parser.error(
            "Use either --export-dir or --weights FILE --latents DDS0 DDS1 DDS2 DDS3"
        )
    app = QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    window = Studio(args.viewer.resolve())
    for folder in args.maps:
        window.set_folder(folder.resolve())
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
