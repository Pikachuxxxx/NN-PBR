"""Qt widgets for the loss timeline and decoded material inspection."""

import json
import math
import struct
import sys
import tempfile
from pathlib import Path

from PySide6.QtCore import QPointF, QProcess, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QLabel,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


def panel_splitter(orientation):
    splitter = QSplitter(orientation)
    splitter.setChildrenCollapsible(False)
    splitter.setHandleWidth(6)
    return splitter


class LossGraph(QWidget):
    colors = ("#55b8ef", "#e8a352", "#63c6a5")

    def __init__(self):
        super().__init__()
        self.points = []
        self.setMinimumHeight(112)
        self.setToolTip(
            "Each point is one logged training batch. Vertical scale is logarithmic."
        )

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#181b21"))
        area = QRectF(53, 13, self.width() - 73, self.height() - 43)
        logs = [math.log10(max(loss, 1e-12)) for _, loss, _ in self.points]
        low, high = (math.floor(min(logs)), math.ceil(max(logs))) if logs else (-5, 0)
        high = max(low + 1, high)
        end = self.points[-1][0] if self.points else 1
        for power in range(low, high + 1):
            y = area.bottom() - (power - low) * area.height() / (high - low)
            painter.setPen(QColor("#333841"))
            painter.drawLine(QPointF(area.left(), y), QPointF(area.right(), y))
            painter.setPen(QColor("#aab2bf"))
            painter.drawText(QPointF(7, y + 4), f"1e{power}")
        painter.drawText(QPointF(area.left(), self.height() - 8), "0")
        painter.drawText(
            QPointF(area.right() - 90, self.height() - 8),
            f"{end:,} updates" if logs else "Updates",
        )
        for phase, color in enumerate(self.colors, 1):
            path = QPainterPath()
            for (step, _, p), val in zip(self.points, logs):
                if p != phase:
                    continue
                point = QPointF(
                    area.left() + step * area.width() / end,
                    area.bottom() - (val - low) * area.height() / (high - low),
                )
                if path.elementCount() == 0:
                    path.moveTo(point)
                    painter.setPen(QColor(color))
                    painter.drawEllipse(point, 1.6, 1.6)
                else:
                    path.lineTo(point)
            painter.setPen(QPen(QColor(color), 1.4))
            painter.drawPath(path)
        if not logs:
            painter.setPen(QColor("#aab2bf"))
            painter.drawText(
                area, Qt.AlignCenter, "Waiting for the first training batch"
            )


class ModelData(QWidget):
    """Compact inline inspection of the selected runtime assets."""

    def __init__(self, weights, textures, parent):
        super().__init__(parent)
        self.cache = tempfile.TemporaryDirectory()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        splitter = panel_splitter(Qt.Vertical)
        layout.addWidget(splitter)
        texture_panel = QWidget()
        splitter.addWidget(texture_panel)
        body = QVBoxLayout(texture_panel)
        body.addWidget(QLabel("Latent textures"))
        self.layer = QComboBox()
        self.layer.addItems(
            [f"Texture {i}  ·  {p.name}" for i, p in enumerate(textures)]
        )
        self.mip = QComboBox()
        body.addWidget(self.layer)
        body.addWidget(self.mip)
        self.image = QLabel("Decoding texture…")
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.image.setMinimumHeight(50)
        body.addWidget(self.image)
        self.range = QLabel()
        self.range.setWordWrap(True)
        body.addWidget(self.range)
        self.inventory = []
        self.layer.currentIndexChanged.connect(self.select_layer)
        self.mip.currentIndexChanged.connect(self.show_mip)
        weights_tab = QWidget()
        weights_layout = QVBoxLayout(weights_tab)
        weights_layout.addWidget(QLabel("Decoder weights"))
        self.tensor = QComboBox()
        self.tensor.addItems(
            [
                "Layer 1 weights   ·   16 × 12",
                "Layer 1 bias   ·   16",
                "Layer 2 weights   ·   8 × 16",
                "Layer 2 bias   ·   8",
            ]
        )
        self.weight_table = QTableWidget()
        self.weight_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.weight_table.setMinimumWidth(120)
        self.weight_table.horizontalHeader().setDefaultSectionSize(70)
        weights_layout.addWidget(self.tensor)
        weights_layout.addWidget(self.weight_table)
        weights_layout.addWidget(
            QLabel(f"{weights.name}   ·   FP16   ·   344 parameters")
        )
        splitter.addWidget(weights_tab)
        splitter.setSizes([240, 260])
        splitter.splitterMoved.connect(
            lambda *_: self.show_mip(self.mip.currentIndex())
        )
        self.values = struct.unpack("<344e", weights.read_bytes())
        self.tensor.currentIndexChanged.connect(self.show_tensor)
        self.show_tensor(0)
        self.decoder = QProcess(self)
        self.decoder.setWorkingDirectory(str(Path(__file__).resolve().parents[1]))
        self.decoder.finished.connect(self.decoded)
        self.decoder.start(
            sys.executable,
            ["-m", "nnpbr.studio_latents", self.cache.name, *map(str, textures)],
        )

    def decoded(self, code, status):
        if code:
            self.image.setText(
                bytes(self.decoder.readAllStandardError()).decode("utf8", "replace")[
                    -400:
                ]
            )
            return
        self.inventory = json.loads((Path(self.cache.name) / "levels.json").read_text())
        self.select_layer(self.layer.currentIndex())

    def select_layer(self, i):
        if self.inventory:
            self.mip.clear()
            self.mip.addItems(
                [
                    f"Mip {j}   ·   {entry['size']} px"
                    for j, entry in enumerate(self.inventory[i])
                ]
            )

    def show_mip(self, j):
        if j < 0 or not self.inventory:
            return
        i = self.layer.currentIndex()
        entry = self.inventory[i][j]
        pix = QPixmap(str(Path(self.cache.name) / f"{i}-{j}.png"))
        self.image.setPixmap(
            pix.scaled(self.image.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
        )
        self.range.setText(
            f"Decoded RGB range   {entry['min']:.5g} … {entry['max']:.5g}     ·     Scaled to fit the preview"
        )

    def show_tensor(self, index):
        offset, rows, cols = [(0, 16, 12), (192, 16, 1), (208, 8, 16), (336, 8, 1)][
            index
        ]
        values = self.values[offset : offset + rows * cols]
        peak = max(max(abs(v) for v in values), 1e-10)
        self.weight_table.setRowCount(rows)
        self.weight_table.setColumnCount(cols)
        self.weight_table.setHorizontalHeaderLabels(
            ["Bias"] if cols == 1 else [f"in {i}" for i in range(cols)]
        )
        for i, v in enumerate(values):
            item = QTableWidgetItem(f"{v:.5g}")
            color = QColor("#2d6a8d" if v < 0 else "#91652e")
            color.setAlpha(int(220 * abs(v) / peak))
            item.setToolTip(f"Output {i // cols}, input {i % cols}: {v}")
            item.setBackground(color)
            self.weight_table.setItem(i // cols, i % cols, item)

    def dispose(self):
        if self.decoder.state() != QProcess.NotRunning:
            self.decoder.kill()
            self.decoder.waitForFinished(3000)
        self.cache.cleanup()
