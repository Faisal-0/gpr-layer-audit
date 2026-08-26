from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from gpr_layer_audit.models import AnalysisResult

from .theme import LAYER_COLOURS


class ProfileView(QWidget):
    """Linked cumulative-depth and individual-thickness engineering profiles."""

    chainageRequested = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result: AnalysisResult | None = None
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["millimetres", "inches"])
        self.unit_combo.currentTextChanged.connect(self._draw)
        header = QHBoxLayout()
        header.addWidget(QLabel("PROFILE UNITS"))
        header.addWidget(self.unit_combo)
        header.addStretch(1)
        self.cumulative = self._plot("Cumulative interface depth")
        self.individual = self._plot("Individual layer thickness")
        self.cumulative.setXLink(self.individual)
        self._cumulative_cursor = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen("#edf7fa", width=1)
        )
        self._individual_cursor = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen("#edf7fa", width=1)
        )
        self.cumulative.addItem(self._cumulative_cursor)
        self.individual.addItem(self._individual_cursor)
        self.cumulative.scene().sigMouseClicked.connect(self._clicked)
        self.individual.scene().sigMouseClicked.connect(self._clicked)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(header)
        layout.addWidget(self.cumulative, 1)
        layout.addWidget(self.individual, 1)

    @staticmethod
    def _plot(title: str) -> pg.PlotWidget:
        plot = pg.PlotWidget(background="#071016", title=title)
        plot.setLabel("bottom", "Chainage", units="m")
        plot.setLabel("left", "Depth / thickness", units="mm")
        plot.showGrid(x=True, y=True, alpha=0.13)
        plot.addLegend(offset=(12, 12))
        return plot

    def set_result(self, result: AnalysisResult) -> None:
        self.result = result
        self._draw()

    def set_cursor(self, chainage_m: float, *_unused) -> None:
        self._cumulative_cursor.setPos(chainage_m)
        self._individual_cursor.setPos(chainage_m)

    def _draw(self) -> None:
        for plot, cursor in (
            (self.cumulative, self._cumulative_cursor),
            (self.individual, self._individual_cursor),
        ):
            plot.clear()
            plot.addItem(cursor)
        if self.result is None:
            return
        factor = 1.0 / 25.4 if self.unit_combo.currentText() == "inches" else 1.0
        unit = "in" if factor != 1.0 else "mm"
        self.cumulative.setLabel("left", "Cumulative depth", units=unit)
        self.individual.setLabel("left", "Layer thickness", units=unit)
        for order in sorted({item.layer_order for item in self.result.profile}):
            points = sorted(
                (item for item in self.result.profile if item.layer_order == order),
                key=lambda item: item.chainage_m,
            )
            if not points:
                continue
            x = np.asarray([item.chainage_m for item in points], dtype=float)
            cumulative = np.asarray(
                [
                    item.cumulative_depth_mm * factor
                    if item.cumulative_depth_mm is not None
                    and not item.anomaly
                    and not item.interpolated
                    else np.nan
                    for item in points
                ]
            )
            individual = np.asarray(
                [
                    item.individual_thickness_mm * factor
                    if item.individual_thickness_mm is not None
                    and not item.anomaly
                    and not item.interpolated
                    else np.nan
                    for item in points
                ]
            )
            pen = pg.mkPen(LAYER_COLOURS.get(order, "#ffffff"), width=1.8)
            name = points[0].layer_name
            self.cumulative.plot(x, cumulative, pen=pen, name=name, connect="finite")
            self.individual.plot(x, individual, pen=pen, name=name, connect="finite")
            dashed_pen = pg.mkPen(
                LAYER_COLOURS.get(order, "#ffffff"),
                width=1.3,
                style=Qt.PenStyle.DashLine,
            )
            for plot, field_name in (
                (self.cumulative, "cumulative_depth_mm"),
                (self.individual, "individual_thickness_mm"),
            ):
                interpolated = np.asarray(
                    [
                        getattr(item, field_name) * factor
                        if item.interpolated
                        and not item.anomaly
                        and getattr(item, field_name) is not None
                        else np.nan
                        for item in points
                    ]
                )
                if np.any(np.isfinite(interpolated)):
                    plot.plot(
                        x,
                        interpolated,
                        pen=dashed_pen,
                        symbol="o",
                        symbolSize=3,
                        connect="finite",
                    )
            colour = pg.mkColor(LAYER_COLOURS.get(order, "#ffffff"))
            colour.setAlpha(35)
            for plot, low_name, high_name in (
                (self.cumulative, "cumulative_low_mm", "cumulative_high_mm"),
                (self.individual, "individual_low_mm", "individual_high_mm"),
            ):
                low = np.asarray(
                    [
                        getattr(item, low_name) * factor
                        if getattr(item, low_name) is not None
                        else np.nan
                        for item in points
                    ]
                )
                high = np.asarray(
                    [
                        getattr(item, high_name) * factor
                        if getattr(item, high_name) is not None
                        else np.nan
                        for item in points
                    ]
                )
                low_curve = pg.PlotDataItem(x, low, pen=pg.mkPen(None), connect="finite")
                high_curve = pg.PlotDataItem(x, high, pen=pg.mkPen(None), connect="finite")
                plot.addItem(low_curve)
                plot.addItem(high_curve)
                plot.addItem(
                    pg.FillBetweenItem(low_curve, high_curve, brush=pg.mkBrush(colour))
                )
            design = np.asarray(
                [
                    item.design_thickness_mm * factor
                    if item.design_thickness_mm is not None
                    else np.nan
                    for item in points
                ]
            )
            if np.any(np.isfinite(design)):
                design_pen = pg.mkPen(
                    LAYER_COLOURS.get(order, "#ffffff"),
                    width=1,
                    style=Qt.PenStyle.DashLine,
                )
                self.individual.plot(x, design, pen=design_pen, connect="finite")
        for region in self.result.anomaly_regions:
            for plot in (self.cumulative, self.individual):
                shade = pg.LinearRegionItem(
                    values=(region.start_chainage_m, region.end_chainage_m),
                    movable=False,
                    brush=pg.mkBrush(255, 107, 107, 40),
                    pen=pg.mkPen(None),
                    orientation="vertical",
                )
                shade.setZValue(-5)
                plot.addItem(shade)

    def _clicked(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        plot = (
            self.cumulative
            if self.cumulative.sceneBoundingRect().contains(event.scenePos())
            else self.individual
        )
        point = plot.plotItem.vb.mapSceneToView(event.scenePos())
        self.chainageRequested.emit(float(point.x()))
