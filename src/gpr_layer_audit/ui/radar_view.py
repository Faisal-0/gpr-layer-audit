from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from gpr_layer_audit.models import AnalysisResult

from .theme import LAYER_COLOURS, LAYER_DASHES


class RadarView(QWidget):
    locationChanged = Signal(float, float, int)
    anchorRequested = Signal(int, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result: AnalysisResult | None = None
        self.active_layer = 1
        self.view_mode = "Clean"
        self.image = pg.ImageItem(axisOrder="row-major")
        self.radar_plot = pg.PlotWidget(background="#071016")
        self.radar_plot.addItem(self.image)
        self._candidate_ridges = pg.ScatterPlotItem(size=3, pen=None)
        self.radar_plot.addItem(self._candidate_ridges)
        self.radar_plot.setLabel("bottom", "Chainage", units="m")
        self.radar_plot.setLabel("left", "Two-way time", units="ns")
        self.radar_plot.showGrid(x=True, y=True, alpha=0.13)
        self.radar_plot.getViewBox().setMouseMode(pg.ViewBox.PanMode)
        self.radar_plot.getViewBox().invertY(True)
        self.a_scan = pg.PlotWidget(background="#071016")
        self.a_scan.setMaximumHeight(175)
        self.a_scan.setLabel("bottom", "Amplitude")
        self.a_scan.setLabel("left", "Time", units="ns")
        self.a_scan.showGrid(x=True, y=True, alpha=0.12)
        self.a_scan.getViewBox().invertY(True)
        self._a_curve = self.a_scan.plot(pen=pg.mkPen("#dbe7ed", width=1.2))
        self._candidate_markers = pg.ScatterPlotItem(
            size=7,
            pen=pg.mkPen("#071016", width=1),
            brush=pg.mkBrush("#ffc857"),
        )
        self.a_scan.addItem(self._candidate_markers)
        self._current_index = 0
        self._candidate_by_layer_chainage: dict[tuple[int, float], list] = {}
        self._cursor = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#8aa1ac", width=1))
        self.radar_plot.addItem(self._cursor)
        self._pick_curves: list[pg.PlotDataItem] = []
        self._corridor_items: list[pg.GraphicsObject] = []
        self._guide_lines: list[pg.InfiniteLine] = []
        self.readout = QLabel(
            "Move across the radargram to inspect an A-scan. Ctrl+click adds an anchor."
        )
        self.readout.setStyleSheet("color: #9eb3bd; padding: 3px 2px;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(self.radar_plot, 1)
        layout.addWidget(self.readout)
        layout.addWidget(self.a_scan)
        self.radar_plot.scene().sigMouseMoved.connect(self._mouse_moved)
        self.radar_plot.scene().sigMouseClicked.connect(self._mouse_clicked)

    def set_active_layer(self, order: int) -> None:
        self.active_layer = order
        if self.result is not None:
            self._set_candidate_preview()
            self._show_trace(self._current_index)

    def set_result(self, result: AnalysisResult) -> None:
        self.result = result
        self._candidate_by_layer_chainage = {}
        for event in result.candidate_events:
            self._candidate_by_layer_chainage.setdefault(
                (event.layer_order, event.chainage_m), []
            ).append(event)
        self._set_image()
        length = max(float(result.chainage_m[-1]), 1.0)
        self.image.setRect(QRectF(0.0, 0.0, length, float(result.header.range_ns)))
        for curve in self._pick_curves:
            self.radar_plot.removeItem(curve)
        self._pick_curves.clear()
        for item in self._corridor_items:
            self.radar_plot.removeItem(item)
        self._corridor_items.clear()
        for order, corridor in sorted(result.search_corridors.items()):
            colour = pg.mkColor(LAYER_COLOURS.get(order, "#ffffff"))
            colour.setAlpha(22)
            lower = pg.PlotDataItem(
                corridor.chainage_m,
                corridor.lower_sample * result.header.sample_interval_ns,
                pen=pg.mkPen(None),
                connect="finite",
            )
            upper = pg.PlotDataItem(
                corridor.chainage_m,
                corridor.upper_sample * result.header.sample_interval_ns,
                pen=pg.mkPen(None),
                connect="finite",
            )
            band = pg.FillBetweenItem(lower, upper, brush=pg.mkBrush(colour))
            centre = pg.PlotDataItem(
                corridor.chainage_m,
                corridor.centre_sample * result.header.sample_interval_ns,
                pen=pg.mkPen(
                    LAYER_COLOURS.get(order, "#ffffff"),
                    width=0.8,
                    style=Qt.PenStyle.DashLine,
                ),
                connect="finite",
            )
            for item in (lower, upper, band, centre):
                self.radar_plot.addItem(item)
                self._corridor_items.append(item)
        for order in sorted({item.layer_order for item in result.picks}):
            items = [item for item in result.picks if item.layer_order == order]
            dash = LAYER_DASHES.get(order)
            pen = pg.mkPen(LAYER_COLOURS.get(order, "#ffffff"), width=1.7)
            if dash:
                pen.setDashPattern(dash)
            curve = self.radar_plot.plot(
                [item.chainage_m for item in items],
                [
                    (
                        item.selected_lobe_sample
                        if item.selected_lobe_sample is not None
                        else item.sample_index
                    )
                    * result.header.sample_interval_ns
                    if item.sample_index >= 0
                    else np.nan
                    for item in items
                ],
                pen=pen,
                name=items[0].layer_name,
                connect="finite",
            )
            self._pick_curves.append(curve)
            design = result.design_guided_paths.get(order)
            signal = result.signal_only_paths.get(order)
            if design is not None and signal is not None and np.any(design != signal):
                design_colour = pg.mkColor(LAYER_COLOURS.get(order, "#ffffff"))
                design_colour.setAlpha(105)
                design_pen = pg.mkPen(design_colour, width=1.0)
                design_curve = self.radar_plot.plot(
                    result.chainage_m,
                    np.where(
                        design >= 0,
                        design * result.header.sample_interval_ns,
                        np.nan,
                    ),
                    pen=design_pen,
                    connect="finite",
                )
                self._pick_curves.append(design_curve)
        self._set_guides()
        self._set_candidate_preview()
        self.radar_plot.setXRange(0, min(length, 250), padding=0)
        self.radar_plot.setYRange(0, result.header.range_ns, padding=0)
        self._show_trace(0)

    def set_view_mode(self, mode: str) -> None:
        self.view_mode = mode
        if self.result is not None:
            self._set_image()
            self._show_trace(0)

    def set_enhanced_view(self, enabled: bool) -> None:
        self.set_view_mode("Clean" if enabled else "Raw")

    def available_views(self) -> list[str]:
        if self.result is None:
            return ["Raw", "Clean"]
        preferred = [
            "Raw",
            "Clean",
            "Reflectivity",
            "Phase",
            "Gradient",
            "Oriented ridge",
            "Candidates",
        ]
        available = list(self.result.display_radargrams)
        ordered = [name for name in preferred if name in available]
        ordered.extend(name for name in available if name not in ordered)
        return ordered

    def _display_radargram(self) -> np.ndarray:
        if self.result is None:
            raise RuntimeError("No analysis result loaded")
        if self.view_mode in self.result.display_radargrams:
            return self.result.display_radargrams[self.view_mode]
        if self.view_mode == "Raw" and self.result.interpretation_input_radargram is not None:
            return self.result.interpretation_input_radargram
        return self.result.calibrated_radargram

    def _set_image(self) -> None:
        if self.result is None:
            return
        data = np.asarray(self._display_radargram().T, dtype=np.float32)
        if self.view_mode in {"Gradient", "Candidates"}:
            limit = float(np.percentile(data, 99.0)) or 1.0
            self.image.setImage(data, autoLevels=False, levels=(0.0, limit))
        else:
            limit = float(np.percentile(np.abs(data), 98.5)) or 1.0
            self.image.setImage(data, autoLevels=False, levels=(-limit, limit))

    def _set_guides(self) -> None:
        for line in self._guide_lines:
            self.radar_plot.removeItem(line)
        self._guide_lines.clear()
        if self.result is None:
            return
        seeded = {round(item.chainage_m, 3) for item in self.result.seed_stations}
        for chainage in self.result.proposed_seed_chainages:
            if round(chainage, 3) in seeded:
                continue
            line = pg.InfiniteLine(
                pos=chainage,
                angle=90,
                movable=False,
                pen=pg.mkPen("#6f8792", width=1, style=Qt.PenStyle.DashLine),
            )
            self.radar_plot.addItem(line)
            self._guide_lines.append(line)

    def _set_candidate_preview(self) -> None:
        if self.result is None:
            self._candidate_ridges.clear()
            return
        events = [
            item
            for item in self.result.candidate_events
            if item.layer_order == self.active_layer and item.rank <= 3
        ]
        stride = max(1, len(events) // 12_000)
        spots = []
        colours = {
            1: (255, 200, 87, 115),
            2: (132, 216, 255, 85),
            3: (219, 231, 237, 55),
        }
        for event in events[::stride]:
            spots.append(
                {
                    "pos": (
                        event.chainage_m,
                        event.sample_index * self.result.header.sample_interval_ns,
                    ),
                    "brush": pg.mkBrush(*colours[event.rank]),
                    "data": event,
                }
            )
        self._candidate_ridges.setData(spots)
        for station in self.result.seed_stations:
            line = pg.InfiniteLine(
                pos=station.chainage_m,
                angle=90,
                movable=False,
                pen=pg.mkPen("#edf7fa", width=1.4),
            )
            self.radar_plot.addItem(line)
            self._guide_lines.append(line)

    def refresh_guides(self) -> None:
        self._set_guides()

    def focus_range(self, start_m: float, end_m: float) -> None:
        margin = max(5.0, (end_m - start_m) * 0.25)
        self.radar_plot.setXRange(max(0.0, start_m - margin), end_m + margin, padding=0)

    def focus_chainage(self, chainage_m: float, width_m: float = 20.0) -> None:
        self.focus_range(max(0.0, chainage_m - width_m / 2), chainage_m + width_m / 2)
        if self.result is not None:
            index = int(np.argmin(np.abs(self.result.chainage_m - chainage_m)))
            self._cursor.setPos(float(self.result.chainage_m[index]))
            self._show_trace(index)

    def _position(self, scene_position) -> tuple[float, float] | None:
        if self.result is None or not self.radar_plot.sceneBoundingRect().contains(scene_position):
            return None
        point = self.radar_plot.plotItem.vb.mapSceneToView(scene_position)
        return float(point.x()), float(point.y())

    def _mouse_moved(self, scene_position) -> None:
        position = self._position(scene_position)
        if position is None or self.result is None:
            return
        chainage, time_ns = position
        index = int(np.argmin(np.abs(self.result.chainage_m - chainage)))
        self._show_trace(index)
        self._cursor.setPos(self.result.chainage_m[index])
        sample = int(
            np.clip(
                time_ns / self.result.header.sample_interval_ns,
                0,
                self.result.header.samples_per_trace - 1,
            )
        )
        self.locationChanged.emit(float(self.result.chainage_m[index]), time_ns, sample)

    def _mouse_clicked(self, event) -> None:
        if self.result is None or event.button() != Qt.MouseButton.LeftButton:
            return
        if not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            return
        position = self._position(event.scenePos())
        if position is None:
            return
        chainage, time_ns = position
        sample = time_ns / self.result.header.sample_interval_ns
        self.anchorRequested.emit(self.active_layer, chainage, sample)

    def _show_trace(self, index: int) -> None:
        if self.result is None:
            return
        index = int(np.clip(index, 0, len(self.result.chainage_m) - 1))
        self._current_index = index
        trace = self._display_radargram()[index]
        time = np.arange(len(trace)) * self.result.header.sample_interval_ns
        self._a_curve.setData(trace, time)
        chainage = float(self.result.chainage_m[index])
        events = self._candidate_by_layer_chainage.get((self.active_layer, chainage), [])
        self._candidate_markers.setData(
            [trace[item.sample_index] for item in events],
            [item.sample_index * self.result.header.sample_interval_ns for item in events],
        )
        self.a_scan.setYRange(0, self.result.header.range_ns, padding=0)
        candidates = ", ".join(
            f"#{item.rank} s{item.sample_index} corr {item.waveform_correlation:.2f} "
            f"phase {item.phase_class}"
            for item in sorted(events, key=lambda value: value.rank)[:3]
        )
        self.readout.setText(
            f"Chainage {self.result.chainage_m[index]:,.2f} m  •  "
            f"stack {index:,}  •  {self.view_mode} view  •  "
            f"Ctrl+click to seed layer {self.active_layer}"
            + (f"  •  {candidates}" if candidates else "")
        )
