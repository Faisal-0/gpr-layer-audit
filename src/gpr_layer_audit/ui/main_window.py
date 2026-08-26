"""GPR Layer Audit main workbench.

THESIS: The radar evidence is the workspace; controls and summaries never displace it.
OWN-WORLD: A strip-chart inspection bench with ink-black signal fields, cool instrument chrome,
and cyan/amber/coral interface channels.
STORY: Import, calibrate, auto-track, resolve exceptions, then reveal design compliance.
FIRST VIEWPORT: Project scope at left, the full radargram and A-scan in the centre, and the
exception queue at right; Run Analysis is the single primary action.
FORM: Oscilloscope and continuous strip-chart workbench, seed 6dd06bf2.
FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review,
the verdict, DESIGN.md, and every shipping raster carrying its provenance.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from threading import Event

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal, Slot
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from gpr_layer_audit.design import compare_with_design, read_design_schedule
from gpr_layer_audit.export import export_audit_package
from gpr_layer_audit.models import AcquisitionFileSet, AnalysisResult, LayerSpec
from gpr_layer_audit.processing import (
    AnalysisCancelled,
    AnalysisOptions,
    analyze_acquisition,
    retrack_segment,
)
from gpr_layer_audit.project import ProjectStore
from gpr_layer_audit.reference import evaluate_manual_reference, read_manual_reference

from .radar_view import RadarView
from .theme import LAYER_COLOURS


class WorkerSignals(QObject):
    progress = Signal(int, str)
    result = Signal(object)
    error = Signal(str)
    cancelled = Signal()


class AnalysisWorker(QRunnable):
    def __init__(self, road, plate, options):
        super().__init__()
        self.road = road
        self.plate = plate
        self.options = options
        self.signals = WorkerSignals()
        self.cancel_event = Event()

    @Slot()
    def run(self):
        try:
            result = analyze_acquisition(
                self.road,
                self.plate,
                self.options,
                progress=lambda value, text: self.signals.progress.emit(value, text),
                cancel=self.cancel_event.is_set,
            )
        except AnalysisCancelled:
            self.signals.cancelled.emit()
        except Exception:
            self.signals.error.emit(traceback.format_exc())
        else:
            self.signals.result.emit(result)

    def cancel(self):
        self.cancel_event.set()


class ProjectDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import GSSI surveys and create an audit project")
        self.setMinimumWidth(680)
        self.road_edit = QLineEdit()
        self.plate_edit = QLineEdit()
        self.project_edit = QLineEdit()
        self.accept_dielectric = QCheckBox("Use scan dielectric as an explicitly assumed value")
        self.interval = QComboBox()
        self.interval.addItems(["1", "5", "10"])
        self.interval.setCurrentText("5")
        form = QFormLayout()
        form.addRow("Road GSSI survey", self._path_row(self.road_edit))
        form.addRow("Metal-plate calibration", self._path_row(self.plate_edit))
        form.addRow("Audit project record", self._save_row(self.project_edit))
        form.addRow("Report interval (m)", self.interval)
        form.addRow("Dielectric policy", self.accept_dielectric)
        notice = QLabel(
            "For each GSSI input, select either its .PRJ folder or the .DZT inside it. "
            "The application automatically attaches matching .DZG GPS and .DZX metadata. "
            "The .gprproj file below is this application's resumable audit record.\n\n"
            "The scan dielectric is an acquisition assumption. Leave this unchecked to "
            "retain travel-time picks without forcing physical thickness where reflection "
            "calibration fails."
        )
        notice.setWordWrap(True)
        notice.setStyleSheet("color: #9eb3bd; padding: 8px 0;")
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(notice)
        layout.addWidget(buttons)

    def _path_row(self, edit):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        file_button = QPushButton("DZT…")
        file_button.setToolTip("Choose the waveform .DZT inside a GSSI .PRJ folder")
        file_button.clicked.connect(lambda: self._choose_open(edit))
        folder_button = QPushButton("PRJ folder…")
        folder_button.setToolTip("Choose the GSSI .PRJ survey folder")
        folder_button.clicked.connect(lambda: self._choose_prj_folder(edit))
        layout.addWidget(edit, 1)
        layout.addWidget(file_button)
        layout.addWidget(folder_button)
        return widget

    def _save_row(self, edit):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        button = QPushButton("Browse…")
        button.clicked.connect(lambda: self._choose_save(edit))
        layout.addWidget(edit, 1)
        layout.addWidget(button)
        return widget

    def _choose_open(self, edit):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select GSSI waveform", "", "GSSI waveform (*.DZT *.dzt)"
        )
        if path:
            edit.setText(path)

    def _choose_prj_folder(self, edit):
        folder = QFileDialog.getExistingDirectory(self, "Select GSSI .PRJ folder")
        if not folder:
            return
        try:
            edit.setText(str(self._dzt_from_folder(Path(folder))))
        except ValueError as exc:
            QMessageBox.warning(self, "GSSI survey not found", str(exc))

    def _dzt_from_folder(self, folder: Path) -> Path:
        candidates = sorted(
            path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == ".dzt"
        )
        if not candidates:
            raise ValueError(f"No .DZT waveform was found in {folder}")
        if len(candidates) == 1:
            return candidates[0]
        labels = [path.name for path in candidates]
        selected, accepted = QInputDialog.getItem(
            self,
            "Choose acquisition",
            "This .PRJ folder contains multiple DZT files:",
            labels,
            0,
            False,
        )
        if not accepted:
            raise ValueError("No DZT acquisition was selected.")
        return candidates[labels.index(selected)]

    def _choose_save(self, edit):
        path, _ = QFileDialog.getSaveFileName(self, "Create project", "", "GPR project (*.gprproj)")
        if path:
            edit.setText(path if path.lower().endswith(".gprproj") else path + ".gprproj")

    def accept(self):
        road_path = Path(self.road_edit.text())
        if road_path.is_dir():
            try:
                road_path = self._dzt_from_folder(road_path)
                self.road_edit.setText(str(road_path))
            except ValueError as exc:
                QMessageBox.warning(self, "Road survey required", str(exc))
                return
        if not road_path.is_file() or road_path.suffix.lower() != ".dzt":
            QMessageBox.warning(
                self,
                "Road survey required",
                "Choose the road .PRJ folder or its waveform .DZT file.",
            )
            return
        plate_path = Path(self.plate_edit.text()) if self.plate_edit.text() else None
        if plate_path and plate_path.is_dir():
            try:
                plate_path = self._dzt_from_folder(plate_path)
                self.plate_edit.setText(str(plate_path))
            except ValueError as exc:
                QMessageBox.warning(self, "Plate calibration not found", str(exc))
                return
        if plate_path and (not plate_path.is_file() or plate_path.suffix.lower() != ".dzt"):
            QMessageBox.warning(
                self,
                "Plate calibration not found",
                "Choose the metal-plate .PRJ folder or its waveform .DZT file.",
            )
            return
        if not self.project_edit.text():
            QMessageBox.warning(
                self, "Project file required", "Choose where the project record will be saved."
            )
            return
        super().accept()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GPR Layer Audit")
        self.resize(1500, 900)
        self.setMinimumSize(1180, 720)
        self.thread_pool = QThreadPool.globalInstance()
        self.worker: AnalysisWorker | None = None
        self.result: AnalysisResult | None = None
        self.project_store: ProjectStore | None = None
        self.road: AcquisitionFileSet | None = None
        self.plate: AcquisitionFileSet | None = None
        self.options = AnalysisOptions()
        self.design_segments = []
        self.reference_points = []
        self._build_toolbar()
        self._build_workspace()
        self.statusBar().showMessage("Create a project to begin.")

    def _build_toolbar(self):
        toolbar = QToolBar("Analysis")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        new_action = QAction("Import GSSI survey", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self.new_project)
        toolbar.addAction(new_action)
        open_action = QAction("Open audit project", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_project)
        toolbar.addAction(open_action)
        self.run_action = QAction("Run analysis", self)
        self.run_action.setShortcut("Ctrl+R")
        self.run_action.triggered.connect(self.run_analysis)
        self.run_action.setEnabled(False)
        toolbar.addAction(self.run_action)
        self.cancel_action = QAction("Cancel", self)
        self.cancel_action.triggered.connect(self.cancel_analysis)
        self.cancel_action.setEnabled(False)
        toolbar.addAction(self.cancel_action)
        self.enhanced_action = QAction("Enhanced view", self)
        self.enhanced_action.setCheckable(True)
        self.enhanced_action.setChecked(True)
        self.enhanced_action.setToolTip(
            "Toggle between the automatic interpretation preprocessing and its input"
        )
        self.enhanced_action.toggled.connect(lambda enabled: self.radar.set_enhanced_view(enabled))
        toolbar.addAction(self.enhanced_action)
        undo_action = QAction("Undo anchor", self)
        undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        undo_action.triggered.connect(self.undo_anchor)
        toolbar.addAction(undo_action)
        toolbar.addSeparator()
        design_action = QAction("Import design", self)
        design_action.triggered.connect(self.import_design)
        toolbar.addAction(design_action)
        reference_action = QAction("Import manual reference", self)
        reference_action.triggered.connect(self.import_reference)
        toolbar.addAction(reference_action)
        self.export_action = QAction("Export audit package", self)
        self.export_action.setShortcut("Ctrl+E")
        self.export_action.triggered.connect(self.export_package)
        self.export_action.setEnabled(False)
        toolbar.addAction(self.export_action)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)
        self.progress = QProgressBar()
        self.progress.setFixedWidth(250)
        self.progress.setVisible(False)
        toolbar.addWidget(self.progress)

    def _build_workspace(self):
        self.radar = RadarView()
        self.radar.anchorRequested.connect(self.add_anchor)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._project_panel())
        splitter.addWidget(self.radar)
        splitter.addWidget(self._review_panel())
        splitter.setSizes([245, 1000, 300])
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

    def _project_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        title = QLabel("PROJECT SCOPE")
        title.setStyleSheet("font-weight: 700; color: #9eb3bd; letter-spacing: 1px;")
        layout.addWidget(title)
        self.source_label = QLabel("No road file loaded")
        self.source_label.setWordWrap(True)
        layout.addWidget(self.source_label)
        layers = QGroupBox("Interfaces")
        layer_layout = QVBoxLayout(layers)
        self.layer_checks = {}
        for layer in LayerSpec.defaults():
            check = QCheckBox(layer.name)
            check.setChecked(True)
            check.setStyleSheet(f"color: {LAYER_COLOURS[layer.order]};")
            check.clicked.connect(
                lambda checked, order=layer.order: self.radar.set_active_layer(order)
            )
            self.layer_checks[layer.order] = check
            layer_layout.addWidget(check)
        layout.addWidget(layers)
        self.assumption_label = QLabel(
            "Dielectric: unresolved until calibration or explicit acceptance"
        )
        self.assumption_label.setWordWrap(True)
        self.assumption_label.setStyleSheet("color: #ffc857; padding: 8px 0;")
        layout.addWidget(self.assumption_label)
        self.calibration_label = QLabel("Calibration has not run.")
        self.calibration_label.setWordWrap(True)
        layout.addWidget(self.calibration_label)
        layout.addStretch(1)
        return panel

    def _review_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        title = QLabel("EXCEPTION QUEUE")
        title.setStyleSheet("font-weight: 700; color: #9eb3bd; letter-spacing: 1px;")
        layout.addWidget(title)
        self.issue_list = QListWidget()
        self.issue_list.itemActivated.connect(self.focus_issue)
        layout.addWidget(self.issue_list, 1)
        self.tabs = QTabWidget()
        self.thickness_table = QTableWidget(0, 5)
        self.thickness_table.setHorizontalHeaderLabels(
            ["Chainage", "Layer", "mm", "Confidence", "Status"]
        )
        self.thickness_table.setAlternatingRowColors(True)
        self.tabs.addTab(self.thickness_table, "Results")
        self.method_text = QLabel("Run analysis to view calibration provenance.")
        self.method_text.setWordWrap(True)
        self.method_text.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.tabs.addTab(self.method_text, "Method")
        layout.addWidget(self.tabs, 1)
        return panel

    @Slot()
    def new_project(self):
        dialog = ProjectDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.road = AcquisitionFileSet(Path(dialog.road_edit.text()))
        self.plate = (
            AcquisitionFileSet(Path(dialog.plate_edit.text())) if dialog.plate_edit.text() else None
        )
        self.options = AnalysisOptions(
            report_interval_m=float(dialog.interval.currentText()),
            accept_scan_dielectric=dialog.accept_dielectric.isChecked(),
        )
        self.project_store = ProjectStore.create(
            dialog.project_edit.text(), self.road.dzt_path.stem
        )
        self.project_store.set_file("road", self.road.dzt_path)
        if self.plate:
            self.project_store.set_file("plate", self.plate.dzt_path)
        self.project_store.set_layers(self.options.layer_specs)
        self.source_label.setText(f"{self.road.dzt_path.name}\n{self.road.dzt_path.parent}")
        self.assumption_label.setText(
            "Dielectric: scan value accepted as assumed"
            if self.options.accept_scan_dielectric
            else "Dielectric: reflection calibration required"
        )
        self.run_action.setEnabled(True)
        self.statusBar().showMessage("Project created. Run the automatic analysis.")

    @Slot()
    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open GPR project", "", "GPR project (*.gprproj)"
        )
        if not path:
            return
        store = ProjectStore(path)
        try:
            files = store.file_paths()
            road_path = files["road"]
            if not road_path.is_file():
                raise FileNotFoundError(f"Road source is no longer available: {road_path}")
            plate_path = files.get("plate")
            if plate_path is not None and not plate_path.is_file():
                raise FileNotFoundError(f"Plate source is no longer available: {plate_path}")
            parameters = store.latest_parameters()
        except Exception as exc:
            QMessageBox.warning(self, "Project could not be opened", str(exc))
            return
        self.project_store = store
        self.road = AcquisitionFileSet(road_path)
        self.plate = AcquisitionFileSet(plate_path) if plate_path else None
        self.options = AnalysisOptions(
            stack_size=int(parameters.get("stack_size", 10)),
            report_interval_m=float(parameters.get("report_interval_m", 5.0)),
            accept_scan_dielectric=bool(parameters.get("accept_scan_dielectric", False)),
        )
        self.source_label.setText(f"{road_path.name}\n{road_path.parent}")
        self.assumption_label.setText(
            "Dielectric: scan value accepted as assumed"
            if self.options.accept_scan_dielectric
            else "Dielectric: reflection calibration required"
        )
        self.run_action.setEnabled(True)
        self.statusBar().showMessage(
            "Project opened. Saved anchors will be applied on the next analysis."
        )

    @Slot()
    def undo_anchor(self):
        if not self.project_store:
            return
        removed = self.project_store.remove_last_anchor()
        if removed is None:
            self.statusBar().showMessage("There is no saved anchor to undo.")
            return
        layer, chainage, _ = removed
        self.statusBar().showMessage(
            f"Removed the last layer {layer} anchor at {chainage:.2f} m; retracking…"
        )
        self._local_retrack(chainage)

    @Slot()
    def run_analysis(self):
        if not self.road or self.worker:
            return
        layers = LayerSpec.defaults()
        for layer in layers:
            layer.analysis_enabled = self.layer_checks[layer.order].isChecked()
            layer.audit_enabled = layer.analysis_enabled
        self.options.layer_specs = layers
        if self.project_store:
            self.options.anchors = self.project_store.anchors()
        self.worker = AnalysisWorker(self.road, self.plate, self.options)
        self.worker.signals.progress.connect(self._progress)
        self.worker.signals.result.connect(self._analysis_complete)
        self.worker.signals.error.connect(self._analysis_error)
        self.worker.signals.cancelled.connect(self._analysis_cancelled)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.run_action.setEnabled(False)
        self.cancel_action.setEnabled(True)
        self.thread_pool.start(self.worker)

    @Slot()
    def cancel_analysis(self):
        if self.worker:
            self.worker.cancel()
            self.statusBar().showMessage("Cancelling after the current processing block…")

    @Slot(int, str)
    def _progress(self, value, message):
        self.progress.setValue(value)
        self.progress.setFormat(f"{message}  %p%")
        self.statusBar().showMessage(message)

    @Slot(object)
    def _analysis_complete(self, result):
        self.result = result
        self.worker = None
        self.progress.setVisible(False)
        self.run_action.setEnabled(True)
        self.cancel_action.setEnabled(False)
        self.export_action.setEnabled(True)
        if self.design_segments:
            self.result.thickness = compare_with_design(self.result.thickness, self.design_segments)
        if self.reference_points:
            evaluate_manual_reference(self.result, self.reference_points)
        self.radar.set_result(result)
        self._populate_results()
        self._populate_issues()
        self._show_method()
        if self.project_store:
            self.project_store.save_analysis(
                result, str(self.plate.dzt_path) if self.plate else None
            )
        self.statusBar().showMessage(
            f"Analysis complete: {len(result.review_issues)} exception groups require review."
        )

    @Slot(str)
    def _analysis_error(self, details):
        self.worker = None
        self.progress.setVisible(False)
        self.run_action.setEnabled(bool(self.road))
        self.cancel_action.setEnabled(False)
        QMessageBox.critical(self, "Analysis failed", details)
        self.statusBar().showMessage("Analysis failed. Review the error and source files.")

    @Slot()
    def _analysis_cancelled(self):
        self.worker = None
        self.progress.setVisible(False)
        self.run_action.setEnabled(bool(self.road))
        self.cancel_action.setEnabled(False)
        self.statusBar().showMessage("Analysis cancelled; previous accepted results are unchanged.")

    def _populate_results(self):
        self.thickness_table.setRowCount(0)
        if not self.result:
            return
        for item in self.result.thickness:
            row = self.thickness_table.rowCount()
            self.thickness_table.insertRow(row)
            values = [
                f"{item.chainage_m:.1f}",
                item.layer_name,
                "—" if item.thickness_mm is None else f"{item.thickness_mm:.1f}",
                f"{item.confidence:.0%}",
                str(item.status),
            ]
            for column, value in enumerate(values):
                self.thickness_table.setItem(row, column, QTableWidgetItem(value))
        self.thickness_table.resizeColumnsToContents()

    def _populate_issues(self):
        self.issue_list.clear()
        if not self.result:
            return
        for issue in self.result.review_issues:
            item = QListWidgetItem(
                f"{issue.layer_name}\n{issue.start_chainage_m:.1f}–{issue.end_chainage_m:.1f} m"
            )
            item.setData(Qt.ItemDataRole.UserRole, issue)
            item.setToolTip("; ".join(issue.reasons))
            self.issue_list.addItem(item)
        if not self.result.review_issues:
            item = QListWidgetItem(
                "No exception groups\nAll tracked paths meet the current confidence rule."
            )
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.issue_list.addItem(item)

    def _show_method(self):
        if not self.result:
            return
        notes = "\n• ".join(self.result.diagnostics.messages)
        preprocessing = "\n  - ".join(self.result.diagnostics.preprocessing_steps)
        reference_note = ""
        if self.result.reference_diagnostics:
            diagnostic = [
                item
                for item in self.result.reference_diagnostics
                if not item.interpolated_reference and item.absolute_error_mm is not None
            ]
            passing = sum(item.within_release_target is True for item in diagnostic)
            reference_note = (
                f"\n\nManual reference diagnostic: {passing}/{len(diagnostic)} "
                "non-interpolated points meet the layer-specific release target. "
                "Reference values do not influence picking."
            )
        self.method_text.setText(
            f"Plate valid for dielectric: {self.result.diagnostics.valid_for_dielectric}\n"
            f"Gain compatible: {self.result.diagnostics.gain_compatible}\n"
            f"Reference surface sample: {self.result.reference_surface_sample}\n\n"
            f"Automatic preprocessing:\n  - {preprocessing}\n\n"
            f"• {notes}{reference_note}"
        )
        self.calibration_label.setText(
            "Calibration passed for amplitude dielectric."
            if self.result.diagnostics.valid_for_dielectric
            else "Amplitude dielectric unavailable; inspect Method for the fail-closed reason."
        )

    @Slot(QListWidgetItem)
    def focus_issue(self, item):
        issue = item.data(Qt.ItemDataRole.UserRole)
        if issue:
            self.radar.set_active_layer(issue.layer_order)
            self.radar.focus_range(issue.start_chainage_m, issue.end_chainage_m)

    @Slot(int, float, float)
    def add_anchor(self, layer_order, chainage_m, sample_index):
        if not self.project_store:
            return
        self.project_store.add_anchor(layer_order, chainage_m, sample_index)
        self.statusBar().showMessage(
            f"Anchor saved for layer {layer_order} at {chainage_m:.2f} m; retracking…"
        )
        self._local_retrack(chainage_m)

    def _local_retrack(self, chainage_m: float):
        if not self.result or not self.project_store:
            self.run_analysis()
            return
        self.options.anchors = self.project_store.anchors()
        retrack_segment(
            self.result,
            self.options,
            max(0.0, chainage_m - 25.0),
            chainage_m + 25.0,
        )
        if self.design_segments:
            self.result.thickness = compare_with_design(self.result.thickness, self.design_segments)
        if self.reference_points:
            evaluate_manual_reference(self.result, self.reference_points)
        self.radar.set_result(self.result)
        self._populate_results()
        self._populate_issues()
        self._show_method()
        self.project_store.save_analysis(
            self.result, str(self.plate.dzt_path) if self.plate else None
        )
        self.statusBar().showMessage(
            f"Re-tracked only {max(0.0, chainage_m - 25.0):.1f}–"
            f"{chainage_m + 25.0:.1f} m; outside picks were preserved."
        )

    @Slot()
    def import_design(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import design schedule", "", "Design schedule (*.csv *.xlsx *.xlsm)"
        )
        if not path:
            return
        try:
            self.design_segments = read_design_schedule(path)
        except Exception as exc:
            QMessageBox.warning(self, "Design import failed", str(exc))
            return
        if self.project_store:
            self.project_store.add_design_segments(self.design_segments)
        if self.result:
            self.result.thickness = compare_with_design(self.result.thickness, self.design_segments)
            self._populate_results()
        self.statusBar().showMessage(
            f"Imported {len(self.design_segments)} design segments. Measurements were not changed."
        )

    @Slot()
    def import_reference(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import manual reference", "", "Reference workbook (*.xlsx *.xlsm)"
        )
        if not path:
            return
        try:
            self.reference_points = read_manual_reference(path)
            if self.result:
                evaluate_manual_reference(self.result, self.reference_points)
                self._show_method()
        except Exception as exc:
            QMessageBox.warning(self, "Reference import failed", str(exc))
            return
        excluded = sum(item.interpolated for item in self.reference_points)
        self.statusBar().showMessage(
            f"Imported {len(self.reference_points)} reference values; {excluded} "
            "interpolated values are flagged and excluded from release diagnostics."
        )

    @Slot()
    def export_package(self):
        if not self.result:
            return
        directory = QFileDialog.getExistingDirectory(self, "Choose audit export folder")
        if not directory:
            return
        try:
            package = export_audit_package(self.result, directory)
            if self.project_store:
                self.project_store.record_export(package, self.result.manifest())
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        QMessageBox.information(self, "Audit package exported", f"Created:\n{package}")
        self.statusBar().showMessage(f"Audit package exported to {package}")
