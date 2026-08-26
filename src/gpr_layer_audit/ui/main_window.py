from __future__ import annotations

import traceback
from pathlib import Path
from threading import Event
from uuid import uuid4

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

from gpr_layer_audit.catalog import calibration_candidates_for, discover_survey_catalog
from gpr_layer_audit.design import compare_with_design, read_design_schedule
from gpr_layer_audit.export import export_audit_package
from gpr_layer_audit.models import (
    AcquisitionFileSet,
    AnalysisResult,
    LayerSpec,
    SeedStation,
    SurveyCatalog,
    VisibilityState,
)
from gpr_layer_audit.processing import (
    AnalysisCancelled,
    AnalysisOptions,
    analyze_acquisition,
    resolve_review_issue,
    retrack_segment,
)
from gpr_layer_audit.project import ProjectStore
from gpr_layer_audit.reference import evaluate_manual_reference, read_manual_reference
from gpr_layer_audit.seeds import MAX_SEED_STATIONS

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
                progress=lambda value, message: self.signals.progress.emit(value, message),
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


class CatalogDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Catalog GSSI directory")
        self.setMinimumWidth(780)
        self.catalog: SurveyCatalog | None = None
        self.root_edit = QLineEdit()
        self.road_combo = QComboBox()
        self.plate_combo = QComboBox()
        self.reference_combo = QComboBox()
        self.design_combo = QComboBox()
        self.project_edit = QLineEdit()
        self.accept_dielectric = QCheckBox("Use scan dielectric as an explicitly assumed value")
        self.interval = QComboBox()
        self.interval.addItems(["1", "5", "10"])
        self.interval.setCurrentText("5")
        root_row = QWidget()
        root_layout = QHBoxLayout(root_row)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(self.root_edit, 1)
        browse = QPushButton("Choose directory…")
        browse.clicked.connect(self._choose_root)
        root_layout.addWidget(browse)
        scan = QPushButton("Scan")
        scan.clicked.connect(self.scan)
        root_layout.addWidget(scan)
        project_row = QWidget()
        project_layout = QHBoxLayout(project_row)
        project_layout.setContentsMargins(0, 0, 0, 0)
        project_layout.addWidget(self.project_edit, 1)
        project_button = QPushButton("Save as…")
        project_button.clicked.connect(self._choose_project)
        project_layout.addWidget(project_button)
        form = QFormLayout()
        form.addRow("Survey directory", root_row)
        form.addRow("Road acquisition", self.road_combo)
        form.addRow("Proposed calibration", self.plate_combo)
        form.addRow("Reference workbook", self.reference_combo)
        form.addRow("Design schedule", self.design_combo)
        form.addRow("Project record", project_row)
        form.addRow("Report interval (m)", self.interval)
        form.addRow("Dielectric policy", self.accept_dielectric)
        self.summary = QLabel(
            "Choose a directory. The catalog will classify road and plate scans, attach "
            "matching GPS/metadata, and show any reference or design files for confirmation."
        )
        self.summary.setWordWrap(True)
        self.summary.setObjectName("secondaryText")
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.summary)
        layout.addWidget(buttons)
        self.road_combo.currentIndexChanged.connect(self._populate_calibrations)

    def _choose_root(self):
        path = QFileDialog.getExistingDirectory(self, "Select directory containing GSSI surveys")
        if path:
            self.root_edit.setText(path)
            self.scan()

    def _choose_project(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Create prototype project", "", "GPR project (*.gprproj)"
        )
        if path:
            self.project_edit.setText(
                path if path.casefold().endswith(".gprproj") else path + ".gprproj"
            )

    @Slot()
    def scan(self):
        try:
            self.catalog = discover_survey_catalog(self.root_edit.text())
        except Exception as exc:
            QMessageBox.warning(self, "Directory could not be cataloged", str(exc))
            return
        self.road_combo.blockSignals(True)
        self.road_combo.clear()
        for survey in self.catalog.roads:
            distance = (
                survey.trace_count / survey.scans_per_meter
                if survey.scans_per_meter > 0
                else 0.0
            )
            self.road_combo.addItem(
                f"{survey.survey_id}  ·  {distance / 1000:.2f} km",
                survey.survey_id,
            )
        self.road_combo.blockSignals(False)
        self.reference_combo.clear()
        self.reference_combo.addItem("None", None)
        for path in self.catalog.reference_files:
            self.reference_combo.addItem(str(path.relative_to(self.catalog.root)), str(path))
        self.design_combo.clear()
        self.design_combo.addItem("None", None)
        for path in self.catalog.design_files:
            self.design_combo.addItem(str(path.relative_to(self.catalog.root)), str(path))
        self._populate_calibrations()
        self.summary.setText(
            f"Found {len(self.catalog.roads)} road acquisition(s), "
            f"{len(self.catalog.calibrations)} calibration acquisition(s), and "
            f"{len(self.catalog.reference_files)} candidate reference file(s). "
            f"{len(self.catalog.orphan_files)} companion file(s) are unmatched. "
            "Review the proposed pairing before creating the project."
        )
        if self.catalog.roads and not self.project_edit.text():
            road = self.catalog.roads[self.road_combo.currentIndex()]
            default_name = self.catalog.root / f"{Path(road.survey_id).stem}.gprproj"
            self.project_edit.setText(str(default_name))

    @Slot()
    def _populate_calibrations(self):
        self.plate_combo.clear()
        self.plate_combo.addItem("No calibration", None)
        if self.catalog is None or self.road_combo.currentIndex() < 0:
            return
        road_id = self.road_combo.currentData()
        for candidate in calibration_candidates_for(self.catalog, road_id):
            problems = ", ".join(candidate.problems) or "fully compatible"
            self.plate_combo.addItem(
                f"{candidate.calibration_survey_id}  ·  score {candidate.compatibility_score:.2f} "
                f"· {problems}",
                candidate.calibration_survey_id,
            )
        candidates = calibration_candidates_for(self.catalog, road_id)
        if self.plate_combo.count() > 1 and candidates and candidates[0].gain_compatible:
            self.plate_combo.setCurrentIndex(1)

    def selected_road(self):
        if self.catalog is None:
            return None
        survey_id = self.road_combo.currentData()
        return next((item for item in self.catalog.roads if item.survey_id == survey_id), None)

    def selected_plate(self):
        if self.catalog is None or not self.plate_combo.currentData():
            return None
        survey_id = self.plate_combo.currentData()
        return next(
            (item for item in self.catalog.calibrations if item.survey_id == survey_id), None
        )

    def accept(self):
        if self.catalog is None or self.selected_road() is None:
            QMessageBox.warning(self, "Road acquisition required", "Scan and select a road survey.")
            return
        project_path = Path(self.project_edit.text())
        if not self.project_edit.text():
            QMessageBox.warning(self, "Project path required", "Choose a project record path.")
            return
        if project_path.exists():
            QMessageBox.warning(
                self,
                "Project already exists",
                "Choose a new project filename; existing project data will not be overwritten.",
            )
            return
        super().accept()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GPR Layer Audit · Seeded Research Prototype")
        self.resize(1580, 940)
        self.setMinimumSize(1220, 760)
        self.thread_pool = QThreadPool.globalInstance()
        self.worker: AnalysisWorker | None = None
        self.result: AnalysisResult | None = None
        self.project_store: ProjectStore | None = None
        self.catalog: SurveyCatalog | None = None
        self.road: AcquisitionFileSet | None = None
        self.plate: AcquisitionFileSet | None = None
        self.options = AnalysisOptions()
        self.design_segments = []
        self.reference_points = []
        self._build_toolbar()
        self._build_workspace()
        self.statusBar().showMessage("Catalog a survey directory to begin.")

    def _build_toolbar(self):
        toolbar = QToolBar("Analysis")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        new_action = QAction("Catalog directory", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self.new_project)
        toolbar.addAction(new_action)
        open_action = QAction("Open project", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_project)
        toolbar.addAction(open_action)
        self.run_action = QAction("Build preview", self)
        self.run_action.setShortcut("Ctrl+R")
        self.run_action.triggered.connect(self.run_analysis)
        self.run_action.setEnabled(False)
        toolbar.addAction(self.run_action)
        self.cancel_action = QAction("Cancel", self)
        self.cancel_action.triggered.connect(self.cancel_analysis)
        self.cancel_action.setEnabled(False)
        toolbar.addAction(self.cancel_action)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel("View"))
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Raw", "Clean"])
        self.view_combo.setCurrentText("Clean")
        self.view_combo.currentTextChanged.connect(self._change_view)
        toolbar.addWidget(self.view_combo)
        undo_action = QAction("Undo station", self)
        undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        undo_action.triggered.connect(self.undo_seed_station)
        toolbar.addAction(undo_action)
        toolbar.addSeparator()
        design_action = QAction("Import design", self)
        design_action.triggered.connect(self.import_design)
        toolbar.addAction(design_action)
        reference_action = QAction("Import reference", self)
        reference_action.triggered.connect(self.import_reference)
        toolbar.addAction(reference_action)
        self.export_action = QAction("Export evidence package", self)
        self.export_action.setShortcut("Ctrl+E")
        self.export_action.triggered.connect(self.export_package)
        self.export_action.setEnabled(False)
        toolbar.addAction(self.export_action)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)
        self.progress = QProgressBar()
        self.progress.setFixedWidth(280)
        self.progress.setVisible(False)
        toolbar.addWidget(self.progress)

    def _build_workspace(self):
        self.radar = RadarView()
        self.radar.anchorRequested.connect(self.add_seed_pick)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._project_panel())
        splitter.addWidget(self.radar)
        splitter.addWidget(self._review_panel())
        splitter.setSizes([285, 980, 330])
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

    def _project_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        title = QLabel("PROJECT & SEEDS")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        self.source_label = QLabel("No survey cataloged")
        self.source_label.setWordWrap(True)
        layout.addWidget(self.source_label)
        layers = QGroupBox("Interfaces to track")
        layer_layout = QVBoxLayout(layers)
        self.layer_checks = {}
        for layer in LayerSpec.defaults():
            check = QCheckBox(layer.name)
            check.setChecked(True)
            check.setStyleSheet(f"color: {LAYER_COLOURS[layer.order]};")
            self.layer_checks[layer.order] = check
            layer_layout.addWidget(check)
        layer_layout.addWidget(QLabel("Interface currently being seeded"))
        self.active_layer_combo = QComboBox()
        for layer in LayerSpec.defaults():
            self.active_layer_combo.addItem(layer.name, layer.order)
        self.active_layer_combo.currentIndexChanged.connect(
            lambda: self.radar.set_active_layer(int(self.active_layer_combo.currentData()))
        )
        layer_layout.addWidget(self.active_layer_combo)
        layout.addWidget(layers)
        seeds = QGroupBox("Guided seed stations")
        seed_layout = QVBoxLayout(seeds)
        self.seed_help = QLabel(
            "Build a preview, focus each suggested station, then Ctrl+click all visible interfaces."
        )
        self.seed_help.setWordWrap(True)
        seed_layout.addWidget(self.seed_help)
        self.seed_combo = QComboBox()
        self.seed_combo.currentIndexChanged.connect(self.focus_selected_seed)
        seed_layout.addWidget(self.seed_combo)
        self.seed_list = QListWidget()
        self.seed_list.setMaximumHeight(145)
        seed_layout.addWidget(self.seed_list)
        visibility_row = QHBoxLayout()
        not_visible = QPushButton("Not visible")
        not_visible.clicked.connect(lambda: self.mark_seed_visibility(VisibilityState.NOT_VISIBLE))
        absent = QPushButton("Absent")
        absent.clicked.connect(lambda: self.mark_seed_visibility(VisibilityState.ABSENT))
        visibility_row.addWidget(not_visible)
        visibility_row.addWidget(absent)
        seed_layout.addLayout(visibility_row)
        self.track_button = QPushButton("Track from completed seeds")
        self.track_button.setProperty("primary", True)
        self.track_button.setEnabled(False)
        self.track_button.clicked.connect(self.run_analysis)
        seed_layout.addWidget(self.track_button)
        layout.addWidget(seeds)
        self.assumption_label = QLabel(
            "Dielectric unresolved until calibration or explicit assumption."
        )
        self.assumption_label.setWordWrap(True)
        self.assumption_label.setObjectName("warningText")
        layout.addWidget(self.assumption_label)
        self.calibration_label = QLabel("Calibration has not run.")
        self.calibration_label.setWordWrap(True)
        layout.addWidget(self.calibration_label)
        layout.addStretch(1)
        return panel

    def _review_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        title = QLabel("PRIORITIZED REVIEW")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        self.issue_list = QListWidget()
        self.issue_list.itemActivated.connect(self.focus_issue)
        layout.addWidget(self.issue_list, 1)
        row1 = QHBoxLayout()
        accept = QPushButton("Accept evidence")
        accept.clicked.connect(lambda: self.resolve_selected_issue("accept"))
        correct = QPushButton("Correct point")
        correct.clicked.connect(self.prepare_issue_correction)
        row1.addWidget(accept)
        row1.addWidget(correct)
        layout.addLayout(row1)
        row2 = QHBoxLayout()
        invisible = QPushButton("Not visible")
        invisible.clicked.connect(lambda: self.resolve_selected_issue("not_visible"))
        absent = QPushButton("Layer absent")
        absent.clicked.connect(lambda: self.resolve_selected_issue("absent"))
        row2.addWidget(invisible)
        row2.addWidget(absent)
        layout.addLayout(row2)
        structural = QPushButton("Add structural break and retrack")
        structural.clicked.connect(self.add_structural_break)
        layout.addWidget(structural)
        self.tabs = QTabWidget()
        self.thickness_table = QTableWidget(0, 5)
        self.thickness_table.setHorizontalHeaderLabels(
            ["Chainage", "Layer", "mm", "Confidence", "Status"]
        )
        self.thickness_table.setAlternatingRowColors(True)
        self.tabs.addTab(self.thickness_table, "Results")
        self.method_text = QLabel("Run the preview to inspect processing provenance.")
        self.method_text.setWordWrap(True)
        self.method_text.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.tabs.addTab(self.method_text, "Method")
        layout.addWidget(self.tabs, 1)
        return panel

    @Slot()
    def new_project(self):
        dialog = CatalogDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.catalog = dialog.catalog
        road = dialog.selected_road()
        plate = dialog.selected_plate()
        if road is None:
            return
        self.road = AcquisitionFileSet(road.dzt_path)
        self.plate = AcquisitionFileSet(plate.dzt_path) if plate else None
        self.options = AnalysisOptions(
            survey_id=road.survey_id,
            report_interval_m=float(dialog.interval.currentText()),
            accept_scan_dielectric=dialog.accept_dielectric.isChecked(),
        )
        self.design_segments = []
        self.reference_points = []
        if dialog.design_combo.currentData():
            self.design_segments = read_design_schedule(dialog.design_combo.currentData())
            self.options.design_segments = self.design_segments
        if dialog.reference_combo.currentData():
            self.reference_points = read_manual_reference(dialog.reference_combo.currentData())
        self.project_store = ProjectStore.create(dialog.project_edit.text(), road.survey_id)
        self.project_store.set_meta("catalog_root", str(self.catalog.root))
        self.project_store.set_meta("survey_id", road.survey_id)
        self.project_store.set_file("road", road.dzt_path)
        if plate:
            self.project_store.set_file("plate", plate.dzt_path)
        self.project_store.set_layers(self.options.layer_specs)
        if self.design_segments:
            self.project_store.add_design_segments(self.design_segments)
        self.result = None
        self.source_label.setText(f"{road.survey_id}\n{road.trace_count:,} traces · {road.antenna}")
        self.assumption_label.setText(
            "Scan εr accepted as an assumption"
            if self.options.accept_scan_dielectric
            else "Physical thickness requires valid calibration or analyst εr"
        )
        self.run_action.setText("Build preview")
        self.run_action.setEnabled(True)
        self.statusBar().showMessage("Catalog confirmed. Build the processing preview.")

    @Slot()
    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open prototype project", "", "GPR project (*.gprproj)"
        )
        if not path:
            return
        store = ProjectStore(path)
        try:
            store.validate()
            files = store.file_paths()
            road_path = files["road"]
            plate_path = files.get("plate")
            parameters = store.latest_parameters()
            seeds = store.seed_stations()
            design_segments = store.design_segments()
            if not road_path.is_file() or (plate_path and not plate_path.is_file()):
                raise FileNotFoundError("One or more source acquisitions are no longer available.")
        except Exception as exc:
            QMessageBox.warning(self, "Project could not be opened", str(exc))
            return
        self.project_store = store
        self.road = AcquisitionFileSet(road_path)
        self.plate = AcquisitionFileSet(plate_path) if plate_path else None
        self.options = AnalysisOptions(
            survey_id=store.get_meta("survey_id") or road_path.stem,
            stack_size=int(parameters.get("stack_size", 0)),
            report_interval_m=float(parameters.get("report_interval_m", 5.0)),
            accept_scan_dielectric=bool(parameters.get("accept_scan_dielectric", False)),
            seed_stations=seeds,
            design_segments=design_segments,
            structural_breaks_m=list(parameters.get("structural_breaks_m", [])),
        )
        self.design_segments = design_segments
        self.source_label.setText(f"{road_path.name}\n{road_path.parent}")
        self.run_action.setText("Track from saved seeds" if seeds else "Build preview")
        self.run_action.setEnabled(True)
        self.statusBar().showMessage("Project opened. Run tracking to reconstruct the workbench.")

    @Slot()
    def run_analysis(self):
        if not self.road or self.worker:
            return
        layers = LayerSpec.defaults()
        for layer in layers:
            layer.analysis_enabled = self.layer_checks[layer.order].isChecked()
            layer.audit_enabled = layer.analysis_enabled
        self.options.layer_specs = layers
        self.options.design_segments = self.design_segments
        self.worker = AnalysisWorker(self.road, self.plate, self.options)
        self.worker.signals.progress.connect(self._progress)
        self.worker.signals.result.connect(self._analysis_complete)
        self.worker.signals.error.connect(self._analysis_error)
        self.worker.signals.cancelled.connect(self._analysis_cancelled)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.run_action.setEnabled(False)
        self.track_button.setEnabled(False)
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
        self._update_view_modes()
        self._populate_seed_controls()
        self._populate_results()
        self._populate_issues()
        self._show_method()
        if self.project_store:
            self.project_store.save_analysis(
                result, str(self.plate.dzt_path) if self.plate else None
            )
        completed = self._completed_initial_stations()
        self.run_action.setText("Re-run seeded tracker" if completed else "Re-run preview")
        if completed >= 3:
            message = f"Seeded tracking complete; {len(result.review_issues)} regions need review."
        else:
            message = (
                f"Preview ready. Complete three suggested stations ({completed}/3) before "
                "accepting automatic paths."
            )
        self.statusBar().showMessage(message)

    @Slot(str)
    def _analysis_error(self, details):
        self.worker = None
        self.progress.setVisible(False)
        self.run_action.setEnabled(True)
        self.cancel_action.setEnabled(False)
        QMessageBox.critical(self, "Analysis failed", details)
        self.statusBar().showMessage("Analysis failed; the source project was not changed.")

    @Slot()
    def _analysis_cancelled(self):
        self.worker = None
        self.progress.setVisible(False)
        self.run_action.setEnabled(True)
        self.cancel_action.setEnabled(False)
        self.statusBar().showMessage("Analysis cancelled.")

    def _update_view_modes(self):
        current = self.view_combo.currentText()
        self.view_combo.blockSignals(True)
        self.view_combo.clear()
        self.view_combo.addItems(self.radar.available_views())
        mode = current if current in self.radar.available_views() else "Clean"
        self.view_combo.setCurrentText(mode)
        self.view_combo.blockSignals(False)
        self.radar.set_view_mode(self.view_combo.currentText())

    @Slot(str)
    def _change_view(self, mode):
        if mode:
            self.radar.set_view_mode(mode)

    def _station_complete(self, station: SeedStation) -> bool:
        enabled = [order for order, check in self.layer_checks.items() if check.isChecked()]
        return all(
            order in station.samples
            or station.visibility.get(order)
            in {VisibilityState.NOT_VISIBLE, VisibilityState.ABSENT}
            for order in enabled
        )

    def _completed_initial_stations(self) -> int:
        return sum(
            station.role == "initial" and self._station_complete(station)
            for station in self.options.seed_stations
        )

    def _populate_seed_controls(self):
        self.seed_combo.blockSignals(True)
        selected = self.seed_combo.currentData()
        self.seed_combo.clear()
        proposed = self.result.proposed_seed_chainages if self.result else []
        for index, chainage in enumerate(proposed, 1):
            self.seed_combo.addItem(f"Suggested {index} · {chainage:.1f} m", float(chainage))
        if self.result and self._completed_initial_stations() >= 3:
            self.seed_combo.addItem("Correction at clicked chainage", None)
        if selected is not None:
            nearest = min(
                range(self.seed_combo.count()),
                key=lambda index: abs(
                    float(self.seed_combo.itemData(index)) - float(selected)
                    if self.seed_combo.itemData(index) is not None
                    else 1e12
                ),
                default=0,
            )
            self.seed_combo.setCurrentIndex(nearest)
        elif self.seed_combo.count():
            self.seed_combo.setCurrentIndex(0)
        self.seed_combo.blockSignals(False)
        self.seed_list.clear()
        names = {item.order: item.name for item in LayerSpec.defaults()}
        for station in sorted(self.options.seed_stations, key=lambda item: item.chainage_m):
            states = []
            for order in sorted(names):
                if order in station.samples:
                    state = "picked"
                else:
                    state = str(station.visibility.get(order, "missing"))
                states.append(f"{names[order]}: {state}")
            item = QListWidgetItem(
                f"{station.chainage_m:.1f} m · {station.role}\n" + " · ".join(states)
            )
            self.seed_list.addItem(item)
        complete = self._completed_initial_stations()
        self.seed_help.setText(
            f"{complete}/3 initial stations complete · {len(self.options.seed_stations)}/"
            f"{MAX_SEED_STATIONS} total. Ctrl+click the selected interface in each window."
        )
        self.track_button.setEnabled(complete >= 3 and not self.worker)
        if self.result:
            self.result.seed_stations = list(self.options.seed_stations)
            self.radar.refresh_guides()
            if not selected and proposed:
                self.focus_selected_seed()

    @Slot()
    def focus_selected_seed(self):
        if not self.result:
            return
        chainage = self.seed_combo.currentData()
        if chainage is not None:
            self.radar.focus_chainage(float(chainage), 24.0)

    def _selected_or_clicked_station(self, clicked_chainage: float) -> SeedStation | None:
        selected = self.seed_combo.currentData()
        target = float(selected) if selected is not None else float(clicked_chainage)
        existing = next(
            (
                item
                for item in self.options.seed_stations
                if abs(item.chainage_m - target) <= max(0.5, self.result.stack_size / 20)
            ),
            None,
        )
        if existing:
            return existing
        if len(self.options.seed_stations) >= MAX_SEED_STATIONS:
            QMessageBox.warning(
                self,
                "Five-station limit reached",
                "Remove a station before adding another correction.",
            )
            return None
        station = SeedStation(
            station_id=str(uuid4()),
            chainage_m=target,
            role="initial" if selected is not None else "correction",
        )
        self.options.seed_stations.append(station)
        return station

    @Slot(int, float, float)
    def add_seed_pick(self, layer_order, chainage_m, sample_index):
        if not self.result or not self.project_store:
            return
        station = self._selected_or_clicked_station(chainage_m)
        if station is None:
            return
        station.samples[layer_order] = float(sample_index)
        station.visibility[layer_order] = VisibilityState.VISIBLE
        self.project_store.save_seed_station(station)
        self.options.seed_stations.sort(key=lambda item: item.chainage_m)
        correction = station.role == "correction"
        self._populate_seed_controls()
        if correction:
            self._local_retrack(station.chainage_m)
        else:
            self.statusBar().showMessage(
                f"Saved {LayerSpec.defaults()[layer_order - 1].name} at "
                f"{station.chainage_m:.1f} m. Complete all enabled interfaces."
            )

    def mark_seed_visibility(self, visibility: VisibilityState):
        if not self.result or not self.project_store:
            return
        chainage = self.seed_combo.currentData()
        if chainage is None:
            self.statusBar().showMessage("Select a suggested station before marking visibility.")
            return
        station = self._selected_or_clicked_station(float(chainage))
        if station is None:
            return
        order = int(self.active_layer_combo.currentData())
        station.samples.pop(order, None)
        station.visibility[order] = visibility
        self.project_store.save_seed_station(station)
        self._populate_seed_controls()

    @Slot()
    def undo_seed_station(self):
        if not self.project_store:
            return
        removed = self.project_store.remove_last_seed_station()
        if removed is None:
            self.statusBar().showMessage("There is no seed station to remove.")
            return
        self.options.seed_stations = [
            item for item in self.options.seed_stations if item.station_id != removed.station_id
        ]
        self._populate_seed_controls()
        self.statusBar().showMessage(f"Removed station at {removed.chainage_m:.1f} m.")

    def _local_retrack(self, chainage_m: float):
        if not self.result:
            return
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
        if self.project_store:
            self.project_store.save_analysis(
                self.result, str(self.plate.dzt_path) if self.plate else None
            )
        self.statusBar().showMessage(
            f"Correction applied around {chainage_m:.1f} m; outside picks were preserved."
        )

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
                f"Priority {issue.priority:.0%} · {issue.layer_name}\n"
                f"{issue.start_chainage_m:.1f}–{issue.end_chainage_m:.1f} m"
            )
            item.setData(Qt.ItemDataRole.UserRole, issue)
            item.setToolTip("; ".join(issue.reasons))
            self.issue_list.addItem(item)
        if not self.result.review_issues:
            item = QListWidgetItem("Review queue complete\nNo unresolved automatic regions remain.")
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
                f"\n\nBlocked reference diagnostic input: {passing}/{len(diagnostic)} "
                "non-interpolated values meet layer targets."
            )
        self.method_text.setText(
            f"Stacking: {self.result.stack_size} traces per coarse bin\n"
            f"Plate valid for dielectric: {self.result.diagnostics.valid_for_dielectric}\n"
            f"Gain compatible: {self.result.diagnostics.gain_compatible}\n"
            f"Reference surface sample: {self.result.reference_surface_sample}\n"
            f"Seed stations: {len(self.result.seed_stations)}\n"
            f"Design-guided dual pass: {bool(self.design_segments)}\n\n"
            f"Processing branches:\n  - {preprocessing}\n\n• {notes}{reference_note}"
        )
        self.calibration_label.setText(
            "Calibration passed for amplitude dielectric."
            if self.result.diagnostics.valid_for_dielectric
            else (
                "Amplitude dielectric unavailable; interface TWTT remains independently "
                "reviewable."
            )
        )

    def _selected_issue(self):
        item = self.issue_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    @Slot(QListWidgetItem)
    def focus_issue(self, item):
        issue = item.data(Qt.ItemDataRole.UserRole)
        if issue:
            self.radar.set_active_layer(issue.layer_order)
            index = self.active_layer_combo.findData(issue.layer_order)
            self.active_layer_combo.setCurrentIndex(index)
            self.radar.focus_range(issue.start_chainage_m, issue.end_chainage_m)

    def resolve_selected_issue(self, action: str):
        if not self.result:
            return
        issue = self._selected_issue()
        if issue is None:
            self.statusBar().showMessage("Select a review region first.")
            return
        resolve_review_issue(self.result, self.options, issue.issue_id, action)
        if self.project_store:
            self.project_store.record_review_event(
                action,
                issue_id=issue.issue_id,
                layer_order=issue.layer_order,
                start_chainage_m=issue.start_chainage_m,
                end_chainage_m=issue.end_chainage_m,
            )
        self.radar.set_result(self.result)
        self._populate_results()
        self._populate_issues()
        self.statusBar().showMessage(f"Recorded review decision: {action.replace('_', ' ')}.")

    def prepare_issue_correction(self):
        issue = self._selected_issue()
        if issue is None:
            self.statusBar().showMessage("Select a review region first.")
            return
        index = self.active_layer_combo.findData(issue.layer_order)
        self.active_layer_combo.setCurrentIndex(index)
        correction_index = self.seed_combo.findData(None)
        if correction_index >= 0:
            self.seed_combo.setCurrentIndex(correction_index)
        target = issue.suggested_chainage_m or (issue.start_chainage_m + issue.end_chainage_m) / 2
        self.radar.focus_chainage(target, 20.0)
        self.statusBar().showMessage(
            f"Ctrl+click the corrected {issue.layer_name} interface near {target:.1f} m."
        )

    def add_structural_break(self):
        issue = self._selected_issue()
        if issue is None:
            self.statusBar().showMessage("Select a review region first.")
            return
        target = issue.suggested_chainage_m or (issue.start_chainage_m + issue.end_chainage_m) / 2
        if not any(abs(value - target) < 0.1 for value in self.options.structural_breaks_m):
            self.options.structural_breaks_m.append(target)
        if self.project_store:
            self.project_store.record_review_event(
                "structural_break",
                issue_id=issue.issue_id,
                layer_order=issue.layer_order,
                start_chainage_m=target,
                end_chainage_m=target,
            )
        self.statusBar().showMessage(f"Structural break added at {target:.1f} m; retracking…")
        self.run_analysis()

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
        self.options.design_segments = self.design_segments
        if self.project_store:
            self.project_store.add_design_segments(self.design_segments)
        self.statusBar().showMessage(
            f"Imported {len(self.design_segments)} design segments; running the recorded dual pass."
        )
        if self.result:
            self.run_analysis()

    @Slot()
    def import_reference(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import analysis reference", "", "Reference workbook (*.xlsx *.xlsm)"
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
            f"Imported {len(self.reference_points)} values; {excluded} derived values are excluded "
            "from primary validation."
        )

    @Slot()
    def export_package(self):
        if not self.result:
            return
        directory = QFileDialog.getExistingDirectory(self, "Choose evidence export folder")
        if not directory:
            return
        try:
            package = export_audit_package(self.result, directory)
            if self.project_store:
                self.project_store.record_export(package, self.result.manifest())
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        QMessageBox.information(self, "Evidence package exported", f"Created:\n{package}")
        self.statusBar().showMessage(f"Evidence package exported to {package}")
