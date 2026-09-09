from __future__ import annotations

import traceback
from copy import deepcopy
from pathlib import Path
from threading import Event
from uuid import uuid4

import numpy as np
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
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from scipy.signal import hilbert

from gpr_layer_audit.catalog import (
    calibration_candidates_for,
    calibration_pairing_is_ambiguous,
    discover_survey_catalog,
)
from gpr_layer_audit.checkpoints import load_checkpoint_file, save_checkpoint_file
from gpr_layer_audit.design import (
    compare_with_design,
    quick_layer_designs,
    read_design_schedule,
)
from gpr_layer_audit.export import export_audit_package
from gpr_layer_audit.models import (
    AcquisitionFileSet,
    AnalysisResult,
    LayerDesign,
    LayerSpec,
    PickStatus,
    SeedRequest,
    SeedStation,
    SurveyCatalog,
    ValidationCheckpoint,
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

from .profile_view import ProfileView
from .radar_view import RadarView
from .theme import LAYER_COLOURS


def _review_proposal_snapshot(result: AnalysisResult, issue) -> dict:
    """Freeze source, family identity, and coordinates shown at confirmation."""

    proposal = []
    for item in result.picks:
        if not (
            item.layer_order == issue.layer_order
            and issue.start_chainage_m <= item.chainage_m <= issue.end_chainage_m
        ):
            continue
        display = (
            item.selected_lobe_sample
            if item.selected_lobe_sample is not None
            else item.evidence.guided_graph_selected_sample
        )
        canonical = (
            item.canonical_event_sample
            if item.canonical_event_sample is not None
            else item.evidence.canonical_event_sample
        )
        if not (
            np.isfinite(display) and display >= 0 and np.isfinite(canonical) and canonical >= 0
        ):
            continue
        proposal.append(
            {
                "chainage_m": float(item.chainage_m),
                "display_sample": float(display),
                "canonical_sample": float(canonical),
                "event_family_id": getattr(item, "event_family_id", None),
                "selected_lobe": getattr(item, "selected_lobe", None),
            }
        )
    source = getattr(result, "source", None)
    return {
        "schema_version": 2,
        "source_fingerprint": getattr(source, "fingerprint", None),
        "proposal": proposal,
    }


def _review_proposal_matches(result: AnalysisResult, issue, details: dict) -> bool:
    """Replay an acceptance only when the rerun shows the same frozen path."""

    expected = details.get("proposal") if isinstance(details, dict) else None
    if not expected or details.get("schema_version") != 2:
        return False
    snapshot = _review_proposal_snapshot(result, issue)
    if details.get("source_fingerprint") != snapshot.get("source_fingerprint"):
        return False
    current = snapshot["proposal"]
    if len(current) != len(expected):
        return False
    return all(
        abs(float(left["chainage_m"]) - float(right["chainage_m"])) <= 1e-6
        and abs(float(left["display_sample"]) - float(right["display_sample"])) <= 0.5
        and abs(float(left["canonical_sample"]) - float(right["canonical_sample"])) <= 0.5
        and left.get("event_family_id") == right.get("event_family_id")
        and left.get("selected_lobe") == right.get("selected_lobe")
        for left, right in zip(current, expected, strict=True)
    )


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
        self.options = deepcopy(options)
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


class ProcessedCorrectionWorker(AnalysisWorker):
    """Keep native processed corrections cancellable without blocking the UI."""

    def __init__(self, result, options, chainage, *, layer_orders=None):
        super().__init__(result.source, None, options)
        self.previous_result = result
        self.chainage = chainage
        self.layer_orders = layer_orders

    @Slot()
    def run(self):
        try:
            result = deepcopy(self.previous_result)
            self.signals.progress.emit(10, "Retracking the declared correction window")
            retrack_segment(
                result,
                self.options,
                max(0.0, self.chainage - 25),
                self.chainage + 25,
                cancel=self.cancel_event.is_set,
                layer_orders=self.layer_orders,
            )
            if self.cancel_event.is_set():
                raise InterruptedError("Analysis cancelled")
        except (AnalysisCancelled, InterruptedError):
            self.signals.cancelled.emit()
        except Exception:
            self.signals.error.emit(traceback.format_exc())
        else:
            self.signals.result.emit(result)


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
        self.accept_dielectric = QCheckBox(
            "Use recorded DZX/header εr; if missing, explicitly assume εr = 7"
        )
        self.accept_dielectric.setChecked(False)
        self.design_unit = QComboBox()
        self.design_unit.addItems(["inches", "millimetres"])
        self.asphalt_design = QLineEdit()
        self.asphalt_design.setPlaceholderText("Optional")
        self.base_design = QLineEdit()
        self.base_design.setPlaceholderText("Optional")
        self.subbase_design = QLineEdit()
        self.subbase_design.setPlaceholderText("Optional")
        self.asphalt_dielectric = QLineEdit()
        self.asphalt_dielectric.setPlaceholderText("Optional")
        self.base_dielectric = QLineEdit()
        self.base_dielectric.setPlaceholderText("Optional")
        self.subbase_dielectric = QLineEdit()
        self.subbase_dielectric.setPlaceholderText("Optional")
        self.design_targets = QLabel("Cumulative targets: no design thickness supplied")
        self.design_targets.setObjectName("secondaryText")
        for edit in (self.asphalt_design, self.base_design, self.subbase_design):
            edit.textChanged.connect(self._update_design_targets)
        self.design_unit.currentTextChanged.connect(self._update_design_targets)
        self.interval = QComboBox()
        self.interval.addItems(["1", "5", "10"])
        self.interval.setCurrentText("1")
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
        form.addRow("Design units", self.design_unit)
        form.addRow("Asphalt thickness", self.asphalt_design)
        form.addRow("Base thickness", self.base_design)
        form.addRow("Subbase thickness", self.subbase_design)
        form.addRow("Asphalt εr", self.asphalt_dielectric)
        form.addRow("Base εr", self.base_dielectric)
        form.addRow("Subbase εr", self.subbase_dielectric)
        form.addRow("Interface targets", self.design_targets)
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
                survey.trace_count / survey.scans_per_meter if survey.scans_per_meter > 0 else 0.0
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
        ambiguous = calibration_pairing_is_ambiguous(candidates)
        if ambiguous:
            self.plate_combo.setToolTip(
                "The leading calibration candidates are tied. Select the acquisition "
                "that belongs to this road; none has been chosen automatically."
            )
        else:
            self.plate_combo.setToolTip("")
        if (
            self.plate_combo.count() > 1
            and candidates
            and candidates[0].waveform_compatible
            and not ambiguous
        ):
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

    def selected_layer_designs(self) -> list[LayerDesign]:
        unit = "in" if self.design_unit.currentText() == "inches" else "mm"
        dielectric_by_layer: dict[int, float] = {}
        for order, (name, edit) in enumerate(
            (
                ("Asphalt", self.asphalt_dielectric),
                ("Base", self.base_dielectric),
                ("Subbase", self.subbase_dielectric),
            ),
            1,
        ):
            text = edit.text().strip()
            if not text:
                continue
            value = float(text)
            if not 1.0 < value <= 40.0:
                raise ValueError(f"{name} dielectric must be between 1 and 40.")
            dielectric_by_layer[order] = value
        return quick_layer_designs(
            self.asphalt_design.text(),
            self.base_design.text(),
            self.subbase_design.text(),
            default_unit=unit,
            dielectric_by_layer=dielectric_by_layer,
        )

    @Slot()
    def _update_design_targets(self):
        try:
            designs = self.selected_layer_designs()
        except ValueError:
            self.design_targets.setText("Enter valid positive layer thicknesses.")
            return
        cumulative = 0.0
        values: list[str] = []
        inches = self.design_unit.currentText() == "inches"
        for design in designs:
            if design.thickness_mm is None:
                values.append(f"{design.layer_name}: unknown")
                continue
            cumulative += design.thickness_mm
            shown = cumulative / 25.4 if inches else cumulative
            values.append(f"{design.layer_name}: {shown:.1f} {'in' if inches else 'mm'}")
        self.design_targets.setText("Cumulative targets: " + " · ".join(values))

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
        try:
            self.selected_layer_designs()
        except ValueError as exc:
            QMessageBox.warning(self, "Design inputs need attention", str(exc))
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
        self._training_observations_this_session: set[tuple[str, int]] = set()
        self.design_segments = []
        self.reference_points = []
        self.validation_checkpoints: list[ValidationCheckpoint] = []
        self.capture_checkpoint_mode = False
        self._analyzed_training_station_ids: set[str] = set()
        self._proposed_seed_layers: dict[float, int] = {}
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
        self.checkpoint_action = QAction("Capture validation checkpoints", self)
        self.checkpoint_action.setCheckable(True)
        self.checkpoint_action.toggled.connect(self._set_checkpoint_mode)
        toolbar.addAction(self.checkpoint_action)
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
        self.profile = ProfileView()
        self.radar.anchorRequested.connect(self.add_seed_pick)
        self.radar.locationChanged.connect(self.profile.set_cursor)
        self.profile.chainageRequested.connect(self.radar.focus_chainage)
        self.workspace_tabs = QTabWidget()
        self.workspace_tabs.addTab(self.radar, "Radar & A-scan")
        self.workspace_tabs.addTab(self.profile, "Depth profiles")
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._project_panel())
        splitter.addWidget(self.workspace_tabs)
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
            # Asphalt and base have enough supplied cross-road evidence for the
            # normal seed-assisted workflow.  Subbase is materially weaker and
            # has reference coverage on only a small subset of roads, so make
            # it an explicit analyst choice instead of silently adding a slow,
            # unvalidated third-interface fit to every new project.
            check.setChecked(layer.order < 3)
            if layer.order == 3:
                check.setToolTip(
                    "Enable only when a distinct subbase reflector can be manually "
                    "identified. Design thickness alone is not evidence of this interface."
                )
            check.setStyleSheet(f"color: {LAYER_COLOURS[layer.order]};")
            self.layer_checks[layer.order] = check
            layer_layout.addWidget(check)
        for order, check in self.layer_checks.items():
            check.toggled.connect(
                lambda checked, layer_order=order: self._enforce_layer_dependencies(
                    layer_order, checked
                )
            )
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
            "Build a preview, then Ctrl+click the interface you want to trace. "
            "Start with three distributed observations per interface."
        )
        self.seed_help.setWordWrap(True)
        seed_layout.addWidget(self.seed_help)
        self.seed_combo = QComboBox()
        self.seed_combo.currentIndexChanged.connect(self.focus_selected_seed)
        seed_layout.addWidget(self.seed_combo)
        next_seed = QPushButton("Next useful seed")
        next_seed.setToolTip("Inspect the highest-priority unresolved reflector interval")
        next_seed.clicked.connect(self.focus_next_useful_seed)
        seed_layout.addWidget(next_seed)
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
        engine_box = QGroupBox("Tracing method")
        engine_layout = QVBoxLayout(engine_box)
        self.engine_combo = QComboBox()
        self.engine_combo.addItem("Established tracker", "joint_seed_adaptive")
        self.engine_combo.addItem("Hybrid tracing · experimental", "seed_hybrid")
        self.engine_combo.setToolTip(
            "Hybrid combines waveform correspondence with optional learned evidence"
        )
        self.engine_combo.currentIndexChanged.connect(self._change_tracking_method)
        engine_layout.addWidget(self.engine_combo)
        self.input_mode_combo = QComboBox()
        self.input_mode_combo.addItem("Raw acquisition", "raw")
        self.input_mode_combo.addItem("Processed DZT · preserve coordinates", "processed")
        self.input_mode_combo.setToolTip(
            "Processed mode uses stored samples and the DZT time origin"
        )
        engine_layout.addWidget(self.input_mode_combo)
        self.processed_stride_combo = QComboBox()
        for stride in (1, 4, 16):
            self.processed_stride_combo.addItem(f"Processed trace stride: {stride}", stride)
        engine_layout.addWidget(self.processed_stride_combo)
        self.query_layers_combo = QComboBox()
        for label, orders in (
            ("Base and subbase", [2, 3]),
            ("Base", [2]),
            ("Subbase", [3]),
            ("Asphalt", [1]),
            ("All interfaces", [1, 2, 3]),
        ):
            self.query_layers_combo.addItem(f"Request observations: {label}", orders)
        engine_layout.addWidget(self.query_layers_combo)
        load_config = QPushButton("Load tracing configuration…")
        load_config.clicked.connect(self.load_tracing_configuration)
        engine_layout.addWidget(load_config)
        load_seeds = QPushButton("Load native seed observations…")
        load_seeds.clicked.connect(self.load_native_seed_observations)
        engine_layout.addWidget(load_seeds)
        self.model_label = QLabel("ML inactive · no validated model loaded")
        self.model_label.setWordWrap(True)
        engine_layout.addWidget(self.model_label)
        load_model = QPushButton("Load model bundle…")
        load_model.clicked.connect(self.load_model_bundle)
        engine_layout.addWidget(load_model)
        export_labels = QPushButton("Export confirmed training picks…")
        export_labels.clicked.connect(self.export_training_picks)
        engine_layout.addWidget(export_labels)
        layout.addWidget(engine_box)
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
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(285)
        return scroll

    def _change_tracking_method(self):
        if self.worker:
            return
        self.options.tracker_method = self.engine_combo.currentData()

    def load_tracing_configuration(self):
        if self.worker:
            return
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load tracing configuration", "", "JSON (*.json)"
        )
        if not filename:
            return
        try:
            import json
            from dataclasses import asdict

            from gpr_layer_audit.processing.conventional_config import resolve_config

            self.options.conventional_config = asdict(
                resolve_config(json.loads(Path(filename).read_text()))
            )
        except (ValueError, OSError) as exc:
            QMessageBox.information(self, "Configuration not loaded", str(exc))
            return
        self.statusBar().showMessage(f"Tracing configuration loaded: {Path(filename).name}")

    def load_native_seed_observations(self):
        if self.worker or not self.road or not self.project_store:
            return
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load native seed observations", "", "JSON (*.json)"
        )
        if not filename:
            return
        try:
            from gpr_layer_audit.native_seed_io import load_native_observations

            stations = load_native_observations(filename, self.road.dzt_path)
            if self.options.seed_stations:
                raise ValueError(
                    "Open a fresh project to import operating observations "
                    "without replacing existing work"
                )
        except (ValueError, OSError) as exc:
            QMessageBox.information(self, "Observations not loaded", str(exc))
            return
        self.options.seed_stations = stations
        for order in {o for s in stations for o in s.samples | s.visibility}:
            if order in self.layer_checks:
                self.layer_checks[order].setChecked(True)
        for station in stations:
            self.project_store.save_seed_station(station)
        self.input_mode_combo.setCurrentIndex(self.input_mode_combo.findData("processed"))
        self._analyzed_training_station_ids.clear()
        self.run_analysis()

    def load_model_bundle(self):
        if self.worker:
            return
        filename, _ = QFileDialog.getOpenFileName(self, "Load model manifest", "", "JSON (*.json)")
        if not filename:
            return
        try:
            import json

            from gpr_layer_audit.processing.hybrid_evidence import PREPROCESSING_VERSION

            manifest = json.loads(Path(filename).read_text(encoding="utf-8"))
            if manifest.get("preprocessing_version") != PREPROCESSING_VERSION:
                raise ValueError("This model uses an incompatible radar preprocessing version.")
            if manifest.get("promotion", {}).get("passed") is not True:
                raise ValueError(
                    "This model has not passed held-out hybrid evaluation. "
                    "Use the CLI for research evaluation."
                )
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Model unavailable", str(exc))
            return
        self.options.ml_model = filename
        self.engine_combo.setCurrentIndex(self.engine_combo.findData("seed_hybrid"))
        self.model_label.setText(f"Validated model: {Path(filename).parent.name}")

    def focus_next_useful_seed(self):
        if not self.result or self.worker:
            return
        requests = self.result.proposed_seed_requests
        if requests:
            target = requests[0].chainage_m
            for index in range(self.seed_combo.count()):
                value = self.seed_combo.itemData(index)
                if isinstance(value, (float, int)) and abs(value - target) < 1e-6:
                    self.seed_combo.setCurrentIndex(index)
                    self.focus_selected_seed(index)
                    return
        self.statusBar().showMessage(
            "Choose a review interval or click the reflector at a new station."
        )

    def export_training_picks(self):
        if not self.result or self.worker:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export confirmed training observations", "", "CSV (*.csv)"
        )
        if not filename:
            return
        try:
            from gpr_layer_audit.ml.dataset import export_confirmed_annotations

            count = export_confirmed_annotations(
                self.result,
                self.options.seed_stations,
                filename,
                current_station_ids=self._training_observations_this_session,
            )
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Training observations not exported", str(exc))
            return
        self.statusBar().showMessage(
            f"Exported {count} confirmed picks with their source coordinate contract."
        )

    def _review_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        title = QLabel("PRIORITIZED REVIEW")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        self.review_help = QLabel(
            "Dashed paths are proposals only and produce no TWTT or thickness. "
            "Inspect the radargram and A-scan. Correct the point with Ctrl+click, "
            "or confirm the proposed reflector only after checking its identity. "
            "Use Not visible when the layer is present but no reliable reflector "
            "is visible; use Layer absent only when it is genuinely absent."
        )
        self.review_help.setWordWrap(True)
        self.review_help.setObjectName("secondaryText")
        layout.addWidget(self.review_help)
        self.issue_list = QListWidget()
        self.issue_list.itemActivated.connect(self.focus_issue)
        layout.addWidget(self.issue_list, 1)
        row1 = QHBoxLayout()
        accept = QPushButton("Confirm proposed reflector")
        accept.setToolTip(
            "Accept the current proposal across the selected review interval only "
            "after checking the radargram and A-scan."
        )
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
        structure_row = QHBoxLayout()
        anomaly = QPushButton("Confirm anomaly")
        anomaly.clicked.connect(lambda: self.resolve_selected_issue("anomaly"))
        structural = QPushButton("Add structural break")
        structural.clicked.connect(self.add_structural_break)
        structure_row.addWidget(anomaly)
        structure_row.addWidget(structural)
        layout.addLayout(structure_row)
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
            layer_designs=dialog.selected_layer_designs(),
        )
        new_layers = LayerSpec.defaults()
        new_layers[2].analysis_enabled = False
        new_layers[2].audit_enabled = False
        self.options.layer_specs = new_layers
        self._restore_layer_controls(new_layers)
        self.design_segments = []
        self.reference_points = []
        if dialog.design_combo.currentData():
            self.design_segments = read_design_schedule(dialog.design_combo.currentData())
            self.options.design_segments = self.design_segments
        if dialog.reference_combo.currentData():
            self.reference_points = read_manual_reference(dialog.reference_combo.currentData())
        self.project_store = ProjectStore.create(dialog.project_edit.text(), road.survey_id)
        self.validation_checkpoints = []
        self.project_store.set_meta("catalog_root", str(self.catalog.root))
        self.project_store.set_meta("survey_id", road.survey_id)
        self.project_store.set_file("road", road.dzt_path)
        if plate:
            self.project_store.set_file("plate", plate.dzt_path)
        self.project_store.set_layers(self.options.layer_specs)
        self.project_store.set_layer_designs(self.options.layer_designs)
        if self.design_segments:
            self.project_store.add_design_segments(self.design_segments)
        self.result = None
        self._training_observations_this_session.clear()
        self._analyzed_training_station_ids.clear()
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
            layer_designs = store.layer_designs()
            stored_layers = store.layer_specs()
            if not stored_layers:
                stored_layers = LayerSpec.defaults()
                stored_layers[2].analysis_enabled = False
                stored_layers[2].audit_enabled = False
            if not road_path.is_file() or (plate_path and not plate_path.is_file()):
                raise FileNotFoundError("One or more source acquisitions are no longer available.")
        except Exception as exc:
            QMessageBox.warning(self, "Project could not be opened", str(exc))
            return
        self.project_store = store
        checkpoint_path = store.path.with_suffix(".checkpoints.json")
        self.validation_checkpoints = []
        if checkpoint_path.is_file():
            try:
                checkpoint_survey, checkpoints = load_checkpoint_file(checkpoint_path)
                expected_survey = store.get_meta("survey_id") or road_path.stem
                if checkpoint_survey != expected_survey:
                    raise ValueError(
                        f"Checkpoint survey {checkpoint_survey!r} does not match "
                        f"project survey {expected_survey!r}."
                    )
                self.validation_checkpoints = checkpoints
            except Exception as exc:
                QMessageBox.warning(
                    self,
                    "Validation checkpoints were not loaded",
                    str(exc),
                )
        self.road = AcquisitionFileSet(road_path)
        self.plate = AcquisitionFileSet(plate_path) if plate_path else None
        self.options = AnalysisOptions(
            survey_id=store.get_meta("survey_id") or road_path.stem,
            input_mode=parameters.get("input_mode", "raw"),
            processed_stride=int(parameters.get("processed_stride", 1)),
            query_layer_orders=list(parameters.get("query_layer_orders", [2, 3])),
            local_correction_order=list(parameters.get("local_correction_order", [])),
            tracker_method=parameters.get("tracker_method", "joint_seed_adaptive"),
            ml_model=parameters.get("ml_model"),
            ml_policy=parameters.get("ml_policy", "auto"),
            hybrid_calibration=parameters.get("hybrid_calibration"),
            conventional_config=parameters.get("conventional_config", {}),
            stack_size=int(parameters.get("stack_size", 0)),
            report_interval_m=float(parameters.get("report_interval_m", 5.0)),
            accept_scan_dielectric=bool(parameters.get("accept_scan_dielectric", False)),
            analyst_dielectric={
                int(k): v for k, v in parameters.get("analyst_dielectric", {}).items()
            },
            seed_stations=seeds,
            design_segments=design_segments,
            layer_designs=layer_designs,
            layer_specs=stored_layers,
            structural_breaks_m=list(parameters.get("structural_breaks_m", [])),
            auto_fine_retrack=bool(parameters.get("auto_fine_retrack", True)),
            max_auto_fine_regions=parameters.get("max_auto_fine_regions", 1),
        )
        self._restore_layer_controls(self.options.layer_specs)
        self.input_mode_combo.setCurrentIndex(
            max(0, self.input_mode_combo.findData(self.options.input_mode))
        )
        if self.processed_stride_combo.findData(self.options.processed_stride) < 0:
            self.processed_stride_combo.addItem(
                str(self.options.processed_stride), self.options.processed_stride
            )
        self.processed_stride_combo.setCurrentIndex(
            self.processed_stride_combo.findData(self.options.processed_stride)
        )
        if self.query_layers_combo.findData(self.options.query_layer_orders) < 0:
            self.query_layers_combo.addItem(
                f"Request interfaces: {self.options.query_layer_orders}",
                self.options.query_layer_orders,
            )
        self.query_layers_combo.setCurrentIndex(
            self.query_layers_combo.findData(self.options.query_layer_orders)
        )
        self.engine_combo.setCurrentIndex(
            max(0, self.engine_combo.findData(self.options.tracker_method))
        )
        self.model_label.setText(
            Path(self.options.ml_model).name
            if self.options.ml_model
            else "ML inactive · no validated model loaded"
        )
        self._training_observations_this_session.clear()
        self.design_segments = design_segments
        self._analyzed_training_station_ids.clear()
        self.source_label.setText(f"{road_path.name}\n{road_path.parent}")
        self.run_action.setText("Track from saved seeds" if seeds else "Build preview")
        self.run_action.setEnabled(True)
        checkpoint_note = (
            f" {len(self.validation_checkpoints)} blinded checkpoints loaded."
            if self.validation_checkpoints
            else ""
        )
        self.statusBar().showMessage(
            "Project opened. Run tracking to reconstruct the workbench." + checkpoint_note
        )

    def _enforce_layer_dependencies(self, layer_order: int, checked: bool) -> None:
        """Keep enabled interfaces contiguous from asphalt downward."""

        if checked:
            affected = range(1, layer_order)
            value = True
        else:
            affected = range(layer_order + 1, max(self.layer_checks) + 1)
            value = False
        for order in affected:
            self.layer_checks[order].setChecked(value)

    def _layers_from_controls(self, layers: list[LayerSpec] | None = None) -> list[LayerSpec]:
        """Copy layer definitions and apply the current contiguous UI selection."""

        output = deepcopy(layers or LayerSpec.defaults())
        for layer in output:
            check = self.layer_checks.get(layer.order)
            if check is None:
                continue
            layer.analysis_enabled = check.isChecked()
            layer.audit_enabled = layer.analysis_enabled
        return output

    def _restore_layer_controls(self, layers: list[LayerSpec]) -> None:
        """Restore persisted enablement while keeping upper interfaces enabled."""

        enabled = {
            layer.order
            for layer in layers
            if layer.analysis_enabled and layer.order in self.layer_checks
        }
        if enabled:
            deepest = max(enabled)
            enabled = {order for order in self.layer_checks if order <= deepest}
        else:
            # A tracker run cannot interpret a deeper interface without the
            # surface/asphalt boundary. Keep malformed legacy projects usable.
            enabled = {min(self.layer_checks)}
        for order, check in self.layer_checks.items():
            check.blockSignals(True)
            check.setChecked(order in enabled)
            check.blockSignals(False)
        self.options.layer_specs = self._layers_from_controls(layers)

    @Slot()
    def run_analysis(self):
        if not self.road or self.worker:
            return
        self.options.tracker_method = self.engine_combo.currentData()
        self.options.input_mode = self.input_mode_combo.currentData()
        self.options.processed_stride = self.processed_stride_combo.currentData()
        self.options.query_layer_orders = self.query_layers_combo.currentData()
        layers = self._layers_from_controls(self.options.layer_specs)
        self.options.layer_specs = layers
        self.options.design_segments = self.design_segments
        if self.project_store:
            self.project_store.set_layers(layers)
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

    def _replay_saved_review_decisions(self) -> tuple[int, int]:
        """Restore durable decisions without accepting a changed proposal."""

        if not self.result or not self.project_store:
            return 0, 0
        latest = {
            event["issue_id"]: event
            for event in self.project_store.review_events()
            if event.get("issue_id")
        }
        applied = skipped = 0
        for issue_id, event in latest.items():
            issue = next(
                (item for item in self.result.review_issues if item.issue_id == issue_id),
                None,
            )
            if issue is None:
                continue
            action = str(event.get("action"))
            if action == "accept" and not _review_proposal_matches(
                self.result, issue, event.get("details", {})
            ):
                skipped += 1
                continue
            try:
                resolve_review_issue(
                    self.result,
                    self.options,
                    issue.issue_id,
                    action,
                )
            except ValueError:
                skipped += 1
            else:
                applied += 1
        self.result.parameters["review_decision_replay"] = {
            "applied": applied,
            "skipped_changed_proposals": skipped,
        }
        return applied, skipped

    @Slot(object)
    def _analysis_complete(self, result):
        self.result = result
        replayed_decisions, changed_decisions = self._replay_saved_review_decisions()
        self._analyzed_training_station_ids = {
            station.station_id
            for station in self.options.seed_stations
            if station.role != "correction"
        }
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
        self.profile.set_result(result)
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
        required = self._required_initial_station_count()
        self.run_action.setText("Re-run seeded tracker" if completed else "Re-run preview")
        if required == 0:
            message = (
                f"Automatic tracking complete; {len(result.review_issues)} regions need review."
            )
        elif completed >= required:
            message = f"Seeded tracking complete; {len(result.review_issues)} regions need review."
        else:
            message = (
                f"Automatic pass ready. Complete requested stations ({completed}/{required}) "
                "to establish manual reflector identity."
            )
        if replayed_decisions:
            message += f" Restored {replayed_decisions} saved review decision(s)."
        if changed_decisions:
            message += (
                f" {changed_decisions} saved confirmation(s) need review because "
                "the tracker proposal changed."
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
        required_orders = self._required_seed_orders()
        if not required_orders:
            return any(station.user_confirmed.values())
        enabled = required_orders
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

    def _required_seed_orders(self) -> list[int]:
        if self.result:
            return [int(order) for order in self.result.parameters.get("required_seed_orders", [])]
        return [
            design.layer_order
            for design in self.options.layer_designs
            if design.thickness_mm is None
        ]

    def _required_initial_station_count(self) -> int:
        if self.result:
            return int(self.result.parameters.get("required_seed_count", 0))
        return 2 if self._required_seed_orders() else 0

    def _populate_seed_controls(self):
        self.seed_combo.blockSignals(True)
        selected = self.seed_combo.currentData()
        selected_mode = self.seed_combo.currentData(Qt.ItemDataRole.UserRole + 1)
        self.seed_combo.clear()
        proposed = self.result.proposed_seed_chainages if self.result else []
        requests = getattr(self.result, "proposed_seed_requests", []) if self.result else []
        required_orders = self._required_seed_orders()
        self._proposed_seed_layers = {}
        if not requests:
            requests = [
                SeedRequest(
                    chainage_m=float(chainage),
                    layer_orders=[self._suggested_layer_order(float(chainage))],
                    reason="Inspect the highest-information review location",
                    priority=0.0,
                )
                for chainage in proposed
            ]
        for index, request in enumerate(requests, 1):
            chainage = float(request.chainage_m)
            orders = request.layer_orders or [self._suggested_layer_order(chainage)]
            order = int(orders[0])
            self._proposed_seed_layers[round(float(chainage), 6)] = order
            layer_names = ", ".join(
                LayerSpec.defaults()[layer_order - 1].name for layer_order in orders
            )
            label = "Requested seed" if required_orders else "Resolve ambiguity"
            self.seed_combo.addItem(
                f"{label} {index} · {layer_names} · {chainage:.1f} m",
                float(chainage),
            )
            self.seed_combo.setItemData(
                self.seed_combo.count() - 1,
                request.reason,
                Qt.ItemDataRole.ToolTipRole,
            )
            if self.result and self.result.parameters.get("input_mode") == "processed":
                self.seed_combo.setItemData(
                    self.seed_combo.count() - 1, "correction", Qt.ItemDataRole.UserRole + 1
                )
        required = self._required_initial_station_count()
        if self.result:
            self.seed_combo.addItem("Model seed at clicked chainage", None)
            self.seed_combo.setItemData(
                self.seed_combo.count() - 1, "model", Qt.ItemDataRole.UserRole + 1
            )
        if self.result and self._completed_initial_stations() >= required:
            self.seed_combo.addItem("Correction at clicked chainage", None)
            self.seed_combo.setItemData(
                self.seed_combo.count() - 1, "correction", Qt.ItemDataRole.UserRole + 1
            )
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
            mode_index = self.seed_combo.findData(selected_mode, Qt.ItemDataRole.UserRole + 1)
            self.seed_combo.setCurrentIndex(max(0, mode_index))
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
        station_count = sum(station.role != "correction" for station in self.options.seed_stations)
        self.seed_help.setText(
            f"{complete}/{required} requested stations complete · {station_count} stations. "
            "Aim for three observations per interface; add more where needed."
        )
        pending_training_seed = any(
            station.role != "correction"
            and any(station.user_confirmed.values())
            and station.station_id not in self._analyzed_training_station_ids
            for station in self.options.seed_stations
        )
        self.track_button.setText(
            "Re-run with new model seed" if pending_training_seed else "Track from completed seeds"
        )
        self.track_button.setEnabled(
            not self.worker and (pending_training_seed or (required > 0 and complete >= required))
        )
        if self.result:
            self.result.seed_stations = list(self.options.seed_stations)
            self.radar.refresh_guides()
            if selected is None and selected_mode is None and proposed:
                self.focus_selected_seed()

    @Slot()
    def focus_selected_seed(self):
        if not self.result:
            return
        chainage = self.seed_combo.currentData()
        if chainage is not None:
            order = self._proposed_seed_layers.get(round(float(chainage), 6))
            if order is not None:
                layer_index = self.active_layer_combo.findData(order)
                if layer_index >= 0:
                    self.active_layer_combo.setCurrentIndex(layer_index)
            self.radar.focus_chainage(float(chainage), 24.0)

    def _suggested_layer_order(self, chainage: float) -> int:
        required = self._required_seed_orders()
        if required:
            return int(required[0])
        if self.result and self.result.review_issues:
            issue = min(
                self.result.review_issues,
                key=lambda item: abs(
                    float(
                        item.suggested_chainage_m
                        if item.suggested_chainage_m is not None
                        else 0.5 * (item.start_chainage_m + item.end_chainage_m)
                    )
                    - chainage
                ),
            )
            return int(issue.layer_order)
        return int(self.active_layer_combo.currentData())

    def _selected_or_clicked_station(self, clicked_chainage: float) -> SeedStation | None:
        # Suggestions are navigation aids, never substitutes for the actual
        # clicked trace. Keep the sample and waveform metadata at one chainage.
        row = int(np.argmin(np.abs(self.result.chainage_m - clicked_chainage)))
        target = float(self.result.chainage_m[row])
        existing = next(
            (
                item
                for item in self.options.seed_stations
                if int(np.argmin(np.abs(self.result.chainage_m - item.chainage_m))) == row
            ),
            None,
        )
        if existing:
            return existing
        # Suggested and arbitrary model seeds train the whole road. Only the
        # explicit correction mode limits the update to a local segment.
        adding_training_station = (
            self.seed_combo.currentData(Qt.ItemDataRole.UserRole + 1) != "correction"
        )
        station = SeedStation(
            station_id=str(uuid4()),
            chainage_m=target,
            role=("initial" if adding_training_station else "correction"),
        )
        self.options.seed_stations.append(station)
        return station

    @Slot(int, float, float)
    def add_seed_pick(self, layer_order, chainage_m, sample_index):
        if not self.result or not self.project_store:
            return
        if self.worker:
            self.statusBar().showMessage("Wait for tracking to finish before changing seeds.")
            return
        if (
            not np.isfinite(chainage_m)
            or not np.isfinite(sample_index)
            or not self.result.chainage_m[0] <= chainage_m <= self.result.chainage_m[-1]
            or not 0 <= sample_index < self.result.calibrated_radargram.shape[1]
        ):
            self.statusBar().showMessage("Pick inside the recorded radargram.")
            return
        selected_chainage = float(
            self.result.chainage_m[int(abs(self.result.chainage_m - chainage_m).argmin())]
        )
        if self.result.parameters.get("input_mode") == "processed":
            # The display reports discrete native sample cells in this mode.
            sample_index = float(round(sample_index))
        events = [
            item
            for item in self.result.candidate_events
            if item.layer_order == layer_order and abs(item.chainage_m - selected_chainage) < 1e-6
        ]
        nearest = (
            min(events, key=lambda item: abs(item.sample_index - sample_index)) if events else None
        )
        if self.capture_checkpoint_mode:
            self._capture_validation_checkpoint(
                layer_order,
                selected_chainage,
                sample_index,
                nearest,
            )
            return
        warnings: list[str] = []
        phase_regime_change = False
        if nearest is None or abs(nearest.sample_index - sample_index) > 4:
            warnings.append("The click is a free pick more than four samples from a candidate.")
        elif any(
            item.waveform_correlation > nearest.waveform_correlation + 0.18
            and abs(item.sample_index - nearest.sample_index) <= 24
            and item.phase_class != nearest.phase_class
            for item in events
        ):
            warnings.append(
                "A neighboring wavelet cycle has materially stronger seed-waveform similarity."
            )
        existing_phases = [
            station.phase_class[layer_order]
            for station in self.options.seed_stations
            if layer_order in station.phase_class
        ]
        if nearest is not None and existing_phases:
            distances = [
                min(abs(nearest.phase_class - value), 8 - abs(nearest.phase_class - value))
                for value in existing_phases
            ]
            if all(distance >= 3 for distance in distances):
                warnings.append("The phase cycle conflicts with the existing confirmed regime.")
                phase_regime_change = True
        if warnings:
            answer = QMessageBox.question(
                self,
                "Confirm unusual seed",
                "\n".join(warnings)
                + "\n\nThe ±25 m preview should follow the same physical reflector. "
                "Save this seed anyway?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Save:
                return
        station = self._selected_or_clicked_station(chainage_m)
        if station is None:
            return
        station.samples[layer_order] = float(sample_index)
        self._training_observations_this_session.add((station.station_id, layer_order))
        station.visibility[layer_order] = VisibilityState.VISIBLE
        station.user_confirmed[layer_order] = True
        station.preview_status[layer_order] = "confirmed_with_warning" if warnings else "confirmed"
        if nearest is not None and abs(nearest.sample_index - sample_index) <= 4:
            station.phase_class[layer_order] = nearest.phase_class
            station.analytic_phase_rad[layer_order] = nearest.analytic_phase_rad
            station.polarity[layer_order] = nearest.polarity
            station.selected_lobe[layer_order] = nearest.selected_lobe
            station.canonical_samples[layer_order] = float(
                nearest.canonical_sample_index
                if nearest.canonical_sample_index is not None
                else nearest.sample_index
            )
            station.pulse_width_samples[layer_order] = max(
                1.0, float(nearest.pulse_width_samples or 7.0)
            )
            station.event_ids[layer_order] = nearest.event_id or (
                f"L{layer_order}:{station.chainage_m:.3f}:{nearest.sample_index}"
            )
            if nearest.event_family_id:
                station.family_ids[layer_order] = nearest.event_family_id
            if nearest.competing_family_id:
                station.competing_family_ids[layer_order] = nearest.competing_family_id
            station.competing_samples[layer_order] = [
                float(item.sample_index)
                for item in events
                if item.event_id in nearest.competing_event_ids
            ]
            preview_events = [
                item
                for item in self.result.candidate_events
                if item.layer_order == layer_order
                and abs(item.chainage_m - station.chainage_m) <= 25.0
            ]
            preview_paths: dict[str, list[float]] = {}
            for family_id in filter(None, (nearest.event_family_id, nearest.competing_family_id)):
                family_events = sorted(
                    (
                        item
                        for item in preview_events
                        if item.event_family_id == family_id
                        or (item.event_family_id is None and item.event_id == family_id)
                    ),
                    key=lambda item: item.chainage_m,
                )
                preview_paths[str(family_id)] = [float(item.sample_index) for item in family_events]
            station.preview_paths[layer_order] = preview_paths
        else:
            row = int(np.argmin(np.abs(self.result.chainage_m - selected_chainage)))
            sample = int(
                np.clip(
                    round(sample_index),
                    0,
                    self.result.calibrated_radargram.shape[1] - 1,
                )
            )
            trace = self.result.calibrated_radargram[row]
            phase = float(np.angle(hilbert(trace)[sample]))
            station.phase_class[layer_order] = int(
                np.floor(((phase + np.pi) % (2.0 * np.pi)) * 8.0 / (2.0 * np.pi))
            )
            station.analytic_phase_rad[layer_order] = phase
            station.polarity[layer_order] = int(np.sign(trace[sample]))
            station.selected_lobe[layer_order] = (
                "negative_trough" if trace[sample] < 0 else "positive_peak"
            )
            station.canonical_samples[layer_order] = float(sample)
            station.pulse_width_samples[layer_order] = 7.0
            station.event_ids[layer_order] = (
                f"free:L{layer_order}:{station.chainage_m:.3f}:{sample}"
            )
            station.family_ids[layer_order] = f"free:L{layer_order}:{station.chainage_m:.3f}"
            station.competing_samples[layer_order] = []
            station.preview_paths[layer_order] = {}
        existing_regimes = [
            item.regime_ids[layer_order]
            for item in self.options.seed_stations
            if item is not station and layer_order in item.regime_ids
        ]
        station.regime_ids[layer_order] = (
            f"change@{station.chainage_m:.3f}"
            if phase_regime_change
            else (existing_regimes[0] if existing_regimes else "default")
        )
        if warnings:
            station.warnings[layer_order] = " ".join(warnings)
        else:
            station.warnings.pop(layer_order, None)
        station.preview_start_chainage_m[layer_order] = max(0.0, station.chainage_m - 25.0)
        station.preview_end_chainage_m[layer_order] = station.chainage_m + 25.0
        self.project_store.save_seed_station(station)
        self._analyzed_training_station_ids.discard(station.station_id)
        self.options.seed_stations.sort(key=lambda item: item.chainage_m)
        correction = station.role == "correction"
        if correction:
            self.options.local_correction_order = [
                value
                for value in self.options.local_correction_order
                if value != station.station_id
            ] + [station.station_id]
            self.project_store.record_review_event(
                "correction",
                layer_order=layer_order,
                start_chainage_m=max(0, station.chainage_m - 25),
                end_chainage_m=station.chainage_m + 25,
                details={
                    "station_id": station.station_id,
                    "sample": sample_index,
                    "operation": "local_correction",
                    "source_sha256": self.result.source.fingerprint,
                },
            )
        self._populate_seed_controls()
        if correction:
            self._local_retrack(station.chainage_m, layer_orders={layer_order})
        elif any(station.user_confirmed.values()):
            self.statusBar().showMessage(
                f"Saved model seed at {station.chainage_m:.1f} m. "
                "Re-run to propagate it across supported reflector segments."
            )
        else:
            self.statusBar().showMessage(
                f"Saved {LayerSpec.defaults()[layer_order - 1].name} at "
                f"{station.chainage_m:.1f} m. Complete all enabled interfaces."
            )

    def _set_checkpoint_mode(self, enabled: bool) -> None:
        if enabled and self.reference_points:
            self.capture_checkpoint_mode = False
            self.checkpoint_action.blockSignals(True)
            self.checkpoint_action.setChecked(False)
            self.checkpoint_action.blockSignals(False)
            QMessageBox.warning(
                self,
                "Blinded capture unavailable",
                "A manual reference workbook is already loaded in this session. "
                "Start or reopen the project without importing a reference before "
                "capturing validation checkpoints.",
            )
            return
        self.capture_checkpoint_mode = bool(enabled)
        self.statusBar().showMessage(
            "Validation mode: Ctrl+click blinded radar events; checkpoints never train the tracker."
            if enabled
            else "Seed mode restored."
        )

    def _capture_validation_checkpoint(
        self,
        layer_order: int,
        chainage_m: float,
        sample_index: float,
        nearest,
    ) -> None:
        if self.result is None or self.project_store is None:
            return
        if nearest is not None and abs(nearest.sample_index - sample_index) <= 4:
            sample = float(nearest.sample_index)
            canonical = float(
                nearest.canonical_sample_index
                if nearest.canonical_sample_index is not None
                else nearest.sample_index
            )
            lobe = nearest.selected_lobe
            family_id = nearest.event_family_id
            pulse_width = max(1.0, float(nearest.pulse_width_samples or 7.0))
        else:
            sample = float(sample_index)
            canonical = sample
            lobe = "free_click"
            family_id = None
            pulse_width = 7.0
        checkpoint = ValidationCheckpoint(
            checkpoint_id=f"L{layer_order}-checkpoint-{len(self.validation_checkpoints) + 1:03d}",
            layer_order=layer_order,
            chainage_m=chainage_m,
            sample_index=sample,
            canonical_sample_index=canonical,
            visibility=VisibilityState.VISIBLE,
            user_confirmed=True,
            selected_lobe=lobe,
            event_family_id=family_id,
            pulse_width_samples=pulse_width,
        )
        self.validation_checkpoints.append(checkpoint)
        survey_id = str(self.result.parameters.get("survey_id") or self.result.source.dzt_path.stem)
        output = self.project_store.path.with_suffix(".checkpoints.json")
        save_checkpoint_file(output, survey_id, self.validation_checkpoints)
        count = sum(item.layer_order == layer_order for item in self.validation_checkpoints)
        self.statusBar().showMessage(
            f"Saved blinded layer-{layer_order} checkpoint {count}/30 to {output.name}."
        )

    def mark_seed_visibility(self, visibility: VisibilityState):
        if not self.result or not self.project_store or self.worker:
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
        station.user_confirmed[order] = True
        station.preview_status[order] = "explicit_visibility_state"
        station.preview_start_chainage_m[order] = max(0.0, station.chainage_m - 25.0)
        station.preview_end_chainage_m[order] = station.chainage_m + 25.0
        self.project_store.save_seed_station(station)
        self.project_store.record_review_event(
            visibility.value,
            layer_order=order,
            start_chainage_m=station.chainage_m,
            end_chainage_m=station.chainage_m,
            details={
                "station_id": station.station_id,
                "operation": station.role,
                "source_sha256": self.result.source.fingerprint,
            },
        )
        self._analyzed_training_station_ids.discard(station.station_id)
        self._populate_seed_controls()
        if station.role == "correction":
            self._local_retrack(station.chainage_m, layer_orders={order})

    @Slot()
    def undo_seed_station(self):
        if not self.project_store or self.worker:
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

    def _local_retrack(self, chainage_m: float, *, layer_orders=None):
        if not self.result:
            return
        if self.result.parameters.get("input_mode") == "processed":
            self.options.query_layer_orders = self.query_layers_combo.currentData()
            for station in self.options.seed_stations:
                if station.role != "correction" or abs(station.chainage_m - chainage_m) > 1e-6:
                    continue
                self.options.local_correction_order = [
                    value
                    for value in self.options.local_correction_order
                    if value != station.station_id
                ] + [station.station_id]
            self.worker = ProcessedCorrectionWorker(
                self.result, self.options, chainage_m, layer_orders=layer_orders
            )
            self.worker.signals.progress.connect(self._progress)
            self.worker.signals.result.connect(self._analysis_complete)
            self.worker.signals.error.connect(self._analysis_error)
            self.worker.signals.cancelled.connect(self._analysis_cancelled)
            self.progress.setVisible(True)
            self.run_action.setEnabled(False)
            self.track_button.setEnabled(False)
            self.cancel_action.setEnabled(True)
            self.thread_pool.start(self.worker)
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
        self.profile.set_result(self.result)
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
            status_text = {
                PickStatus.HIGH_CONFIDENCE: "Automatic · accepted",
                PickStatus.ACCEPTED: "Analyst-confirmed",
                PickStatus.REVIEW: "Review · measurement withheld",
                PickStatus.UNRESOLVED: "Unresolved · measurement withheld",
            }.get(item.status, str(item.status).replace("_", " ").title())
            values = [
                f"{item.chainage_m:.1f}",
                item.layer_name,
                "—" if item.thickness_mm is None else f"{item.thickness_mm:.1f}",
                f"{item.confidence:.0%}",
                status_text,
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
                f"{issue.start_chainage_m:.1f}–{issue.end_chainage_m:.1f} m\n"
                f"Next: {issue.suggested_action}"
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
        design_aid_used = bool(
            self.design_segments
            or any(item.thickness_mm is not None for item in self.options.layer_designs)
        )
        identity_orders = [
            int(order) for order in self.result.parameters.get("required_seed_orders", [])
        ]
        identity_state = (
            "provisional; TWTT and physical results withheld for layers "
            + ", ".join(map(str, identity_orders))
            if identity_orders
            else "manual observation requirement complete"
        )
        fine_plan = self.result.parameters.get("automatic_fine_retrack_plan", {})
        fine_state = (
            f"{fine_plan.get('policy', 'disabled')} · "
            f"{len(fine_plan.get('selected_windows_m', []))} window(s)"
        )
        self.method_text.setText(
            f"Stacking: {self.result.stack_size} traces per coarse bin\n"
            f"Plate valid for dielectric: {self.result.diagnostics.valid_for_dielectric}\n"
            f"Gain compatible: {self.result.diagnostics.gain_compatible}\n"
            f"Reference surface sample: {self.result.reference_surface_sample}\n"
            f"Seed stations: {len(self.result.seed_stations)}\n"
            f"Reflector identity: {identity_state}\n"
            f"Optional design aid used: {design_aid_used}\n\n"
            "Path display: solid segments are accepted measurements. Dashed "
            "segments are provisional graph proposals only; they never produce "
            "TWTT or thickness until an analyst confirms or corrects them. "
            "Dotted/dash-dot overlays are search or design aids, not measurements.\n\n"
            f"Automatic fine retracking: {fine_state}\n\n"
            f"Processing branches:\n  - {preprocessing}\n\n• {notes}{reference_note}"
        )
        self.calibration_label.setText(
            "Calibration passed for amplitude dielectric."
            if self.result.diagnostics.valid_for_dielectric
            else (
                "Amplitude dielectric unavailable; interface TWTT remains independently reviewable."
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
        decision_details = _review_proposal_snapshot(self.result, issue)
        try:
            resolve_review_issue(self.result, self.options, issue.issue_id, action)
        except ValueError as exc:
            QMessageBox.information(self, "Review decision not applied", str(exc))
            self.statusBar().showMessage(str(exc))
            return
        if self.project_store:
            self.project_store.record_review_event(
                action,
                issue_id=issue.issue_id,
                layer_order=issue.layer_order,
                start_chainage_m=issue.start_chainage_m,
                end_chainage_m=issue.end_chainage_m,
                details=decision_details,
            )
        self.radar.set_result(self.result)
        self.profile.set_result(self.result)
        self._populate_results()
        self._populate_issues()
        if self.project_store:
            self.project_store.save_analysis(
                self.result, str(self.plate.dzt_path) if self.plate else None
            )
        self.statusBar().showMessage(f"Recorded review decision: {action.replace('_', ' ')}.")

    def prepare_issue_correction(self):
        issue = self._selected_issue()
        if issue is None:
            self.statusBar().showMessage("Select a review region first.")
            return
        index = self.active_layer_combo.findData(issue.layer_order)
        self.active_layer_combo.setCurrentIndex(index)
        correction_index = self.seed_combo.findData("correction", Qt.ItemDataRole.UserRole + 1)
        if correction_index >= 0:
            self.seed_combo.setCurrentIndex(correction_index)
        target = issue.suggested_chainage_m or (issue.start_chainage_m + issue.end_chainage_m) / 2
        self.radar.focus_chainage(target, 20.0)
        self.statusBar().showMessage(
            f"Ctrl+click the actual {issue.layer_name} reflector near {target:.1f} m; "
            "the dashed path is only a proposal and produces no measurement."
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
            f"Imported {len(self.design_segments)} chainage overrides; rebuilding design corridors."
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
