from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "windows" if sys.platform == "win32" else "offscreen")

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from gpr_layer_audit.models import AcquisitionFileSet, LayerDesign
from gpr_layer_audit.processing import AnalysisOptions, analyze_acquisition
from gpr_layer_audit.ui.main_window import CatalogDialog, MainWindow
from gpr_layer_audit.ui.theme import APP_STYLESHEET


def main() -> int:
    road = Path("GPR Data/talagang/TALAGANG.PRJ/TALAGANG_001.DZT")
    plate = Path(
        "GPR Data/talagang/TALAGANG METAL PLATE.PRJ/"
        "TALAGANG METAL PLATE_001.DZT"
    )
    designs = [
        LayerDesign(1, "Asphalt", 50.8, 7.0),
        LayerDesign(2, "Base course", 101.6, 7.0),
        LayerDesign(3, "Sub-base course", None, 7.0),
    ]
    result = analyze_acquisition(
        AcquisitionFileSet(road),
        AcquisitionFileSet(plate),
        AnalysisOptions(
            stack_size=20,
            layer_designs=designs,
            auto_fine_retrack=False,
        ),
    )
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLESHEET)
    pg.setConfigOptions(imageAxisOrder="row-major", antialias=True, background="#071016")
    window = MainWindow()
    window.options.layer_designs = designs
    window.resize(1500, 900)
    window.source_label.setText(f"{road.name}\n{road.parent}")
    window.assumption_label.setText("Dielectric: scan value accepted as assumed")
    window._analysis_complete(result)
    window.show()
    app.processEvents()
    output = Path(".impeccable/review/gpr-layer-audit-workbench.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not window.grab().save(str(output)):
        raise RuntimeError("Could not save UI capture")
    window.workspace_tabs.setCurrentIndex(1)
    app.processEvents()
    profile_output = Path(".impeccable/review/gpr-layer-profiles.png")
    if not window.grab().save(str(profile_output)):
        raise RuntimeError("Could not save profile capture")
    catalog = CatalogDialog(window)
    catalog.root_edit.setText("GPR Data/talagang")
    catalog.scan()
    catalog.show()
    app.processEvents()
    catalog_output = Path(".impeccable/review/gpr-quick-design.png")
    if not catalog.grab().save(str(catalog_output)):
        raise RuntimeError("Could not save catalog capture")
    print(output.resolve())
    print(profile_output.resolve())
    print(catalog_output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
