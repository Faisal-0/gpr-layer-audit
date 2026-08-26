from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from gpr_layer_audit.models import AcquisitionFileSet
from gpr_layer_audit.processing import AnalysisOptions, analyze_acquisition
from gpr_layer_audit.ui.main_window import MainWindow
from gpr_layer_audit.ui.theme import APP_STYLESHEET


def main() -> int:
    road = Path("GPR Data/talagang/TALAGANG.PRJ/TALAGANG_001.DZT")
    plate = Path("GPR Data/talagang/TALAGANG METAL PLATE.PRJ/TALAGANG METAL PLATE_001.DZT")
    result = analyze_acquisition(
        AcquisitionFileSet(road),
        AcquisitionFileSet(plate),
        AnalysisOptions(stack_size=20, accept_scan_dielectric=True),
    )
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLESHEET)
    pg.setConfigOptions(imageAxisOrder="row-major", antialias=True, background="#071016")
    window = MainWindow()
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
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
