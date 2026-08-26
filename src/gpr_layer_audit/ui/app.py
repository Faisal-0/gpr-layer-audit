from __future__ import annotations

import sys

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from .main_window import MainWindow
from .theme import APP_STYLESHEET


def main() -> int:
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("GPR Layer Audit")
    app.setOrganizationName("GPR Layer Audit Team")
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLESHEET)
    pg.setConfigOptions(imageAxisOrder="row-major", antialias=True, background="#071016")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
