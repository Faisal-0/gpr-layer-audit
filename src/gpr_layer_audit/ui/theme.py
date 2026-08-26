from __future__ import annotations

APP_STYLESHEET = """
QWidget {
    background: #0d1720;
    color: #dbe7ed;
    font-family: "Segoe UI Variable", "Segoe UI", sans-serif;
    font-size: 13px;
}
QLabel#sectionTitle {
    color: #9eb3bd;
    font-weight: 700;
    letter-spacing: 1px;
    padding-top: 8px;
    padding-bottom: 3px;
}
QLabel#secondaryText { color: #9eb3bd; padding: 7px 0; }
QLabel#warningText { color: #ffc857; padding: 8px 0; }
QMainWindow, QDialog { background: #0d1720; }
QToolBar {
    background: #13222d;
    border: 0;
    border-bottom: 1px solid #2b4350;
    spacing: 4px;
    padding: 6px 10px;
}
QToolButton, QPushButton {
    background: #1a303d;
    border: 1px solid #365463;
    border-radius: 4px;
    padding: 7px 12px;
    color: #edf7fa;
}
QToolButton:hover, QPushButton:hover { background: #234453; border-color: #4a7180; }
QToolButton:pressed, QPushButton:pressed { background: #102630; }
QToolButton:disabled, QPushButton:disabled {
    color: #718691;
    background: #14232c;
    border-color: #263c47;
}
QPushButton[primary="true"] { background: #087f8c; border-color: #28d7e5; font-weight: 600; }
QPushButton[primary="true"]:hover { background: #0c98a5; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background: #101f29;
    border: 1px solid #365463;
    border-radius: 3px;
    padding: 6px;
    selection-background-color: #087f8c;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 2px solid #28d7e5;
}
QCheckBox:focus, QToolButton:focus, QPushButton:focus,
QListWidget:focus, QTableWidget:focus {
    border: 1px solid #28d7e5;
}
QGroupBox {
    border: 1px solid #2b4350;
    margin-top: 14px;
    padding: 12px 8px 8px 8px;
    font-weight: 600;
    color: #b9ccd5;
}
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
QListWidget, QTableWidget, QTreeWidget {
    background: #0a131a;
    alternate-background-color: #101f29;
    border: 1px solid #2b4350;
    outline: none;
}
QListWidget::item, QTreeWidget::item { padding: 7px 5px; }
QListWidget::item:selected, QTreeWidget::item:selected { background: #164b58; color: white; }
QListWidget::item:disabled { color: #9eb3bd; }
QHeaderView::section {
    background: #182a35;
    color: #bcd0d9;
    border: 0;
    border-right: 1px solid #2b4350;
    border-bottom: 1px solid #2b4350;
    padding: 6px;
    font-weight: 600;
}
QTabWidget::pane { border: 1px solid #2b4350; }
QTabBar::tab { background: #13222d; padding: 8px 12px; border: 1px solid #2b4350; }
QTabBar::tab:selected { background: #1a303d; border-bottom-color: #28d7e5; }
QProgressBar {
    background: #101f29;
    border: 1px solid #2b4350;
    border-radius: 3px;
    text-align: center;
}
QProgressBar::chunk { background: #28d7e5; }
QStatusBar { background: #13222d; border-top: 1px solid #2b4350; }
QSplitter::handle { background: #2b4350; width: 1px; height: 1px; }
QScrollBar:vertical { background: #0d1720; width: 12px; }
QScrollBar::handle:vertical { background: #365463; min-height: 24px; border-radius: 4px; }
QScrollBar:horizontal { background: #0d1720; height: 12px; }
QScrollBar::handle:horizontal { background: #365463; min-width: 24px; border-radius: 4px; }
QToolTip { background: #edf7fa; color: #102029; border: 1px solid #5f7782; padding: 4px; }
"""

LAYER_COLOURS = {1: "#28d7e5", 2: "#ffc857", 3: "#ff6b6b"}
LAYER_DASHES = {1: None, 2: [8, 4], 3: [3, 3]}
