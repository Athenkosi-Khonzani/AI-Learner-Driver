APP_QSS = r"""
QWidget {
    background: #171613;
    color: #ece9e3;
    font-family: "Segoe UI", "Noto Sans", sans-serif;
    font-size: 12px;
}
QMainWindow, QDialog { background: #171613; }
/* Labels inherit QWidget styling in Qt. Keep them transparent so text sits cleanly
   on cards, the top bar and status pills instead of drawing dark rectangles. */
QLabel { background: transparent; border: none; }
QFrame#topBar { background: #1d1c18; border-bottom: 1px solid #2d2b27; }
QFrame#panel { background: #1b1a16; }
QFrame#card { background: #1c1b17; border: 1px solid #34322e; border-radius: 11px; }
QLabel#appTitle { color: #aaa7a0; font-size: 13px; font-weight: 600; }
QLabel#sectionTitle { color: #ece9e3; font-size: 20px; font-weight: 700; }
QLabel#eyebrow { color: #77746d; font-family: Consolas, "Courier New", monospace; font-size: 10px; font-weight: 600; }
QLabel#muted { color: #85827b; }
QLabel#metricValue { font-family: Consolas, "Courier New", monospace; font-size: 22px; font-weight: 600; }
QLabel#metricLabel { color: #78756e; font-family: Consolas, "Courier New", monospace; font-size: 9px; font-weight: 600; }
QPushButton {
    background: #24231f;
    border: 1px solid #393732;
    border-radius: 8px;
    padding: 7px 12px;
    color: #d9d6cf;
}
QPushButton:hover { background: #2d2b26; border-color: #4a4741; }
QPushButton:pressed { background: #181714; }
QPushButton:disabled { color: #5f5c56; background: #1c1b18; border-color: #292824; }
QPushButton#primary {
    background: #78beff;
    border-color: #78beff;
    color: #0d1a2a;
    font-weight: 700;
    padding: 9px 18px;
}
QPushButton#primary:hover { background: #92cbff; }
QPushButton#navButton { background: transparent; border: none; color: #85827b; padding: 6px 14px; }
QPushButton#navButton:checked { background: #e9e5dd; color: #17161a; border-radius: 6px; font-weight: 600; }
QPushButton#chip { border-radius: 15px; padding: 5px 11px; color: #9f9c95; background: #1d1c19; }
QLineEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background: #181714;
    border: 1px solid #393732;
    border-radius: 8px;
    padding: 7px 10px;
    color: #ece9e3;
    selection-background-color: #355b7e;
}
QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #608bb2; }
QListWidget, QTreeWidget {
    background: transparent;
    border: none;
    outline: none;
}
QListWidget::item { padding: 9px 10px; border-radius: 7px; color: #a4a199; }
QListWidget::item:selected { background: #24313d; color: #ece9e3; }
QListWidget::item:hover { background: #22211d; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: #171613; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #3a3833; border-radius: 5px; min-height: 25px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QCheckBox { color: #c5c2bb; spacing: 8px; }
QCheckBox::indicator { width: 30px; height: 17px; }
QCheckBox::indicator:unchecked { image: none; background: #3a3833; border-radius: 8px; }
QCheckBox::indicator:checked { image: none; background: #78beff; border-radius: 8px; }
QGroupBox { border: 1px solid #34322e; border-radius: 11px; margin-top: 12px; padding: 14px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; color: #ece9e3; }
QTabBar::tab { background: #1b1a16; padding: 8px 14px; color: #88857f; }
QTabBar::tab:selected { color: #ece9e3; background: #262520; }
QToolTip { background: #24231f; color: #ece9e3; border: 1px solid #44413b; }
"""

ACCENT = "#78beff"
SUCCESS = "#59b662"
DANGER = "#e0655a"
MUTED = "#85827b"
