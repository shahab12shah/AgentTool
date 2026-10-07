"""Application stylesheet built from a small colour palette (dark and light)."""

from __future__ import annotations

PALETTES: dict[str, dict[str, str]] = {
    "dark": {
        "bg": "#17191d", "panel": "#1f2227", "panel2": "#262a31", "border": "#33383f",
        "text": "#e4e7ec", "muted": "#8d95a3", "accent": "#4aa3ff", "accent_text": "#08121f",
        "danger": "#ff6b6b", "ok": "#4cc38a", "clip_video": "#2f6fb3", "clip_image": "#8a5cc2",
        "clip_audio": "#2d9a78", "selection": "#ffd166",
    },
    "light": {
        "bg": "#f3f4f6", "panel": "#ffffff", "panel2": "#eceef2", "border": "#cfd4dc",
        "text": "#1c2230", "muted": "#667085", "accent": "#1769d1", "accent_text": "#ffffff",
        "danger": "#c62828", "ok": "#1b8a5a", "clip_video": "#5b9bd8", "clip_image": "#a98ad6",
        "clip_audio": "#5cc2a0", "selection": "#d98a00",
    },
}


def palette(theme: str) -> dict[str, str]:
    return PALETTES.get(theme, PALETTES["dark"])


def stylesheet(theme: str) -> str:
    c = palette(theme)
    return f"""
* {{ font-size: 13px; }}
QWidget {{ background: {c['bg']}; color: {c['text']}; }}
QMainWindow, QDialog {{ background: {c['bg']}; }}
QLabel {{ background: transparent; }}
QToolBar QWidget {{ background: transparent; }}
QListWidget#nav::item:disabled {{ color: {c['muted']}; }}
QToolBar {{ background: {c['panel']}; border: none; border-bottom: 1px solid {c['border']}; spacing: 6px; padding: 4px 8px; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 5px; padding: 5px 10px; }}
QToolButton:hover {{ background: {c['panel2']}; border-color: {c['border']}; }}
QToolButton:disabled {{ color: {c['muted']}; }}
QPushButton {{ background: {c['panel2']}; border: 1px solid {c['border']}; border-radius: 5px; padding: 6px 14px; }}
QPushButton:hover {{ border-color: {c['accent']}; }}
QPushButton:disabled {{ color: {c['muted']}; background: {c['panel']}; }}
QPushButton#primary {{ background: {c['accent']}; color: {c['accent_text']}; border-color: {c['accent']}; font-weight: 600; }}
QPushButton#primary:disabled {{ background: {c['panel2']}; color: {c['muted']}; }}
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
  background: {c['panel']}; border: 1px solid {c['border']}; border-radius: 5px; padding: 4px 6px;
  selection-background-color: {c['accent']}; selection-color: {c['accent_text']}; }}
QLineEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {c['accent']}; }}
QComboBox QAbstractItemView {{ background: {c['panel']}; selection-background-color: {c['accent']}; selection-color: {c['accent_text']}; }}
QListWidget, QTableWidget, QTreeWidget {{ background: {c['panel']}; border: 1px solid {c['border']}; border-radius: 5px; }}
QListWidget::item {{ border-radius: 5px; padding: 4px; }}
QListWidget::item:selected {{ background: {c['panel2']}; border: 1px solid {c['accent']}; color: {c['text']}; }}
QListWidget#nav {{ background: {c['panel']}; border: none; border-right: 1px solid {c['border']}; padding-top: 8px; }}
QListWidget#nav::item {{ padding: 10px 16px; margin: 1px 6px; }}
QListWidget#nav::item:selected {{ background: {c['panel2']}; border: none; border-left: 3px solid {c['accent']}; }}
QHeaderView::section {{ background: {c['panel2']}; border: none; padding: 4px 8px; color: {c['muted']}; }}
QGroupBox {{ border: 1px solid {c['border']}; border-radius: 6px; margin-top: 12px; padding: 10px 8px 8px 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {c['muted']}; }}
QLabel#title {{ font-size: 22px; font-weight: 600; }}
QLabel#muted {{ color: {c['muted']}; }}
QLabel#placeholder {{ color: {c['muted']}; font-size: 16px; }}
QLabel#logo {{ font-weight: 700; font-size: 15px; color: {c['accent']}; padding-right: 8px; }}
QLabel#projectname {{ font-weight: 600; font-size: 14px; padding: 0 12px; }}
QFrame#failureBox {{ background: {c['panel2']}; border: 1px solid {c['danger']}; border-radius: 6px; padding: 6px; }}
QStatusBar, QFrame#jobbar {{ background: {c['panel']}; border-top: 1px solid {c['border']}; }}
QProgressBar {{ background: {c['panel2']}; border: 1px solid {c['border']}; border-radius: 4px; text-align: center; height: 14px; }}
QProgressBar::chunk {{ background: {c['accent']}; border-radius: 3px; }}
QSlider::groove:horizontal {{ height: 4px; background: {c['panel2']}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {c['accent']}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 12px; margin: -5px 0; border-radius: 6px; background: {c['text']}; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {c['muted']}; border-radius: 3px; background: {c['panel']}; }}
QCheckBox::indicator:checked {{ background: {c['accent']}; border-color: {c['accent']}; }}
QCheckBox::indicator:hover {{ border-color: {c['accent']}; }}
QScrollArea {{ border: none; }}
QScrollBar:horizontal {{ background: {c['panel']}; height: 12px; }}
QScrollBar:vertical {{ background: {c['panel']}; width: 12px; }}
QScrollBar::handle {{ background: {c['border']}; border-radius: 5px; min-width: 24px; min-height: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QSplitter::handle {{ background: {c['border']}; }}
QMenu {{ background: {c['panel']}; border: 1px solid {c['border']}; padding: 4px; }}
QMenu::item {{ padding: 6px 22px; border-radius: 4px; }}
QMenu::item:selected {{ background: {c['accent']}; color: {c['accent_text']}; }}
QMenu::item:disabled {{ color: {c['muted']}; }}
QToolTip {{ background: {c['panel2']}; color: {c['text']}; border: 1px solid {c['border']}; }}
"""
