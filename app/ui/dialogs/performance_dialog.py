"""Performance settings: profile, preview, proxies, render backend, cache, background work and diagnostics.

The dialog only talks to ``ws.performance`` (the service owns every decision). Settings can be edited globally or as overrides for the open project;
options the machine cannot honour are disabled with the reason beside them (a graphics card alone never enables hardware rendering).
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit, QPushButton, QSpinBox,
    QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from app.performance import bottleneck_report as report
from app.performance.settings import CACHE_CATEGORIES, MB, PROXY_PROFILE_RESOLUTION, PerformanceSettings
from app.ui.dialogs.message import show_error

PROFILE_LABELS = {"power_saver": "Power Saver", "balanced": "Balanced", "performance": "Performance"}
PROFILE_HELP = {
    "power_saver": "Fewer background workers, conservative caching and lightweight previews. Best on small or busy computers.",
    "balanced": "Moderate background work and standard preview quality.",
    "performance": "More background work and preview prefetching within safe limits (never every core or all memory); validated hardware may be used.",
}
PREVIEW_LABELS = {"draft": "Draft (fastest, smaller frames)", "balanced": "Balanced", "high": "High (largest frames)"}
PROXY_POLICY_LABELS = {"off": "Off", "manual": "Manual", "automatic": "Automatic"}
PROXY_PROFILE_LABELS = {"performance": "Performance (540p)", "balanced": "Balanced (720p)", "quality": "Quality (1080p)"}
BACKEND_LABELS = {"auto": "Auto (recommended)", "cpu": "CPU", "hardware": "Hardware"}
CATEGORY_LABELS = {"thumbnails": "Thumbnails", "preview_frames": "Preview frames", "proxies": "Proxies", "waveforms": "Waveforms", "analysis": "Analysis results",
                   "render_previews": "Render previews", "temporary": "Temporary files"}
NOTE = "Changing the editing profile never silently lowers final export quality: exports read the original media at the settings chosen on the Export page."


def _human(n: int | float | None) -> str:
    if n is None:
        return "n/a"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


class PerformanceDialog(QDialog):
    def __init__(self, ws, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ws = ws
        self.svc = ws.performance
        self.setWindowTitle("Performance")
        self.setMinimumSize(640, 560)
        self._has_project = ws.project is not None
        self.tabs = QTabWidget()

        # ---- scope ----
        self.scope = QComboBox()
        self.scope.addItem("All projects (global)", "global")
        self.scope.addItem("This project only (overrides)", "project")
        if not self._has_project:
            self.scope.model().item(1).setEnabled(False)
            self.scope.setItemData(1, "Open a project to set project-specific values.", Qt.ItemDataRole.ToolTipRole)
        self.scope.currentIndexChanged.connect(self._load_scope)
        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("Edit settings for:"))
        scope_row.addWidget(self.scope, 1)

        self.tabs.addTab(self._build_general(), "General")
        self.tabs.addTab(self._build_cache(), "Cache")
        self.tabs.addTab(self._build_diagnostics(), "Diagnostics")

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Close)
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.Save).clicked.connect(self.save)
        buttons.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(scope_row)
        lay.addWidget(self.tabs, 1)
        lay.addWidget(buttons)
        self._load_scope()
        self.refresh_cache()
        self.refresh_diagnostics()

    # ------------------------------------------------------------------ general
    def _build_general(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.profile = QComboBox()
        for k, v in PROFILE_LABELS.items():
            self.profile.addItem(v, k)
        self.profile_help = QLabel(PROFILE_HELP["balanced"])
        self.profile_help.setWordWrap(True)
        self.profile.currentIndexChanged.connect(lambda: self.profile_help.setText(PROFILE_HELP[self.profile.currentData()]))
        self.note = QLabel(NOTE)
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color: gray")
        self.preview = QComboBox()
        for k, v in PREVIEW_LABELS.items():
            self.preview.addItem(v, k)
        self.preview.setToolTip("Preview resolution only. It never changes the export resolution.")
        self.proxy_policy = QComboBox()
        for k, v in PROXY_POLICY_LABELS.items():
            self.proxy_policy.addItem(v, k)
        self.proxy_profile = QComboBox()
        for k, v in PROXY_PROFILE_LABELS.items():
            self.proxy_profile.addItem(v, k)
        self.backend = QComboBox()
        for k, v in BACKEND_LABELS.items():
            self.backend.addItem(v, k)
        self.backend_note = QLabel("")
        self.backend_note.setWordWrap(True)
        self.workers = QSpinBox()
        self.workers.setRange(0, 64)
        self.workers.setSpecialValueText("Automatic")
        self.workers.setToolTip("Maximum background worker threads (never more than the CPU count). Automatic follows the profile.")
        self.bg_proxy = QCheckBox("Allow proxy generation in the background")
        self.idle_cache = QCheckBox("Prepare thumbnails, waveforms and frames when the app is idle")
        self.metrics = QCheckBox("Collect performance metrics (lightweight timers; no content is recorded)")
        self.limits_label = QLabel("")
        self.limits_label.setWordWrap(True)
        f.addRow("Performance profile", self.profile)
        f.addRow("", self.profile_help)
        f.addRow("", self.note)
        f.addRow("Preview quality", self.preview)
        f.addRow("Proxy media", self.proxy_policy)
        f.addRow("Proxy profile", self.proxy_profile)
        f.addRow("Render backend", self.backend)
        f.addRow("", self.backend_note)
        f.addRow("Background workers", self.workers)
        f.addRow("", self.bg_proxy)
        f.addRow("", self.idle_cache)
        f.addRow("", self.metrics)
        f.addRow("In effect", self.limits_label)
        self.proxy_policy.currentIndexChanged.connect(self._sync_proxy_enabled)
        return w

    def _sync_proxy_enabled(self) -> None:
        on = self.proxy_policy.currentData() != "off"
        self.proxy_profile.setEnabled(on)
        self.bg_proxy.setEnabled(on)
        self.proxy_profile.setToolTip("" if on else "Proxy media is switched off.")

    def _hardware_ok(self) -> tuple[bool, str]:
        s = self.svc.hardware_summary()
        enc = (s.get("encoders") or {}).get("hardware") or {}
        working = [k for k, v in enc.items() if v]
        if working:
            return True, "Hardware encoders that passed a test on this computer: " + ", ".join(sorted(working)) + "."
        pending = s.get("pending") or []
        if pending:
            return False, "Hardware capabilities have not been checked yet. Use Diagnostics → Check hardware."
        return False, "No hardware video encoder passed a test on this computer (a graphics card alone is not enough). CPU rendering is used."

    def _load_scope(self) -> None:
        project_scope = self.scope.currentData() == "project"
        s = self.svc.settings() if project_scope else self.svc.global_settings()
        self.profile.setCurrentIndex(self.profile.findData(s.profile))
        self.profile_help.setText(PROFILE_HELP[s.profile])
        self.preview.setCurrentIndex(self.preview.findData(s.preview_quality))
        self.proxy_policy.setCurrentIndex(self.proxy_policy.findData(s.proxy_policy))
        self.proxy_profile.setCurrentIndex(self.proxy_profile.findData(s.proxy_profile))
        ok, why = self._hardware_ok()
        item = self.backend.model().item(self.backend.findData("hardware"))
        item.setEnabled(ok)
        self.backend.setItemData(self.backend.findData("hardware"), why, Qt.ItemDataRole.ToolTipRole)
        self.backend_note.setText(why)
        self.backend.setCurrentIndex(self.backend.findData(s.render_backend if (ok or s.render_backend != "hardware") else "auto"))
        self.workers.setValue(s.max_background_workers)
        self.bg_proxy.setChecked(s.background_proxy)
        self.idle_cache.setChecked(s.idle_cache_generation)
        self.metrics.setChecked(s.metrics_enabled)
        self._sync_proxy_enabled()
        lim = self.svc.limits()
        self.limits_label.setText(f"{lim.background_workers} background worker(s), {lim.foreground_workers} reserved for foreground work; in-memory caches up to {_human(lim.memory_cache_bytes)}; "
                                  f"rebuildable cache up to {_human(lim.cache_total_bytes)}.")

    def collect(self) -> PerformanceSettings:
        cur = self.svc.global_settings() if self.scope.currentData() == "global" else self.svc.settings()
        d = cur.to_dict()
        d.update(profile=self.profile.currentData(), preview_quality=self.preview.currentData(), proxy_policy=self.proxy_policy.currentData(), proxy_profile=self.proxy_profile.currentData(),
                 render_backend=self.backend.currentData(), max_background_workers=self.workers.value(), background_proxy=self.bg_proxy.isChecked(),
                 idle_cache_generation=self.idle_cache.isChecked(), metrics_enabled=self.metrics.isChecked())
        d["cache_limit_mb"] = self.cache_limit.value()
        return PerformanceSettings.from_dict(d)

    def save(self) -> None:
        try:
            new = self.collect()
            if self.scope.currentData() == "global":
                self.svc.update_global(new)
            else:
                base = self.svc.global_settings().to_dict()
                self.svc.set_project_overrides({k: v for k, v in new.to_dict().items() if base.get(k) != v})
            self._load_scope()
            self.refresh_cache()
        except Exception as exc:  # noqa: BLE001
            show_error(self, "The performance settings could not be saved.", str(exc))

    # ------------------------------------------------------------------ cache
    def _build_cache(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        top = QFormLayout()
        self.cache_limit = QSpinBox()
        self.cache_limit.setRange(0, 10_000_000)
        self.cache_limit.setSuffix(" MB")
        self.cache_limit.setSpecialValueText("Automatic (a share of free disk space)")
        self.cache_limit.setValue(self.svc.global_settings().cache_limit_mb)
        self.cache_loc = QLabel("")
        self.cache_loc.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.cache_loc.setWordWrap(True)
        self.disk_label = QLabel("")
        self.total_label = QLabel("")
        self.cleanup_label = QLabel("")
        top.addRow("Rebuildable cache limit", self.cache_limit)
        top.addRow("Location", self.cache_loc)
        top.addRow("Current size", self.total_label)
        top.addRow("Disk space free", self.disk_label)
        top.addRow("Last cleanup", self.cleanup_label)
        lay.addLayout(top)
        self.table = QTableWidget(len(CACHE_CATEGORIES), 4)
        self.table.setHorizontalHeaderLabels(["Category", "Entries", "Size", "Limit"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        self.btn_tmp = QPushButton("Clean temporary files")
        self.btn_cache = QPushButton("Clean rebuildable cache")
        self.btn_proxies = QPushButton("Remove selected proxies…")
        self.btn_tmp.clicked.connect(self._clean_temp)
        self.btn_cache.clicked.connect(self._clean_cache)
        self.btn_proxies.clicked.connect(self._remove_proxies)
        for b in (self.btn_tmp, self.btn_cache, self.btn_proxies):
            row.addWidget(b)
        lay.addLayout(row)
        self.cache_msg = QLabel("Cleaning removes only rebuildable cache files; imported media and exports are never deleted.")
        self.cache_msg.setWordWrap(True)
        lay.addWidget(self.cache_msg)
        return w

    def refresh_cache(self) -> None:
        st = self.svc.cache_stats()
        avail = st.get("available", False)
        cache = self.svc.cache
        self.cache_loc.setText(str(cache.root) if cache is not None and cache.root else "Open a project to see its cache.")
        cats = st.get("categories", {})
        for r, c in enumerate(CACHE_CATEGORIES):
            info = cats.get(c, {})
            for col, text in enumerate((CATEGORY_LABELS[c], str(info.get("entries", 0)), _human(info.get("bytes", 0)), _human(info.get("limit_bytes")) if info.get("limit_bytes") else "—")):
                self.table.setItem(r, col, QTableWidgetItem(text))
        self.total_label.setText(_human(st.get("total_bytes", 0)) + (f" of {_human(st['total_limit_bytes'])}" if st.get("total_limit_bytes") else ""))
        self.disk_label.setText(_human(st.get("disk_free_bytes")) if st.get("disk_free_bytes") is not None else _human((self.svc.monitor.latest() or self.svc.monitor.sample()).disk_free_bytes))
        self.cleanup_label.setText(st.get("last_cleanup") or "never")
        for b in (self.btn_tmp, self.btn_cache):
            b.setEnabled(bool(avail))
        self.btn_proxies.setEnabled(self._has_project and bool(self.ws.render.proxies.records()))

    def _clean_temp(self) -> None:
        r = self.svc.clear_rebuildable_cache(["temporary"]) or {}
        self.svc.cleanup_cache(force=True)
        self.cache_msg.setText(f"Temporary files cleaned ({_human(r.get('freed_bytes', 0))} freed).")
        self.refresh_cache()

    def _clean_cache(self) -> None:
        if QMessageBox.question(self, "Clean rebuildable cache", "Remove thumbnails, preview frames, waveforms and other rebuildable cache files? They are recreated when needed. "
                                "Your media, proxies in use and exports are not touched.") != QMessageBox.StandardButton.Yes:
            return
        r = self.svc.clear_rebuildable_cache() or {}
        self.cache_msg.setText(f"Cache cleaned ({_human(r.get('freed_bytes', 0))} freed); skipped: {r.get('skipped') or 'nothing'}.")
        self.refresh_cache()

    def _remove_proxies(self) -> None:
        recs = self.ws.render.proxies.records()
        dlg = QDialog(self)
        dlg.setWindowTitle("Remove proxies")
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel("Select proxies to remove. Originals are never touched; proxies are regenerated on request."))
        lst = QListWidget()
        lst.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        for aid, r in sorted(recs.items()):
            asset = self.ws.project.assets.get(aid) if self.ws.project else None
            it = QListWidgetItem(f"{asset.name if asset else aid}  —  {r.proxy_resolution}, {_human(r.size_bytes)}, {r.proxy_status}")
            it.setData(Qt.ItemDataRole.UserRole, aid)
            lst.addItem(it)
        lay.addWidget(lst)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        if dlg.exec() and lst.selectedItems():
            ids = [i.data(Qt.ItemDataRole.UserRole) for i in lst.selectedItems()]
            remover = getattr(self.ws.render.proxies, "remove_selected", None) or self.ws.render.proxies.delete
            n = remover(ids)
            self.cache_msg.setText(f"Removed {n if isinstance(n, int) else len(ids)} proxy file(s).")
            self.refresh_cache()

    # ------------------------------------------------------------------ diagnostics
    def _build_diagnostics(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.diag = QPlainTextEdit()
        self.diag.setReadOnly(True)
        lay.addWidget(self.diag, 1)
        row = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh")
        self.btn_hw = QPushButton("Check hardware")
        self.btn_export = QPushButton("Export report…")
        self.btn_refresh.clicked.connect(self.refresh_diagnostics)
        self.btn_hw.clicked.connect(self._check_hardware)
        self.btn_export.clicked.connect(self._export)
        for b in (self.btn_refresh, self.btn_hw, self.btn_export):
            row.addWidget(b)
        lay.addLayout(row)
        return w

    def refresh_diagnostics(self) -> None:
        try:
            rep = self.svc.diagnostics()
            hw = self.svc.hardware_summary()
            text = report.render_text(rep)
            text += "\n\nHardware & FFmpeg\n" + (hw.get("text") or "not checked yet")
            if hw.get("pending"):
                text += "\n(pending: " + ", ".join(hw["pending"]) + ")"
            self.diag.setPlainText(text)
        except Exception as exc:  # noqa: BLE001 - diagnostics must never break the dialog
            self.diag.setPlainText(f"Diagnostics are unavailable: {exc}")

    def _check_hardware(self) -> None:
        job = self.svc.detect_hardware_async(force=True)
        self.diag.setPlainText("Checking hardware capabilities in the background…" if job else "Hardware detection is not available.")
        if job is not None:
            self.ws.bus.subscribe("performance.updated", lambda t, p: (self.refresh_diagnostics(), self._load_scope()) if p.get("kind") == "hardware" else None)

    def _export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export performance report", str(Path.home() / "agenttool-performance.json"), "JSON (*.json);;Text (*.txt)")
        if path:
            try:
                self.svc.export_diagnostics(Path(path), as_text=path.lower().endswith(".txt"))
                self.diag.appendPlainText(f"\nReport saved to {path}")
            except Exception as exc:  # noqa: BLE001
                show_error(self, "The report could not be saved.", str(exc))
