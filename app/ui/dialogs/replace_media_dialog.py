"""Pick the media that replaces a clip's media: the project's library, and - for clips that belong to a scene - the research candidates that are already in the project.

The dialog only chooses; the caller replaces the media through the engine (``ws.editing.replace_clip_asset`` for AI scene clips, ``ws.edit.replace_clip`` otherwise).
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget

from app.media.asset import Asset, AssetType
from app.ui.context import UiContext

MAX_ICONS = 150  # thumbnails are read from disk; a very large library shows plain rows


def compatible_assets(ctx: UiContext, clip_id: str) -> list[Asset]:
    """Library assets that can replace the clip's media: pictures for picture clips, sound for audio clips (the clip's current asset is left out)."""
    project = ctx.ws.project
    if project is None:
        return []
    try:
        track, clip = project.timeline.find_clip(clip_id)
    except Exception:  # noqa: BLE001
        return []
    want = {AssetType.AUDIO} if track.is_audio else {AssetType.VIDEO, AssetType.IMAGE}
    return sorted((a for a in project.assets.all() if a.type in want and a.id != clip.asset_id), key=lambda a: (a.type.value, a.name.lower()))


def scene_candidates(ctx: UiContext, scene_id: str) -> list[dict]:
    """Research candidates of the scene: ``{candidate_id, title, asset_id (None = not in the project), role, score, assigned}`` best first. Read from the project data only."""
    project = ctx.ws.project
    if project is None or not scene_id:
        return []
    assigned = project.visual_assignments.get(scene_id)
    state = project.research_status.get(scene_id)
    roles = {r.candidate_id: r.role for r in (state.ranked if state else [])}
    out = []
    for cid, c in project.visual_candidates.items():
        if c.scene_id != scene_id:
            continue
        score = project.candidate_scores.get(cid)
        out.append({"candidate_id": cid, "title": c.title or c.provider_id or cid, "asset_id": c.asset_id if c.asset_id and project.assets.get(c.asset_id) else None,
                    "role": roles.get(cid, ""), "score": score.overall if score else None, "kind": c.kind, "assigned": bool(assigned and assigned.candidate_id == cid),
                    "status": c.status.value})
    out.sort(key=lambda r: (not r["assigned"], r["role"] != "BEST", -(r["score"] or 0.0)))
    return out


class ReplaceMediaDialog(QDialog):
    """Two lists - candidates of the clip's scene and the whole library. ``asset_id`` is the chosen replacement after the dialog was accepted."""

    def __init__(self, ctx: UiContext, clip_id: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx, self.clip_id = ctx, clip_id
        self.asset_id: str | None = None
        self.setWindowTitle("Replace media")
        project = ctx.ws.project
        clip = project.timeline.get_clip(clip_id) if project else None
        current = project.assets.get(clip.asset_id) if project and clip else None
        lay = QVBoxLayout(self)
        self.heading = QLabel(f"Replace “{current.name}”" if current else "Replace media")
        self.heading.setWordWrap(True)
        lay.addWidget(self.heading)
        self.note = QLabel("The clip keeps its place and timing. If the new media is shorter, the engine tells you.")
        self.note.setWordWrap(True)
        lay.addWidget(self.note)

        self.cand_label = QLabel()
        self.cand_list = QListWidget()
        self.cand_list.setObjectName("replace_candidates")
        self.lib_list = QListWidget()
        self.lib_list.setObjectName("replace_library")
        lay.addWidget(self.cand_label)
        lay.addWidget(self.cand_list, 1)
        lay.addWidget(QLabel("Project library"))
        lay.addWidget(self.lib_list, 2)

        shown: set[str] = set()
        cands = scene_candidates(ctx, clip.scene_id) if clip else []
        usable = 0
        for r in cands:
            label = ("Assigned · " if r["assigned"] else "Best · " if r["role"] == "BEST" else "") + f"{r['title']}" + (f" · {r['score']:.0f}%" if r["score"] is not None else "")
            item = QListWidgetItem(label)
            if r["asset_id"] and r["asset_id"] != (clip.asset_id if clip else ""):
                item.setData(Qt.ItemDataRole.UserRole, r["asset_id"])
                item.setToolTip("A candidate for this scene that is already in the project.")
                usable += 1
                shown.add(r["asset_id"])
            else:
                item.setFlags(Qt.ItemFlag.NoItemFlags)
                item.setToolTip("This is the media the clip already uses." if r["asset_id"] else
                                "Not in the project yet: acquire it on the Research page (Search alternatives), then it can be used here.")
                item.setText(label + (" (in use)" if r["asset_id"] else " (not downloaded)"))
            self.cand_list.addItem(item)
        self.cand_label.setText(f"Candidates for this scene ({usable} usable of {len(cands)})" if cands else "No research candidates exist for this clip.")
        self.cand_list.setVisible(bool(cands))

        assets = compatible_assets(ctx, clip_id)
        use_icons = len(assets) <= MAX_ICONS
        for a in assets:
            dur = f" · {a.duration:.1f} s" if a.duration else ""
            item = QListWidgetItem(f"{a.name}  ({a.type.value}{dur})")
            item.setData(Qt.ItemDataRole.UserRole, a.id)
            if use_icons:
                path = ctx.ws.media.thumbnail_file(a)
                if path is not None and path.is_file():
                    item.setIcon(QIcon(QPixmap(str(path))))
            self.lib_list.addItem(item)
        if not assets:
            self.lib_list.addItem(QListWidgetItem("There is no other compatible media in the project library."))
            self.lib_list.item(0).setFlags(Qt.ItemFlag.NoItemFlags)

        row = QHBoxLayout()
        self.ok = QPushButton("Replace")
        self.ok.setEnabled(False)
        self.ok.setToolTip("Choose the media to use first.")
        self.ok.clicked.connect(self._accept)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        row.addStretch(1)
        row.addWidget(self.ok)
        row.addWidget(cancel)
        lay.addLayout(row)
        for lst, other in ((self.cand_list, self.lib_list), (self.lib_list, self.cand_list)):
            lst.itemSelectionChanged.connect(lambda lst=lst, other=other: self._changed(lst, other))
            lst.itemDoubleClicked.connect(lambda _i: self._accept())

    def _changed(self, lst: QListWidget, other: QListWidget) -> None:
        if lst.selectedItems():
            other.blockSignals(True)
            other.clearSelection()
            other.blockSignals(False)
        self.ok.setEnabled(self.selected_asset() is not None)

    def selected_asset(self) -> str | None:
        for lst in (self.cand_list, self.lib_list):
            for it in lst.selectedItems():
                aid = it.data(Qt.ItemDataRole.UserRole)
                if aid:
                    return str(aid)
        return None

    def _accept(self) -> None:
        self.asset_id = self.selected_asset()
        if self.asset_id:
            self.accept()


def choose_replacement(parent: QWidget | None, ctx: UiContext, clip_id: str) -> str | None:
    """Ask which media should replace the clip's media (``None`` = cancelled). Looked up through the module at call time, so tests can replace it."""
    dlg = ReplaceMediaDialog(ctx, clip_id, parent)
    return dlg.asset_id if dlg.exec() == QDialog.DialogCode.Accepted else None
