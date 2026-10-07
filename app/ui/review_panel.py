"""Visual Research / Review: research a scene, compare the best visual with its alternatives, and decide."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QBrush, QColor, QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTableWidgetSelectionRange,
    QVBoxLayout,
    QWidget,
)

from app.analysis.models import Scene
from app.core.exceptions import AppError
from app.research.models import Acquisition, Candidate, CandidateScore, CandidateStatus, ResearchStatus
from app.ui.context import UiContext
from app.ui.theme import palette

COLUMNS = ("Scene", "Topic", "Research", "Best", "Visual")
STATUS_TEXT = {
    ResearchStatus.NOT_STARTED: "Not started", ResearchStatus.RESEARCHING: "Researching…", ResearchStatus.CANDIDATES_READY: "Candidates ready",
    ResearchStatus.NEEDS_REVIEW: "Needs review", ResearchStatus.APPROVED: "Approved ✓", ResearchStatus.LOW_CONFIDENCE: "Low confidence",
    ResearchStatus.ERROR: "Error",
}
CATEGORY_COLOR = {"EXCELLENT": "#3fb27f", "GOOD": "#7ac35a", "REVIEW": "#e0a030", "WEAK": "#e07a30", "REJECT": "#d9534f"}
MAX_ALTERNATIVES = 5
JOB_TYPES = ("visual_research", "candidate_evaluation", "candidate_download", "screenshot", "ai_image_generation")
UNAVAILABLE = "Visual research unavailable."


def thumb_path(project_root: Path | None, c: Candidate) -> Path | None:
    for raw in (c.thumbnail_path, c.local_path if c.kind == "IMAGE" else ""):
        if not raw:
            continue
        p = Path(raw)
        if not p.is_absolute() and project_root:
            p = project_root / p
        if p.is_file():
            return p
    return None


def duration_text(c: Candidate) -> str:
    return f"{c.duration:.1f}s" if c.duration else ("image" if c.kind == "IMAGE" else "—")


class PreviewDialog(QDialog):
    """Look at one candidate before choosing it. Local videos play; everything else shows its preview and a link to the source."""

    def __init__(self, ctx: UiContext, c: Candidate, score: CandidateScore | None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(c.title or "Candidate preview")
        self.resize(720, 520)
        lay = QVBoxLayout(self)
        self.player = None
        local = Path(c.local_path) if c.local_path else None
        if c.kind == "VIDEO" and local and local.is_file():
            from PySide6.QtMultimediaWidgets import QVideoWidget

            from app.preview.player import QtPreviewPlayer

            video = QVideoWidget()
            video.setMinimumHeight(300)
            lay.addWidget(video, 1)
            self.player = QtPreviewPlayer(self)
            self.player.set_video_output(video)
            self.player.load(local)
            row = QHBoxLayout()
            play, pause = QPushButton("Play"), QPushButton("Pause")
            play.clicked.connect(self.player.play)
            pause.clicked.connect(self.player.pause)
            row.addWidget(play)
            row.addWidget(pause)
            row.addStretch(1)
            lay.addLayout(row)
        else:
            pic = QLabel()
            pic.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tp = thumb_path(ctx.ws.project.root if ctx.ws.project else None, c)
            if tp:
                pic.setPixmap(QPixmap(str(tp)).scaled(680, 340, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            else:
                pic.setText("No preview available for this candidate.")
            lay.addWidget(pic, 1)
        info = [f"{c.title}", f"Source: {c.source_type.value.replace('_', ' ').title()} · {c.provider}", f"Duration: {duration_text(c)}",
                f"Licence (as stated by the source, not verified): {c.license.name or 'unknown'}"]
        if score:
            info.append(f"Score {score.overall:.0f}/100 — {score.reason}")
        if c.acquisition is Acquisition.REFERENCE_ONLY:
            info.append("Reference only: this source cannot be downloaded into the project.")
        label = QLabel("\n".join(info))
        label.setWordWrap(True)
        lay.addWidget(label)
        row = QHBoxLayout()
        if c.source_reference.startswith(("http://", "https://")):
            open_btn = QPushButton("Open source page")
            open_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(c.source_reference)))
            row.addWidget(open_btn)
        row.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        lay.addLayout(row)

    def done(self, result: int) -> None:
        if self.player is not None:
            self.player.pause()
        super().done(result)


class CandidateCard(QFrame):
    """Preview, source, score, reason, duration and status for one candidate, plus its actions."""

    def __init__(self, panel: "ReviewPanel", c: Candidate, score: CandidateScore | None, role: str, chosen: bool) -> None:
        super().__init__()
        self.setObjectName("candidateCard")
        self.setProperty("role", role)
        root = panel.ctx.ws.project.root if panel.ctx.ws.project else None
        lay = QHBoxLayout(self)
        pic = QLabel()
        pic.setFixedSize(200 if role == "BEST" else 140, 112 if role == "BEST" else 80)
        pic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pic.setObjectName("candidateThumb")
        tp = thumb_path(root, c)
        if tp:
            pic.setPixmap(QPixmap(str(tp)).scaled(pic.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            pic.setText("no preview")
        lay.addWidget(pic)
        col = QVBoxLayout()
        head = QLabel(f"<b>{'Best visual' if role == 'BEST' else 'Alternative'}</b> · {self._esc(c.title or c.provider_id or 'Untitled')}")
        head.setWordWrap(True)
        head.setTextFormat(Qt.TextFormat.RichText)
        col.addWidget(head)
        if score:
            color = CATEGORY_COLOR.get(score.category.name, "#888")
            badge = QLabel(f"<span style='color:{color}; font-weight:bold'>{score.overall:.0f}/100 {score.category.name.title()}</span> · "
                           f"confidence {score.confidence.value.title()}")
            badge.setTextFormat(Qt.TextFormat.RichText)
            col.addWidget(badge)
        meta = f"{c.source_type.value.replace('_', ' ').title()} via {c.provider} · {duration_text(c)} · {c.evidence_kind.value.title()}"
        status = c.status.value.replace("_", " ").title()
        if chosen:
            status = "Chosen · " + status
        if c.acquisition is Acquisition.REFERENCE_ONLY:
            status += " · reference only"
        m = QLabel(f"{meta} · {status}")
        m.setObjectName("muted")
        m.setWordWrap(True)
        col.addWidget(m)
        if score and score.reason:
            r = QLabel(score.reason)
            r.setWordWrap(True)
            col.addWidget(r)
        if role == "BEST" and score and score.factors:
            why = QLabel("<b>Why this visual?</b><br>" + "<br>".join(self._esc(f) for f in score.factors[:8]))
            why.setTextFormat(Qt.TextFormat.RichText)
            why.setWordWrap(True)
            why.setObjectName("whyBox")
            col.addWidget(why)
        if score and (score.basis == "METADATA"):
            note = QLabel("Scored from titles, descriptions and tags — the pixels were not inspected.")
            note.setObjectName("muted")
            note.setWordWrap(True)
            col.addWidget(note)
        lay.addLayout(col, 1)
        btns = QVBoxLayout()
        use = QPushButton("Chosen ✓" if chosen else ("Use" if role != "BEST" else "Use this"))
        use.setEnabled(not chosen)
        use.setObjectName("primary")
        use.clicked.connect(lambda: panel.act(lambda: panel.ctx.ws.research.choose_candidate(panel.scene_id, c.candidate_id), "Use visual"))
        prev = QPushButton("Preview")
        prev.clicked.connect(lambda: PreviewDialog(panel.ctx, c, score, panel).exec())
        rej = QPushButton("Reject")
        rej.clicked.connect(lambda: panel.act(lambda: panel.ctx.ws.research.reject_candidate(panel.scene_id, c.candidate_id), "Reject"))
        more = QPushButton("More like this")
        more.setToolTip("Search again with a more specific wording around this kind of visual")
        more.clicked.connect(lambda: panel.act(lambda: panel.ctx.ws.research.search_again(panel.scene_id, "specific"), "Search"))
        for b in (use, prev, rej, more):
            btns.addWidget(b)
        if role != "BEST":
            rep = QPushButton("Replace best")
            rep.setToolTip("Make this the selected visual instead of the AI's best")
            rep.setEnabled(not chosen)
            rep.clicked.connect(lambda: panel.act(lambda: panel.ctx.ws.research.choose_candidate(panel.scene_id, c.candidate_id), "Replace"))
            btns.addWidget(rep)
        btns.addStretch(1)
        lay.addLayout(btns)

    @staticmethod
    def _esc(t: str) -> str:
        return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class ReviewPanel(QWidget):
    def __init__(self, ctx: UiContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.scene_id: str | None = None
        self._loading = False

        title = QLabel("Visual Research")
        title.setObjectName("title")
        self.summary = QLabel()
        self.summary.setObjectName("muted")
        self.research_btn = QPushButton("Research Selected")
        self.research_btn.setObjectName("primary")
        self.research_all_btn = QPushButton("Research All Scenes")
        self.approve_sel_btn = QPushButton("Approve Selected…")
        self.reject_sel_btn = QPushButton("Reject Best For Selected…")
        top = QHBoxLayout()
        top.addWidget(title)
        top.addWidget(self.summary, 1)
        for b in (self.research_btn, self.research_all_btn, self.approve_sel_btn, self.reject_sel_btn):
            top.addWidget(b)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setObjectName("researchTable")
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for col, w in ((0, 50), (2, 130), (3, 60), (4, 110)):
            self.table.setColumnWidth(col, w)
        self.table.setMinimumWidth(460)

        self.empty = QLabel()
        self.empty.setWordWrap(True)
        self.empty.setObjectName("muted")
        self.detail = QWidget()
        self.dl = QVBoxLayout(self.detail)
        self.dl.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(self.detail)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.table)
        split.addWidget(scroll)
        split.setStretchFactor(1, 1)
        split.setSizes([520, 760])
        split.setChildrenCollapsible(False)
        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.empty)
        lay.addWidget(split, 1)

        self.table.itemSelectionChanged.connect(self._on_select)
        self.research_btn.clicked.connect(lambda: self.act(lambda: self.ctx.ws.research.research_many(self.selected_ids()), "Research"))
        self.research_all_btn.clicked.connect(self._research_all)
        self.approve_sel_btn.clicked.connect(self._approve_selected)
        self.reject_sel_btn.clicked.connect(self._reject_selected)
        b = ctx.bridge
        for topic in ("project.opened", "project.closed"):
            b.on(topic, lambda p: self._reset())
        b.on("project.changed", lambda p: self.refresh() if p.get("scope") in ("research", "scenes", "assets", "script", "voice_over") else None)
        b.on("job.updated", lambda p: self.refresh() if p["job"].type in JOB_TYPES else None)
        self._reset()

    # ------------------------------------------------------------ helpers
    def selected_ids(self) -> list[str]:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        return [self.table.item(r, 0).data(Qt.ItemDataRole.UserRole) for r in rows if self.table.item(r, 0)]

    def act(self, action, title: str = "Visual research") -> bool:
        ok = self.ctx.guard(self, action, modal=True, title=title)
        self.refresh()
        return ok

    def _reset(self) -> None:
        self.scene_id = None
        self.refresh()

    def _confirm(self, title: str, text: str) -> bool:
        return QMessageBox.question(self, title, text, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                                    QMessageBox.StandardButton.Cancel) == QMessageBox.StandardButton.Yes

    # ------------------------------------------------------------ bulk actions
    def _research_all(self) -> None:
        ids = self.ctx.ws.research.scenes_ready_for_research()
        if len(ids) > 1 and not self._confirm("Research all scenes", f"Run visual research for {len(ids)} scenes? Each scene runs as its own job."):
            return
        self.act(lambda: self.ctx.ws.research.research_many(ids), "Research")

    def _approve_selected(self) -> None:
        ids = self.selected_ids()
        if not ids:
            return
        if not self._confirm("Approve selected scenes",
                             f"Approve the best visual for {len(ids)} scene(s)?\nScenes below your minimum score are skipped — nothing weak is approved for you."):
            return
        result: list = []
        self.act(lambda: result.extend(self.ctx.ws.research.approve_scenes(ids)), "Approve")
        if result:
            approved, skipped = result
            self.ctx.status(f"Approved {len(approved)} scene(s); skipped {len(skipped)} that need your decision.")

    def _reject_selected(self) -> None:
        ids = self.selected_ids()
        if ids and self._confirm("Reject best visual", f"Reject the current best candidate for {len(ids)} scene(s) and promote the next one?"):
            self.act(lambda: self.ctx.ws.research.reject_best(ids), "Reject")

    # ------------------------------------------------------------ table
    def refresh(self) -> None:
        project = self.ctx.ws.project
        research = self.ctx.ws.research
        scenes = [s for s in project.scenes] if project else []
        has = bool(scenes)
        for b in (self.research_btn, self.research_all_btn, self.approve_sel_btn, self.reject_sel_btn):
            b.setEnabled(has)
        if project is None:
            self.empty.setText("")
        elif not has:
            self.empty.setText("There are no scenes yet. Analyse the voice-over into scenes first (Scenes page), then research visuals here.")
        else:
            self.empty.setText("")
        self.empty.setVisible(bool(self.empty.text()))
        c = palette(self.ctx.ws.settings.theme)
        self._loading = True
        keep = set(self.selected_ids())
        try:
            self.table.setRowCount(len(scenes))
            for r, s in enumerate(scenes):
                st = project.research_status.get(s.id)
                status = st.status if st else ResearchStatus.NOT_STARTED
                stale = bool(st and research.is_stale(s.id))
                best = project.candidate_scores.get(st.best_id) if st and st.best_id else None
                a = project.visual_assignments.get(s.id)
                chosen = "Skipped" if a and a.skipped else (project.visual_candidates[a.candidate_id].title[:22] if a and a.candidate_id in project.visual_candidates else "—")
                text = STATUS_TEXT[status] + (" (scene changed)" if stale else "")
                color = {ResearchStatus.APPROVED: c["ok"], ResearchStatus.LOW_CONFIDENCE: "#e0a030", ResearchStatus.ERROR: c["danger"],
                         ResearchStatus.NEEDS_REVIEW: "#e0a030"}.get(status)
                for col, value in enumerate((s.label, s.topic or "—", text, f"{best.overall:.0f}" if best else "—", chosen)):
                    item = QTableWidgetItem(value)
                    item.setData(Qt.ItemDataRole.UserRole, s.id)
                    if color and col == 2:
                        item.setForeground(QBrush(QColor(color)))
                    self.table.setItem(r, col, item)
                if s.id in keep or (not keep and s.id == self.scene_id):
                    self.table.setRangeSelected(QTableWidgetSelectionRange(r, 0, r, len(COLUMNS) - 1), True)
        finally:
            self._loading = False
        done = sum(1 for s in scenes if (project.research_status.get(s.id) and project.research_status[s.id].status is ResearchStatus.APPROVED))
        self.summary.setText(f"{done}/{len(scenes)} scenes approved" if has else "")
        self._on_select()

    def _on_select(self) -> None:
        if self._loading:
            return
        ids = self.selected_ids()
        self.scene_id = ids[0] if len(ids) == 1 else (self.scene_id if ids else None)
        self._show_detail(multi=len(ids) > 1)

    # ------------------------------------------------------------ detail
    def _clear(self) -> None:
        while self.dl.count():
            item = self.dl.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    def _label(self, text: str, muted: bool = False, rich: bool = False) -> QLabel:
        lab = QLabel(text)
        lab.setWordWrap(True)
        if muted:
            lab.setObjectName("muted")
        if rich:
            lab.setTextFormat(Qt.TextFormat.RichText)
        return lab

    def _button(self, text: str, slot, primary: bool = False, tip: str = "") -> QPushButton:
        b = QPushButton(text)
        if primary:
            b.setObjectName("primary")
        if tip:
            b.setToolTip(tip)
        b.clicked.connect(slot)
        return b

    def _show_detail(self, multi: bool = False) -> None:
        self._clear()
        project = self.ctx.ws.project
        if project is None:
            return
        if multi:
            self.dl.addWidget(self._label(f"{len(self.selected_ids())} scenes selected. Use the buttons above for bulk actions, or select one scene to review it."))
            return
        scene = next((s for s in project.scenes if s.id == self.scene_id), None)
        if scene is None:
            self.dl.addWidget(self._label("Select a scene to see its research brief, queries and candidates.", muted=True))
            return
        sid = scene.id
        research = self.ctx.ws.research
        st = project.research_status.get(sid)
        status = st.status if st else ResearchStatus.NOT_STARTED
        self.dl.addWidget(self._label(f"<h3>Scene {scene.label} — {scene.topic or 'untitled'}</h3>", rich=True))
        self.dl.addWidget(self._label(scene.narration, muted=True))
        if st and research.is_stale(sid):
            self.dl.addWidget(self._label("⚠ This scene changed after it was researched. Search again to refresh the candidates."))
        self.dl.addWidget(self._banner(scene, status, st))
        self._add_brief(scene)
        if st and st.best_id and st.best_id in project.visual_candidates:
            a = project.visual_assignments.get(sid)
            best = project.visual_candidates[st.best_id]
            self.dl.addWidget(CandidateCard(self, best, project.candidate_scores.get(best.candidate_id), "BEST", bool(a and a.candidate_id == best.candidate_id)))
            for cid in st.alternatives[:MAX_ALTERNATIVES]:
                alt = project.visual_candidates.get(cid)
                if alt is not None and alt.status is not CandidateStatus.REJECTED:
                    self.dl.addWidget(CandidateCard(self, alt, project.candidate_scores.get(cid), "ALTERNATIVE", bool(a and a.candidate_id == cid)))
            chosen = a.candidate_id if a else None
            if chosen and chosen not in [st.best_id, *st.alternatives] and chosen in project.visual_candidates:  # a manual/AI pick outside the ranking
                c = project.visual_candidates[chosen]
                self.dl.addWidget(CandidateCard(self, c, project.candidate_scores.get(chosen), "ALTERNATIVE", True))
        elif st and status in (ResearchStatus.LOW_CONFIDENCE, ResearchStatus.ERROR):
            pass
        self.dl.addWidget(self._actions(scene, status))
        self.dl.addStretch(1)

    def _banner(self, scene: Scene, status: ResearchStatus, st) -> QWidget:
        box = QFrame()
        box.setObjectName("failureBox" if status in (ResearchStatus.ERROR, ResearchStatus.LOW_CONFIDENCE) else "infoBox")
        lay = QVBoxLayout(box)
        text = STATUS_TEXT[status]
        if st and st.message:
            text += " — " + st.message
        elif status is ResearchStatus.NOT_STARTED:
            text += " — press “Research Selected” to find visuals for this scene."
        if status is ResearchStatus.ERROR and not text.split(" — ", 1)[-1].startswith(UNAVAILABLE):
            text = UNAVAILABLE
        lay.addWidget(self._label(text))
        if status in (ResearchStatus.ERROR, ResearchStatus.LOW_CONFIDENCE):
            row = QHBoxLayout()
            sid = scene.id
            row.addWidget(self._button("Retry" if status is ResearchStatus.ERROR else "Search Again",
                                       lambda: self.act(lambda: self.ctx.ws.research.search_again(sid) if status is not ResearchStatus.ERROR
                                                        else self.ctx.ws.research.research_scene(sid, fresh=True), "Search"), primary=True))
            row.addWidget(self._button("Expand Sources…", self._expand))
            row.addWidget(self._button("Generate AI Visual", self._generate_ai))
            row.addWidget(self._button("Manual Select…", self._manual))
            row.addWidget(self._button("Skip", lambda: self.act(lambda: self.ctx.ws.research.skip_visual(sid), "Skip")))
            row.addStretch(1)
            lay.addLayout(row)
        return box

    def _add_brief(self, scene: Scene) -> None:
        project = self.ctx.ws.project
        group = QGroupBox("Research brief and queries")
        group.setCheckable(True)
        group.setChecked(True)
        gl = QVBoxLayout(group)
        try:
            b = self.ctx.ws.research.brief(scene.id)
        except AppError as exc:
            gl.addWidget(self._label(exc.user_message))
            self.dl.addWidget(group)
            return
        lines = [f"Subject: {b.primary_subject or '—'}" + (f" / {b.secondary_subject}" if b.secondary_subject else ""),
                 f"Visual type: {b.visual_type.replace('_', ' ').title()} · Evidence: {b.evidence_level.value.title()}",
                 "Preferred sources: " + (", ".join(s.replace('_', ' ').title() for s in b.preferred_sources) or "—")]
        if b.action:
            lines.append(f"Action: {b.action}")
        if b.context_terms:
            lines.append("Context carried over: " + ", ".join(b.context_terms[:8]))
        if b.avoid_terms:
            lines.append("Avoid: " + ", ".join(b.avoid_terms[:8]))
        gl.addWidget(self._label("\n".join(lines)))
        st = project.research_status.get(scene.id)
        session = next((s for s in reversed(project.research_sessions) if s.scene_id == scene.id), None)
        if session and session.query_ids:
            qs = [project.research_queries[q] for q in session.query_ids if q in project.research_queries]
            box = QPlainTextEdit("\n".join(f"[{q.type.value}] {q.text}  —  {q.purpose}" for q in qs))
            box.setReadOnly(True)
            box.setMaximumHeight(110)
            gl.addWidget(box)
            reports = [f"{r.provider}: {r.status.lower()} ({r.candidates} found)" + (f" — {r.error}" if r.error else "") for r in session.provider_reports
                       if not r.provider.startswith("(")]
            if reports:
                gl.addWidget(self._label("Providers: " + "; ".join(reports), muted=True))
        elif st is None:
            gl.addWidget(self._label("No queries yet.", muted=True))
        self.dl.addWidget(group)

    def _actions(self, scene: Scene, status: ResearchStatus) -> QWidget:
        sid = scene.id
        r = self.ctx.ws.research
        project = self.ctx.ws.project
        a = project.visual_assignments.get(sid)
        box = QFrame()
        lay = QHBoxLayout(box)
        lay.setContentsMargins(0, 8, 0, 0)
        lay.addWidget(self._button("Search Again", lambda: self.act(lambda: r.search_again(sid), "Search"), tip="New query variations, using the cache where possible"))
        lay.addWidget(self._button("Fresh Search", lambda: self.act(lambda: r.research_scene(sid, fresh=True), "Search"), tip="Ignore cached results"))
        lay.addWidget(self._button("Generate More", lambda: self.act(lambda: r.generate_more(sid), "Search"), tip="Look for a different interpretation of the scene"))
        lay.addWidget(self._button("Expand Sources…", self._expand))
        lay.addWidget(self._button("Generate AI Visual", self._generate_ai))
        lay.addWidget(self._button("Manual Select…", self._manual))
        lay.addWidget(self._button("Add Page Screenshot…", self._screenshot))
        lay.addWidget(self._button("Skip", lambda: self.act(lambda: r.skip_visual(sid), "Skip")))
        if a and a.approved:
            lay.addWidget(self._button("Unapprove", lambda: self.act(lambda: r.unapprove(sid), "Unapprove")))
        else:
            lay.addWidget(self._button("Approve", lambda: self.act(lambda: r.approve(sid), "Approve"), primary=True))
        return box

    # ------------------------------------------------------------ dialogs
    def _expand(self) -> None:
        sid = self.scene_id
        if sid is None:
            return
        options = self.ctx.ws.research.expandable_sources()
        if not options:
            QMessageBox.information(self, "Expand Sources", "Every source with an available provider is already enabled.")
            return
        names = "\n".join(f"• {o['kind'].replace('_', ' ').title()} ({', '.join(o['providers'])})" for o in options)
        if self._confirm("Expand sources?", f"These sources are turned off in your Visual preferences:\n\n{names}\n\n"
                                            "Search them for this scene once? Your preferences will not change."):
            self.act(lambda: self.ctx.ws.research.search_again(sid, expand=True), "Expand Sources")

    def _generate_ai(self) -> None:
        sid = self.scene_id
        if sid and self._confirm("Generate AI visual", "Generate an AI image for this scene? This calls your configured image service and may cost money.\n"
                                                      "AI visuals are never treated as evidence."):
            self.act(lambda: self.ctx.ws.research.generate_ai_visual(sid), "Generate AI Visual")

    def _manual(self) -> None:
        sid = self.scene_id
        if sid is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Choose a visual for this scene", "", "Media (*.png *.jpg *.jpeg *.webp *.mp4 *.mov *.mkv *.webm);;All files (*)")
        if path:
            self.act(lambda: self.ctx.ws.research.add_manual_visual(sid, Path(path)), "Manual Select")

    def _screenshot(self) -> None:
        sid = self.scene_id
        if sid is None:
            return
        url, ok = QInputDialog.getText(self, "Add page screenshot", "Page URL (http or https):")
        if ok and url.strip():
            self.act(lambda: self.ctx.ws.research.add_page_screenshot(sid, url.strip()), "Screenshot")
