"""Export page: validates the project and shows the render plan. Rendering itself is not built yet."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

from app.core.timecode import format_timecode
from app.ui.context import UiContext


class ExportPanel(QWidget):
    def __init__(self, ctx: UiContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        title = QLabel("Export")
        title.setObjectName("title")
        info = QLabel(
            "Rendering is not implemented yet (planned for Phase 3). "
            "You can check whether the current timeline is ready to render."
        )
        info.setObjectName("muted")
        info.setWordWrap(True)
        self.check_btn = QPushButton("Check project")
        self.render_btn = QPushButton("Render — Coming in Phase 3")
        self.render_btn.setEnabled(False)
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setObjectName("exportReport")
        layout = QVBoxLayout(self)
        for w in (title, info, self.check_btn, self.render_btn, self.output):
            layout.addWidget(w)
        layout.setStretchFactor(self.output, 1)
        self.check_btn.clicked.connect(self.check)

    def check(self) -> None:
        project = self.ctx.ws.project
        if project is None:
            self.output.setPlainText("Open a project first.")
            return
        renderer = self.ctx.ws.renderer
        issues = renderer.validate(project)
        plan = renderer.build_render_plan(project)
        lines = [
            f"Output: {plan.width}×{plan.height} @ {plan.fps} fps, length {format_timecode(plan.duration)}",
            f"Clips in plan: {len(plan.segments)}",
            "",
        ]
        lines += [f"[{i.severity.upper()}] {i.message}" for i in issues] or ["No problems found."]
        self.output.setPlainText("\n".join(lines))
