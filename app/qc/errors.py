"""QC exceptions (kept apart so the fix engine and the service can both raise them without importing each other)."""

from __future__ import annotations

from app.core.exceptions import AppError


class QCError(AppError):
    """A QC action could not be carried out (the message is safe to show to the user)."""
