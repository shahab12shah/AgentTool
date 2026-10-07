"""Structured (JSON-lines) application logging with secret redaction.

Note: this package is named ``app.logging``. Always import it as ``app.logging.logger``;
never put the ``app/`` directory itself on ``sys.path``.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_LOGGER = "agenttool"
_SECRET_RE = re.compile(r"(?i)\b(api[_-]?key|token|secret|password|authorization)\b(\s*[=:]\s*)([^\s,;\"']+)")
_STD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


def redact(text: str) -> str:
    """Mask values that look like credentials."""
    return _SECRET_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}***", text)


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = redact(record.getMessage())
            record.args = ()
        except Exception:
            pass
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STD_ATTRS and not key.startswith("_"):
                payload[key] = _jsonable(value)
        if record.exc_info:
            payload["exception"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=False)


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return redact(str(value))


def get_logger(name: str) -> logging.Logger:
    """Return a logger below the application root (``app.x.y`` -> ``agenttool.x.y``)."""
    short = name[4:] if name.startswith("app.") else name
    return logging.getLogger(f"{ROOT_LOGGER}.{short}" if short else ROOT_LOGGER)


def setup_logging(log_dir: Path | None, level: int = logging.INFO, console: bool = True) -> None:
    """Configure file + console handlers. Safe to call more than once."""
    root = logging.getLogger(ROOT_LOGGER)
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.propagate = False
    flt = RedactingFilter()
    if log_dir is not None:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(
                log_dir / "agenttool.log", maxBytes=2_000_000, backupCount=5, encoding="utf-8"
            )
            fh.setFormatter(JsonFormatter())
            fh.addFilter(flt)
            root.addHandler(fh)
        except OSError:
            pass  # logging problems must never prevent startup
    if console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
        ch.addFilter(flt)
        root.addHandler(ch)
    if not root.handlers:
        root.addHandler(logging.NullHandler())


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Log a structured event, e.g. ``log_event(log, "project.saved", path=...)``."""
    safe = {(f"{k}_" if k in _STD_ATTRS else k): _jsonable(v) for k, v in fields.items()}
    logger.info(event, extra={"event": event, **safe})
