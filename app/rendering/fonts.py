"""FontResolver: turns a requested font family into an actual font *file* the renderer can hand to libass.

A missing font never crashes a render: the configured fallback is used and the substitution is reported (preflight warning + render log).
Files are copied into the render's own ``fonts`` folder, so FFmpeg is pointed at a relative directory with no path escaping.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

GENERIC = {"sans": "sans-serif", "sans-serif": "sans-serif", "serif": "serif", "mono": "monospace", "monospace": "monospace", "": "sans-serif"}
FALLBACK_FAMILIES = ["Inter", "DejaVu Sans", "Liberation Sans", "Arial", "Noto Sans", "Helvetica", "Segoe UI", "Roboto"]
FONT_EXT = {".ttf", ".otf", ".ttc"}


@dataclass
class FontMatch:
    requested: str
    family: str  # the name to put in the ASS style
    path: str = ""
    bold_path: str = ""
    substituted: bool = False
    reason: str = ""

    @property
    def files(self) -> list[str]:
        return [p for p in (self.path, self.bold_path) if p]


def _font_dirs() -> list[Path]:
    home = Path.home()
    if sys.platform.startswith("win"):
        win = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        return [win, Path(os.environ.get("LOCALAPPDATA", home / "AppData/Local")) / "Microsoft/Windows/Fonts"]
    if sys.platform == "darwin":
        return [Path("/System/Library/Fonts"), Path("/Library/Fonts"), home / "Library/Fonts"]
    return [Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), home / ".fonts", home / ".local/share/fonts"]


class FontResolver:
    def __init__(self, fallback_family: str = "", extra_dirs: list[Path] | None = None) -> None:
        self.fallback_family = fallback_family
        self.dirs = list(extra_dirs or []) + _font_dirs()
        self._index: dict[str, dict[str, str]] | None = None  # normalised family -> {"regular": path, "bold": path}
        self._lock = threading.Lock()
        self._fc = shutil.which("fc-match")
        self._cache: dict[tuple[str, bool], FontMatch] = {}

    # ------------------------------------------------------------------ file index (no fontconfig)
    def _build_index(self) -> dict[str, dict[str, str]]:
        idx: dict[str, dict[str, str]] = {}
        try:
            from PIL import ImageFont
        except ImportError:  # no Pillow: name guessing from file names only
            ImageFont = None  # type: ignore[assignment]
        for d in self.dirs:
            if not d.is_dir():
                continue
            for root, _dirs, files in os.walk(d):
                for f in files:
                    p = Path(root) / f
                    if p.suffix.lower() not in FONT_EXT:
                        continue
                    fam, style = p.stem, "regular"
                    if ImageFont is not None:
                        try:
                            n = ImageFont.truetype(str(p), 12).getname()
                            fam, style = n[0], n[1].lower()
                        except Exception:
                            continue
                    slot = "bold" if "bold" in style and "italic" not in style and "oblique" not in style else "regular" if style in ("regular", "book", "roman", "normal") else None
                    if slot:
                        entry = idx.setdefault(_norm(fam), {})
                        if style == "bold":  # the plain Bold face beats ExtraBold/SemiBold variants
                            entry["bold"] = str(p)
                        else:
                            entry.setdefault(slot, str(p))
        return idx

    def _lookup(self, family: str) -> dict[str, str] | None:
        with self._lock:
            if self._index is None:
                self._index = self._build_index()
        return self._index.get(_norm(family))

    # ------------------------------------------------------------------ resolution
    def resolve(self, family: str, bold: bool = False) -> FontMatch:
        key = (family.strip().lower(), bold)
        if key in self._cache:
            return self._cache[key]
        m = self._resolve(family.strip(), bold)
        self._cache[key] = m
        return m

    def _resolve(self, family: str, bold: bool) -> FontMatch:
        generic = GENERIC.get(family.lower())
        if self._fc:
            got = self._fc_match(generic or family, bold)
            if got:
                path, fam = got
                bpath = self._fc_match(fam, True)
                subst = False if generic else _norm(fam) != _norm(family)
                return FontMatch(family, fam, path, bpath[0] if bpath and bpath[0] != path else "", subst,
                                 f"“{family}” is not installed; using “{fam}”." if subst else "")
        if not generic:
            hit = self._lookup(family)
            if hit:
                return FontMatch(family, family, hit.get("regular") or hit.get("bold", ""), hit.get("bold", "") if hit.get("regular") else "")
        for cand in ([self.fallback_family] if self.fallback_family else []) + FALLBACK_FAMILIES:
            hit = self._lookup(cand)
            if hit:
                return FontMatch(family, cand, hit.get("regular") or hit.get("bold", ""), hit.get("bold", "") if hit.get("regular") else "", not generic,
                                 "" if generic else f"“{family}” is not installed; using “{cand}”.")
        return FontMatch(family, "sans-serif", "", "", True, f"No font file was found for “{family}”; the system default is used.")

    def _fc_match(self, name: str, bold: bool) -> tuple[str, str] | None:
        pattern = f"{name}:weight=bold" if bold else name
        try:
            r = subprocess.run([self._fc, "-f", "%{file}|%{family}", pattern], capture_output=True, text=True, timeout=5)  # type: ignore[list-item]
        except (OSError, subprocess.SubprocessError):
            return None
        if r.returncode != 0 or "|" not in r.stdout:
            return None
        path, fam = r.stdout.split("|", 1)
        if not path or not Path(path).is_file():
            return None
        return path, fam.split(",")[0].strip()

    # ------------------------------------------------------------------ staging
    @staticmethod
    def stage(matches: list[FontMatch], fonts_dir: Path) -> list[str]:
        """Copy the font files into ``fonts_dir`` (a render's own folder); returns the file names copied."""
        fonts_dir.mkdir(parents=True, exist_ok=True)
        done: list[str] = []
        for m in matches:
            for f in m.files:
                src = Path(f)
                dest = fonts_dir / src.name
                if not dest.exists():
                    try:
                        shutil.copy2(src, dest)
                    except OSError:
                        continue
                done.append(src.name)
        return done


def _norm(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())
