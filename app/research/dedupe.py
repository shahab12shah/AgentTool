"""Candidate deduplication: identical URLs/assets, the same media from several queries, and near-identical thumbnails."""

from __future__ import annotations

import subprocess
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.core.textutil import STOPWORDS, stem, tokenize
from app.media.media_probe import locate_binary, run_process
from app.research.models import Candidate

TRACKING = ("utm_", "fbclid", "gclid", "ref", "source")
NEAR_DUPLICATE_BITS = 6  # max Hamming distance of 64-bit difference hashes


def normalize_url(url: str) -> str:
    if not url:
        return ""
    parts = urlsplit(url.strip())
    q = [(k, v) for k, v in parse_qsl(parts.query) if not k.lower().startswith(TRACKING)]
    host = (parts.hostname or "").lower().removeprefix("www.")
    return urlunsplit((parts.scheme.lower(), host + (f":{parts.port}" if parts.port else ""), parts.path.rstrip("/"), urlencode(q), ""))


def identity_keys(c: Candidate) -> set[str]:
    keys = {f"{c.provider}:{c.provider_id}"} if c.provider_id else set()
    for u in (c.media_url, c.source_reference):
        if u:
            keys.add("url:" + normalize_url(u) if u.startswith("http") else "path:" + u)
    if c.local_path:
        keys.add("path:" + str(Path(c.local_path)))
    return keys


def dhash_gray(pixels: bytes) -> int | None:
    """64-bit difference hash from a 9x8 grayscale buffer (72 bytes)."""
    if len(pixels) < 72:
        return None
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (1 if pixels[row * 9 + col] > pixels[row * 9 + col + 1] else 0)
    return bits


def fingerprint_file(path: Path, ffmpeg: str = "") -> str:
    """Perceptual hash (hex) of an image file, or '' when it cannot be computed."""
    try:
        r = subprocess.run([locate_binary("ffmpeg", ffmpeg), "-v", "error", "-i", str(path), "-vf", "scale=9:8:flags=area,format=gray",
                            "-frames:v", "1", "-f", "rawvideo", "-"], capture_output=True, timeout=30)
    except (subprocess.SubprocessError, OSError, Exception):
        return ""
    h = dhash_gray(r.stdout) if r.returncode == 0 else None
    return f"{h:016x}" if h is not None else ""


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def _title_terms(c: Candidate) -> frozenset[str]:
    return frozenset(stem(t.norm) for t in tokenize(c.title) if t.norm not in STOPWORDS)


def _merge(keep: Candidate, other: Candidate, reason: str) -> None:
    for q in other.query_ids:
        if q not in keep.query_ids:
            keep.query_ids.append(q)
    if len(other.description) > len(keep.description):
        keep.description = other.description
    for t in other.tags:
        if t not in keep.tags:
            keep.tags.append(t)
    keep.metadata.setdefault("duplicates", []).append({"candidate": other.candidate_id or other.provider_id, "reason": reason,
                                                       "provider": other.provider, "reference": other.source_reference})


def deduplicate(candidates: list[Candidate]) -> tuple[list[Candidate], int]:
    """Keep the first of every duplicate group (input order = priority). Returns ``(kept, removed_count)``."""
    kept: list[Candidate] = []
    by_key: dict[str, Candidate] = {}
    removed = 0
    for c in candidates:
        dup: Candidate | None = None
        reason = ""
        for k in identity_keys(c):
            if k in by_key:
                dup, reason = by_key[k], "same URL/asset"
                break
        if dup is None and c.fingerprint:
            for k in kept:
                if k.fingerprint and k.kind == c.kind and hamming(k.fingerprint, c.fingerprint) <= NEAR_DUPLICATE_BITS:
                    dup, reason = k, "visually near-identical"
                    break
        if dup is None and c.title:  # same title and similar length from the same provider (e.g. re-listed item)
            for k in kept:
                if k.provider == c.provider and k.kind == c.kind and k.title and k.title.lower() == c.title.lower() \
                        and abs((k.duration or 0) - (c.duration or 0)) < 0.5:
                    dup, reason = k, "same title and duration"
                    break
        if dup is not None:
            _merge(dup, c, reason)
            removed += 1
            continue
        kept.append(c)
        for k in identity_keys(c):
            by_key[k] = c
    return kept, removed
