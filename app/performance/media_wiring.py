"""One call that connects the media pipelines to the performance service (thumbnails, proxies, waveforms, probe results).

Every media component works without the performance service (fixed small limits, no index, manual proxies); this plugs the real limits, the cache index and the
project's policy in. Safe to call more than once. ``Workspace.__init__`` calls it right after it creates ``self.performance``.
"""

from __future__ import annotations

from typing import Any


def wire_media_performance(ws: Any) -> None:
    perf = lambda: getattr(ws, "performance", None)  # noqa: E731
    cache = lambda: getattr(perf(), "cache", None)  # noqa: E731
    ws.media.attach_performance(perf)
    ws.render.proxies.attach_performance(perf)
    ws.presentation.attach_performance(perf)
    seen: set[int] = set()
    for owner in (getattr(ws.render, "engine", None), getattr(ws, "qc", None)):
        probe = getattr(owner, "probe", None)
        if probe is not None and id(probe) not in seen and hasattr(probe, "attach_cache"):
            seen.add(id(probe))
            probe.attach_cache(cache)
    p = perf()
    if p is not None and hasattr(p, "_proxy_in_use"):
        p._proxy_in_use = ws.render.proxies.is_protected  # only proxies a render / preview is reading (or that are being made) are protected from clean-up; the rest is rebuildable
