"""dependencies.py: stable keys, file fingerprints, dependency graph."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path, PureWindowsPath

import pytest

from app.performance.dependencies import (TRANSCRIPT_DEP, DependencyGraph, asset_dep, file_fingerprint, normalize_path, scene_dep, settings_dep, stable_key,
                                          timeline_dep)


class Mode(Enum):
    A = "a"


def test_stable_key_ignores_dict_order_float_noise_and_path_separators():
    assert stable_key("t", {"a": 1, "b": [1, 2]}) == stable_key("t", {"b": [1, 2], "a": 1})
    assert stable_key("t", 0.1 + 0.2) == stable_key("t", 0.3)
    assert stable_key("t", 2.0) == stable_key("t", 2)
    assert stable_key("t", -0.0) == stable_key("t", 0)
    assert stable_key("t", PureWindowsPath("a\\b")) == stable_key("t", Path("a/b"))
    assert stable_key("t", {1, 3, 2}) == stable_key("t", {3, 2, 1})
    assert stable_key("t", Mode.A) == stable_key("t", "a")


def test_stable_key_is_sensitive_to_what_matters():
    base = stable_key("thumbnails", "asset1", 320)
    assert base != stable_key("proxies", "asset1", 320)
    assert base != stable_key("thumbnails", "asset1", 321)
    assert base != stable_key("thumbnails", 320, "asset1")  # positional
    assert base != stable_key("thumbnails", "asset1", 320.5)
    assert len(base) == 40 and base == stable_key("thumbnails", "asset1", 320)


def test_normalize_path_casefold_is_explicit():
    assert normalize_path(PureWindowsPath("C:\\Users\\Ü\\a.mp4"), casefold=True) == "c:/users/ü/a.mp4"
    assert normalize_path("a\\b\\", casefold=False) == "a/b"


def test_dep_ids():
    assert asset_dep("a") == "asset:a" and scene_dep("s") == "scene:s" and settings_dep("x") == "settings:x" and timeline_dep("t") == "timeline:t"
    assert TRANSCRIPT_DEP == "transcript"


def test_fingerprint_modes(tmp_path):
    f = tmp_path / "dir with space ü" / "m.bin"
    f.parent.mkdir()
    data = bytearray(os.urandom(300_000))
    f.write_bytes(bytes(data))
    stat0, quick0, full0 = file_fingerprint(f), file_fingerprint(f, "quick"), file_fingerprint(f, "full")
    st = f.stat()
    # a change in the tail, same size and same mtime: stat cannot see it, quick and full can
    data[-10] ^= 0xFF
    f.write_bytes(bytes(data))
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert file_fingerprint(f) == stat0
    assert file_fingerprint(f, "quick") != quick0 and file_fingerprint(f, "full") != full0
    # a change in the middle is below quick's resolution (documented trade-off) but full sees it
    q1, f1 = file_fingerprint(f, "quick"), file_fingerprint(f, "full")
    data[150_000] ^= 0xFF
    f.write_bytes(bytes(data))
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert file_fingerprint(f, "quick") == q1 and file_fingerprint(f, "full") != f1
    # an mtime change alone is seen by stat
    s1 = file_fingerprint(f)
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    assert file_fingerprint(f) != s1


def test_fingerprint_missing_unreadable_and_small_files(tmp_path):
    assert file_fingerprint(tmp_path / "nope") == "missing"
    small = tmp_path / "s.bin"
    small.write_bytes(b"abc")
    assert file_fingerprint(small, "quick") != file_fingerprint(small, "stat")
    with pytest.raises(ValueError):
        file_fingerprint(small, "huge")


def test_graph_transitive_affected_and_cycles():
    g = DependencyGraph()
    g.add_edge("asset:a", "scene:1")
    g.add_edge("scene:1", "qc:1")
    g.add_edge("qc:1", "report")
    g.add_dependency("preview:1", "scene:1", "settings:q")
    g.add_edge("report", "scene:1")  # cycle must terminate
    assert g.affected("asset:a") == {"scene:1", "qc:1", "report", "preview:1"}
    assert g.affected("settings:q") == {"preview:1"}
    assert g.affected("asset:a", include_self=True) >= {"asset:a"}
    assert g.affected("unknown") == set()
    assert g.dependants("scene:1") == {"qc:1", "preview:1"} and g.sources("preview:1") == {"scene:1", "settings:q"}


def test_graph_neighbours_via_callback():
    order = ["s0", "s1", "s2", "s3", "s4"]

    def nb(s: str):
        i = order.index(s)
        return [order[j] for j in (i - 1, i + 1) if 0 <= j < len(order)]

    g = DependencyGraph(nb)
    for s in order:
        g.add_edge(s, f"qc:{s}")
    assert g.affected("s2") == {"qc:s2"}
    assert g.affected("s2", include_neighbours=1) == {"qc:s2", "s1", "s3", "qc:s1", "qc:s3"}
    assert g.affected("s2", include_neighbours=2) >= {"s0", "s4", "qc:s0", "qc:s4"}
    assert g.affected("s0", include_neighbours=1) == {"s1", "qc:s0", "qc:s1"}
    g.set_neighbours(lambda _s: 1 / 0)  # a broken callback degrades to "no neighbours"
    assert g.affected("s2", include_neighbours=1) == {"qc:s2"}


def test_graph_removal_and_clear():
    g = DependencyGraph()
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.remove_node("b")
    assert g.affected("a") == set() and "b" not in g.nodes()
    g.add_edge("a", "b")
    g.remove_edge("a", "b")
    assert g.affected("a") == set()
    g.add_edge("a", "z")
    g.clear()
    assert g.nodes() == set() and g.affected_many(["a"]) == {"a"}
