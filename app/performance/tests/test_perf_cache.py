"""MediaCacheManager: validation, invalidation, eviction, safety, recovery, concurrency and hot-path cost."""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from app.performance.cache_index import INDEX_NAME
from app.performance.cache_manager import MediaCacheManager, tmp_for
from app.performance.dependencies import asset_dep, scene_dep, stable_key
from app.performance.profiler import Profiler
from app.performance.settings import ResolvedLimits

ROOT_NAMES = ["cache", "my cache ü 日本 dir"]


def limits(**cats: int):
    total = cats.pop("total", 10**12)
    base = {c: 10**12 for c in ("thumbnails", "preview_frames", "proxies", "waveforms", "analysis", "render_previews", "temporary")}
    base.update(cats)
    return lambda: ResolvedLimits("balanced", 1, 1, 1, 1, 4, 64 * 1024 * 1024, total, base)


@pytest.fixture(params=ROOT_NAMES)
def root(request, tmp_path) -> Path:
    return tmp_path / request.param


def new(root: Path, **kw) -> MediaCacheManager:
    kw.setdefault("profiler", Profiler())
    return MediaCacheManager(root, **kw)


def put_file(m: MediaCacheManager, cat: str, key: str, size: int = 100, deps=None, suffix=".bin"):
    p = m.path_for(cat, key, suffix)
    assert p is not None
    p.write_bytes(b"x" * size)
    e = m.put(key, category=cat, path=p, deps=deps)
    assert e is not None
    return p, e


def test_inline_roundtrip_and_miss_hit_counters(root):
    prof = Profiler()
    m = new(root, profiler=prof)
    assert m.get("k1", category="analysis") is None
    assert m.put("k1", {"a": [1, 2, 3], "u": "ü"}, category="analysis", data_type="scene", metadata={"w": 3}) is not None
    e = m.get("k1")
    assert e is not None and e.inline == {"a": [1, 2, 3], "u": "ü"} and e.category == "analysis" and e.metadata == {"w": 3}
    assert prof.metrics.counter("cache.analysis.hit") == 1 and prof.metrics.counter("cache.analysis.miss") == 1
    assert m.get_cache_stats()["categories"]["analysis"]["hits"] == 1


def test_file_roundtrip_size_measured_and_persisted(root):
    m = new(root)
    p, e = put_file(m, "waveforms", "w1", 321)
    assert e.size == 321 and not e.external and p.is_relative_to(root)
    m.close()
    m2 = new(root)  # new process: the entry comes from SQLite
    e2 = m2.get("w1")
    assert e2 is not None and e2.path == p and e2.size == 321


def test_missing_or_empty_file_is_a_miss_and_dropped(root):
    m = new(root)
    p, _ = put_file(m, "thumbnails", "t1")
    p.unlink()
    assert m.get("t1") is None and m.peek("t1") is None
    assert m.get_cache_stats()["categories"]["thumbnails"]["entries"] == 0
    z = m.path_for("thumbnails", "t2", ".jpg")
    z.write_bytes(b"")
    assert m.put("t2", category="thumbnails", path=z) is None  # an empty file is never a cache hit
    assert m.put("t3", category="thumbnails", path=root / "nope.jpg") is None


def test_version_mismatch_is_a_miss(root):
    m = new(root)
    m.put("k", 1, category="analysis", version=2)
    assert m.get("k", version=1) is None
    assert m.get("k", version=2) is not None


def test_stale_by_dependency_fingerprint(root):
    m = new(root)
    p, _ = put_file(m, "proxies", "p1", deps={asset_dep("a1"): "fp1", "settings:q": "hi"})
    assert m.get("p1", deps={asset_dep("a1"): "fp1", "settings:q": "hi"}) is not None
    assert m.get("p1") is not None  # no deps given: no dependency check
    assert m.get("p1", deps={asset_dep("a1"): "fp2"}) is None  # changed fingerprint -> stale, removed
    assert not p.exists() and m.peek("p1") is None
    p, _ = put_file(m, "proxies", "p2", deps={asset_dep("a1"): "fp1"})
    assert m.get("p2", deps={asset_dep("a1"): "fp1", "settings:new": "x"}) is None  # built without knowing a dependency the caller now cares about


def test_invalidate_by_key_dependency_prefix_and_category(root):
    m = new(root)
    for i in range(3):
        put_file(m, "analysis", f"s{i}", 10, deps={scene_dep(f"S{i}"): "v", asset_dep("A"): "v"})
    put_file(m, "waveforms", "w", 10, deps={asset_dep("B"): "v"})
    m.put("inl", [1], category="analysis", deps={scene_dep("S1"): "v"})
    assert m.invalidate("s0") and not m.invalidate("s0")
    assert m.invalidate_by_dependency(scene_dep("S1")) == 2  # file entry + inline entry
    assert m.invalidate_by_dependency_prefix("scene:") == 1
    assert m.peek("s2") is None and m.has("w")
    assert m.invalidate_by_dependency(asset_dep("A")) == 0
    assert m.invalidate_category("waveforms") == 1 and not m.has("w")
    assert m.get_cache_stats()["categories"]["analysis"]["entries"] == 0


def test_lru_eviction_by_category_limit(root):
    m = new(root, limits=limits(thumbnails=250))
    paths = {k: put_file(m, "thumbnails", k, 100)[0] for k in ("a", "b", "c")}
    assert m.get("a") is not None  # a becomes the most recently used
    rep = m.cleanup()
    assert rep.evicted == 1 and rep.freed_bytes == 100 and rep.by_category == {"thumbnails": 1}
    assert not paths["b"].exists() and paths["a"].exists() and paths["c"].exists()
    st = m.get_cache_stats()
    assert st["categories"]["thumbnails"]["bytes"] == 200 and st["categories"]["thumbnails"]["evictions"] == 1


def test_eviction_by_total_limit_crosses_categories(root):
    m = new(root, limits=limits(total=250))
    put_file(m, "proxies", "old", 100)
    put_file(m, "waveforms", "mid", 100)
    put_file(m, "thumbnails", "new", 100)
    m.cleanup()
    assert not m.has("old") and m.has("mid") and m.has("new")
    assert m.get_cache_stats()["total_bytes"] == 200


def test_protected_entries_survive_and_veto_is_honoured(root):
    keep = {"p_used"}
    m = new(root, limits=limits(proxies=50), protected=lambda e: e.key in keep, can_remove=lambda e: e.key != "p_vetoed")
    for k in ("p_used", "p_vetoed", "p_free"):
        put_file(m, "proxies", k, 100)
    rep = m.cleanup()
    assert m.has("p_used") and m.has("p_vetoed") and not m.has("p_free")
    assert rep.skipped.get("protected", 0) >= 1 and rep.skipped.get("vetoed", 0) >= 1
    clear = m.clear_rebuildable_cache()
    assert clear.evicted == 0 and m.has("p_used") and m.has("p_vetoed")
    assert {e.key for e in m.eligible_for_removal()} == set()
    keep.clear()
    assert [e.key for e in m.eligible_for_removal()] == ["p_used"]
    m.protect(m.peek("p_used").path)  # explicit protection by path
    assert m.eligible_for_removal() == []
    m.unprotect(m.peek("p_used").path)
    assert m.clear_rebuildable_cache().evicted == 1


def test_invalidating_a_protected_entry_keeps_the_file_but_never_serves_it(root):
    m = new(root, protected=lambda e: True)
    p, _ = put_file(m, "proxies", "inuse", 40, deps={"asset:x": "1"})
    assert m.invalidate_by_dependency("asset:x") == 1
    assert p.exists() and m.get("inuse") is None
    assert m.peek("inuse").status == "stale" and m.get_cache_stats()["categories"]["proxies"]["bytes"] == 40
    m.set_protected(None)
    assert m.clear_rebuildable_cache().evicted == 1 and not p.exists()  # stale rows go first once nothing protects them


def test_cleanup_removes_orphans_and_old_temp_files_only(root):
    m = new(root, temp_max_age_s=60)
    p, _ = put_file(m, "thumbnails", "gone", 10)
    q, _ = put_file(m, "thumbnails", "kept", 10)
    p.unlink()
    d = root / "proxies" / "ab"
    d.mkdir(parents=True)
    old_tmp, new_tmp, plain = d / ".x.part.mp4", d / "y.tmp", d / "unrelated.mp4"
    for f in (old_tmp, new_tmp, plain):
        f.write_bytes(b"z")
    old = time.time() - 3600
    os.utime(old_tmp, (old, old))
    os.utime(plain, (old, old))
    rep = m.cleanup(force=True)
    assert rep.orphans_removed == 1 and rep.temp_files_removed == 1
    assert not old_tmp.exists() and new_tmp.exists() and plain.exists() and q.exists()
    assert m.peek("gone") is None


def test_temporary_category_unregistered_old_files_are_swept(root):
    m = new(root, temp_max_age_s=60)
    m.path_for("temporary", "init")  # open the index first: files present at open time are adopted by the rescan, not swept
    f = root / "temporary" / "zz"
    f.mkdir(parents=True)
    old = f / "scratch.dat"
    old.write_bytes(b"1")
    os.utime(old, (time.time() - 999, time.time() - 999))
    registered, _ = put_file(m, "temporary", "reg", 5)
    os.utime(registered, (time.time() - 999, time.time() - 999))
    m.cleanup(force=True)
    assert not old.exists() and registered.exists()


def test_register_external_accounts_without_moving_and_clear_removes_it(tmp_path):
    proj = tmp_path / "Projekt ü 1"
    (proj / "thumbnails").mkdir(parents=True)
    (proj / "media").mkdir()
    thumb = proj / "thumbnails" / "a1.jpg"
    thumb.write_bytes(b"j" * 30)
    original = proj / "media" / "clip.mp4"
    original.write_bytes(b"orig")
    m = new(tmp_path / "cache", deny_roots=[proj / "media"])
    e = m.register_external("thumbnails", thumb, deps={asset_dep("a1"): "s"})
    assert e is not None and e.external and e.path == thumb and e.size == 30
    assert m.register_external("thumbnails", original) is None  # under a denied root: refused
    assert m.register_external("thumbnails", thumb) is not None and m.get_cache_stats()["categories"]["thumbnails"]["entries"] == 1  # idempotent
    assert m.get(e.key, deps={asset_dep("a1"): "s"}) is not None
    assert m.clear_rebuildable_cache(["proxies"]).evicted == 0 and thumb.exists()
    assert m.clear_rebuildable_cache(["thumbnails"]).evicted == 1
    assert not thumb.exists() and original.exists()


def test_non_rebuildable_and_unregistered_files_are_never_deleted(tmp_path):
    outside = tmp_path / "exports"
    outside.mkdir()
    export, bystander = outside / "final.mp4", outside / "other.mp4"
    export.write_bytes(b"e" * 20)
    bystander.write_bytes(b"b" * 20)
    m = new(tmp_path / "cache", limits=limits(total=1))
    assert m.register_external("exports", export) is not None  # accounted, but not a rebuildable category
    m.cleanup(force=True)
    m.clear_rebuildable_cache()
    m.invalidate_category("exports")
    assert export.exists() and bystander.exists()
    assert m.get_cache_stats()["unmanaged_bytes"] >= 0


def test_symlinks_are_not_followed_for_deletion(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "precious.jpg"
    target.write_bytes(b"p" * 50)
    m = new(tmp_path / "cache")
    thumbs = m.path_for("thumbnails", "t", ".jpg").parent
    link = thumbs / "link.jpg"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    assert m.put("linked", category="thumbnails", path=link) is not None
    m.clear_rebuildable_cache()
    assert target.exists() and not link.is_symlink()  # only the link was removed
    # a directory symlink inside the cache pointing outside: files reached through it are not ours to delete
    (tmp_path / "cache" / "proxies").mkdir(parents=True, exist_ok=True)
    os.symlink(outside, tmp_path / "cache" / "proxies" / "via", target_is_directory=True)
    assert m.put("through", category="proxies", path=tmp_path / "cache" / "proxies" / "via" / "precious.jpg") is not None
    rep = m.clear_rebuildable_cache()
    assert target.exists() and rep.skipped.get("unsafe", 0) == 1 and m.peek("through") is None


def test_garbage_index_is_set_aside_and_files_rediscovered(root):
    m = new(root)
    p, _ = put_file(m, "thumbnails", stable_key("thumbnails", "a"), 64)
    m.close()
    (root / INDEX_NAME).write_bytes(b"this is not a database" * 100)
    m2 = new(root)
    assert m2.get(stable_key("thumbnails", "a")) is None  # rediscovered but unverified -> miss
    assert any(root.glob(f"{INDEX_NAME}.corrupt-*"))
    st = m2.get_cache_stats()["categories"]["thumbnails"]
    assert st["entries"] == 1 and st["bytes"] == 64  # still accounted
    assert m2.put(stable_key("thumbnails", "a"), category="thumbnails", path=p) is not None
    assert m2.get(stable_key("thumbnails", "a")) is not None
    assert m2.get_cache_stats()["categories"]["thumbnails"]["entries"] == 1


def test_truncated_index_recovers_and_keeps_at_most_two_corrupt_copies(root):
    for round_ in range(4):
        m = new(root)
        for i in range(150):
            m.put(f"k{round_}-{i}", {"i": i, "pad": "x" * 200}, category="analysis")
        m.close()
        db = root / INDEX_NAME
        db.write_bytes(db.read_bytes()[: db.stat().st_size // 2])
        m = new(root)
        assert m.get("k0-0") is None
        m.put("fresh", 1, category="analysis")
        assert m.get("fresh") is not None
        m.close()
        db.write_bytes(b"\x00" * 10) if round_ % 2 else None
    assert len(list(root.glob(f"{INDEX_NAME}.corrupt-*"))) <= 2


def test_failure_at_runtime_rewrites_the_index_from_memory(root):
    m = new(root)
    put_file(m, "waveforms", "w", 10)
    m._st.db.conn.close()  # simulate the connection dying underneath us
    assert m.put("after", 1, category="analysis") is not None  # no exception into the caller
    assert m.has("w") and m.has("after")
    m.close()
    m2 = new(root)
    assert m2.has("w") and m2.has("after")


def test_unwritable_root_degrades_to_misses_without_raising(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    m = new(blocker / "cache")  # cannot be created
    assert m.put("k", 1, category="analysis") is not None or m.get("k") is None  # in-memory only, but never raises
    assert m.get("zzz") is None
    assert m.cleanup().evicted == 0
    assert isinstance(m.get_cache_stats(), dict)
    none_mgr = new(None)
    assert none_mgr.put("k", 1, category="analysis") is None and none_mgr.get("k") is None and none_mgr.path_for("a", "b") is None


def test_root_can_change_between_projects(tmp_path):
    cur = {"root": tmp_path / "p1"}
    m = new(lambda: cur["root"])
    m.put("k", 1, category="analysis")
    cur["root"] = tmp_path / "p2"
    assert m.get("k") is None
    m.put("k", 2, category="analysis")
    cur["root"] = tmp_path / "p1"
    assert m.get("k").inline == 1
    cur["root"] = None
    assert m.get("k") is None


def test_last_access_is_persisted_by_flush(root):
    m = new(root)
    m.put("a", 1, category="analysis")
    m.put("b", 2, category="analysis")
    m.get("a")
    la = m.peek("a").last_access
    m.close()
    m2 = new(root)
    assert m2.peek("a").last_access == pytest.approx(la) and m2.peek("a").last_access > m2.peek("b").last_access


def test_stats_numbers(root):
    m = new(root, limits=limits(thumbnails=1000, total=5000))
    put_file(m, "thumbnails", "a", 100)
    put_file(m, "thumbnails", "b", 200)
    put_file(m, "waveforms", "w", 50)
    m.get("a")
    m.get("nothing", category="thumbnails")
    s = m.get_cache_stats()
    t = s["categories"]["thumbnails"]
    assert (t["entries"], t["bytes"], t["limit_bytes"], t["hits"], t["misses"]) == (2, 300, 1000, 1, 1)
    assert s["total_bytes"] == 350 and s["total_limit_bytes"] == 5000 and s["eligible_bytes"] == 350
    assert s["disk_free_bytes"] and set(s["categories"]) >= {"proxies", "preview_frames", "render_previews", "temporary"}
    assert s["last_cleanup"] is None
    m.cleanup()
    assert m.get_cache_stats()["last_cleanup"] is not None


def test_concurrent_put_get_invalidate(root):
    m = new(root)
    errors: list[BaseException] = []

    def worker(n: int) -> None:
        try:
            for i in range(60):
                k = f"t{n}-{i}"
                p = m.path_for("analysis", k, ".dat")
                p.write_bytes(b"d" * 16)
                assert m.put(k, category="analysis", path=p, deps={f"scene:{i % 5}": "v"}) is not None
                m.get(k, deps={f"scene:{i % 5}": "v"})  # may legitimately miss: another thread can invalidate the dependency
                if i % 7 == 0:
                    m.invalidate_by_dependency(f"scene:{i % 5}")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    ts = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
    [t.start() for t in ts]
    [t.join(30) for t in ts]
    assert not errors
    st = m.get_cache_stats()["categories"]["analysis"]
    live = m.entries("analysis")
    assert st["entries"] == len(live) and st["bytes"] == sum(e.size for e in live)
    m.close()
    assert len(new(root).entries("analysis")) == len(live)


def test_memory_tier_is_bounded_and_releases(root):
    released = []
    lim = lambda: ResolvedLimits("balanced", 1, 1, 1, 1, 4, 1000, 10**9, {})  # noqa: E731
    m = new(root, limits=lim)
    tier = m.memory_tier("thumbnails", on_evict=lambda k, v: released.append(k), share=0.5)
    assert m.memory_tier("thumbnails") is tier and tier.max_weight == 500
    for i in range(10):
        tier.put(f"k{i}", b"x" * 100)
    assert tier.weight <= 500 and released and "k0" in released
    m.put("k9", 1, category="thumbnails")
    m.invalidate("k9")
    assert tier.peek("k9") is None
    m.trim_memory(0.5)
    assert tier.weight <= 250
    m.clear_memory()
    assert len(tier) == 0


def test_tmp_for_keeps_extension_and_is_treated_as_temp(root):
    f = Path("a/b/video.mp4")
    assert tmp_for(f).suffix == ".mp4" and tmp_for(f).name.startswith(".")


def test_hot_get_is_cheap_and_does_not_touch_sqlite(root):
    m = new(root)
    p, _ = put_file(m, "thumbnails", "hot", 10, deps={"asset:1": "a"})
    statements: list[str] = []
    m._st.db.conn.set_trace_callback(statements.append)
    t0 = time.perf_counter()
    for _ in range(5000):
        assert m.get("hot", deps={"asset:1": "a"}) is not None
    elapsed = time.perf_counter() - t0
    assert statements == []
    assert elapsed < 5.0  # generous: typically a few tens of ms
    m.get("nope", category="thumbnails")
    assert statements == []  # a plain miss does not write either
    m.flush()
    assert any("UPDATE entries SET last_access" in s for s in statements)


def test_large_index_lookups_scale(root):
    m = new(root)
    for i in range(3000):
        m.put(f"k{i}", i, category="analysis", deps={f"scene:{i % 100}": "v"})
    assert m.invalidate_by_dependency("scene:7") == 30  # via the dependency index, not a scan of every row
    assert m.get_cache_stats()["categories"]["analysis"]["entries"] == 2970


def test_sqlite_schema_is_what_the_spec_lists(root):
    m = new(root)
    m.put("k", {"a": 1}, category="analysis", deps={"d": "1"})
    m.close()
    con = sqlite3.connect(str(root / INDEX_NAME))
    cols = {r[1] for r in con.execute("PRAGMA table_info(entries)")}
    con.close()
    assert {"key", "category", "data_type", "path", "inline", "deps", "created", "last_access", "size", "version", "status"} <= cols
