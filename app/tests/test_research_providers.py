"""Providers. LocalStock and Screenshot run for real here; Wikimedia/YouTube/Pexels/AI-image are tested ONLY against
a local mock server that imitates the documented response shapes (the real services were unreachable)."""

from __future__ import annotations

import base64
import json
import urllib.parse
from pathlib import Path

import pytest

from app.core.exceptions import AcquisitionError, ProviderError
from app.media.asset import SourceType as S
from app.research.http import HttpClient, is_public_url
from app.research.models import Acquisition, CandidateStatus, EvidenceKind, QueryType, ResearchQuery
from app.research.providers.ai_image import AIImageProvider, build_prompt
from app.research.providers.base import SearchContext
from app.research.providers.local_stock import LocalStockProvider
from app.research.providers.pexels import PexelsProvider
from app.research.providers.screenshot import ScreenshotProvider, find_chromium
from app.research.providers.wikimedia import WikimediaProvider
from app.research.providers.youtube import YouTubeProvider, parse_chapters, parse_iso_duration
from app.tests.conftest import needs_ffmpeg
from app.tests.helpers import MockWeb, make_image, make_video, solar_brief


@pytest.fixture
def web():
    w = MockWeb()
    yield w
    w.close()


@pytest.fixture
def local_http():
    return HttpClient(allow_private=True, retries=1, min_interval=0)


def q(text, qtype=QueryType.LITERAL):
    return ResearchQuery("query_00001", "scene_014", text, qtype, "test", 1, [])


CTX = SearchContext(solar_brief())


# ================================================================== http safety
def test_private_and_non_http_addresses_are_refused_by_default():
    assert not is_public_url("http://127.0.0.1:8000/x") and not is_public_url("http://localhost/x") and not is_public_url("file:///etc/passwd")
    assert not is_public_url("http://192.168.1.5/") and not is_public_url("http://169.254.169.254/latest/meta-data")
    with pytest.raises(ProviderError, match="private network"):
        HttpClient().request_json("http://127.0.0.1:9/x")
    with pytest.raises(ProviderError, match="Only http"):
        HttpClient().request_json("ftp://example.com/x")


def test_http_retries_transient_errors_then_succeeds_and_caps_downloads(web, local_http, tmp_path):
    state = {"n": 0}

    def flaky(h, b):
        state["n"] += 1
        return (503, "text/plain", b"x") if state["n"] < 2 else (200, "application/json", b'{"ok": true}')

    web.routes["/flaky"] = flaky
    assert local_http.request_json(web.base + "/flaky") == {"ok": True} and state["n"] == 2
    web.routes["/big"] = lambda h, b: (200, "application/octet-stream", b"x" * 5000)
    with pytest.raises(ProviderError, match="larger"):
        local_http.download(web.base + "/big", tmp_path / "f.bin", max_bytes=1000)
    assert not (tmp_path / "f.bin").exists() and not list(tmp_path.glob("*.part"))
    web.json("/forbidden", {}, status=403)
    with pytest.raises(ProviderError, match="403"):
        local_http.request_json(web.base + "/forbidden")


# ================================================================== local stock folder (real)
@pytest.fixture
def stock_dir(tmp_path):
    d = tmp_path / "stock"
    (d / "energy").mkdir(parents=True)
    (d / "misc").mkdir()
    if __import__("shutil").which("ffmpeg"):
        make_video(d / "energy" / "solar_factory_line.mp4", 3.0)
        make_video(d / "misc" / "clip_0042.mp4", 2.0, "testsrc2")
        make_image(d / "energy" / "rooftop.jpg", "mandel")
    (d / "catalog.json").write_text(json.dumps({
        "misc/clip_0042.mp4": {"title": "Silver paste applied to solar cells", "description": "close-up of screen printing", "tags": ["silver", "photovoltaic"],
                               "license": {"name": "Studio licence #42", "attribution": "Stock Co"}}}))
    return d


@needs_ffmpeg
def test_local_stock_search_matches_catalog_filename_and_folders(stock_dir):
    p = LocalStockProvider(lambda: str(stock_dir))
    assert p.is_available() == (True, "")
    vids = p.search(q("silver solar cells screen printing"), S.STOCK_VIDEO, 5, CTX)
    assert [c.title for c in vids][0] == "Silver paste applied to solar cells"  # catalog metadata wins over the filename
    c = vids[0]
    assert c.source_type is S.STOCK_VIDEO and c.kind == "VIDEO" and c.acquisition is Acquisition.LOCAL and Path(c.local_path).is_file()
    assert c.duration == pytest.approx(2.0, abs=0.2) and (c.width, c.height) == (640, 360)
    assert c.license.name == "Studio licence #42" and c.license.status == "PROVIDER_STATED" and c.license.verified is False
    by_folder = p.search(q("energy solar factory"), S.STOCK_VIDEO, 5, CTX)
    assert by_folder and by_folder[0].title == "solar factory line"  # folder + file name become metadata
    imgs = p.search(q("rooftop energy"), S.STOCK_IMAGE, 5, CTX)
    assert imgs and imgs[0].kind == "IMAGE" and imgs[0].duration is None and imgs[0].license.status == "UNKNOWN"
    assert p.search(q("completely unrelated quantum"), S.STOCK_VIDEO, 5, CTX) == []
    assert p.acquire(c, stock_dir, CTX) == Path(c.local_path)


@needs_ffmpeg
def test_local_stock_thumbnail_and_unavailable_states(stock_dir, tmp_path, local_http):
    p = LocalStockProvider(lambda: str(stock_dir))
    c = p.search(q("silver solar cells"), S.STOCK_VIDEO, 1, CTX)[0]
    assert p.fetch_thumbnail(c, tmp_path / "t.jpg", local_http) and (tmp_path / "t.jpg").stat().st_size > 0
    assert LocalStockProvider(lambda: "").is_available()[0] is False
    ok, why = LocalStockProvider(lambda: str(tmp_path / "nope")).is_available()
    assert not ok and "does not exist" in why
    (stock_dir / "catalog.json").write_text("{ broken")  # a corrupt catalog must not break search
    assert p.search(q("solar factory"), S.STOCK_VIDEO, 3, CTX)


# ================================================================== Wikimedia (mock server)
def commons_payload(web):
    return {"query": {"pages": {
        "11": {"pageid": 11, "title": "File:Solar cell silver contacts.jpg", "index": 1, "imageinfo": [{
            "url": web.base + "/media/solar.jpg", "descriptionurl": "https://commons.wikimedia.org/wiki/File:Solar_cell_silver_contacts.jpg",
            "thumburl": web.base + "/media/solar_thumb.jpg", "width": 3000, "height": 2000, "mime": "image/jpeg", "mediatype": "BITMAP",
            "extmetadata": {"ObjectName": {"value": "Silver contacts on a solar cell"},
                            "ImageDescription": {"value": "<p>A <b>photovoltaic</b> cell with silver busbars &amp; fingers</p>"},
                            "Artist": {"value": "<a href='x'>Jane Doe</a>"}, "LicenseShortName": {"value": "CC BY-SA 4.0"},
                            "LicenseUrl": {"value": "https://creativecommons.org/licenses/by-sa/4.0"}, "Categories": {"value": "Solar cells|Silver|Photovoltaics"}}}]},
        "12": {"pageid": 12, "title": "File:Factory tour.webm", "index": 2, "imageinfo": [{
            "url": web.base + "/media/tour.webm", "descriptionurl": "https://commons.wikimedia.org/wiki/File:Factory_tour.webm", "thumburl": "",
            "width": 1280, "height": 720, "mime": "video/webm", "mediatype": "VIDEO", "duration": 41.5,
            "extmetadata": {"LicenseShortName": {"value": "CC0"}}}]}}}}


def test_wikimedia_normalises_images_and_videos_with_stated_licences(web, local_http):
    web.json("/w/api.php", commons_payload(web))
    p = WikimediaProvider(lambda: web.base + "/w/api.php", http=local_http)
    imgs = p.search(q("solar cell silver"), S.WEB_IMAGE, 5, CTX)
    assert len(imgs) == 1  # the video entry is filtered out of image searches
    c = imgs[0]
    assert c.source_type is S.WEB_IMAGE and c.title == "Silver contacts on a solar cell" and c.description == "A photovoltaic cell with silver busbars & fingers"
    assert c.tags == ["Solar cells", "Silver", "Photovoltaics"] and (c.width, c.height) == (3000, 2000) and c.provider == "wikimedia"
    assert c.license.name == "CC BY-SA 4.0" and c.license.attribution == "Jane Doe" and c.license.status == "PROVIDER_STATED" and not c.license.verified
    assert c.acquisition is Acquisition.DOWNLOAD and c.media_url.endswith("solar.jpg") and "wikipedia" not in c.source_reference
    vids = p.search(q("factory tour"), S.WEB_VIDEO, 5, CTX)
    assert len(vids) == 1 and vids[0].kind == "VIDEO" and vids[0].duration == 41.5 and vids[0].title == "Factory tour"
    req = urllib.parse.parse_qs(urllib.parse.urlsplit(web.requests[0]["path"]).query)
    assert "solar cell silver" in req["gsrsearch"][0] and "filetype:bitmap" in req["gsrsearch"][0] and req["gsrnamespace"] == ["6"]
    assert "filetype:video" in urllib.parse.parse_qs(urllib.parse.urlsplit(web.requests[1]["path"]).query)["gsrsearch"][0]


def test_wikimedia_downloads_and_reports_failures(web, local_http, tmp_path):
    web.json("/w/api.php", commons_payload(web))
    web.routes["/media/solar.jpg"] = lambda h, b: (200, "image/jpeg", b"\xff\xd8fakejpeg")
    p = WikimediaProvider(lambda: web.base + "/w/api.php", http=local_http)
    c = p.search(q("solar"), S.WEB_IMAGE, 5, CTX)[0]
    c.candidate_id = "candidate_00007"
    f = p.acquire(c, tmp_path, CTX)
    assert f.name == "candidate_00007.jpg" and f.read_bytes().startswith(b"\xff\xd8")
    web.json("/w/api.php", {}, status=500)
    with pytest.raises(ProviderError):
        p.search(q("solar"), S.WEB_IMAGE, 5, CTX)
    assert p.search.__self__.http is local_http


# ================================================================== YouTube (mock server)
DESC = "Factory tour\n0:00 Intro\n1:10 Silver paste screen printing\n2:30 Cell testing\n"


def yt_routes(web):
    web.json("/youtube/v3/search", {"items": [{"id": {"videoId": "AAA111"}}, {"id": {"videoId": "BBB222"}}, {"id": {"kind": "youtube#channel"}}]})
    web.json("/youtube/v3/videos", {"items": [
        {"id": "AAA111", "snippet": {"title": "How solar panels are made", "description": DESC, "channelTitle": "Factory Channel",
                                     "thumbnails": {"high": {"url": web.base + "/thumb/a.jpg"}}, "tags": ["solar", "manufacturing"]},
         "contentDetails": {"duration": "PT12M45S"}, "status": {"license": "youtube"}},
        {"id": "BBB222", "snippet": {"title": "Silver price today", "description": "no chapters here", "channelTitle": "News"},
         "contentDetails": {"duration": "PT3M"}, "status": {"license": "creativeCommon"}}]})


def test_youtube_parsing_helpers():
    assert parse_iso_duration("PT12M45S") == 765 and parse_iso_duration("PT1H2M3S") == 3723 and parse_iso_duration("PT45S") == 45 and parse_iso_duration("bad") is None
    ch = parse_chapters(DESC, 900.0)
    assert ch == [(0.0, 70.0, "Intro"), (70.0, 150.0, "Silver paste screen printing"), (150.0, 900.0, "Cell testing")]
    assert parse_chapters("no timestamps", 100) == []


def test_youtube_candidates_have_references_and_chapter_based_segments(web, local_http, monkeypatch):
    yt_routes(web)
    monkeypatch.setenv("TEST_YT_KEY", "yt-secret-key")
    p = YouTubeProvider(lambda: web.base + "/youtube/v3", lambda: "TEST_YT_KEY", http=local_http)
    assert p.is_available() == (True, "")
    got = p.search(q("silver paste screen printing"), S.YOUTUBE, 5, SearchContext(solar_brief(), default_clip_seconds=5.0))
    a, b = got
    assert (a.provider_id, a.source_reference, a.duration) == ("AAA111", "https://www.youtube.com/watch?v=AAA111", 765.0)
    assert a.title == "How solar panels are made" and a.thumbnail_url.endswith("a.jpg") and a.metadata["channel"] == "Factory Channel"
    assert a.segment.basis == "CHAPTER" and a.segment.start == 70.0 and a.segment.end == 75.0  # the useful section, not the whole video
    assert b.segment.basis == "UNKNOWN" and b.segment.start is None  # no chapters: we do not pretend to know
    assert a.acquisition is Acquisition.REFERENCE_ONLY and a.license.name == "Standard YouTube License" and b.license.name.startswith("Creative Commons")
    assert all(not c.license.verified for c in got)
    assert "yt-secret-key" in web.requests[0]["path"]  # sent to the API, and only there
    with pytest.raises(AcquisitionError, match="not implemented"):
        p.acquire(a, Path("."), CTX)
    monkeypatch.delenv("TEST_YT_KEY")
    ok, why = p.is_available()
    assert not ok and "TEST_YT_KEY" in why and "yt-secret" not in why


def test_youtube_no_results_and_quota_error(web, local_http, monkeypatch):
    monkeypatch.setenv("K", "x")
    p = YouTubeProvider(lambda: web.base + "/youtube/v3", lambda: "K", http=local_http)
    web.json("/youtube/v3/search", {"items": []})
    assert p.search(q("nothing"), S.YOUTUBE, 5, CTX) == []
    web.json("/youtube/v3/search", {"error": "quota"}, status=403)
    with pytest.raises(ProviderError, match="403"):
        p.search(q("anything"), S.YOUTUBE, 5, CTX)


# ================================================================== Pexels (mock server)
def test_pexels_images_and_videos(web, local_http, monkeypatch, tmp_path):
    monkeypatch.setenv("PEXELS_TEST", "px-key")
    web.json("/v1/search", {"photos": [{"id": 1, "width": 6000, "height": 4000, "url": "https://www.pexels.com/photo/solar-panels-on-roof-1/", "photographer": "Ann",
                                        "alt": "Solar panels on a roof", "src": {"large2x": web.base + "/img/l.jpg", "medium": web.base + "/img/m.jpg"}}]})
    web.json("/videos/search", {"videos": [{"id": 2, "width": 3840, "height": 2160, "url": "https://www.pexels.com/video/solar-panel-factory-assembly-2/", "image": web.base + "/img/v.jpg",
                                           "duration": 14, "user": {"name": "Bob"},
                                           "video_files": [{"link": web.base + "/vid/uhd.mp4", "width": 3840, "height": 2160}, {"link": web.base + "/vid/hd.mp4", "width": 1920, "height": 1080}]}]})
    web.routes["/vid/hd.mp4"] = lambda h, b: (200, "video/mp4", b"mp4bytes")
    p = PexelsProvider(lambda: web.base, lambda: "PEXELS_TEST", http=local_http)
    img = p.search(q("solar roof"), S.STOCK_IMAGE, 5, CTX)[0]
    assert img.kind == "IMAGE" and img.title == "Solar panels on a roof" and img.media_url.endswith("l.jpg") and img.license.name == "Pexels License"
    assert img.license.attribution == "Ann" and img.license.url == "https://www.pexels.com/license/" and not img.license.verified
    vid = p.search(q("solar factory"), S.STOCK_VIDEO, 5, CTX)[0]
    assert vid.kind == "VIDEO" and vid.duration == 14.0 and vid.media_url.endswith("hd.mp4") and (vid.width, vid.height) == (1920, 1080)
    assert "solar panel factory assembly" in vid.description  # words recovered from the page slug when the API gives no text
    assert web.requests[0]["headers"].get("Authorization") == "px-key"
    vid.candidate_id = "candidate_00002"
    assert p.acquire(vid, tmp_path, CTX).read_bytes() == b"mp4bytes"
    monkeypatch.delenv("PEXELS_TEST")
    assert p.is_available()[0] is False


# ================================================================== AI image (mock server)
def test_ai_provider_only_proposes_during_search_and_prompt_forbids_fabrication(web, local_http, monkeypatch):
    monkeypatch.setenv("AI_TEST", "ai-key")
    p = AIImageProvider(lambda: web.base + "/v1", lambda: "gpt-image-1", lambda: "AI_TEST", http=local_http)
    b = solar_brief()
    got = p.search(q("ai concept"), S.AI_GENERATED, 1, SearchContext(b))
    assert len(web.requests) == 0  # searching costs nothing and calls nothing
    c = got[0]
    assert c.status is CandidateStatus.PROPOSED and c.acquisition is Acquisition.GENERATE and c.evidence_kind is EvidenceKind.DECORATIVE
    assert "solar installations" in c.prompt and "Do not render documents" in c.prompt and "identifiable people" in c.prompt
    assert "Generic silver coins" in c.prompt  # the brief's avoid list reaches the prompt
    assert c.license.status == "UNKNOWN" and "AI-generated" in c.license.name
    assert p.search(q("x"), S.AI_GENERATED, 1, SearchContext(None)) == []
    ev = build_prompt(solar_brief(visual_type="EVIDENCE"))
    assert "Nothing in the image may be presented as factual evidence" in ev


def test_ai_generation_saves_image_and_handles_errors(web, local_http, monkeypatch, tmp_path):
    monkeypatch.setenv("AI_TEST", "ai-key")
    png = base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode()
    web.json("/v1/images/generations", {"data": [{"b64_json": png}]})
    p = AIImageProvider(lambda: web.base + "/v1", lambda: "m", lambda: "AI_TEST", http=local_http)
    c = p.search(q("ai"), S.AI_GENERATED, 1, SearchContext(solar_brief()))[0]
    c.candidate_id = "candidate_00005"
    out = p.generate(c, tmp_path / "generated")
    assert out.name == "ai_candidate_00005.png" and out.read_bytes().startswith(b"\x89PNG")
    sent = json.loads(web.requests[0]["body"])
    assert sent["prompt"] == c.prompt and sent["model"] == "m" and web.requests[0]["headers"]["Authorization"] == "Bearer ai-key"
    web.routes["/files/x.png"] = lambda h, b: (200, "image/png", b"\x89PNGurl")
    web.json("/v1/images/generations", {"data": [{"url": web.base + "/files/x.png"}]})
    assert p.generate(c, tmp_path / "g2").read_bytes() == b"\x89PNGurl"
    web.json("/v1/images/generations", {"data": []})
    with pytest.raises(AcquisitionError, match="no image"):
        p.generate(c, tmp_path / "g3")
    web.json("/v1/images/generations", {}, status=401)
    with pytest.raises(ProviderError, match="401"):
        p.generate(c, tmp_path / "g4")
    monkeypatch.delenv("AI_TEST")
    ok, why = p.is_available()
    assert not ok and "AI_TEST" in why
    assert AIImageProvider(lambda: "", http=local_http).is_available()[0] is False


# ================================================================== screenshots (REAL Chromium, local pages)
chromium = pytest.mark.skipif(find_chromium() is None, reason="Chromium not installed")


@chromium
def test_screenshot_provider_captures_a_page_as_an_evidence_candidate(web, tmp_path):
    web.html("/notice", "<html><body style='background:#fff'><h1>IRS Notice CP2000</h1><p>Official test page</p></body></html>")
    p = ScreenshotProvider(allow_private=True, extra_sites={"irs": ("Internal Revenue Service (IRS)", web.base + "/notice")})
    assert p.is_available()[0]
    ctx = SearchContext(solar_brief(), cache_dir=tmp_path)
    got = p.search(q("IRS notice silver sellers", QueryType.DOCUMENT), S.SCREENSHOT, 3, ctx)
    assert len(got) == 1
    c = got[0]
    assert c.source_type is S.SCREENSHOT and c.evidence_kind is EvidenceKind.EVIDENCE and c.acquisition is Acquisition.CAPTURE
    assert c.title.endswith("official website (homepage)") and "Verify" in c.description  # honest: a homepage, needs a human check
    png = Path(c.local_path)
    assert png.is_file() and png.read_bytes()[:4] == b"\x89PNG" and (c.width, c.height) == (1280, 720) and c.thumbnail_path == c.local_path
    assert "not a licence" in c.license.name.lower() and c.license.verified is False
    hits = web.count("/notice")
    assert hits >= 1
    again = p.search(q("IRS notice", QueryType.DOCUMENT), S.SCREENSHOT, 3, ctx)
    assert len(again) == 1 and web.count("/notice") == hits  # the capture is cached on disk: no new requests
    assert p.search(q("solar panels"), S.SCREENSHOT, 3, ctx) == []  # no known official site for the query
    assert p.acquire(c, tmp_path, ctx) == png


@chromium
def test_screenshot_refuses_private_addresses_by_default_and_reports_unreachable_pages(web, tmp_path):
    p = ScreenshotProvider()
    ctx = SearchContext(cache_dir=tmp_path)
    with pytest.raises(ProviderError, match="private network"):
        p.capture_url(web.base + "/x", q("x"), ctx)
    with pytest.raises(ProviderError, match="http"):
        p.capture_url("file:///etc/passwd", q("x"), ctx)
    web.routes["/missing"] = lambda h, b: (404, "text/html", b"<h1>Not found</h1>")
    p2 = ScreenshotProvider(allow_private=True)
    with pytest.raises(ProviderError, match="404"):  # a 404 page is not evidence; Chromium would happily screenshot it
        p2.capture_url(web.base + "/missing", q("x"), ctx)
    assert not list((tmp_path / "screenshots").glob("*.png")) if (tmp_path / "screenshots").exists() else True
    refused = ScreenshotProvider(allow_private=True, extra_sites={"irs": ("IRS", "http://127.0.0.1:9/")})
    with pytest.raises(ProviderError, match="could not be captured"):
        refused.search(q("IRS notice"), S.SCREENSHOT, 2, SearchContext(cache_dir=tmp_path))
