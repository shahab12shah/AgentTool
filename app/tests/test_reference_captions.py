"""Phase 7 caption / on-screen text analysis tests (``CaptionTextAnalyzer`` + ``TextRegionDetector``): no OCR, geometry and timing only.

Ground truth comes from two kinds of synthetic media, both with *known* text (what, where, when, how it appears):
* real FFmpeg videos (``reference_helpers.overlay_video`` and drawtext chains built here) that go through the real ``FrameSampler`` -> ``analyze_video`` path,
  with one steady picture or hard-cut shots underneath (the caption must stay put while the picture changes);
* pure-numpy RGB frame lists drawn with PIL (``analyze_frames``) for what FFmpeg cannot make cheaply or precisely: busy textured pictures, text-like textures
  (bricks, blinds, barcodes, printed posters), style matrices, watermarks, flat graphics, degenerate inputs.

Nothing here reads a word: the tests assert counts, positions, sizes, timings, rates and classes against the numbers worked out from the script of each scene.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from app.core.serialization import to_plain
from app.reference.caption_analyzer import (CAPTION_STYLES, CAPTION_TRAITS, EVENT_KINDS, FIXED_VOCABULARY, CaptionTextAnalyzer, CaptionTextResult, position_of)
from app.reference.signals import AnalysisCancelled, FrameSampler, SampledFrame
from app.reference.style_model import CaptionStats, TextEvent
from app.reference.text_regions import TextRegionDetector
from app.rendering.ffmpeg_service import FFmpegService
from app.tests.conftest import needs_ffmpeg
from app.tests.reference_helpers import blob_image, camera_video, font_path, overlay_video, run_ffmpeg, shots_video

W, H, FPS = 480, 270, 4.0  # the analyzer's default sampling
VIDEO_SIZE = (320, 180)


# ---------------------------------------------------------------------------------------------- shared helpers
def _pairs(events: list[TextEvent], truth: list[tuple[float, float]], tol: float = 0.5) -> int:
    """How many ground-truth (start, end) intervals have a detected event whose start AND end are within ``tol`` seconds."""
    return sum(any(abs(e.start - a) <= tol and abs(e.end - b) <= tol for e in events) for a, b in truth)


def _kinds(res: CaptionTextResult, kind: str) -> list[TextEvent]:
    return [e for e in res.events if e.kind == kind]


@pytest.fixture(scope="module")
def pic(tmp_path_factory):
    """``pic(seed)`` -> a smooth random picture (the same generator the FFmpeg shots use) as uint8 (270, 480, 3)."""
    d = tmp_path_factory.mktemp("cap_pics")

    @lru_cache(maxsize=None)
    def make(seed: int) -> np.ndarray:
        return np.asarray(Image.open(blob_image(d / f"p{seed}.png", seed, (W, H), boxes=3)).convert("RGB"))

    return make


@lru_cache(maxsize=None)
def _font(px: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_path(True), px)


def put_text(img: np.ndarray, text: str, cx: float, cy: float, size: int, *, color=(255, 255, 255), alpha: float = 1.0, box: bool = False, stroke: int = 2,
             anchor: str = "mm") -> np.ndarray:
    """Draw ``text`` (outlined like burned-in captions) on a copy of ``img``; ``size`` is the font size in pixels at 480x270, ``alpha`` fades it in."""
    base = Image.fromarray(img).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    if box:
        bb = d.textbbox((cx, cy), text, font=_font(size), anchor=anchor)
        d.rectangle([bb[0] - 10, bb[1] - 8, bb[2] + 10, bb[3] + 8], fill=(0, 0, 0, int(255 * 0.65 * alpha)))
    d.text((cx, cy), text, font=_font(size), fill=(*color, int(255 * alpha)), stroke_width=stroke, stroke_fill=(0, 0, 0, int(255 * alpha)), anchor=anchor)
    return np.asarray(Image.alpha_composite(base, layer).convert("RGB"))


def make_frames(seconds: float, background, overlay=None, *, fps: float = FPS, noise: float = 1.5, seed: int = 0) -> list[SampledFrame]:
    """RGB frames at ``fps``: ``background(t)`` -> picture, ``overlay(img, t)`` -> picture with text/graphics, plus a little sensor noise."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(int(round(seconds * fps))):
        t = i / fps
        img = background(t)
        if overlay is not None:
            img = overlay(img, t)
        if noise:
            img = np.clip(img.astype(np.float32) + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
        out.append(SampledFrame(i, t, img))
    return out


def captions_overlay(events: list[tuple[float, float, str, dict]]):
    """events = (start, end, text, put_text kwargs); ``fade`` in the kwargs is a fade-in time in seconds."""
    def f(img: np.ndarray, t: float) -> np.ndarray:
        for a, b, text, kw in events:
            if a <= t < b:
                kk = dict(kw)
                fade = kk.pop("fade", 0.0)
                img = put_text(img, text, alpha=min(1.0, max(0.0, (t - a) / fade)) if fade else 1.0, **kk)
        return img
    return f


def analyze(frames: list[SampledFrame], seconds: float, **kw) -> CaptionTextResult:
    """``analyze_frames`` plus the invariants every result must satisfy (probabilities, ordered times, fixed vocabulary), so every scene below checks them."""
    res = CaptionTextAnalyzer().analyze_frames(frames, seconds, **kw)
    assert 0.0 <= res.caption_confidence <= 1.0 and 0.0 <= res.text_confidence <= 1.0
    for e in res.events:
        assert 0.0 <= e.start < e.end and 0.0 <= e.confidence <= 1.0 and e.kind in EVENT_KINDS and e.position in ("top", "center", "bottom", "lower_left")
        assert 0.0 <= e.relative_height <= 1.0 and 0.0 <= e.relative_width <= 1.0
    assert res.captions.style_class in CAPTION_STYLES and set(res.captions.traits) <= set(CAPTION_TRAITS)
    assert 0.0 <= res.captions.caption_coverage <= 1.0 and res.captions.captions_per_minute >= 0
    return res


def texture(seed: int, cell: int, amp: float) -> np.ndarray:
    """Colour noise at ``cell`` px (bicubic) with amplitude ``amp`` around mid-grey: foliage / confetti like texture for busy-picture tests."""
    rng = np.random.default_rng(seed)
    small = 0.5 + amp * (rng.random((H // cell + 2, W // cell + 2, 3)) - 0.5)
    im = Image.fromarray((np.clip(small, 0, 1) * 255).astype(np.uint8)).resize(((W // cell + 2) * cell, (H // cell + 2) * cell), Image.Resampling.BICUBIC).crop((0, 0, W, H))
    return np.asarray(im)


SHORT_WORDS = ["LOOK AT THIS", "SO MUCH FUN", "REALLY GREAT", "ONE MORE TIME", "KEEP WATCHING", "THE END NOW", "COME ON", "BIG NEWS"]
LONG_LINES = ["the quick brown fox jumps over the lazy dog", "a second long caption with many more words in it", "and this third line keeps on going for quite a while",
              "finally the last subtitle comes to an end here"]


def series(texts: list[str], n: int, period: float, dur: float, start: float = 0.5, **kw) -> list[tuple[float, float, str, dict]]:
    return [(start + i * period, start + i * period + dur, texts[i % len(texts)], dict(kw)) for i in range(n)]


# ---------------------------------------------------------------------------------------------- real FFmpeg videos
@pytest.fixture(scope="module")
def vid(tmp_path_factory):
    """``vid(name, seconds, events, shots=None, filters=None)`` -> (path, result); videos are made once, analysed through the real FrameSampler."""
    d = tmp_path_factory.mktemp("cap_videos")
    sampler = FrameSampler(FFmpegService())
    cache: dict[str, tuple[Path, CaptionTextResult]] = {}

    def make(name: str, seconds: float, events: list[dict] | None = None, shots: list[float] | None = None, filters: str | None = None):
        if name not in cache:
            p = d / f"{name}.mp4"
            if filters is None:
                overlay_video(p, seconds, events or [], fps=12, shots=shots)
            else:  # a custom drawtext chain over the same kind of base picture
                base = d / f"_{name}_b.mp4"
                shots_video(base, shots, fps=12) if shots else camera_video(base, seconds, "static", fps=12)
                run_ffmpeg(["-i", str(base), "-vf", filters, "-c:v", "libx264", "-preset", "ultrafast", "-crf", "16", "-pix_fmt", "yuv420p", str(p)])
            cache[name] = (p, CaptionTextAnalyzer().analyze_video(sampler, p, seconds))
        return cache[name]

    return make


def drawtext(text: str, a: float, b: float, *, x: str = "(w-text_w)/2", y: str = "h-h*0.10-text_h", size: float = 0.07, color: str = "white") -> str:
    fp = font_path(True).replace(":", "\\:")
    return (f"drawtext=fontfile='{fp}':text='{text}':fontsize={int(VIDEO_SIZE[1] * size)}:fontcolor={color}:borderw=2:bordercolor=black:x={x}:y={y}:"
            f"enable='between(t,{a},{b})'")


def short_caption_events(n: int = 6) -> list[dict]:
    return [dict(start=0.5 + i * 1.5, end=1.5 + i * 1.5, text=SHORT_WORDS[i % len(SHORT_WORDS)], pos="bottom") for i in range(n)]


@needs_ffmpeg
class TestVideoScenes:
    def test_clean_video_has_no_captions(self, vid):
        _, res = vid("clean_shots", 10, [], shots=[2.5, 2.5, 2.5, 2.5])
        assert not res.captions.caption_present
        assert res.events == []
        assert res.text.text_events_per_minute == 0 and res.text.graphic_events_per_minute == 0
        assert res.caption_confidence >= 0.6 and res.text_confidence >= 0.6  # a clean absence is still a confident one
        assert "no captions detected" in res.notes

    @pytest.mark.parametrize("shots", [None, [2.5, 2.5, 2.5, 2.5]])
    def test_frequent_short_bottom_captions(self, vid, shots):
        truth = [(0.5 + i * 1.5, 1.5 + i * 1.5) for i in range(6)]
        _, res = vid(f"freq_{bool(shots)}", 10, short_caption_events(), shots=shots)
        c = res.captions
        caps = _kinds(res, "CAPTION")
        assert c.caption_present and abs(len(caps) - 6) <= 1
        assert _pairs(caps, truth) >= 5  # timing within 0.5 s
        assert 30 <= c.captions_per_minute <= 42  # truth: 6 captions in 10 s = 36 / min
        assert 0.51 <= c.caption_coverage <= 0.69  # truth 0.60, within 15 %
        assert c.caption_position == "bottom" and all(e.position == "bottom" for e in caps)
        assert 0.9 <= c.caption_line_count <= 1.2
        assert 1.0 <= c.average_words_per_caption <= 3.5  # truth: 2-3 words each (an estimate)
        assert 0.06 <= c.relative_text_height <= 0.11  # font 0.07 of the height, outline included
        assert "bottom_position" in c.traits and "short_caption_segments" in c.traits
        assert res.caption_confidence >= 0.6
        assert c.caption_animation_rate == 0.0 and c.caption_emphasis_rate == 0.0 and not c.has_background_box

    def test_few_long_two_line_captions(self, vid):
        filters = ",".join(drawtext(t, a, b, y=y, size=0.055) for a, b, l1, l2 in ((1.0, 5.0, "The quick brown fox jumps over", "the lazy dog by the river"),
                                                                               (6.0, 10.0, "A second long caption that runs", "on across two full lines"))
                           for t, y in ((l1, "h-h*0.20-text_h"), (l2, "h-h*0.12-text_h")))
        _, res = vid("two_line", 12, filters=filters, shots=[3, 3, 3, 3])
        c = res.captions
        caps = _kinds(res, "CAPTION")
        assert c.caption_present and len(caps) == 2
        assert _pairs(caps, [(1.0, 5.0), (6.0, 10.0)]) == 2
        assert all(e.lines == 2 for e in caps) and c.caption_line_count >= 1.9
        assert 6.0 <= c.captions_per_minute <= 14.0  # truth 10 / min
        assert 0.567 <= c.caption_coverage <= 0.767  # truth 8 s of 12
        assert 5.0 <= c.average_words_per_caption <= 16.0  # truth 11-12 words: an estimate from the width / height ratio, mixed case runs short
        assert c.caption_position == "bottom"

    def test_number_cards_are_centred_large_and_brief(self, vid):
        ev = [dict(start=1.5, end=3.0, text="128", pos="center", size=0.30), dict(start=6.0, end=7.5, text="42", pos="center", size=0.30)]
        _, res = vid("num_shots", 10, ev, shots=[2.5, 2.5, 2.5, 2.5])
        cards = _kinds(res, "NUMBER_CARD")
        assert len(cards) == 2 and _pairs(cards, [(1.5, 3.0), (6.0, 7.5)]) == 2
        assert all(e.position == "center" and e.relative_height >= 0.14 and e.lines == 1 for e in cards)
        assert all(e.confidence <= 0.5 for e in cards)  # a size / shape heuristic: the digits are not read
        assert not res.captions.caption_present  # two big centred cards are not a caption series
        assert 9.0 <= res.text.number_graphic_frequency <= 15.0  # truth: 2 in 10 s = 12 / min
        assert res.text.position_share["center"] == 1.0
        assert any("digits are not read" in n for n in res.notes)

    def test_top_headline(self, vid):
        _, res = vid("headline", 8, [dict(start=1.0, end=6.0, text="BREAKING STORY OF THE DAY", pos="top", size=0.09)])
        heads = _kinds(res, "HEADLINE")
        assert len(heads) == 1 and _pairs(heads, [(1.0, 6.0)]) == 1
        e = heads[0]
        assert e.position == "top" and e.lines == 1 and 0.08 <= e.relative_height <= 0.13 and e.relative_width >= 0.6
        assert not res.captions.caption_present
        assert 6.0 <= res.text.headline_frequency <= 9.0  # 1 in 8 s = 7.5 / min
        assert res.text.position_share["top"] == 1.0 and 4.0 <= res.text.average_duration <= 5.6

    def test_lower_third_with_box_and_without(self, vid):
        _, res = vid("lower_box", 8, [dict(start=1.5, end=5.5, text="Jane Example", pos="lower_third", size=0.06, box=True)])
        lt = _kinds(res, "LOWER_THIRD")
        assert len(lt) == 1 and _pairs(lt, [(1.5, 5.5)]) == 1
        assert lt[0].position == "lower_left" and lt[0].has_box and lt[0].relative_height <= 0.075
        assert 6.0 <= res.text.lower_third_frequency <= 9.0 and not res.captions.caption_present
        _, res2 = vid("lower_nobox", 8, [dict(start=1.5, end=5.5, text="Jane Example", pos="lower_third", size=0.06)])
        assert _kinds(res2, "LOWER_THIRD") == []  # the same text without a plate is plain TEXT in the same place
        t = _kinds(res2, "TEXT")
        assert len(t) == 1 and t[0].position == "lower_left" and not t[0].has_box

    def test_fade_in_captions_are_animated_instant_ones_are_not(self, vid):
        def events(fade):
            return [dict(start=0.5 + i * 1.6, end=1.7 + i * 1.6, text=SHORT_WORDS[i], pos="bottom", fade=fade) for i in range(6)]

        _, fade = vid("fade_caps", 11, events(0.6), shots=[2.75, 2.75, 2.75, 2.75])
        _, inst = vid("inst_caps", 11, events(None), shots=[2.75, 2.75, 2.75, 2.75])
        assert fade.captions.caption_present and inst.captions.caption_present
        assert fade.captions.caption_animation_rate >= 0.8
        assert inst.captions.caption_animation_rate == 0.0
        assert abs(len(_kinds(fade, "CAPTION")) - 6) <= 1 and abs(len(_kinds(inst, "CAPTION")) - 6) <= 1

    def test_emphasised_word_inside_a_caption(self, vid):
        fnt = ImageFont.truetype(font_path(True), int(VIDEO_SIZE[1] * 0.07))
        w1, w2 = fnt.getlength("THIS IS "), fnt.getlength("AMAZING")
        x0 = (VIDEO_SIZE[0] - w1 - w2) / 2
        parts = []
        for i in range(5):
            a, b = 0.5 + i * 2.0, 2.0 + i * 2.0
            parts += [drawtext("THIS IS", a, b, x=f"{x0:.1f}", y="h-h*0.12-text_h"), drawtext("AMAZING", a, b, x=f"{x0 + w1:.1f}", y="h-h*0.12-text_h", color="yellow")]
        _, hi = vid("emph_yellow", 11, filters=",".join(parts), shots=[2.75] * 4)
        plain = [p.replace("color=yellow", "color=white") for p in parts]
        plain = [p.replace("fontcolor=yellow", "fontcolor=white") for p in parts]
        _, lo = vid("emph_white", 11, filters=",".join(plain), shots=[2.75] * 4)
        assert hi.captions.caption_present and lo.captions.caption_present
        assert hi.captions.caption_emphasis_rate >= 0.6 and lo.captions.caption_emphasis_rate == 0.0
        assert "frequent_highlighting" in hi.captions.traits and "frequent_highlighting" not in lo.captions.traits
        assert any(e.emphasized for e in _kinds(hi, "CAPTION")) and not any(e.emphasized for e in lo.events)

    def test_a_caption_that_stays_across_hard_cuts_is_one_event(self, vid):
        # 3 captions; the middle one stays on screen over a cut of the picture (cuts at 2.5, 5, 7.5 s)
        ev = [dict(start=0.5, end=2.0, text="FIRST ONE", pos="bottom"), dict(start=3.5, end=6.5, text="STAYS ACROSS A CUT", pos="bottom"),
              dict(start=8.0, end=9.5, text="LAST ONE", pos="bottom")]
        _, res = vid("across_cuts", 10, ev, shots=[2.5, 2.5, 2.5, 2.5])
        caps = _kinds(res, "CAPTION")
        assert len(caps) == 3 and _pairs(caps, [(0.5, 2.0), (3.5, 6.5), (8.0, 9.5)]) == 3

    def test_analyze_video_equals_analyze_frames_on_the_same_frames(self, vid):
        p, res = vid("freq_False", 10, short_caption_events())
        frames = list(FrameSampler(FFmpegService()).frames(p, FPS, W, H, gray=False))
        again = CaptionTextAnalyzer().analyze_frames(frames, 10)
        assert to_plain(again) == to_plain(res)

    def test_unreadable_video_is_reported_not_raised(self, tmp_path):
        bad = tmp_path / "broken.mp4"
        bad.write_bytes(b"this is not a video" * 50)
        res = CaptionTextAnalyzer().analyze_video(FrameSampler(FFmpegService()), bad, 5.0)
        assert not res.captions.caption_present and res.caption_confidence == 0.0 and res.text_confidence == 0.0 and res.events == []
        assert any("could not be decoded" in n or "Too few frames" in n for n in res.notes)


# ---------------------------------------------------------------------------------------------- no captions: busy and text-like pictures
class TestNoCaptions:
    def test_clean_and_busy_pictures_have_no_captions(self, pic):
        cases = {
            "static": make_frames(10, lambda t: pic(11)),
            "shots": make_frames(10, lambda t: pic(100 + int(t // 2.5))),
            "busy_moving": make_frames(10, lambda t: texture(int(t * 4) + 5, 6, 1.0)),
            "busy_shots": make_frames(10, lambda t: texture(int(t // 2.5) + 5, 4, 1.0)),
            "foliage": make_frames(10, lambda t: texture(int(t * 4) + 5, 14, 0.6)),
        }
        for name, frames in cases.items():
            res = analyze(frames, 10)
            assert not res.captions.caption_present, name
            assert [e for e in res.events if e.kind == "CAPTION"] == [], name
            assert "no captions detected" in res.notes, name

    @pytest.mark.parametrize("kind", ["bricks", "blinds", "barcode", "tiles", "poster"])
    def test_static_text_like_textures_are_not_captions(self, pic, kind):
        def paint(base: np.ndarray) -> np.ndarray:
            img = base.copy()
            rng = np.random.default_rng(3)
            if kind == "bricks":
                for r, y in enumerate(range(180, 250, 9)):
                    for x in range(60 - 12 * (r % 2), 420, 24):
                        img[y:y + 8, max(60, x):min(420, x + 23)] = (rng.integers(120, 200), rng.integers(50, 90), rng.integers(40, 70))
            elif kind == "blinds":
                for y in range(170, 250, 7):
                    img[y:y + 3, 60:420] = 15
            elif kind == "barcode":
                x = 140
                while x < 340:
                    w = int(rng.integers(1, 5))
                    img[190:240, x:x + w] = 0
                    x += w + int(rng.integers(1, 5))
            elif kind == "tiles":
                img[150:250:10, 40:440] = 20
                img[150:250, 40:440:10] = 20
            else:  # a printed poster: text-looking strokes that belong to the scene
                for k, line in enumerate(("TASTY", "OPEN TODAY")):
                    img = put_text(img, line, 240, 190 + k * 26, 20)
            return img

        res = analyze(make_frames(8, lambda t: paint(pic(11))), 8)
        assert not res.captions.caption_present
        assert [e for e in res.events if e.kind in ("CAPTION", "HEADLINE", "NUMBER_CARD", "LOWER_THIRD")] == []
        assert res.captions.captions_per_minute == 0

    def test_moving_text_like_texture_is_not_a_caption(self, pic):
        def drifting_blinds(t: float) -> np.ndarray:
            img = pic(11).copy()
            off = int(t * 9) % 7  # the stripes crawl: the pattern is never at the same place twice in a row
            for y in range(170 + off, 250, 7):
                img[y:y + 3, 60:420] = 15
            return img

        res = analyze(make_frames(8, drifting_blinds), 8)
        assert not res.captions.caption_present and _kinds(res, "CAPTION") == []

    def test_watermark_over_changing_pictures_is_not_an_editing_event(self, pic):
        wm = lambda img, t: put_text(img, "CHANNEL", 420, 20, 14, stroke=1)  # noqa: E731
        res = analyze(make_frames(10, lambda t: pic(100 + int(t // 2.5)), wm), 10)
        assert res.events == [] and not res.captions.caption_present
        assert any("logo / watermark-like" in n for n in res.notes)
        # ... and a watermark does not hide the real captions
        cap = captions_overlay(series(SHORT_WORDS, 4, 2.5, 2.0, cx=240, cy=235, size=24))
        res2 = analyze(make_frames(10, lambda t: pic(100 + int(t // 2.5)), lambda img, t: cap(wm(img, t), t)), 10)
        assert res2.captions.caption_present and len(_kinds(res2, "CAPTION")) == 4
        assert all(e.position == "bottom" for e in _kinds(res2, "CAPTION"))


# ---------------------------------------------------------------------------------------------- in-memory scenes: timing, positions, kinds
class TestCaptionGeometry:
    def test_timing_and_coverage_of_every_caption(self, pic):
        ev = series(SHORT_WORDS, 6, 1.5, 1.0, cx=240, cy=235, size=24)
        res = analyze(make_frames(10, lambda t: pic(100 + int(t // 2.5)), captions_overlay(ev)), 10)
        caps = _kinds(res, "CAPTION")
        assert len(caps) == 6 and _pairs(caps, [(a, b) for a, b, _t, _k in ev], tol=0.35) == 6
        assert abs(res.captions.caption_coverage - 0.6) <= 0.09 and abs(res.captions.captions_per_minute - 36.0) <= 3.0
        assert all(0.08 <= e.relative_height <= 0.11 for e in caps)  # a 24 px font on a 270 px frame, outline included
        assert all(0.4 <= e.confidence <= 1.0 for e in caps)

    def test_caption_position_bottom_center_top_and_mixed(self, pic):
        def run(cy_for):
            ev = [(0.5 + i * 1.2, 1.4 + i * 1.2, SHORT_WORDS[i], dict(cx=240, cy=cy_for(i), size=24)) for i in range(8)]
            return analyze(make_frames(10, lambda t: pic(100 + int(t // 2.5)), captions_overlay(ev)), 10)

        assert run(lambda i: 135).captions.caption_position == "center"
        top = run(lambda i: 40)
        assert top.captions.caption_position == "top" and all(e.position == "top" for e in _kinds(top, "CAPTION"))
        mixed = run(lambda i: 235 if i % 2 else 40)  # four at the bottom, four at the top (a top band needs 4 events to count as captions): neither is "the" position
        assert mixed.captions.caption_position == "mixed" and len(_kinds(mixed, "CAPTION")) == 8
        three = run(lambda i: 235 if i < 5 else 40)  # a top series of only three is a run of headlines, not captions
        assert three.captions.caption_position == "bottom" and len(_kinds(three, "HEADLINE")) == 3

    def test_two_headlines_at_the_top_are_headlines_not_captions(self, pic):
        ev = [(1.0, 4.0, "BREAKING NEWS TODAY", dict(cx=240, cy=40, size=30)), (6.0, 9.0, "SECOND BIG STORY HERE", dict(cx=240, cy=40, size=30))]
        res = analyze(make_frames(11, lambda t: pic(100 + int(t // 2.75)), captions_overlay(ev)), 11)
        heads = _kinds(res, "HEADLINE")
        assert len(heads) == 2 and not res.captions.caption_present
        assert _pairs(heads, [(1.0, 4.0), (6.0, 9.0)], tol=0.35) == 2
        assert res.text.headline_frequency == pytest.approx(2 / 11 * 60, abs=1.0)

    def test_overlapping_text_is_separated_by_kind(self, pic):
        """A series of bottom captions plus a headline, a lower third and a number card in the same video: each lands in its own class."""
        caps = series(SHORT_WORDS, 4, 1.8, 1.4, start=0.5, cx=240, cy=235, size=22)
        extra = [(1.0, 4.5, "BREAKING STORY", dict(cx=240, cy=38, size=30)), (9.0, 11.5, "Jane Example", dict(cx=40, cy=200, size=18, box=True, anchor="lm")),
                 (7.2, 8.6, "128", dict(cx=240, cy=135, size=80))]
        res = analyze(make_frames(12, lambda t: pic(100 + int(t // 3)), captions_overlay(caps + extra)), 12)
        assert len(_kinds(res, "CAPTION")) == 4
        assert len(_kinds(res, "HEADLINE")) == 1 and len(_kinds(res, "LOWER_THIRD")) == 1 and len(_kinds(res, "NUMBER_CARD")) == 1
        assert res.text.headline_frequency > 0 and res.text.lower_third_frequency > 0 and res.text.number_graphic_frequency > 0
        shares = res.text.position_share
        assert pytest.approx(sum(shares.values()), abs=0.01) == 1.0 and shares["top"] > 0 and shares["lower_left"] > 0 and shares["center"] > 0

    def test_line_count_and_estimated_words_scale_with_the_text(self, pic):
        one = [(1.0, 4.0, "the quick brown fox jumps", dict(cx=240, cy=235, size=16)), (5.0, 8.0, "over the lazy dog today", dict(cx=240, cy=235, size=16))]
        two = []
        for a, b, text, kw in one:
            w = text.split()
            two += [(a, b, " ".join(w[:3]), dict(cx=240, cy=222, size=16)), (a, b, " ".join(w[3:]), dict(cx=240, cy=244, size=16))]
        r1 = analyze(make_frames(10, lambda t: pic(100 + int(t // 5)), captions_overlay(one)), 10)
        r2 = analyze(make_frames(10, lambda t: pic(100 + int(t // 5)), captions_overlay(two)), 10)
        assert r1.captions.caption_line_count == pytest.approx(1.0, abs=0.01) and r2.captions.caption_line_count == pytest.approx(2.0, abs=0.01)
        assert 3.0 <= r1.captions.average_words_per_caption <= 7.5  # truth 5 and 4-5 words
        assert 3.0 <= r2.captions.average_words_per_caption <= 7.5
        assert r1.captions.average_chars_per_line > r2.captions.average_chars_per_line  # the same text split in two lines is shorter per line

    def test_larger_text_is_measured_larger(self, pic):
        def height(size: int) -> float:
            ev = series(SHORT_WORDS, 4, 2.0, 1.5, cx=240, cy=200, size=size)
            return analyze(make_frames(9, lambda t: pic(100 + int(t // 3)), captions_overlay(ev)), 9).captions.relative_text_height

        h18, h26, h36 = height(18), height(26), height(36)
        assert h18 < h26 < h36
        assert h18 == pytest.approx(18 * 1.2 / H, rel=0.35) and h36 == pytest.approx(36 * 1.2 / H, rel=0.35)

    def test_event_confidence_is_lower_for_tiny_text(self, pic):
        def conf(size: int) -> float:
            ev = series(SHORT_WORDS, 4, 2.0, 1.5, cx=240, cy=235, size=size)
            r = analyze(make_frames(9, lambda t: pic(100 + int(t // 3)), captions_overlay(ev)), 9)
            return float(np.mean([e.confidence for e in r.events])) if r.events else 0.0

        assert conf(26) > conf(11)  # 11 px text on a 270 px frame is at the edge of what can be measured

    def test_large_text_is_one_line_not_fragments(self, pic):
        """Large glyphs have an empty interior, so a naive row profile cuts one number into several 'lines'; the coarse pass keeps it whole."""
        det = TextRegionDetector(W, H)
        for size in (50, 70, 85):
            img = put_text(pic(11), "128", 240, 135, size)
            lines = det.detect(img).lines
            big = [ln for ln in lines if ln.text_score >= 0.5 and ln.h >= 0.3 * size]  # (small bits of the picture's own rectangles may score too; fragments of the digits would be tall enough to count)
            assert len(big) == 1, (size, [(ln.h, ln.w) for ln in lines])
            assert 0.55 * size <= big[0].h <= 1.15 * size and abs(big[0].cx - 240) <= 8 and abs(big[0].cy - 135) <= 8

    def test_two_tight_lines_stay_two_lines(self, pic):
        det = TextRegionDetector(W, H)
        img = put_text(put_text(pic(11), "the quick brown fox jumps", 240, 214, 16), "over the lazy dog today", 240, 235, 16)
        lines = [ln for ln in det.detect(img).lines if ln.text_score >= 0.5 and ln.y0 > 150]  # (the caption area; the picture's own rectangles are up in the corner)
        assert len(lines) == 2 and max(ln.h for ln in lines) <= 26  # not merged into one big band by the coarse pass


# ---------------------------------------------------------------------------------------------- appearance: animation, emphasis, box
class TestAppearance:
    def test_fade_in_is_animation_instant_is_not(self, pic):
        def rate(fade: float) -> float:
            ev = series(SHORT_WORDS, 6, 1.6, 1.3, cx=240, cy=235, size=22, fade=fade)
            return analyze(make_frames(10, lambda t: pic(100 + int(t // 2.5)), captions_overlay(ev)), 10).captions.caption_animation_rate

        assert rate(0.0) == 0.0
        assert rate(0.6) >= 0.8 and rate(0.5) >= 0.8

    def test_fade_in_text_events_count_for_the_text_stats(self, pic):
        ev = [(1.0, 5.0, "BREAKING STORY", dict(cx=240, cy=38, size=30, fade=0.7)), (7.0, 11.0, "ANOTHER STORY", dict(cx=240, cy=38, size=30))]
        res = analyze(make_frames(13, lambda t: pic(100 + int(t // 3.25)), captions_overlay(ev)), 13)
        heads = sorted(_kinds(res, "HEADLINE"), key=lambda e: e.start)
        assert len(heads) == 2 and heads[0].animated and not heads[1].animated
        assert res.text.animation_rate == pytest.approx(0.5)

    @pytest.mark.parametrize("colour", [(255, 230, 0), (60, 255, 60), (255, 40, 40), (0, 220, 255)])
    def test_highlighted_word_in_any_vivid_colour(self, pic, colour):
        fnt = _font(26)
        w1, w2 = fnt.getlength("THIS IS "), fnt.getlength("AMAZING")
        x0 = 240 - (w1 + w2) / 2

        def overlay(img: np.ndarray, t: float) -> np.ndarray:
            if (t % 2.0) < 1.5 and t >= 0.5:
                img = put_text(img, "THIS IS", x0, 235, 26, anchor="lm")
                img = put_text(img, "AMAZING", x0 + w1, 235, 26, color=colour, anchor="lm")
            return img

        res = analyze(make_frames(10, lambda t: pic(100 + int(t // 2.5)), overlay), 10)
        caps = _kinds(res, "CAPTION")
        assert len(caps) >= 4 and res.captions.caption_emphasis_rate >= 0.7
        assert "frequent_highlighting" in res.captions.traits

    def test_single_colour_caption_has_no_emphasis_even_over_colourful_pictures(self, pic):
        ev = series(SHORT_WORDS, 6, 1.6, 1.3, cx=240, cy=235, size=24)
        for bg in (lambda t: pic(100 + int(t // 2.5)), lambda t: texture(int(t * 4) + 5, 8, 0.8)):
            res = analyze(make_frames(10, bg, captions_overlay(ev)), 10)
            assert res.captions.caption_present and res.captions.caption_emphasis_rate <= 0.2
            assert "frequent_highlighting" not in res.captions.traits

    def test_background_box(self, pic):
        boxed = analyze(make_frames(8, lambda t: pic(100 + int(t // 2)), captions_overlay(series(LONG_LINES, 2, 3.4, 2.8, cx=240, cy=232, size=16, box=True))), 8)
        plain = analyze(make_frames(8, lambda t: pic(100 + int(t // 2)), captions_overlay(series(LONG_LINES, 2, 3.4, 2.8, cx=240, cy=232, size=16))), 8)
        assert boxed.captions.has_background_box and all(e.has_box for e in _kinds(boxed, "CAPTION"))
        assert not plain.captions.has_background_box and not any(e.has_box for e in plain.events)

    def test_lower_third_needs_the_plate(self, pic):
        def run(box: bool):
            ev = [(2.0, 6.0, "Jane Example", dict(cx=40, cy=212, size=18, box=box, anchor="lm"))]
            return analyze(make_frames(9, lambda t: pic(100 + int(t // 3)), captions_overlay(ev)), 9)

        with_plate, without = run(True), run(False)
        assert [e.kind for e in with_plate.events] == ["LOWER_THIRD"] and with_plate.events[0].has_box
        assert [e.kind for e in without.events] == ["TEXT"] and without.events[0].position == "lower_left"


# ---------------------------------------------------------------------------------------------- caption style classes
STYLE_SCENES = {
    # name: (seconds, caption script, expected class)
    "Social": (10, series(SHORT_WORDS, 6, 1.6, 1.2, cx=240, cy=235, size=20, fade=0.5), "Social"),
    "High-Impact": (10, series(SHORT_WORDS, 5, 1.8, 1.4, cx=240, cy=135, size=34, fade=0.4), "High-Impact"),
    "Subtitle-focused": (11, series(LONG_LINES, 4, 2.7, 2.4, cx=240, cy=235, size=15), "Subtitle-focused"),
    "News": (11, series(LONG_LINES, 3, 3.4, 2.8, cx=240, cy=232, size=16, box=True), "News"),
    "Documentary": (12, series(LONG_LINES, 2, 5.0, 2.4, start=1.0, cx=240, cy=235, size=16), "Documentary"),
    "Bold": (12, series(["THE QUICK BROWN FOX JUMPS", "A VERY BOLD CAPTION HERE"], 2, 5.0, 2.4, start=1.0, cx=240, cy=235, size=21), "Bold"),
    "Minimal": (12, series(["HI THERE", "OK"], 2, 5.0, 2.0, start=1.0, cx=240, cy=235, size=14), "Minimal"),
}


@pytest.fixture(scope="module")
def style_result(pic):
    """``style_result(name)`` -> the analysis of one STYLE_SCENES scene (each scene is analysed once for the whole module)."""
    cache: dict[str, CaptionTextResult] = {}

    def get(name: str) -> CaptionTextResult:
        if name not in cache:
            seconds, script, _ = STYLE_SCENES[name]
            cache[name] = analyze(make_frames(seconds, lambda t: pic(100 + int(t // 2.5)), captions_overlay(script)), seconds)
        return cache[name]

    return get


class TestStyleClass:
    @pytest.mark.parametrize("name", list(STYLE_SCENES))
    def test_synthetic_styles(self, style_result, name):
        expected = STYLE_SCENES[name][2]
        res = style_result(name)
        assert res.captions.caption_present
        assert res.captions.style_class == expected, (name, res.captions)
        assert res.captions.style_class in CAPTION_STYLES and set(res.captions.traits) <= set(CAPTION_TRAITS)

    def test_the_documented_rules_on_hand_made_stats(self):
        cls = CaptionTextAnalyzer.style_class

        def mk(**kw) -> CaptionStats:
            base = dict(caption_present=True, relative_text_height=0.05, average_words_per_caption=6.0, captions_per_minute=8.0, caption_coverage=0.3, caption_position="bottom")
            return CaptionStats(**{**base, **kw})

        assert cls(mk(relative_text_height=0.12, caption_position="center", average_words_per_caption=3.0)) == "High-Impact"
        assert cls(mk(relative_text_height=0.09, average_words_per_caption=2.0, caption_animation_rate=0.8)) == "High-Impact"
        assert cls(mk(relative_text_height=0.065, average_words_per_caption=3.0, captions_per_minute=20.0, caption_animation_rate=0.5)) == "Social"
        assert cls(mk(relative_text_height=0.05, average_words_per_caption=3.0, captions_per_minute=20.0, caption_emphasis_rate=0.4)) == "Social"
        assert cls(mk(relative_text_height=0.08)) == "Bold"
        assert cls(mk(has_background_box=True)) == "News"
        assert cls(mk(caption_coverage=0.8, captions_per_minute=15.0)) == "Subtitle-focused"
        assert cls(mk()) == "Documentary"
        assert cls(mk(average_words_per_caption=2.0)) == "Minimal"
        assert cls(mk(relative_text_height=0.03)) == "Minimal"
        # precedence: size wins over the box (a large boxed series is Bold, not News); a calm, sparse series with animation is not Social (it needs short, frequent captions)
        assert cls(mk(has_background_box=True, relative_text_height=0.08)) == "Bold"
        assert cls(mk(caption_animation_rate=0.9)) == "Minimal"

    def test_traits_follow_the_measurements(self):
        st = CaptionStats(caption_present=True, relative_text_height=0.1, average_words_per_caption=2.0, caption_emphasis_rate=0.5, caption_position="center")
        assert set(CaptionTextAnalyzer.traits(st, 0.7)) == {"large_text", "high_contrast", "frequent_highlighting", "short_caption_segments", "center_position"}
        calm = CaptionStats(caption_present=True, relative_text_height=0.05, average_words_per_caption=9.0, caption_position="bottom")
        assert CaptionTextAnalyzer.traits(calm, 0.3) == ["bottom_position"]

    def test_the_scenes_come_out_as_at_least_three_distinct_styles(self, style_result):
        found = {style_result(name).captions.style_class for name in STYLE_SCENES}
        assert len(found) == len(STYLE_SCENES) >= 3  # every synthetic style is told apart from every other


# ---------------------------------------------------------------------------------------------- graphics (flat plates / bars)
class TestGraphics:
    @staticmethod
    def rects(spec):
        def f(img: np.ndarray, t: float) -> np.ndarray:
            img = img.copy()
            for a, b, (x0, y0, x1, y1), col in spec:
                if a <= t < b:
                    img[y0:y1, x0:x1] = col
            return img
        return f

    def test_flat_bars_popping_in_over_a_steady_picture(self, pic):
        spec = [(2.0, 4.5, (40, 100, 440, 140), (20, 20, 120)), (7.0, 10.0, (300, 185, 450, 240), (230, 230, 40))]
        res = analyze(make_frames(12, lambda t: pic(11), self.rects(spec)), 12)
        g = _kinds(res, "GRAPHIC")
        assert len(g) == 2 and _pairs(g, [(2.0, 4.5), (7.0, 10.0)], tol=0.35) == 2
        assert res.text.graphic_events_per_minute == pytest.approx(2 / 12 * 60, abs=0.5)
        assert not res.captions.caption_present and res.text.text_events_per_minute == 0
        assert all(e.lines == 0 for e in g) and g[1].position == "bottom"

    def test_graphics_over_moving_footage_are_not_claimed(self, pic):
        spec = [(2.0, 5.5, (40, 100, 440, 140), (20, 20, 120))]
        res = analyze(make_frames(10, lambda t: np.roll(pic(5), int(t * 40), axis=1), self.rects(spec)), 10)  # a fast pan: every frame differs
        assert res.text.graphic_events_per_minute == 0
        assert any("lower bound" in n for n in res.notes)  # the limit is stated, not hidden

    def test_a_text_plate_is_not_also_a_graphic(self, pic):
        ev = [(2.0, 6.0, "Jane Example", dict(cx=40, cy=212, size=18, box=True, anchor="lm"))]
        res = analyze(make_frames(9, lambda t: pic(11), captions_overlay(ev)), 9)
        assert _kinds(res, "GRAPHIC") == [] and res.text.graphic_events_per_minute == 0


# ---------------------------------------------------------------------------------------------- confidence
class TestConfidence:
    def test_clean_beats_busy_when_nothing_is_found(self, pic):
        clean = analyze(make_frames(12, lambda t: pic(11)), 12)
        busy = analyze(make_frames(12, lambda t: texture(int(t * 4) + 5, 6, 1.0)), 12)
        assert not clean.captions.caption_present and not busy.captions.caption_present
        assert clean.caption_confidence >= 0.6 and busy.caption_confidence <= 0.4
        assert clean.caption_confidence > busy.caption_confidence + 0.3 and clean.text_confidence > busy.text_confidence + 0.3
        assert any("busy" in n for n in busy.notes)

    def test_clean_beats_busy_when_captions_are_found(self, pic):
        ev = series(SHORT_WORDS, 5, 1.5, 1.0, cx=240, cy=235, size=24)
        clean = analyze(make_frames(8, lambda t: pic(100 + int(t // 2)), captions_overlay(ev)), 8)
        by_busyness = []
        for cell, amp in ((14, 0.6), (8, 0.8), (6, 1.0)):
            r = analyze(make_frames(8, lambda t: texture(int(t * 4) + 5, cell, amp), captions_overlay(ev)), 8)
            assert r.captions.caption_present, (cell, amp)  # strong outlined captions survive texture, but are believed less
            by_busyness.append(r.caption_confidence)
        assert clean.caption_confidence > by_busyness[0] > by_busyness[1] > by_busyness[2]

    def test_few_frames_and_low_resolution_lower_the_confidence(self, pic):
        long_clean = analyze(make_frames(12, lambda t: pic(11)), 12)
        short_clean = analyze(make_frames(3, lambda t: pic(11)), 3)
        assert 0 < short_clean.caption_confidence < long_clean.caption_confidence
        small = [SampledFrame(f.index, f.time, np.asarray(Image.fromarray(f.pixels).resize((160, 90), Image.Resampling.BILINEAR))) for f in make_frames(12, lambda t: pic(11))]
        low = analyze(small, 12)
        assert 0 < low.caption_confidence < long_clean.caption_confidence and any("resolution" in n for n in low.notes)


# ---------------------------------------------------------------------------------------------- degenerate input
class TestRobustness:
    def test_unusable_inputs_return_zero_confidence_not_exceptions(self, pic):
        a = CaptionTextAnalyzer()
        frames = make_frames(10, lambda t: pic(11))
        for res, why in ((a.analyze_frames([], 0.0), "Too few"), (a.analyze_frames(frames[:6], 1.5), "Too few"),
                         (a.analyze_frames([SampledFrame(f.index, f.time, f.pixels[:40, :60]) for f in frames], 10), "too small")):
            assert not res.captions.caption_present and res.caption_confidence == 0.0 and res.text_confidence == 0.0 and res.events == []
            assert any(why in n for n in res.notes) and any(n.startswith("no captions detected") for n in res.notes)

    def test_constant_and_black_frames_report_nothing_and_say_so(self):
        for value in (128, 0):
            frames = [SampledFrame(i, i / 4, np.full((H, W, 3), value, np.uint8)) for i in range(40)]
            res = analyze(frames, 10)
            assert not res.captions.caption_present and res.events == [] and any("single flat colour" in n for n in res.notes)
            assert res.caption_confidence <= 0.4  # nothing measurable: not claimed with full confidence

    def test_gray_float_and_generator_inputs(self, pic):
        ev = captions_overlay(series(SHORT_WORDS, 3, 2.0, 1.5, cx=240, cy=235, size=24))
        frames = make_frames(7, lambda t: pic(100 + int(t // 3.5)), ev)
        ref = analyze(frames, 7)
        gray = analyze([SampledFrame(f.index, f.time, f.pixels.mean(axis=2).astype(np.uint8)) for f in frames], 7)
        flt = analyze([SampledFrame(f.index, f.time, f.pixels.astype(np.float32) / 255.0) for f in frames], 7)
        gen = analyze((f for f in frames), 7)
        assert ref.captions.caption_present and gray.captions.caption_present and flt.captions.caption_present
        assert len(_kinds(gray, "CAPTION")) == len(_kinds(ref, "CAPTION")) == len(_kinds(flt, "CAPTION")) == 3
        assert to_plain(gen) == to_plain(ref)

    def test_wrong_sized_frames_are_skipped_and_reported(self, pic):
        frames = make_frames(10, lambda t: pic(11))
        odd = frames[:20] + [SampledFrame(99, 99.0, np.zeros((100, 100, 3), np.uint8)), SampledFrame(100, 100.0, None)] + frames[20:]  # type: ignore[arg-type]
        res = analyze(odd, 10)
        assert any("2 unusable frames were skipped" in n for n in res.notes) and res.caption_confidence > 0

    def test_duration_is_inferred_when_unknown(self, pic):
        frames = make_frames(7, lambda t: pic(100 + int(t // 3.5)), captions_overlay(series(SHORT_WORDS, 4, 1.5, 1.0, cx=240, cy=235, size=24)))
        assert analyze(frames, 0.0).captions.captions_per_minute == pytest.approx(analyze(frames, 7.0).captions.captions_per_minute, rel=0.1)

    def test_cancel_and_progress(self, pic):
        frames = make_frames(10, lambda t: pic(11))
        ev = threading.Event()
        ev.set()
        with pytest.raises(AnalysisCancelled):
            analyze(frames, 10, cancel=ev)
        seen: list[float] = []
        analyze(frames, 10, progress=seen.append)
        assert seen and seen == sorted(seen) and seen[-1] == 1.0 and all(0.0 <= p <= 1.0 for p in seen)

    def test_deterministic(self, pic):
        frames = make_frames(7, lambda t: pic(100 + int(t // 3.5)), captions_overlay(series(SHORT_WORDS, 4, 1.5, 1.0, cx=240, cy=235, size=24, fade=0.5)))
        assert to_plain(analyze(frames, 7)) == to_plain(analyze(frames, 7))

    def test_position_helper(self):
        assert position_of(0.5, 0.9, 0.3, 0.4) == "bottom" and position_of(0.5, 0.1, 0.3, 0.4) == "top" and position_of(0.5, 0.5, 0.3, 0.4) == "center"
        assert position_of(0.2, 0.8, 0.05, 0.3) == "lower_left"


# ---------------------------------------------------------------------------------------------- originality: geometry and counts only
def _walk(value, found: list):
    """Every leaf (and every container type) reachable in a result object."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for f in dataclasses.fields(value):
            _walk(getattr(value, f.name), found)
    elif isinstance(value, dict):
        for k, v in value.items():
            found.append(k)
            _walk(v, found)
    elif isinstance(value, (list, tuple, set)):
        for v in value:
            _walk(v, found)
    else:
        found.append(value)


NOTE_TEMPLATES = re.compile(
    r"^(Captions and text are found from stroke geometry|Too few frames \(\d+\) to measure|The frames are too small|The video frames could not be decoded|no captions detected|"
    r"\d+ faint or flickering text-like region|\d+ text-like region\(s\) never behaved|\d+ overlay\(s\) stayed on screen|The picture is|Low frame resolution|"
    r"Number cards are a size|Graphic overlays are only found|\d+ unusable frames were skipped)")


@pytest.fixture(scope="module")
def results(pic) -> list[CaptionTextResult]:
    """A few different analyses (captions + headline + lower third + number card, a clean picture, a busy picture, an unusable input)."""
    scenes = []
    caps = series(SHORT_WORDS, 4, 1.8, 1.4, start=0.5, cx=240, cy=235, size=22, fade=0.5)
    extra = [(1.0, 4.5, "BREAKING STORY", dict(cx=240, cy=38, size=30)), (9.0, 11.5, "Jane Example", dict(cx=40, cy=200, size=18, box=True, anchor="lm")),
             (7.2, 8.6, "128", dict(cx=240, cy=135, size=80))]
    scenes.append(analyze(make_frames(12, lambda t: pic(100 + int(t // 3)), captions_overlay(caps + extra)), 12))
    scenes.append(analyze(make_frames(8, lambda t: pic(11)), 8))
    scenes.append(analyze(make_frames(8, lambda t: texture(int(t * 4) + 5, 6, 1.0)), 8))
    scenes.append(analyze([], 0.0))
    return scenes


class TestOriginality:
    WORDS = ("HELLO", "FUN", "GREAT", "Jane", "Example", "quick", "brown", "BREAKING", "STORY", "CHANNEL", "AMAZING")

    def test_results_hold_only_numbers_and_fixed_class_names(self, results):
        for res in results:
            leaves: list = []
            _walk(res.captions, leaves)
            _walk(res.text, leaves)
            _walk(res.events, leaves)
            assert all(isinstance(v, (str, int, float, bool)) for v in leaves)  # no arrays, bytes, images, nested objects
            strings = {v for v in leaves if isinstance(v, str)}
            assert strings <= FIXED_VOCABULARY, strings - FIXED_VOCABULARY
            assert {e.kind for e in res.events} <= set(EVENT_KINDS)
            assert set(res.text.position_share) <= {"top", "center", "bottom", "lower_left"}

    def test_notes_are_fixed_phrases_without_any_content(self, results):
        for res in results:
            assert res.notes and all(NOTE_TEMPLATES.match(n) for n in res.notes), [n for n in res.notes if not NOTE_TEMPLATES.match(n)]
            text = " ".join(res.notes).lower()
            assert not any(re.search(rf"\b{w.lower()}\b", text) for w in self.WORDS)

    def test_the_result_is_small_so_no_pixels_can_hide_in_it(self, results):
        for res in results:
            assert len(repr(res)) < 20_000 and len(res.events) < 100

    def test_nothing_is_logged_while_analysing(self, pic):
        records: list[logging.LogRecord] = []

        class Grab(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        root, handler = logging.getLogger(), Grab(level=logging.DEBUG)
        old_level = root.level
        root.addHandler(handler)
        root.setLevel(logging.DEBUG)
        try:
            ev = captions_overlay(series(SHORT_WORDS, 3, 2.0, 1.5, cx=240, cy=235, size=24, fade=0.5))
            analyze(make_frames(8, lambda t: pic(100 + int(t // 2.5)), ev), 8)
            analyze(make_frames(8, lambda t: texture(int(t * 4) + 5, 6, 1.0)), 8)
        finally:
            root.removeHandler(handler)
            root.setLevel(old_level)
        assert not [r for r in records if r.name.startswith("app.reference")]

    @needs_ffmpeg
    def test_video_analysis_result_has_no_content_either(self, vid):
        _, res = vid("freq_False", 10, short_caption_events())
        leaves: list = []
        _walk(res.captions, leaves)
        _walk(res.text, leaves)
        _walk(res.events, leaves)
        assert {v for v in leaves if isinstance(v, str)} <= FIXED_VOCABULARY
        assert not any(re.search(rf"\b{w.lower()}\b", " ".join(res.notes).lower()) for w in self.WORDS)
