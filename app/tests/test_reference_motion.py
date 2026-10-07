"""Phase 7 motion: the global-motion estimator (zoom + translation) and the MotionAnalyzer (per-shot kinds, zoom/pan events, intensity, stats).

Two kinds of ground truth:
* an *analytic* scene (a sum of random cosines evaluated at exactly warped coordinates, so any zoom/shift is exact and there is no interpolation error),
  used for the estimator and for in-memory signals (fast, no FFmpeg);
* FFmpeg videos from ``reference_helpers`` (static / zoom / zoom_out / pan / shake) and a few built here (centred zoom, a mid-shot punch-in, concatenated
  clips), which go through the real FrameSampler -> compute_signals -> MotionAnalyzer path.

Conventions asserted here: dx/dy are *content* shifts (a camera pan to the right makes dx negative); log_scale > 0 = zoom in; ``scale_change`` is the
end/start ratio of the zoom; ``pan_extent`` the translation as a fraction of the frame width.
"""

from __future__ import annotations

import math
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from app.core.serialization import to_plain
from app.reference.motion_analyzer import MotionAnalyzer, MotionResult
from app.reference.motion_estimator import estimate_global_motion, estimate_motion
from app.reference.signals import SIGNAL_H, SIGNAL_W, FrameSampler, FrameSignals, compute_signals, signals_from_gray_frames
from app.reference.style_model import MotionStats, Shot, StyleFeatures, ZoomStats, compute_scores, motion_class
from app.tests.conftest import needs_ffmpeg

W, H = SIGNAL_W, SIGNAL_H


# ---------------------------------------------------------------------------------------------- analytic scene
class Scene:
    """A smooth random picture defined analytically, so ``frame(s, tx, ty)`` is the *exact* view after zooming by ``s`` about the centre and moving the
    content by (tx, ty) pixels."""

    def __init__(self, seed: int, n: int = 14, size: tuple[int, int] = (W, H)) -> None:
        rng = np.random.default_rng(seed)
        self.size = size
        self.f = rng.uniform(0.5, 7.0, n) / size[0]  # cycles per pixel
        self.th = rng.uniform(0, 2 * np.pi, n)
        self.ph = rng.uniform(0, 2 * np.pi, n)
        self.amp = rng.uniform(0.3, 1.0, n) / (1 + self.f * size[0] * 0.3)
        self.rng = np.random.default_rng(seed + 1000)

    def frame(self, s: float = 1.0, tx: float = 0.0, ty: float = 0.0, noise: float = 0.0) -> np.ndarray:
        w, h = self.size
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
        cx, cy = (w - 1) / 2, (h - 1) / 2
        sx, sy = cx + (xs - cx - tx) / s, cy + (ys - cy - ty) / s
        img = np.zeros((h, w))
        for fi, t, p, a in zip(self.f, self.th, self.ph, self.amp):
            img += a * np.cos(2 * np.pi * fi * (np.cos(t) * sx + np.sin(t) * sy) + p)
        img = 0.5 + 0.15 * img
        if noise:
            img = img + self.rng.normal(0, noise, img.shape)
        return np.clip(img, 0, 1).astype(np.float32)


def smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def clip_frames(scene: Scene, seconds: float, fps: float, cam, noise: float = 0.004) -> list[np.ndarray]:
    """Frames of ``scene`` at ``fps``; ``cam(t) -> (scale, dx_fraction_of_width, dy_fraction_of_height)`` is the camera at time t (content shift from t = 0)."""
    out = []
    for i in range(int(round(seconds * fps))):
        s, dx, dy = cam(i / fps)
        out.append(scene.frame(s, dx * W, dy * H, noise))
    return out


def signals_of(frames: list[np.ndarray], fps: float = 8.0) -> FrameSignals:
    return signals_from_gray_frames(frames, fps)


def shots_of(bounds: list[float], **kw) -> list[Shot]:
    return [Shot(f"S{i + 1}", a, b, **kw) for i, (a, b) in enumerate(zip(bounds, bounds[1:]))]


def analyze(frames: list[np.ndarray], shots: list[Shot] | None = None, fps: float = 8.0) -> MotionResult:
    sig = signals_of(frames, fps)
    return MotionAnalyzer().analyze(sig, shots if shots is not None else [Shot("S1", 0.0, sig.duration)])


def kinds(r: MotionResult) -> list[str]:
    return [s.kind for s in r.per_shot]


# ================================================================================================ estimator on exact synthetic frames
class TestEstimator:
    @pytest.mark.parametrize("s", [1.005, 1.012, 1.02, 1.05, 1.10])
    def test_zoom_in_is_positive_and_accurate(self, s):
        sc = Scene(3)
        e = estimate_motion(sc.frame(noise=0.004), sc.frame(s, noise=0.004))
        truth = math.log(s)
        assert e.log_scale > 0
        assert abs(e.log_scale - truth) <= max(0.0008, 0.08 * truth), (s, e)
        assert abs(e.tx) < 0.1 and abs(e.ty) < 0.1  # a centred zoom is not a translation
        assert e.confidence > 0.8

    @pytest.mark.parametrize("s", [0.98, 0.95])
    def test_zoom_out_is_negative_and_accurate(self, s):
        sc = Scene(5)
        e = estimate_motion(sc.frame(noise=0.004), sc.frame(s, noise=0.004))
        assert e.log_scale < 0
        assert abs(e.log_scale - math.log(s)) <= 0.08 * abs(math.log(s)) + 0.0005

    @pytest.mark.parametrize("tx,ty", [(0.3, 0.0), (3.0, 0.0), (-3.0, 1.0), (8.0, 2.4), (-14.0, 3.0)])
    def test_translation_sign_and_magnitude(self, tx, ty):
        sc = Scene(4)
        e = estimate_motion(sc.frame(noise=0.004), sc.frame(tx=tx, ty=ty, noise=0.004))
        assert abs(e.tx - tx) < 0.15 and abs(e.ty - ty) < 0.15, e
        assert abs(e.log_scale) < 0.003  # a shift must not leak into the zoom (the baseline gave -0.016 for a 1 px shift)

    def test_one_pixel_shift_has_no_zoom_regression(self):
        """The baseline searched zoom before translation: a plain 1 px shift came out as log_scale = -0.016 (wrong sign)."""
        sc = Scene(8)
        for tx in (1.0, -1.0):
            e = estimate_motion(sc.frame(noise=0.004), sc.frame(tx=tx, noise=0.004))
            assert abs(e.log_scale) < 0.001 and abs(e.tx - tx) < 0.05

    def test_zoom_and_translation_are_separated(self):
        """A zoom about an off-centre point is zoom + translation; both must come out right (that coupling is what broke the baseline)."""
        sc = Scene(6)
        s, tx, ty = 1.012, 2.0, 1.0
        e = estimate_motion(sc.frame(noise=0.004), sc.frame(s, tx, ty, noise=0.004))
        assert abs(e.log_scale - math.log(s)) < 0.0012
        assert abs(e.tx - tx) < 0.1 and abs(e.ty - ty) < 0.1

    def test_large_translation_uses_the_initialiser(self):
        sc = Scene(3)
        for tx, ty in ((24.0, 4.0), (-28.0, -6.0)):
            e = estimate_motion(sc.frame(noise=0.004), sc.frame(tx=tx, ty=ty, noise=0.004))
            assert abs(e.tx - tx) < 0.5 and abs(e.ty - ty) < 0.5 and e.confidence > 0.5, (tx, e)

    def test_random_picture_warped_by_known_amounts(self):
        """Independent of the analytic scene: a PIL blob picture warped with our own bilinear sampler by 1.02 and by 3 px."""
        from PIL import Image

        rng = np.random.default_rng(2)
        big = np.asarray(Image.fromarray((rng.random((9, 16)) * 255).astype(np.uint8)).resize((1152, 648), Image.Resampling.BICUBIC), dtype=np.float32) / 255

        def view(s: float = 1.0, tx: float = 0.0) -> np.ndarray:
            ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
            acc = np.zeros((H, W))
            for oy in (-1 / 3, 0, 1 / 3):
                for ox in (-1 / 3, 0, 1 / 3):  # 3x3 supersampling = area downscale of the big picture
                    sx = ((xs + ox - (W - 1) / 2 - tx) / s + (W - 1) / 2) * 3 + (1152 - W * 3) / 2
                    sy = ((ys + oy - (H - 1) / 2) / s + (H - 1) / 2) * 3 + (648 - H * 3) / 2
                    x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
                    fx, fy = sx - x0, sy - y0
                    acc += big[y0, x0] * (1 - fy) * (1 - fx) + big[y0, x0 + 1] * (1 - fy) * fx + big[y0 + 1, x0] * fy * (1 - fx) + big[y0 + 1, x0 + 1] * fy * fx
            return (acc / 9).astype(np.float32)

        a = view()
        e = estimate_motion(a, view(s=1.02))
        assert abs(e.log_scale - math.log(1.02)) < 0.002 and abs(e.tx) < 0.15
        e = estimate_motion(a, view(tx=3.0))
        assert abs(e.tx - 3.0) < 0.15 and abs(e.log_scale) < 0.002

    def test_static_pair_is_confident_and_still(self):
        sc = Scene(3)
        e = estimate_motion(sc.frame(noise=0.004), sc.frame(noise=0.004))
        assert abs(e.tx) < 0.05 and abs(e.ty) < 0.05 and abs(e.log_scale) < 0.0008
        assert e.confidence > 0.95

    def test_cut_has_low_confidence_and_reports_no_motion(self):
        a, b = Scene(3).frame(noise=0.004), Scene(9).frame(noise=0.004)
        e = estimate_motion(a, b)
        assert e.confidence < 0.1
        assert estimate_global_motion(a, b)[:3] == (0.0, 0.0, 0.0)  # never the garbage of a failed fit
        for seed in range(10, 16):  # and for other unrelated pairs
            assert estimate_motion(a, Scene(seed).frame(noise=0.004)).confidence < 0.15

    def test_flat_dark_and_odd_inputs_do_not_crash(self):
        flat = np.full((H, W), 0.4, np.float32)
        assert estimate_motion(flat, flat).confidence == 0.0
        assert estimate_global_motion(flat, flat) == (0.0, 0.0, 0.0, 0.0)
        sc = Scene(3)
        dark = (sc.frame() * 0.02).astype(np.float32)
        assert estimate_motion(dark, dark).confidence == 0.0
        assert estimate_motion(sc.frame(), np.zeros((10, 10), np.float32)).confidence == 0.0  # shape mismatch
        tiny = np.random.default_rng(0).random((8, 8)).astype(np.float32)
        assert estimate_motion(tiny, tiny).confidence == 0.0
        assert estimate_motion(np.zeros((H, W), np.float32), sc.frame()).confidence == 0.0

    def test_small_frames_are_fitted_at_their_own_size(self):
        sc = Scene(3, size=(28, 20))
        e = estimate_motion(sc.frame(noise=0.003), sc.frame(1.03, 0.7, 0.0, noise=0.003))
        assert abs(e.log_scale - math.log(1.03)) < 0.004 and abs(e.tx - 0.7) < 0.1

    def test_noise_robustness(self):
        sc = Scene(4)
        for noise in (0.01, 0.03):
            e = estimate_motion(sc.frame(noise=noise), sc.frame(1.02, 1.0, 0.0, noise=noise))
            assert abs(e.log_scale - math.log(1.02)) < 0.0035 and abs(e.tx - 1.0) < 0.2, (noise, e)

    def test_static_caption_and_a_bright_object_do_not_drag_the_fit(self):
        sc = Scene(4)

        def with_overlay(img: np.ndarray, obj_x: int | None = None) -> np.ndarray:
            img = img.copy()
            img[60:68, 20:108] = 0.95  # a static caption bar
            img[62:66, 24:104:6] = 0.1
            if obj_x is not None:
                img[20:44, obj_x:obj_x + 22] = 0.9  # a moving subject
            return img

        a = with_overlay(sc.frame(noise=0.005), 30)
        b = with_overlay(sc.frame(tx=3.0, noise=0.005), 36)
        e = estimate_motion(a, b)
        assert abs(e.tx - 3.0) < 0.3 and abs(e.log_scale) < 0.003, e

    def test_brightness_change_is_ignored(self):
        sc = Scene(4)
        a = sc.frame(noise=0.004)
        b = np.clip(sc.frame(1.02, noise=0.004) * 0.7 + 0.1, 0, 1).astype(np.float32)
        e = estimate_motion(a, b)
        assert abs(e.log_scale - math.log(1.02)) < 0.002

    def test_values_are_fractions_of_the_frame_and_size_independent(self):
        for size in ((128, 72), (192, 108)):
            sc = Scene(3, size=size)
            w, h = size
            dx, dy, ls, conf = estimate_global_motion(sc.frame(noise=0.004), sc.frame(1.01, 0.04 * w, -0.03 * h, noise=0.004))
            assert abs(dx - 0.04) < 0.002 and abs(dy + 0.03) < 0.002 and abs(ls - math.log(1.01)) < 0.0015 and conf > 0.8, (size, dx, dy, ls)

    def test_cost_per_pair_is_small(self):
        sc = Scene(3)
        a, b = sc.frame(noise=0.004), sc.frame(1.012, 2.0, 1.0, noise=0.004)
        estimate_motion(a, b)
        samples = []
        for _ in range(40):
            t0 = time.perf_counter()
            estimate_motion(a, b)
            samples.append(time.perf_counter() - t0)
        assert float(np.median(samples)) < 0.008  # budget: well under 10 ms (an 18 min video at 8 fps is ~8600 pairs)


class TestSignalsMotionBlock:
    """compute_signals feeds frames to the estimator: the per-SECOND rates must carry the right sign and size."""

    def test_zoom_rate_per_second(self):
        sc = Scene(3)
        fr = clip_frames(sc, 4, 8, lambda t: (1 + 0.10 * t / 4, 0, 0))
        sig = signals_of(fr)
        lag = sig.lag
        assert sig.motion_conf[lag:].min() > 0.8
        # a linear zoom 1 -> 1.10 in 4 s: ~ ln(1.10)/4 per second (the rate rises a little as the zoom grows)
        assert abs(float(sig.log_scale[lag:].mean()) - math.log(1.10) / 4) < 0.0025
        assert abs(float(sig.dx[lag:].mean())) < 0.002 and abs(float(sig.dy[lag:].mean())) < 0.002

    def test_zoom_out_and_pan_signs(self):
        sc = Scene(3)
        out = signals_of(clip_frames(sc, 4, 8, lambda t: (1 + 0.10 * (1 - t / 4), 0, 0)))
        assert out.log_scale[out.lag:].mean() < -0.02
        pan = signals_of(clip_frames(sc, 4, 8, lambda t: (1.0, 0.075 * t, 0.0225 * t)))  # content moves right/down: 0.075 W/s, 0.0225 H/s
        sl = slice(pan.lag, None)
        assert abs(float(pan.dx[sl].mean()) - 0.075) < 0.004 and abs(float(pan.dy[sl].mean()) - 0.0225) < 0.002
        assert abs(float(pan.log_scale[sl].mean())) < 0.003

    def test_static_is_zero_with_high_confidence_and_cut_is_low(self):
        a, b = Scene(3), Scene(9)
        fr = clip_frames(a, 2, 8, lambda t: (1, 0, 0)) + clip_frames(b, 2, 8, lambda t: (1, 0, 0))
        sig = signals_of(fr)
        lag = sig.lag
        assert np.abs(sig.log_scale[lag:16]).max() < 0.001 and sig.motion_conf[lag:16].min() > 0.9
        cut_windows = sig.motion_conf[16:16 + lag]  # windows that straddle the cut
        assert cut_windows.max() < 0.2
        assert np.all(sig.log_scale[16:16 + lag][cut_windows < 0.02] == 0.0)


# ================================================================================================ analyzer on in-memory signals
class TestAnalyzerKinds:
    def test_static(self):
        r = analyze(clip_frames(Scene(3), 6, 8, lambda t: (1, 0, 0)))
        s = r.per_shot[0]
        assert s.kind == "static" and s.intensity < 0.05 and s.scale_change == 1.0 and s.pan_extent == 0.0
        assert r.stats.motion_events_per_minute == 0 and r.stats.motion_class == "Minimal"
        assert r.stats.static_shot_share == 1.0 and r.stats.zoom.events == 0 and r.events == []

    def test_zoom_in_ten_percent_over_four_seconds(self):
        r = analyze(clip_frames(Scene(3), 4, 8, lambda t: (1 + 0.10 * t / 4, 0, 0)))
        s = r.per_shot[0]
        assert s.kind == "zoom_in"
        assert abs(s.scale_change - 1.10) < 0.02 and s.pan_extent == 0.0
        assert 0.2 <= s.intensity <= 0.35  # a gentle zoom
        zs = [e for e in r.events if e.kind in ("zoom_in", "zoom_out")]
        assert len(zs) == 1 and zs[0].kind == "zoom_in" and abs(zs[0].magnitude - 1.10) < 0.02
        assert 3.2 <= zs[0].duration <= 4.1
        z = r.stats.zoom
        assert z.events == 1 and abs(z.average_scale - 1.10) < 0.02 and z.maximum_scale == z.average_scale and z.zoom_in_share == 1.0
        assert abs(r.stats.zoom_frequency - 15.0) < 0.1 and r.stats.pan_frequency == 0.0  # one event in 4 s

    def test_zoom_out(self):
        r = analyze(clip_frames(Scene(3), 4, 8, lambda t: (1 + 0.10 * (1 - t / 4), 0, 0)))
        s = r.per_shot[0]
        assert s.kind == "zoom_out" and abs(s.scale_change - 1 / 1.10) < 0.02
        assert r.stats.zoom.events == 1 and r.stats.zoom.zoom_in_share == 0.0
        assert r.stats.zoom.average_scale > 1.05  # magnitudes are reported as ratios >= 1

    @pytest.mark.parametrize("amount,seconds", [(0.05, 4), (0.20, 4), (0.15, 3), (0.08, 6)])
    def test_zoom_magnitude_across_the_range(self, amount, seconds):
        r = analyze(clip_frames(Scene(4), seconds, 8, lambda t: (1 + amount * t / seconds, 0, 0)))
        s = r.per_shot[0]
        assert s.kind == "zoom_in" and abs(s.scale_change - (1 + amount)) <= max(0.012, 0.15 * amount), (amount, s)

    @pytest.mark.parametrize("fraction,seconds", [(0.10, 6), (0.30, 4), (0.50, 2), (0.25, 3)])
    def test_pan_extent_and_direction(self, fraction, seconds):
        r = analyze(clip_frames(Scene(3), seconds, 8, lambda t: (1.0, fraction * t / seconds, 0.0)))
        s = r.per_shot[0]
        assert s.kind == "pan" and abs(s.pan_extent - fraction) <= 0.12 * fraction + 0.01, (fraction, s)
        assert abs(s.scale_change - 1.0) < 0.02
        assert [e.kind for e in r.events] == ["pan"]
        assert r.stats.pan_frequency > 0 and r.stats.zoom_frequency == 0

    def test_vertical_pan_counts_in_widths(self):
        r = analyze(clip_frames(Scene(3), 4, 8, lambda t: (1.0, 0.0, 0.4 * t / 4)))  # 40 % of the HEIGHT = 22.5 % of the width
        assert r.per_shot[0].kind == "pan" and abs(r.per_shot[0].pan_extent - 0.4 * H / W) < 0.03

    def test_intensity_calibration(self):
        sc = Scene(3)
        gentle = analyze(clip_frames(sc, 4, 8, lambda t: (1 + 0.08 * t / 4, 0, 0))).per_shot[0].intensity
        fast_pan = analyze(clip_frames(sc, 2, 8, lambda t: (1.0, 0.5 * t / 2, 0.0))).per_shot[0].intensity
        punch = analyze(clip_frames(sc, 1.5, 8, lambda t: (1 + 0.15 * smoothstep(t / 1.5), 0, 0))).per_shot[0].intensity
        assert 0.2 <= gentle <= 0.35
        assert 0.5 <= fast_pan <= 0.7
        assert 0.5 <= punch <= 0.75
        assert gentle < punch and gentle < fast_pan

    def test_a_very_short_fast_zoom_shot_is_still_a_zoom(self):
        """A 1.5 s shot has only a few windows (the first half second of any shot is skipped); the zoom must still be found and not be mistaken for chaos."""
        r = analyze(clip_frames(Scene(3), 1.5, 8, lambda t: (1 + 0.15 * smoothstep(t / 1.5), 0, 0)))
        s = r.per_shot[0]
        assert s.kind == "zoom_in" and abs(s.scale_change - 1.15) < 0.04 and s.confidence > 0.5
        assert [e.kind for e in r.events] == ["zoom_in"]

    def test_punch_in_inside_a_longer_shot_is_one_event(self):
        r = analyze(clip_frames(Scene(3), 10, 8, lambda t: (1 + 0.15 * smoothstep((t - 4.0) / 0.8), 0, 0)))
        s = r.per_shot[0]
        assert s.kind == "zoom_in" and abs(s.scale_change - 1.15) < 0.03
        zs = [e for e in r.events if e.kind.startswith("zoom")]
        assert len(zs) == 1 and abs(zs[0].magnitude - 1.15) < 0.03
        assert 0.5 <= zs[0].duration <= 1.3 and 3.3 <= zs[0].start and zs[0].end <= 5.7  # where it happened, to within the measurement window
        assert r.stats.zoom.context["punch_in"] == 1.0 and r.stats.zoom.context["slow_zoom"] == 0.0
        assert r.stats.average_motion_intensity < 0.15  # a short punch-in in a long shot does not make the shot busy

    def test_slow_zoom_inside_a_shot_is_one_event(self):
        r = analyze(clip_frames(Scene(3), 10, 8, lambda t: (1 + 0.08 * min(1.0, max(0.0, (t - 2.0) / 6.0)), 0, 0)))
        zs = [e for e in r.events if e.kind.startswith("zoom")]
        assert len(zs) == 1 and abs(zs[0].magnitude - 1.08) < 0.02 and zs[0].duration > 4.5
        assert r.stats.zoom.context["slow_zoom"] == 1.0 and r.per_shot[0].kind == "zoom_in"

    def test_zoom_in_then_out_is_two_events_and_mixed(self):
        def cam(t: float):
            return (1 + 0.10 * smoothstep(t / 2.5) - 0.10 * smoothstep((t - 4.5) / 2.5), 0, 0)

        r = analyze(clip_frames(Scene(3), 8, 8, cam))
        zs = [e for e in r.events if e.kind.startswith("zoom")]
        assert sorted(e.kind for e in zs) == ["zoom_in", "zoom_out"]
        assert r.per_shot[0].kind == "mixed" and abs(r.per_shot[0].scale_change - 1.0) < 0.03  # net end/start ratio is ~1
        assert r.stats.zoom.events == 2 and r.stats.zoom.zoom_in_share == 0.5

    def test_zoom_and_later_pan_in_one_shot_is_mixed(self):
        def cam(t: float):
            return (1 + 0.10 * smoothstep(t / 2.5), 0.3 * smoothstep((t - 4.0) / 3.0), 0.0)

        # zoom during 0-2.5 s, pan during 4-7 s: in the pan the scale is constant, so the relative translation is exactly the pan
        r = analyze(clip_frames(Scene(3), 8, 8, cam))
        assert sorted(e.kind for e in r.events) == ["pan", "zoom_in"]
        assert r.per_shot[0].kind == "mixed" and r.per_shot[0].pan_extent > 0.2 and r.per_shot[0].scale_change > 1.07

    def test_off_centre_zoom_is_a_zoom_not_a_pan(self):
        """A digital punch-in about a point away from the centre also shifts the picture, in proportion to the zoom."""
        px, py = -40.0, -20.0  # the anchor, relative to the centre (pixels)
        r = analyze(clip_frames(Scene(3), 4, 8, lambda t: (s := 1 + 0.10 * t / 4, (1 - s) * px / W, (1 - s) * py / H)))
        s = r.per_shot[0]
        assert s.kind == "zoom_in" and s.pan_extent == 0.0 and abs(s.scale_change - 1.10) < 0.02

    @pytest.mark.parametrize("seed", [5, 6, 7])
    def test_shake_is_high_motion(self, seed):
        rng = np.random.default_rng(seed)
        sc = Scene(3)
        fr = [sc.frame(1.0, float(rng.uniform(-0.3, 0.3) * W), float(rng.uniform(-0.25, 0.25) * H), 0.004) for _ in range(48)]  # a new random position every sample
        r = analyze(fr)
        s = r.per_shot[0]
        assert s.kind == "high_motion" and s.intensity > 0.75
        assert r.stats.motion_class in ("Strong", "Aggressive") and r.stats.high_motion_share > 0.9
        assert r.stats.zoom.events == 0 and r.stats.pan_frequency == 0  # the aliased 0.5 s camera fit must not invent zooms or pans
        assert all(e.kind == "high_motion" for e in r.events) and s.confidence >= 0.6

    def test_a_shake_burst_inside_a_calm_shot_is_one_high_motion_event(self):
        rng = np.random.default_rng(3)
        sc = Scene(3)
        fr = []
        for i in range(96):  # 12 s: still, 3 s of shake in the middle, still
            if 36 <= i < 60:
                fr.append(sc.frame(1.0, float(rng.uniform(-0.3, 0.3) * W), float(rng.uniform(-0.25, 0.25) * H), 0.004))
            else:
                fr.append(sc.frame(noise=0.004))
        r = analyze(fr)
        hm = [e for e in r.events if e.kind == "high_motion"]
        assert len(hm) == 1 and 3.5 <= hm[0].start <= 5.0 and 6.5 <= hm[0].end <= 8.0
        assert [e for e in r.events if e.kind != "high_motion"] == []
        assert r.per_shot[0].kind in ("mixed", "high_motion") and 0.1 < r.per_shot[0].intensity < 0.45  # busy for a quarter of the shot only

    def test_busy_but_not_chaotic_motion_is_not_called_static(self):
        """A moving subject over a still camera: not a zoom or a pan, not static, not shake either."""
        sc = Scene(3)
        fr = []
        for i in range(64):
            f = sc.frame(noise=0.004)
            x = int(60 + 40 * math.sin(i / 8 * 2 * math.pi * 0.5))  # a big bright block sweeping around
            f[10:50, x - 15:x + 15] = 0.95
            fr.append(f)
        r = analyze(fr)
        s = r.per_shot[0]
        assert s.kind in ("mixed", "high_motion") and s.scale_change < 1.02 and s.pan_extent < 0.05 and [e for e in r.events if e.kind != "high_motion"] == []


class TestAnalyzerShotsAndCuts:
    def test_multi_shot_kinds_and_zoom_frequency(self):
        a, b, c, d, e = (Scene(s) for s in (21, 22, 23, 24, 25))
        fr = (clip_frames(a, 3, 8, lambda t: (1, 0, 0)) + clip_frames(b, 4, 8, lambda t: (1 + 0.10 * t / 4, 0, 0)) + clip_frames(c, 3, 8, lambda t: (1, 0, 0))
              + clip_frames(d, 4, 8, lambda t: (1, -0.3 * t / 4, 0)) + clip_frames(e, 3, 8, lambda t: (1 + 0.08 * (1 - t / 3), 0, 0)))
        r = analyze(fr, shots_of([0, 3, 7, 10, 14, 17]))
        assert kinds(r) == ["static", "zoom_in", "static", "pan", "zoom_out"]
        assert [round(s.scale_change, 2) for s in r.per_shot][1] == pytest.approx(1.10, abs=0.02)
        assert r.per_shot[3].pan_extent == pytest.approx(0.30, abs=0.03)
        st = r.stats
        assert st.zoom_frequency == pytest.approx(2 / (17 / 60), rel=0.01) and st.pan_frequency == pytest.approx(1 / (17 / 60), rel=0.01)
        assert st.motion_events_per_minute == pytest.approx(3 / (17 / 60), rel=0.01)
        assert st.static_shot_share == pytest.approx(6 / 17, abs=0.01) and st.high_motion_share == 0.0
        z = st.zoom
        assert z.events == 2 and z.zoom_in_share == 0.5
        assert z.average_scale == pytest.approx(1.10, abs=0.02) and z.maximum_scale == z.average_scale  # zoom-in events only
        assert 2.5 <= z.average_duration <= 4.1 and z.context["static_shots"] == pytest.approx(0.4)
        assert r.confidence > 0.8 and r.notes == []

    def test_a_cut_never_creates_a_fake_zoom_or_pan(self):
        fr = []
        for seed, secs in ((31, 3), (32, 4), (33, 3), (34, 5)):
            fr += clip_frames(Scene(seed), secs, 8, lambda t: (1, 0, 0))
        for shots in (shots_of([0, 3, 7, 10, 15]), None):  # with the right shots, and with one shot that contains all the cuts
            r = analyze(fr, shots)
            assert all(s.kind == "static" and s.intensity < 0.05 and s.scale_change == 1.0 and s.pan_extent == 0 for s in r.per_shot), r.per_shot
            assert r.stats.motion_events_per_minute == 0 and r.events == [] and r.stats.motion_class == "Minimal"

    def test_zoom_that_ends_at_a_cut_stays_in_its_shot(self):
        a, b = Scene(41), Scene(42)
        fr = clip_frames(a, 4, 8, lambda t: (1 + 0.10 * t / 4, 0, 0)) + clip_frames(b, 4, 8, lambda t: (1, 0, 0))
        r = analyze(fr, shots_of([0, 4, 8]))
        assert kinds(r) == ["zoom_in", "static"] and r.per_shot[1].scale_change == 1.0 and r.stats.zoom.events == 1

    def test_dissolve_between_static_shots_is_not_motion(self):
        a, b = Scene(51), Scene(52)
        fa, fb = clip_frames(a, 5, 8, lambda t: (1, 0, 0)), clip_frames(b, 5, 8, lambda t: (1, 0, 0))
        fr = fa[:32] + [(1 - k / 8) * fa[32 + k] + (k / 8) * fb[k] for k in range(8)] + fb[8:]  # 1 s blend starting at 4.0 s
        guarded = shots_of([0, 4.0], ) + [Shot("S2", 4.0, 9.0, transition_type="DISSOLVE", transition_duration=1.0)]
        r = analyze(fr, guarded)
        assert kinds(r) == ["static", "static"] and r.per_shot[1].intensity < 0.02 and r.events == []
        # even if the detector gave no transition information the blend is too gentle to look like a zoom or a pan
        r2 = analyze(fr, shots_of([0, 4, 9]))
        assert all(k in ("static", "mixed") for k in kinds(r2)) and [e for e in r2.events if e.kind in ("zoom_in", "zoom_out", "pan")] == []

    def test_unsorted_overlong_and_empty_shot_lists(self):
        fr = clip_frames(Scene(3), 6, 8, lambda t: (1 + 0.10 * t / 6, 0, 0))
        sig = signals_of(fr)
        ref = MotionAnalyzer().analyze(sig, [Shot("S1", 0.0, sig.duration)])
        assert MotionAnalyzer().analyze(sig, []).per_shot[0].kind == ref.per_shot[0].kind == "zoom_in"  # no shots -> the whole video is one shot
        r = MotionAnalyzer().analyze(sig, [Shot("B", 3.0, 99.0), Shot("A", 0.0, 3.0), Shot("Z", 5.0, 5.0)])  # unsorted, overlong, zero-length
        assert [s.shot_id for s in r.per_shot] == ["A", "B"] and all(0.0 <= s.intensity <= 1.0 for s in r.per_shot)


class TestAnalyzerRobustness:
    def test_noise(self):
        r = analyze(clip_frames(Scene(3), 4, 8, lambda t: (1 + 0.10 * t / 4, 0, 0), noise=0.02))
        assert r.per_shot[0].kind == "zoom_in" and abs(r.per_shot[0].scale_change - 1.10) < 0.03
        s = analyze(clip_frames(Scene(3), 6, 8, lambda t: (1, 0, 0), noise=0.02)).per_shot[0]
        assert s.kind == "static" and s.intensity < 0.1

    @pytest.mark.parametrize("fps", [4.0, 12.0])
    def test_other_sample_rates(self, fps):
        r = analyze(clip_frames(Scene(3), 5, fps, lambda t: (1 + 0.10 * t / 5, 0, 0)), fps=fps)
        assert r.per_shot[0].kind == "zoom_in" and abs(r.per_shot[0].scale_change - 1.10) < 0.03 and 0.15 <= r.per_shot[0].intensity <= 0.4
        p = analyze(clip_frames(Scene(3), 4, fps, lambda t: (1, 0.3 * t / 4, 0)), fps=fps)
        assert p.per_shot[0].kind == "pan" and abs(p.per_shot[0].pan_extent - 0.3) < 0.04

    def test_flat_frames_report_no_confidence(self):
        flat = [np.full((H, W), 0.4, np.float32)] * 40
        r = analyze(flat)
        assert r.per_shot[0].kind == "static" and r.per_shot[0].confidence == 0.0 and r.confidence == 0.0 and r.stats.motion_events_per_minute == 0
        assert any("flat" in n.lower() or "dark" in n.lower() for n in r.notes)

    def test_dark_low_contrast_material_has_low_confidence(self):
        sc = Scene(3)
        dark = [(0.02 + 0.03 * (f - 0.5)).astype(np.float32) for f in clip_frames(sc, 20, 8, lambda t: (1 + 0.1 * t / 20, 0, 0))]
        r = analyze(dark)
        assert r.confidence < 0.2

    def test_very_short_material_has_low_confidence_and_a_note(self):
        r = analyze(clip_frames(Scene(3), 2, 8, lambda t: (1 + 0.05 * t / 2, 0, 0)))
        assert r.confidence <= 0.2 and any("short" in n.lower() for n in r.notes)

    def test_long_clean_material_has_high_confidence(self):
        r = analyze(clip_frames(Scene(3), 20, 8, lambda t: (1, 0, 0)))
        assert r.confidence > 0.9 and r.notes == []

    def test_tiny_inputs(self):
        for n in (1, 2, 3, 5):
            sig = signals_of([Scene(3).frame()] * n)
            r = MotionAnalyzer().analyze(sig, [Shot("S1", 0.0, sig.duration)])
            assert isinstance(r, MotionResult) and r.stats.motion_events_per_minute == 0 and r.confidence <= 0.2
        sig = signals_of([Scene(3).frame()])
        assert MotionAnalyzer().analyze(sig, []).per_shot[0].kind == "static"

    def test_shot_shorter_than_the_lag_is_measured_from_the_frame_difference_only(self):
        sc = Scene(3)
        fr = clip_frames(sc, 3, 8, lambda t: (1, 0, 0)) + clip_frames(Scene(8), 0.5, 8, lambda t: (1, 0, 0)) + clip_frames(sc, 3, 8, lambda t: (1, 0, 0))
        r = analyze(fr, shots_of([0, 3, 3.5, 6.5]))
        short = r.per_shot[1]
        assert short.kind == "static" and short.confidence <= 0.2 and short.scale_change == 1.0

    def test_garbage_inputs_never_crash(self):
        """Random arrays (including NaN/inf) and random, overlapping, unsorted, out-of-range shots: always a well-formed result."""
        rng = np.random.default_rng(0)
        for trial in range(25):
            n = int(rng.integers(2, 200))
            fps = float(rng.choice([2.0, 4.0, 8.0, 15.0]))
            lag = max(1, int(round(0.5 * fps)))

            def arr(scale_: float) -> np.ndarray:
                a = rng.normal(0, scale_, n).astype(np.float32)
                a[rng.random(n) < 0.05] = np.nan
                a[rng.random(n) < 0.02] = np.inf
                return a

            sig = FrameSignals(fps, np.arange(n) / fps, rng.random(n).astype(np.float32), (rng.random(n) * 0.3).astype(np.float32), np.abs(arr(0.1)), np.zeros(n, np.float32),
                               np.zeros(n, np.float32), np.zeros((n, 9, 16), np.float32), lag, arr(0.2), arr(0.2), arr(0.05), rng.random(n).astype(np.float32), n / fps)
            shots = [Shot(f"S{i}", float(rng.uniform(-1, n / fps + 2)), float(rng.uniform(-1, n / fps + 2))) for i in range(int(rng.integers(0, 8)))]
            r = MotionAnalyzer().analyze(sig, shots)
            assert 0.0 <= r.confidence <= 1.0 and math.isfinite(r.stats.average_motion_intensity) and 0.0 <= r.stats.average_motion_intensity <= 1.0
            assert all(0.0 <= s.intensity <= 1.0 and 0.0 <= s.confidence <= 1.0 and math.isfinite(s.scale_change) and s.scale_change > 0 and s.pan_extent >= 0 for s in r.per_shot)
            assert all(math.isfinite(v) for _, v in r.motion_series) and r.stats.motion_class in ("Minimal", "Subtle", "Moderate", "Strong", "Aggressive")
            assert math.isfinite(r.stats.motion_events_per_minute) and 0.0 <= r.stats.static_shot_share <= 1.0 + 1e-9

    def test_deterministic(self):
        fr = clip_frames(Scene(3), 6, 8, lambda t: (1 + 0.1 * t / 6, 0.2 * t / 6, 0))
        sig = signals_of(fr)
        shots = [Shot("S1", 0.0, sig.duration)]
        assert MotionAnalyzer().analyze(sig, shots) == MotionAnalyzer().analyze(sig, shots)

    def test_long_video_is_fast(self):
        """An 18 minute reference at 8 fps is ~8600 samples; the analysis of the arrays must stay well under a few seconds."""
        n = 8640
        rng = np.random.default_rng(0)
        times = np.arange(n) / 8.0
        sig = FrameSignals(8.0, times, np.full(n, 0.4, np.float32), np.full(n, 0.2, np.float32), (rng.random(n) * 0.01).astype(np.float32), np.zeros(n, np.float32),
                           np.zeros(n, np.float32), np.zeros((n, 9, 16), np.float32), 4, (rng.normal(0, 0.002, n)).astype(np.float32), np.zeros(n, np.float32),
                           (rng.normal(0, 0.0005, n)).astype(np.float32), np.ones(n, np.float32), n / 8.0)
        shots = [Shot(f"S{i}", i * 4.0, (i + 1) * 4.0) for i in range(n // 32)]
        t0 = time.perf_counter()
        r = MotionAnalyzer().analyze(sig, shots)
        assert time.perf_counter() - t0 < 3.0
        assert len(r.per_shot) == len(shots) and r.stats.motion_events_per_minute < 1.0


class TestStatsContract:
    def _result(self) -> MotionResult:
        a, b = Scene(61), Scene(62)
        fr = clip_frames(a, 5, 8, lambda t: (1, 0, 0)) + clip_frames(b, 5, 8, lambda t: (1 + 0.12 * t / 5, 0, 0))
        return analyze(fr, shots_of([0, 5, 10]))

    def test_motion_class_uses_the_same_formula_as_the_style_model(self):
        r = self._result()
        st = r.stats
        scores = compute_scores(StyleFeatures(motion_events_per_minute=st.motion_events_per_minute, average_motion_intensity=st.average_motion_intensity))
        assert st.motion_class == motion_class(scores.motion_intensity)

    def test_stats_are_aggregates_only(self):
        st = self._result().stats
        assert isinstance(st, MotionStats) and isinstance(st.zoom, ZoomStats)
        plain = to_plain(st)

        def walk(o):
            if isinstance(o, dict):
                for v in o.values():
                    yield from walk(v)
            else:
                yield o

        assert not any(isinstance(v, (list, tuple)) for v in walk(plain))  # no per-event / per-time lists
        assert set(st.zoom.context) <= {"static_shots", "punch_in", "slow_zoom"}
        assert all(0.0 <= v <= 1.0 for v in st.zoom.context.values())
        assert 0.0 <= st.average_motion_intensity <= 1.0 and 0.0 <= st.static_shot_share <= 1.0 and 0.0 <= st.high_motion_share <= 1.0

    def test_series_and_per_shot_values(self):
        r = self._result()
        assert 8 <= len(r.motion_series) <= 12  # about one value a second
        times = [t for t, _ in r.motion_series]
        assert times == sorted(times) and all(0.0 <= v <= 1.0 for _, v in r.motion_series)
        first = np.mean([v for t, v in r.motion_series if t < 5])
        second = np.mean([v for t, v in r.motion_series if t >= 5])
        assert first < 0.05 < 0.15 < second
        for s in r.per_shot:
            assert 0.0 <= s.intensity <= 1.0 and 0.0 <= s.confidence <= 1.0 and s.duration == 5.0 and s.kind in ("static", "zoom_in", "zoom_out", "pan", "high_motion", "mixed")
        assert 0.0 <= r.confidence <= 1.0

    def test_average_intensity_is_duration_weighted(self):
        r = self._result()
        total = sum(s.duration for s in r.per_shot)
        assert r.stats.average_motion_intensity == pytest.approx(sum(s.intensity * s.duration for s in r.per_shot) / total, abs=1e-9)


# ================================================================================================ through FFmpeg
def _ffmpeg(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", *args], check=True)


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    """Videos with known camera work, built once per module. ``get(name)`` returns the path."""
    from app.tests import reference_helpers as rh

    root = tmp_path_factory.mktemp("motion_media")
    made: dict[str, Path] = {}

    def centred(name: str, secs: float, zexpr: str, seed: int, noise: float = 2.0) -> Path:
        """A zoom truly about the frame centre (the helper's ``zoom`` evaluates the crop offset once, so it zooms about the top-left instead)."""
        w, h = rh.SIZE
        big = rh.blob_image(root / f"_{name}_big.png", seed, (w * 3, h * 3))
        vf = (f"scale=w='{w * 3}*({zexpr})':h='{h * 3}*({zexpr})':eval=frame,crop={w}:{h}:x='({w * 3}*({zexpr})-{w})/2':y='({h * 3}*({zexpr})-{h})/2',"
              f"noise=alls={noise}:allf=t,format=yuv420p")
        _ffmpeg(["-loop", "1", "-framerate", "24", "-t", str(secs), "-i", str(big), "-vf", vf, "-r", "24", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "16",
                 "-pix_fmt", "yuv420p", str(root / f"{name}.mp4")])
        return root / f"{name}.mp4"

    def concat(name: str, parts: list[Path]) -> Path:
        ins = [x for p in parts for x in ("-i", str(p))]
        fc = "".join(f"[{i}:v]settb=1/24,fps=24,format=yuv420p[v{i}];" for i in range(len(parts))) + "".join(f"[v{i}]" for i in range(len(parts))) + f"concat=n={len(parts)}:v=1:a=0[v]"
        _ffmpeg([*ins, "-filter_complex", fc, "-map", "[v]", "-r", "24", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "16", "-pix_fmt", "yuv420p", str(root / f"{name}.mp4")])
        return root / f"{name}.mp4"

    class Media:
        def get(self, name: str) -> Path:
            if name not in made:
                made[name] = self._build(name)
            return made[name]

        def _build(self, name: str) -> Path:
            p = root / f"{name}.mp4"
            if name == "static":
                return rh.camera_video(p, 4, "static")
            if name == "zoom":
                return rh.camera_video(p, 4, "zoom", 0.10)
            if name == "zoom_out":
                return rh.camera_video(p, 4, "zoom_out", 0.10)
            if name == "pan":
                return rh.camera_video(p, 4, "pan", 0.10)  # the window slides 3 * amount of the view width = 30 %
            if name == "pan_fast":
                return rh.camera_video(p, 2, "pan", 0.15)  # 45 % of the width in 2 s
            if name == "shake":
                return rh.camera_video(p, 6, "shake")
            if name == "centred_zoom":
                return centred(name, 4, "1+0.1*t/4", 22)
            if name == "punch":
                return centred(name, 8, "1+0.15*min(1,max(0,(t-3.5)/0.8))", 26)
            if name == "multi":
                return concat(name, [rh.camera_video(root / "m1.mp4", 3, "static", seed=21), centred("m2", 4, "1+0.1*t/4", 22), rh.camera_video(root / "m3.mp4", 3, "static", seed=23),
                                     rh.camera_video(root / "m4.mp4", 4, "pan", 0.1, seed=24), centred("m5", 3, "1+0.08*(1-t/3)", 25)])
            if name == "cuts":
                return rh.shots_video(p, [3, 4, 3, 5])
            if name == "dissolves":
                return rh.shots_video(p, [4, 4, 4], ["dissolve", "fade"], trans_dur=1.0)
            raise KeyError(name)

    return Media()


_SIGNALS: dict[tuple[str, float], FrameSignals] = {}


def video_signals(media, name: str, fps: float = 8.0) -> FrameSignals:
    from app.rendering.ffmpeg_service import FFmpegService

    key = (name, fps)
    if key not in _SIGNALS:
        _SIGNALS[key] = compute_signals(FrameSampler(FFmpegService()).frames(media.get(name), fps, SIGNAL_W, SIGNAL_H), fps)
    return _SIGNALS[key]


def video_result(media, name: str, shots: list[Shot] | None = None, fps: float = 8.0) -> MotionResult:
    sig = video_signals(media, name, fps)
    return MotionAnalyzer().analyze(sig, shots if shots is not None else [Shot("S1", 0.0, sig.duration)])


@needs_ffmpeg
class TestVideos:
    def test_zoom_rate_per_second_matches_the_encoded_zoom(self, media):
        """The known problem: on camera_video(4 s, zoom, 0.10) the baseline reported ~ -0.010 per second; the truth is ln(1.10) / 4 = +0.0238."""
        sig = video_signals(media, "zoom")
        rate = float(sig.log_scale[sig.lag:].mean())
        assert abs(rate - math.log(1.10) / 4) < 0.25 * math.log(1.10) / 4, rate
        assert sig.motion_conf[sig.lag:].min() > 0.8
        centred = video_signals(media, "centred_zoom")
        assert abs(float(centred.log_scale[centred.lag:].mean()) - math.log(1.10) / 4) < 0.25 * math.log(1.10) / 4
        assert abs(float(centred.dx[centred.lag:].mean())) < 0.005 and abs(float(centred.dy[centred.lag:].mean())) < 0.005  # centred: no drift

    def test_pan_direction_and_speed(self, media):
        sig = video_signals(media, "pan")
        # the crop window moves right 30 % of the width in 4 s, so the content moves left at 0.075 widths per second
        assert abs(float(sig.dx[sig.lag:].mean()) + 0.075) < 0.006 and abs(float(sig.dy[sig.lag:].mean())) < 0.003 and abs(float(sig.log_scale[sig.lag:].mean())) < 0.004

    def test_static_video(self, media):
        r = video_result(media, "static")
        assert r.per_shot[0].kind == "static" and r.per_shot[0].intensity < 0.05
        assert r.stats.motion_class == "Minimal" and r.stats.motion_events_per_minute == 0 and r.stats.zoom.events == 0

    def test_zoom_ten_percent_over_four_seconds(self, media):
        for name in ("zoom", "centred_zoom"):
            r = video_result(media, name)
            s = r.per_shot[0]
            assert s.kind == "zoom_in" and abs(s.scale_change - 1.10) < 0.03 and s.pan_extent == 0.0, (name, s)
            assert 0.2 <= s.intensity <= 0.35
            zoom_events = [e for e in r.events if e.kind != "high_motion"]
            assert [e.kind for e in zoom_events] == ["zoom_in"], name
            assert r.stats.zoom.events == 1 and abs(r.stats.zoom.average_scale - 1.10) < 0.03

    def test_zoom_out(self, media):
        r = video_result(media, "zoom_out")
        s = r.per_shot[0]
        assert s.kind == "zoom_out" and abs(s.scale_change - 1 / 1.10) < 0.03
        assert r.stats.zoom.events == 1 and r.stats.zoom.zoom_in_share == 0.0

    def test_pan_thirty_percent_of_the_width(self, media):
        r = video_result(media, "pan")
        s = r.per_shot[0]
        assert s.kind == "pan" and abs(s.pan_extent - 0.30) < 0.04 and abs(s.scale_change - 1.0) < 0.02
        assert 0.2 <= s.intensity <= 0.45 and r.stats.pan_frequency > 0 and r.stats.zoom_frequency == 0

    def test_fast_pan_intensity(self, media):
        s = video_result(media, "pan_fast").per_shot[0]
        assert s.kind == "pan" and abs(s.pan_extent - 0.45) < 0.05 and 0.5 <= s.intensity <= 0.7

    def test_shake_is_high_motion(self, media):
        r = video_result(media, "shake")
        s = r.per_shot[0]
        assert s.kind == "high_motion" and s.intensity > 0.7
        assert r.stats.motion_class in ("Strong", "Aggressive") and r.stats.zoom.events == 0 and r.stats.pan_frequency == 0

    def test_punch_in_in_the_middle_of_a_long_shot(self, media):
        r = video_result(media, "punch")
        s = r.per_shot[0]
        zs = [e for e in r.events if e.kind != "high_motion"]
        assert len(zs) == 1 and zs[0].kind == "zoom_in" and abs(zs[0].magnitude - 1.15) < 0.03
        assert s.kind == "zoom_in" and abs(s.scale_change - 1.15) < 0.03 and 0.5 <= zs[0].duration <= 1.3 and 3.0 <= zs[0].start and zs[0].end <= 5.3

    def test_multi_shot_video(self, media):
        r = video_result(media, "multi", shots_of([0, 3, 7, 10, 14, 17]))
        assert kinds(r) == ["static", "zoom_in", "static", "pan", "zoom_out"]
        assert r.per_shot[1].scale_change == pytest.approx(1.10, abs=0.03) and r.per_shot[3].pan_extent == pytest.approx(0.30, abs=0.04) and r.per_shot[4].scale_change < 0.95
        st = r.stats
        assert st.zoom_frequency == pytest.approx(2 / (17 / 60), rel=0.01) and st.pan_frequency == pytest.approx(1 / (17 / 60), rel=0.01)
        assert st.static_shot_share == pytest.approx(6 / 17, abs=0.01) and st.zoom.events == 2 and st.zoom.average_scale == pytest.approx(1.10, abs=0.03)
        assert st.motion_class in ("Subtle", "Moderate") and r.confidence > 0.8

    def test_hard_cuts_do_not_look_like_motion(self, media):
        for shots in (shots_of([0, 3, 7, 10, 15]), None):
            r = video_result(media, "cuts", shots)
            assert all(s.kind == "static" and s.scale_change == 1.0 and s.pan_extent == 0 for s in r.per_shot)
            assert r.stats.motion_events_per_minute == 0 and r.stats.motion_class == "Minimal"
        sig = video_signals(media, "cuts")
        cut_samples = [int(round(t * 8)) for t in (3, 7, 10)]
        for c in cut_samples:  # the windows that straddle a cut have no confidence, so they can be ignored
            assert sig.motion_conf[c:c + sig.lag].max() < 0.3

    def test_dissolves_and_fades_are_not_motion(self, media):
        shots = [Shot("S1", 0.0, 4.0), Shot("S2", 4.0, 8.0, transition_type="DISSOLVE", transition_duration=1.0), Shot("S3", 8.0, 12.0, transition_type="FADE", transition_duration=1.0)]
        r = video_result(media, "dissolves", shots)
        assert kinds(r) == ["static"] * 3 and max(s.intensity for s in r.per_shot) < 0.02 and r.events == []

    @pytest.mark.parametrize("fps", [4.0, 12.0])
    def test_other_sample_rates(self, media, fps):
        r = video_result(media, "centred_zoom", fps=fps)
        assert r.per_shot[0].kind == "zoom_in" and abs(r.per_shot[0].scale_change - 1.10) < 0.03
