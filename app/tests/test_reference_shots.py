"""Phase 7 shot / transition detection tests.

Ground truth comes from synthetic media with *known* editing: ``reference_helpers.shots_video`` / ``camera_video`` through the real
``FrameSampler`` + ``compute_signals`` (8 fps, 128x72 gray), and pure-numpy frame lists through ``signals_from_gray_frames`` for the cases
FFmpeg cannot make precisely (single flash frames, flicker, overlays, lighting drift, per-frame noise...). Statistics are checked against
numbers worked out by hand.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from app.core.serialization import to_plain
from app.reference.shot_detector import ShotDetection, ShotDetector, compute_shot_stats, compute_transition_stats
from app.reference.signals import SIG_H, SIG_W, FrameSampler, FrameSignals, compute_signals, signals_from_gray_frames
from app.reference.style_model import TRANSITION_TYPES, Section, Shot
from app.rendering.ffmpeg_service import FFmpegService
from app.tests.conftest import needs_ffmpeg
from app.tests.reference_helpers import camera_video, shots_video

FPS = 8
H, W = 72, 128


# ---------------------------------------------------------------------------------------------- helpers
def sample(path, fps: int = FPS) -> FrameSignals:
    return compute_signals(FrameSampler(FFmpegService()).frames(path, fps, W, H), fps)


def picture(seed: int, contrast: float = 1.0, base: float = 0.5) -> np.ndarray:
    """A smooth random 'photo' (distinct per seed), float 0..1, 72x128."""
    small = np.random.default_rng(seed).random((9, 16))
    a = np.asarray(Image.fromarray((small * 255).astype(np.uint8)).resize((W, H), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
    return np.clip(base + (a - 0.5) * contrast, 0, 1).astype(np.float32)


def hold(pic: np.ndarray, seconds: float, fps: int = FPS) -> list[np.ndarray]:
    return [pic.copy() for _ in range(int(round(seconds * fps)))]


def signals_of(frames: list[np.ndarray], fps: int = FPS, motion: bool = False) -> FrameSignals:
    """Signals from in-memory frames. The (slow) global-motion estimate is skipped unless a test needs it."""
    return signals_from_gray_frames(frames, fps, **({} if motion else {"lag_seconds": 1e4}))


def detect(frames: list[np.ndarray], fps: int = FPS, sensitivity: float = 1.0, **kw) -> ShotDetection:
    return ShotDetector(sensitivity).detect(signals_of(frames, fps), **kw)


def noisy(frames: list[np.ndarray], sigma: float, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [np.clip(f + rng.normal(0, sigma, f.shape), 0, 1).astype(np.float32) for f in frames]


def assert_tiles(det: ShotDetection, duration: float) -> None:
    """Shots tile [0, duration] with no gap or overlap and carry well-formed ids, samples and confidences."""
    shots = det.shots
    assert shots, "no shots"
    assert shots[0].start == 0.0 and shots[0].transition_type == "CUT" and shots[0].transition_duration == 0.0
    assert shots[-1].end == pytest.approx(duration, abs=1e-3)
    for i, s in enumerate(shots):
        assert s.shot_id == f"shot_{i + 1:04d}"
        assert s.end > s.start
        assert s.start <= s.frame_sample < s.end
        assert 0.0 <= s.confidence <= 1.0
        assert s.transition_type in TRANSITION_TYPES
        if i:
            assert s.start == shots[i - 1].end
            assert (s.transition_duration == 0.0) == (s.transition_type == "CUT")
    assert 0.0 <= det.confidence <= 1.0


def assert_cuts(det: ShotDetection, expected: list[float], tol: float = 0.25) -> None:
    assert len(det.cut_times) == len(expected), f"found {det.cut_times}, expected {expected}"
    for got, want in zip(det.cut_times, expected):
        assert abs(got - want) <= tol, f"cut at {got}, expected {want}"


def mk_shots(durations: list[float], kinds: dict[int, tuple[str, float]] | None = None) -> list[Shot]:
    """Shots with the given lengths; ``kinds[i] = (type, duration)`` says how shot i was entered."""
    out, t = [], 0.0
    for i, d in enumerate(durations):
        kind, td = (kinds or {}).get(i, ("CUT", 0.0))
        out.append(Shot(f"shot_{i + 1:04d}", t, t + d, t + d / 2, kind, td, 1.0))
        t += d
    return out


# ---------------------------------------------------------------------------------------------- real media (FFmpeg)
@pytest.fixture(scope="module")
def media(tmp_path_factory):
    return tmp_path_factory.mktemp("shots_media")


@pytest.fixture(scope="module")
def cuts_clip(media):
    sig = sample(shots_video(media / "cuts.mp4", [3, 4.5, 2.5, 5, 3.5]))
    return sig, ShotDetector().detect(sig)


@pytest.fixture(scope="module")
def transitions_clip(media):
    sig = sample(shots_video(media / "trans.mp4", [3] * 6, ["cut", "dissolve", "fadeblack", "wipe", "slide"]))
    return sig, ShotDetector().detect(sig)


@needs_ffmpeg
class TestRealMedia:
    def test_hard_cuts_at_known_times(self, cuts_clip):
        sig, det = cuts_clip
        assert_cuts(det, [3.0, 7.5, 10.0, 15.0], tol=0.25)
        assert len(det.shots) == 5
        assert all(s.transition_type == "CUT" and s.transition_duration == 0 for s in det.shots)
        assert det.confidence >= 0.85 and all(s.confidence >= 0.8 for s in det.shots)
        assert det.major_change_times == []
        assert_tiles(det, sig.duration)

    def test_cut_stats_match_hand_computed_numbers(self, cuts_clip):
        sig, det = cuts_clip
        st = det.stats  # shot lengths 3, 4.5, 2.5, 5, 3.5 over 18.5 s
        assert st.count == 5
        assert st.average_shot_duration == pytest.approx(3.7, abs=0.1)
        assert st.median_shot_duration == pytest.approx(3.5, abs=0.15)
        assert st.minimum_shot_duration == pytest.approx(2.5, abs=0.15)
        assert st.maximum_shot_duration == pytest.approx(5.0, abs=0.15)
        assert st.cuts_per_minute == pytest.approx(4 / (18.5 / 60), abs=0.1)
        assert st.cut_frequency_class == "Fast"
        assert st.shot_duration_distribution["2-4s"] == 3 and st.shot_duration_distribution["4-8s"] == 2
        assert st.major_change_per_minute == pytest.approx(st.cuts_per_minute)  # no non-cut major changes
        assert not st.alternates_long_and_short
        tr = det.transitions
        assert tr.non_cut_share == 0 and tr.transition_frequency == 0 and tr.average_transition_duration == 0
        assert tr.transition_distribution["CUT"] == pytest.approx(1.0)
        assert sum(tr.transition_distribution.values()) == pytest.approx(1.0)

    def test_cuts_between_sample_instants_are_within_one_sample(self, media):
        sig = sample(shots_video(media / "unaligned.mp4", [2.37, 3.91, 1.63, 4.2]))
        det = ShotDetector().detect(sig)
        assert_cuts(det, [2.37, 6.28, 7.91], tol=1.0 / FPS + 0.01)
        assert_tiles(det, sig.duration)

    def test_every_transition_type_is_found_and_classified(self, transitions_clip):
        sig, det = transitions_clip
        # shot i+1 starts where its transition is half done: cut 3.0, dissolve 6.0-7.0, fadeblack 9.0-10.0, wipe 12.0-13.0, slide 15.0-16.0
        want = [(3.0, "CUT"), (6.5, "DISSOLVE"), (9.5, "FADE"), (12.5, "WIPE"), (15.5, "SLIDE")]
        assert len(det.shots) == 6, [(s.start, s.transition_type) for s in det.shots]
        for shot, (t, kind) in zip(det.shots[1:], want):
            assert shot.start == pytest.approx(t, abs=0.4)
            assert shot.transition_type == kind
            assert shot.transition_duration == (0.0 if kind == "CUT" else pytest.approx(1.0, abs=0.4))
        assert det.confidence >= 0.6
        assert_tiles(det, sig.duration)

    def test_transition_stats_of_the_mixed_clip(self, transitions_clip):
        sig, det = transitions_clip
        tr = det.transitions
        assert tr.non_cut_share == pytest.approx(0.8)
        assert tr.transition_distribution == pytest.approx({"CUT": 0.2, "FADE": 0.2, "DISSOLVE": 0.2, "WIPE": 0.2, "SLIDE": 0.2, "UNKNOWN": 0.0})
        assert tr.transition_frequency == pytest.approx(4 / (18 / 60), abs=0.01)
        assert tr.average_transition_duration == pytest.approx(1.0, abs=0.3)
        assert det.stats.major_change_per_minute == pytest.approx(det.stats.cuts_per_minute)

    @pytest.mark.parametrize("kind,expected,seconds", [("fade", "DISSOLVE", 1.0), ("dissolve", "DISSOLVE", 0.5), ("dissolve", "DISSOLVE", 2.0), ("fadeblack", "FADE", 0.5),
                                                       ("fadeblack", "FADE", 2.0), ("wipe", "WIPE", 0.5), ("wipe", "WIPE", 2.0), ("slide", "SLIDE", 0.5), ("slide", "SLIDE", 1.0)])
    def test_single_transition_class_boundary_and_duration(self, media, kind, expected, seconds):
        """One xfade transition of known kind and length between two 4 s pictures: it spans [4, 4 + seconds], so the boundary is its midpoint."""
        sig = sample(shots_video(media / f"{kind}_{seconds}.mp4", [4, 4], [kind], trans_dur=seconds))
        det = ShotDetector().detect(sig)
        assert len(det.shots) == 2, [(s.start, s.transition_type) for s in det.shots]
        s = det.shots[1]
        assert s.transition_type == expected
        assert s.start == pytest.approx(4 + seconds / 2, abs=0.3)
        assert s.transition_duration == pytest.approx(seconds, abs=0.3)
        assert 0.5 <= s.confidence <= 1.0
        assert_tiles(det, sig.duration)

    def test_a_slide_that_never_stops_is_a_pan_not_a_transition(self, media):
        """Keeps shifting for 2 s (longer than any slide transition): treated as camera movement, so no boundary."""
        sig = sample(shots_video(media / "slide_long.mp4", [4, 4], ["slide"], trans_dur=2.0))
        det = ShotDetector().detect(sig)
        assert all(s.transition_type != "CUT" for s in det.shots[1:])  # never a burst of hard cuts

    def test_rapid_cuts(self, media):
        lengths = [0.5, 0.75, 1, 0.5, 0.75, 1, 0.5, 0.75, 1, 0.5]
        sig = sample(shots_video(media / "rapid.mp4", lengths))
        det = ShotDetector().detect(sig)
        assert_cuts(det, list(np.cumsum(lengths)[:-1]), tol=0.2)
        assert det.stats.cut_frequency_class == "Very Fast"
        assert det.stats.minimum_shot_duration == pytest.approx(0.5, abs=0.15)
        assert det.stats.cuts_per_minute == pytest.approx(9 / (sum(lengths) / 60), abs=1.5)
        assert det.confidence >= 0.8
        assert_tiles(det, sig.duration)

    def test_twenty_second_long_shot_is_one_shot(self, media):
        sig = sample(shots_video(media / "long.mp4", [20]))
        det = ShotDetector().detect(sig)
        assert len(det.shots) == 1 and det.shots[0].start == 0.0 and det.shots[0].end == pytest.approx(20.0, abs=0.01)
        assert det.confidence >= 0.9 and det.shots[0].confidence >= 0.9
        assert det.stats.count == 1 and det.stats.cuts_per_minute == 0.0 and det.stats.shot_duration_distribution[">15s"] == 1
        assert det.transitions.transition_distribution == {} and det.transitions.non_cut_share == 0.0
        assert_tiles(det, sig.duration)

    @pytest.mark.parametrize("mode,amount", [("zoom", 0.10), ("zoom_out", 0.25), ("pan", 0.30), ("pan", 0.9), ("static", 0.0)])
    def test_camera_moves_are_not_cuts(self, media, mode, amount):
        sig = sample(camera_video(media / f"cam_{mode}_{amount}.mp4", 8, mode, amount))
        det = ShotDetector().detect(sig)
        assert len(det.shots) == 1, det.cut_times
        assert det.major_change_times == []  # camera motion is not a change of what is in the picture

    def test_camera_shake_is_not_cut_but_is_flagged_as_unreliable(self, media):
        """Every sample differs a lot from the last (shake faster than the sample rate): no cuts, low confidence and an explanation."""
        sig = sample(camera_video(media / "cam_shake.mp4", 8, "shake"))
        det = ShotDetector().detect(sig)
        assert len(det.shots) == 1
        assert det.confidence < 0.4
        assert any("continuous" in n.lower() or "shake" in n.lower() for n in det.notes)

    def test_static_video_has_high_confidence(self, media):
        sig = sample(camera_video(media / "cam_static.mp4", 8, "static"))
        det = ShotDetector().detect(sig)
        assert len(det.shots) == 1 and det.confidence >= 0.9 and det.notes == []

    def test_cuts_survive_heavy_sensor_noise(self, media):
        sig = sample(shots_video(media / "noisy_cuts.mp4", [3, 4, 3], noise=20))
        det = ShotDetector().detect(sig)
        assert_cuts(det, [3.0, 7.0], tol=0.25)


# ---------------------------------------------------------------------------------------------- no false cuts (numpy)
class TestNoFalseCuts:
    A = picture(1)

    def test_perfectly_static_video_is_exactly_one_shot(self):
        det = detect(hold(self.A, 8))
        assert len(det.shots) == 1 and det.confidence >= 0.9 and det.major_change_times == []

    @pytest.mark.parametrize("sigma", [0.03, 0.08])
    def test_sensor_like_noise(self, sigma):
        det = detect(noisy(hold(self.A, 8), sigma))
        assert len(det.shots) == 1 and det.confidence >= 0.8

    @pytest.mark.parametrize("amount", [0.10, 0.25])
    def test_brightness_flicker(self, amount):
        rng = np.random.default_rng(5)
        frames = [np.clip(f * (1 + amount * rng.uniform(-1, 1)), 0, 1) for f in hold(self.A, 10)]
        det = detect(frames)
        assert len(det.shots) == 1 and det.major_change_times == []

    def test_single_white_flash_frame_is_not_a_cut_but_a_major_change(self):
        """Documented choice: an excursion of one sample that comes back to the same picture is a flash, not two cuts. It is reported as a major visual change."""
        frames = hold(self.A, 6)
        frames[24] = np.ones((H, W), np.float32)
        det = detect(frames)
        assert len(det.shots) == 1
        assert len(det.major_change_times) == 1 and det.major_change_times[0] == pytest.approx(3.0, abs=0.2)
        assert any("flash" in n.lower() for n in det.notes)
        assert det.stats.major_change_per_minute == pytest.approx(1 / (6 / 60), abs=0.1)  # 0 cuts + 1 major change in 6 s

    def test_brightened_flash_and_two_frame_white_flash(self):
        frames = hold(self.A, 6)
        frames[24] = np.clip(frames[24] + 0.4, 0, 1)
        det = detect(frames)
        assert len(det.shots) == 1 and len(det.major_change_times) == 1
        frames = hold(self.A, 6)
        frames[24] = frames[25] = np.ones((H, W), np.float32)
        det = detect(frames)
        assert len(det.shots) == 1 and len(det.major_change_times) == 1 and det.major_change_times[0] == pytest.approx(3.0, abs=0.3)

    def test_one_sample_insert_of_another_picture_that_returns_is_a_glitch_not_a_shot(self):
        frames = hold(self.A, 6)
        frames[24] = picture(9)
        det = detect(frames)
        assert len(det.shots) == 1 and len(det.major_change_times) == 1

    def test_two_sample_insert_is_a_real_short_shot(self):
        """The minimum shot length is two samples: a 0.25 s insert of another picture IS reported (as two cuts)."""
        frames = hold(self.A, 3) + hold(picture(9), 0.25) + hold(self.A, 3)
        det = detect(frames)
        assert_cuts(det, [3.0, 3.25], tol=0.07)
        assert det.shots[1].duration == pytest.approx(0.25, abs=0.01)

    def test_slow_gradual_lighting_change(self):
        frames = [np.clip(self.A * (0.6 + 0.8 * i / 100), 0, 1) for i in range(100)]
        det = detect(frames)
        assert len(det.shots) == 1 and det.major_change_times == [] and det.confidence >= 0.9

    def test_jump_in_exposure_is_not_a_cut(self):
        det = detect(hold(self.A, 3) + hold(np.clip(self.A + 0.2, 0, 1), 3))
        assert len(det.shots) == 1 and det.major_change_times == [pytest.approx(3.0, abs=0.2)]

    @pytest.mark.parametrize("kind", ["noise", "blobs"])
    def test_every_frame_different_does_not_make_hundreds_of_shots(self, kind):
        if kind == "noise":
            frames = [np.random.default_rng(i).random((H, W)).astype(np.float32) for i in range(96)]
        else:
            frames = [picture(100 + i) for i in range(96)]
        det = detect(frames)
        assert len(det.shots) <= 2
        assert det.confidence <= 0.35
        assert any("continuous" in n.lower() for n in det.notes)

    def test_handheld_jitter_of_the_whole_frame(self):
        rng = np.random.default_rng(3)
        frames = [np.roll(np.roll(self.A, int(rng.integers(-3, 4)), 1), int(rng.integers(-2, 3)), 0) for _ in range(80)]
        det = detect(frames)
        assert len(det.shots) == 1

    def test_caption_sized_overlay_is_neither_cut_nor_major_change(self):
        frames = hold(self.A, 6)
        for i in range(16, 30):
            frames[i][58:66, 30:100] = 1.0  # a bright caption strip, ~6% of the frame
        det = detect(frames)
        assert len(det.shots) == 1 and det.major_change_times == [] and det.confidence >= 0.7

    def test_large_overlay_with_the_real_motion_estimate(self):
        """Same as below but with the real global-motion fields (a static camera must not hide a change of content)."""
        frames = hold(self.A, 5)
        for i in range(16, 30):
            frames[i][20:50, 10:60] = 0.05
        det = ShotDetector().detect(signals_of(noisy(frames, 0.02), motion=True))
        assert len(det.shots) == 1 and len(det.major_change_times) == 2

    def test_large_overlay_is_a_major_change_not_a_cut(self):
        frames = hold(self.A, 7)
        for i in range(16, 40):
            frames[i][20:50, 10:60] = 0.05  # a dark 30x50 box (~16% of the frame) appears at 2.0 s and leaves at 5.0 s
        det = detect(noisy(frames, 0.02))
        assert len(det.shots) == 1
        assert len(det.major_change_times) == 2
        assert det.major_change_times[0] == pytest.approx(2.0, abs=0.3) and det.major_change_times[1] == pytest.approx(5.0, abs=0.3)
        assert det.stats.major_change_per_minute == pytest.approx(2 / (7 / 60), abs=0.2)


# ---------------------------------------------------------------------------------------------- cuts and transitions (numpy)
class TestSyntheticEditing:
    A, B, C = picture(1), picture(2), picture(3)

    def test_cuts_every_three_samples_are_all_found(self):
        det = detect([f for i in range(20) for f in hold(picture(50 + i), 3 / 8)])
        assert len(det.shots) == 20
        assert_cuts(det, [i * 3 / 8 for i in range(1, 20)], tol=0.07)
        assert det.confidence >= 0.8

    def test_cuts_every_two_samples_are_the_limit_and_say_so(self):
        det = detect([f for i in range(20) for f in hold(picture(50 + i), 0.25)])
        assert len(det.shots) >= 17
        assert det.confidence <= 0.35
        assert any("limit" in n.lower() or "shorter than" in n.lower() for n in det.notes)

    def test_cuts_in_the_first_and_last_quarter_second(self):
        det = detect(hold(self.A, 0.25) + hold(self.B, 4) + hold(self.C, 0.25))
        assert_cuts(det, [0.25, 4.25], tol=0.07)

    def test_cut_between_flat_cards(self):
        a, b = np.full((H, W), 0.1, np.float32), np.full((H, W), 0.9, np.float32)
        det = detect(hold(a, 3) + hold(b, 3))
        assert_cuts(det, [3.0], tol=0.07)

    def test_low_contrast_cut(self):
        det = detect(hold(picture(1, 0.25), 3) + hold(picture(2, 0.25), 3))
        assert_cuts(det, [3.0], tol=0.07)

    def test_hard_black_gap_is_two_cuts_not_a_fade(self):
        black = np.zeros((H, W), np.float32)
        det = detect(hold(self.A, 3) + hold(black, 0.75) + hold(self.B, 3))
        assert_cuts(det, [3.0, 3.75], tol=0.07)
        assert all(s.transition_type == "CUT" for s in det.shots)

    @pytest.mark.parametrize("hold_frames", [2, 12])
    def test_fade_through_black_in_samples(self, hold_frames):
        black = np.zeros((H, W), np.float32)
        ramp = [i / 4 for i in range(1, 5)]
        frames = hold(self.A, 3) + [self.A * (1 - r) for r in ramp] + [black] * hold_frames + [self.B * r for r in ramp] + hold(self.B, 3)
        det = detect(frames)
        assert len(det.shots) == 2
        s = det.shots[1]
        assert s.transition_type == "FADE" and s.transition_duration == pytest.approx((8 + hold_frames) / 8, abs=0.35)
        assert s.start == pytest.approx(3 + (8 + hold_frames) / 16, abs=0.3)

    def test_fade_through_white(self):
        white = np.ones((H, W), np.float32)
        ramp = [i / 4 for i in range(1, 5)]
        frames = hold(self.A, 3) + [self.A * (1 - r) + r for r in ramp] + [white] * 2 + [self.B * r + (1 - r) for r in ramp] + hold(self.B, 3)
        det = detect(frames)
        assert len(det.shots) == 2 and det.shots[1].transition_type == "FADE"

    @pytest.mark.parametrize("steps", [3, 8, 16])
    def test_dissolve_lengths(self, steps):
        mix = [(1 - i / (steps + 1)) * self.A + (i / (steps + 1)) * self.B for i in range(1, steps + 1)]
        det = detect(hold(self.A, 3) + mix + hold(self.B, 3))
        assert len(det.shots) == 2
        s = det.shots[1]
        assert s.transition_type == "DISSOLVE"
        assert s.transition_duration == pytest.approx((steps + 1) / FPS, abs=0.25)
        assert s.start == pytest.approx(3 + (steps + 1) / (2 * FPS) - 0.0625, abs=0.2)  # the moment half the change has happened

    def test_a_two_sample_cross_fade_is_indistinguishable_from_a_cut(self):
        mix = [0.67 * self.A + 0.33 * self.B, 0.33 * self.A + 0.67 * self.B]
        det = detect(hold(self.A, 3) + mix + hold(self.B, 3))
        assert len(det.shots) == 2  # one boundary, class CUT or DISSOLVE are both honest at 8 samples/s
        assert det.shots[1].start == pytest.approx(3.1, abs=0.2)

    def test_opening_fade_in_and_closing_fade_out_are_not_boundaries(self):
        frames = [self.A * (i / 8) for i in range(9)] + hold(self.A, 5) + [self.A * (1 - i / 8) for i in range(1, 9)] + [np.zeros((H, W), np.float32)] * 4
        det = detect(frames)
        assert len(det.shots) == 1 and det.major_change_times == []
        assert det.confidence >= 0.7

    def test_cuts_are_the_same_at_other_sample_rates(self):
        for fps in (4, 12):
            det = detect(hold(self.A, 3, fps) + hold(self.B, 3, fps) + hold(self.C, 3, fps), fps=fps)
            assert_cuts(det, [3.0, 6.0], tol=1.0 / fps + 0.01)
        det = detect(hold(self.A, 6, 2) + hold(self.B, 6, 2), fps=2)
        assert len(det.shots) == 2 and any("sample rate" in n.lower() for n in det.notes)

    def test_cuts_with_sensor_noise(self):
        det = detect(noisy(hold(self.A, 3) + hold(self.B, 3) + hold(self.C, 3), 0.05))
        assert_cuts(det, [3.0, 6.0], tol=0.07)

    def test_overlay_that_appears_during_a_cut_does_not_hide_it(self):
        frames = hold(self.A, 3) + hold(self.B, 3)
        for i in range(10, 40):
            frames[i][58:66, 30:100] = 1.0
        assert_cuts(detect(frames), [3.0], tol=0.07)


# ---------------------------------------------------------------------------------------------- random edit lists
def random_edit(seed: int) -> tuple[list[np.ndarray], list[tuple[float, str]]]:
    """A random edit decision list with known truth: shots of 0.5-5 s joined by cuts (60%), cross-dissolves (20%) or fades through black (20%), plus sensor noise."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 9))
    pics = [picture(int(rng.integers(0, 10**6)), float(rng.uniform(0.5, 1.0)), float(rng.uniform(0.3, 0.6))) for _ in range(n)]
    frames: list[np.ndarray] = []
    truth: list[tuple[float, str]] = []
    for i in range(n):
        if i:
            kind, prev = str(rng.choice(["CUT", "CUT", "CUT", "DISSOLVE", "FADE"])), frames[-1]
            if kind == "CUT":
                truth.append((len(frames) / FPS, kind))
            elif kind == "DISSOLVE":
                k = int(rng.integers(4, 10))
                frames += [((1 - j / (k + 1)) * prev + (j / (k + 1)) * pics[i]).astype(np.float32) for j in range(1, k + 1)]
                truth.append(((len(frames) - k / 2) / FPS, kind))
            else:
                k = int(rng.integers(3, 6))
                frames += [(prev * (1 - j / k)).astype(np.float32) for j in range(1, k + 1)] + [(pics[i] * (j / k)).astype(np.float32) for j in range(1, k + 1)]
                truth.append((len(frames) / FPS - k / FPS, kind))
        frames += hold(pics[i], float(rng.integers(4, 40)) / FPS)
    sigma = float(rng.choice([0, 0.01, 0.03]))
    return (noisy(frames, sigma, int(rng.integers(0, 1000))) if sigma else frames), truth


class TestRandomEdits:
    @pytest.mark.parametrize("seed", range(14))
    def test_random_edit_list_is_recovered_exactly(self, seed):
        frames, truth = random_edit(seed)
        det = detect(frames)
        found = [(s.start, s.transition_type) for s in det.shots[1:]]
        assert [k for _, k in found] == [k for _, k in truth], (found, truth)  # nothing missed, nothing extra, every class right
        for (t, _), (want, _) in zip(found, truth):
            assert abs(t - want) <= 0.4
        assert det.confidence >= 0.6
        assert_tiles(det, len(frames) / FPS)


# ---------------------------------------------------------------------------------------------- sensitivity
class TestSensitivity:
    @staticmethod
    def frames(alpha: float) -> list[np.ndarray]:
        p1, p2, p3 = picture(11), picture(12), picture(13)
        weak = (1 - alpha) * p1 + alpha * p2  # a cut to a picture that is only partly different
        return hold(p1, 3) + hold(weak, 3) + hold(p3, 3)

    def test_weak_cut_needs_higher_sensitivity(self):
        sig = signals_of(self.frames(0.08))
        counts = [len(ShotDetector(s).detect(sig).shots) for s in (0.5, 1.0, 2.0)]
        assert counts == [2, 2, 3]  # the strong cut is always found; the weak one (change ~0.02) only when eager

    def test_medium_cut_is_dropped_by_conservative_detector(self):
        sig = signals_of(self.frames(0.16))
        counts = [len(ShotDetector(s).detect(sig).shots) for s in (0.5, 1.0, 2.0)]
        assert counts == [2, 3, 3]

    def test_sensitivity_never_adds_cuts_to_static_or_shaky_footage(self):
        sig = signals_of(hold(picture(1), 8))
        assert len(ShotDetector(3.0).detect(sig).shots) == 1
        shaky = signals_of([np.roll(picture(1), int(d), 1) for d in np.random.default_rng(1).integers(-20, 20, 80)])
        assert len(ShotDetector(3.0).detect(shaky).shots) <= 2

    def test_sensitivity_is_clamped(self):
        assert ShotDetector(0).sensitivity == 0.25 and ShotDetector(99).sensitivity == 3.0 and ShotDetector().sensitivity == 1.0


# ---------------------------------------------------------------------------------------------- degenerate input
def raw_signals(n: int, fps: float = 8.0, fill: float | None = None, seed: int = 0) -> FrameSignals:
    rng = np.random.default_rng(seed)
    sig = np.full((n, SIG_H, SIG_W), fill, np.float32) if fill is not None else rng.random((n, SIG_H, SIG_W)).astype(np.float32)
    z = np.zeros(n, np.float32)
    return FrameSignals(fps, np.arange(n) / fps, z + 0.5, z + 0.1, z, z, z, sig, 4, z, z, z, z, n / fps)


class TestDegenerateInput:
    def test_no_samples(self):
        det = ShotDetector().detect(raw_signals(0))
        assert det.shots == [] and det.confidence == 0.0 and det.notes
        assert det.stats.count == 0 and det.transitions.transition_distribution == {}

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 12])
    def test_very_short_signals_report_low_confidence(self, n):
        det = ShotDetector().detect(raw_signals(n, fill=0.4))
        assert len(det.shots) == 1 and det.shots[0].start == 0.0
        assert det.shots[0].end == pytest.approx(n / 8.0)
        assert det.confidence <= 0.55 and any("sample" in note.lower() or "frame" in note.lower() for note in det.notes)
        assert_tiles(det, n / 8.0)

    def test_two_unrelated_frames_do_not_invent_a_cut(self):
        det = ShotDetector().detect(raw_signals(2))
        assert len(det.shots) == 1 and det.confidence <= 0.35

    def test_short_clip_with_one_cut_is_found_but_flagged(self):
        det = detect(hold(picture(1), 1) + hold(picture(2), 1))  # 16 samples
        assert_cuts(det, [1.0], tol=0.07)
        assert det.confidence < 0.8 and any("samples" in n for n in det.notes)

    @pytest.mark.parametrize("fill", [0.0, 1.0, 0.5])
    def test_constant_frames(self, fill):
        det = ShotDetector().detect(raw_signals(40, fill=fill))
        assert len(det.shots) == 1 and det.confidence >= 0.9

    def test_deterministic(self):
        sig = signals_of(hold(picture(1), 3) + hold(picture(2), 3) + hold(picture(3), 2))
        a, b = ShotDetector().detect(sig), ShotDetector().detect(sig)
        assert a == b
        assert ShotDetector().detect(sig) == a

    def test_does_not_modify_its_input(self):
        sig = signals_of(hold(picture(1), 3) + hold(picture(2), 3))
        before = (sig.sig.copy(), sig.times.copy(), sig.luma_std.copy())
        ShotDetector().detect(sig)
        assert np.array_equal(sig.sig, before[0]) and np.array_equal(sig.times, before[1]) and np.array_equal(sig.luma_std, before[2])


# ---------------------------------------------------------------------------------------------- the result type
class TestResultShape:
    def test_result_holds_only_abstract_numbers(self):
        det = detect(hold(picture(1), 3) + hold(picture(2), 3))
        plain = to_plain(det)
        json.dumps(plain)  # JSON-safe: only str / numbers / lists / dicts, no arrays and no pictures
        assert set(plain) == {"shots", "stats", "transitions", "confidence", "major_change_times", "notes"}
        assert set(plain["shots"][0]) == {"shot_id", "start", "end", "frame_sample", "transition_type", "transition_duration", "confidence"}

    def test_events_view_for_the_cache(self):
        det = detect(hold(picture(1), 3) + hold(picture(2), 3))
        assert det.cut_times == [s.start for s in det.shots[1:]]
        assert det.transition_events == [(det.shots[1].start, "CUT", 0.0)]

    def test_sections_are_passed_to_the_statistics(self):
        sec = [Section(0, 3.0), Section(3.0, 6.0), Section(6.0, 9.0)]
        det = ShotDetector().detect(signals_of(hold(picture(1), 2) + hold(picture(2), 2) + hold(picture(3), 2) + hold(picture(4), 3)), sec)
        assert det.stats.cuts_per_section == [1.0, 1.0, 1.0]
        assert det.stats.cuts_per_scene == pytest.approx(1.0)

    def test_frame_sample_stays_away_from_transitions(self):
        mix = [(1 - i / 9) * picture(1) + (i / 9) * picture(2) for i in range(1, 9)]
        det = detect(hold(picture(1), 3) + mix + hold(picture(2), 4))
        s0, s1 = det.shots
        assert s0.frame_sample < 3.0 - 0.1
        assert s1.frame_sample > s1.start + s1.transition_duration / 2 - 0.01


# ---------------------------------------------------------------------------------------------- statistics (hand computed)
class TestShotStats:
    def test_alternating_long_and_short(self):
        st = compute_shot_stats(mk_shots([2, 3, 7, 2, 10]), 24.0, major_changes=2)
        assert st.count == 5
        assert st.average_shot_duration == pytest.approx(4.8)
        assert st.median_shot_duration == pytest.approx(3.0)
        assert st.minimum_shot_duration == pytest.approx(2.0) and st.maximum_shot_duration == pytest.approx(10.0)
        assert st.std_shot_duration == pytest.approx(np.sqrt(10.16))
        assert st.shot_duration_distribution == {"<1s": 0, "1-2s": 0, "2-4s": 3, "4-8s": 1, "8-15s": 1, ">15s": 0}
        assert st.cuts_per_minute == pytest.approx(10.0)  # 4 cuts in 24 s
        assert st.cut_frequency_class == "Moderate"
        assert st.major_change_per_minute == pytest.approx(15.0)  # (4 cuts + 2 other major changes) in 0.4 min
        assert st.alternates_long_and_short is True

    @pytest.mark.parametrize("durations", [[5, 5, 5, 5], [4, 5, 6, 4, 5, 6], [3, 4, 5, 6, 5, 4], [1, 1, 1, 1, 12, 1], [8, 9], [5]])
    def test_even_or_ordinary_rhythm_does_not_alternate(self, durations):
        assert compute_shot_stats(mk_shots(durations), float(sum(durations))).alternates_long_and_short is False

    def test_uniform_shots_have_no_spread(self):
        st = compute_shot_stats(mk_shots([5, 5, 5, 5]), 20.0)
        assert st.std_shot_duration == 0.0 and st.average_shot_duration == st.median_shot_duration == 5.0
        assert st.cuts_per_minute == pytest.approx(3 / (20 / 60)) and st.cut_frequency_class == "Moderate"

    def test_other_bimodal_examples(self):
        assert compute_shot_stats(mk_shots([1, 1.2, 9, 1.1, 10, 1]), 23.3).alternates_long_and_short
        assert compute_shot_stats(mk_shots([0.8, 6, 0.9, 7, 1.0, 6.5]), 22.2).alternates_long_and_short

    @pytest.mark.parametrize("n_shots,seconds,klass", [(2, 60.0, "Slow"), (7, 60.0, "Moderate"), (13, 60.0, "Fast"), (25, 60.0, "Very Fast"), (12, 120.0, "Slow")])
    def test_pacing_class_follows_cuts_per_minute(self, n_shots, seconds, klass):
        st = compute_shot_stats(mk_shots([seconds / n_shots] * n_shots), seconds)
        assert st.cuts_per_minute == pytest.approx((n_shots - 1) / (seconds / 60))
        assert st.cut_frequency_class == klass

    def test_distribution_buckets(self):
        st = compute_shot_stats(mk_shots([0.5, 1.5, 3, 5, 9, 20]), 39.0)
        assert st.shot_duration_distribution == {"<1s": 1, "1-2s": 1, "2-4s": 1, "4-8s": 1, "8-15s": 1, ">15s": 1}

    def test_cuts_per_section_counts_boundaries_per_section(self):
        # boundaries at 5, 10 (= section edge: counts for the section it starts), 14, 22
        shots = mk_shots([5, 5, 4, 8, 6])
        secs = [Section(0, 10), Section(10, 20), Section(20, 28)]
        st = compute_shot_stats(shots, 28.0, secs)
        assert st.cuts_per_section == [1.0, 2.0, 1.0]
        assert st.cuts_per_scene == pytest.approx(4 / 3)

    def test_cuts_per_block_without_sections(self):
        # 60 s -> 4 equal blocks of 15 s; cuts at 5, 20, 22, 50
        st = compute_shot_stats(mk_shots([5, 15, 2, 28, 10]), 60.0)
        assert st.cuts_per_section == [1.0, 2.0, 0.0, 1.0]
        assert sum(st.cuts_per_section) == 4 and st.cuts_per_scene == pytest.approx(1.0)

    def test_empty_and_single(self):
        assert compute_shot_stats([], 10.0).count == 0
        one = compute_shot_stats(mk_shots([12]), 12.0)
        assert one.count == 1 and one.cuts_per_minute == 0.0 and one.cut_frequency_class == "Slow" and one.major_change_per_minute == 0.0
        zero = compute_shot_stats(mk_shots([1, 1]), 0.0)
        assert zero.cuts_per_minute == 0.0  # no duration: no rate, no division by zero


class TestTransitionStats:
    def test_mostly_cuts_plus_one_dissolve_and_one_fade(self):
        kinds = {4: ("DISSOLVE", 0.8), 8: ("FADE", 1.2)}
        shots = mk_shots([6] * 11, kinds)  # 10 boundaries: 8 cuts, 1 dissolve, 1 fade, over 66 s
        tr = compute_transition_stats(shots, 66.0)
        assert tr.transition_distribution == pytest.approx({"CUT": 0.8, "FADE": 0.1, "DISSOLVE": 0.1, "WIPE": 0.0, "SLIDE": 0.0, "UNKNOWN": 0.0})
        assert set(tr.transition_distribution) == set(TRANSITION_TYPES)
        assert sum(tr.transition_distribution.values()) == pytest.approx(1.0)
        assert tr.non_cut_share == pytest.approx(0.2)
        assert tr.transition_frequency == pytest.approx(2 / (66 / 60))
        assert tr.average_transition_duration == pytest.approx(1.0)

    def test_all_cuts(self):
        tr = compute_transition_stats(mk_shots([3, 3, 3]), 9.0)
        assert tr.non_cut_share == 0.0 and tr.transition_frequency == 0.0 and tr.average_transition_duration == 0.0
        assert tr.transition_distribution["CUT"] == pytest.approx(1.0) and sum(tr.transition_distribution.values()) == pytest.approx(1.0)

    def test_no_boundary_means_empty_distribution(self):
        for shots in ([], mk_shots([10])):
            tr = compute_transition_stats(shots, 10.0)
            assert tr.transition_distribution == {} and tr.non_cut_share == 0.0 and tr.transition_frequency == 0.0 and tr.average_transition_duration == 0.0

    def test_only_entering_transitions_count_and_unknown_types_are_mapped(self):
        shots = mk_shots([2, 2, 2], {0: ("DISSOLVE", 5.0), 1: ("WIPE", 0.5), 2: ("MORPH", 0.7)})
        tr = compute_transition_stats(shots, 6.0)  # shot 0's "transition" is not a boundary
        assert tr.transition_distribution["WIPE"] == pytest.approx(0.5) and tr.transition_distribution["UNKNOWN"] == pytest.approx(0.5)
        assert tr.transition_distribution["DISSOLVE"] == 0.0
        assert tr.non_cut_share == 1.0 and tr.average_transition_duration == pytest.approx(0.6)
        assert tr.transition_frequency == pytest.approx(2 / (6 / 60))
