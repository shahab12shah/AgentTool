"""VoicePriorityController: the voice always wins. Music and sound effects may never mask speech."""

from __future__ import annotations

from dataclasses import dataclass

from app.presentation.models import AudioSettings

PRIORITY = {"VOICE": 100, "SFX": 60, "MUSIC": 40}
MASK_STEP = 0.1


@dataclass
class MaskingIssue:
    role: str
    start: float
    end: float
    level: float
    message: str


class VoicePriorityController:
    def __init__(self, settings: AudioSettings) -> None:
        self.s = settings
        # while someone is speaking, music may be at most a modest multiple of the "normal" level; SFX stay clearly below the voice
        self.music_ceiling = min(0.35, max(settings.music_level * 1.6, settings.pause_level))
        self.sfx_ceiling = 0.6

    def clamp(self, role: str, level: float, voice_active: bool) -> float:
        if role == "VOICE":
            return max(level, 0.0)
        if not voice_active:
            return level
        return min(level, self.music_ceiling if role == "MUSIC" else self.sfx_ceiling)

    def masking(self, role: str, gain_at, speech: list[tuple[float, float]], duration: float) -> list[MaskingIssue]:
        """Where ``role`` is louder than allowed while the voice is speaking (``gain_at(t)`` = effective linear gain)."""
        ceiling = self.music_ceiling if role == "MUSIC" else self.sfx_ceiling
        out: list[MaskingIssue] = []
        for a, b in speech:
            t, run = a, None
            while t <= min(b, duration):
                g = gain_at(t)
                if g > ceiling + 1e-6:
                    run = run or [t, t, g]
                    run[1], run[2] = t, max(run[2], g)
                elif run:
                    out.append(MaskingIssue(role, run[0], run[1], run[2], f"{role.title()} is {run[2]:.0%} while the voice speaks (limit {ceiling:.0%})."))
                    run = None
                t += MASK_STEP
            if run:
                out.append(MaskingIssue(role, run[0], run[1], run[2], f"{role.title()} is {run[2]:.0%} while the voice speaks (limit {ceiling:.0%})."))
        return out
