"""OriginalityGuard: the reference is a source of editing *principles*, never of content.

    "Copy the competitor's exact opening."   ->   "Use a fast, high-information opening with strong text emphasis."

* ``convert_request`` reads a free-text style wish and turns every part that asks to reproduce something specific - footage, an exact shot sequence or
  opening, a script or exact words, captions, graphics or animations, a logo or branding, a composition, music, a thumbnail - into an abstract editing
  instruction. What the user asked for is never echoed back (not even quoted words): the result says what was neutralised and what instruction replaced it.
  Wishes that are already abstract ("faster pacing with subtle zooms") pass through unchanged.
* ``audit_project`` checks the isolation rule: no reference video, copy or link of it is in the media library or on the timeline.
* ``audit_overrides`` / ``audit_profile`` check that what leaves the analysis is abstract: numbers, classes and style ids - no free text, no file names.

Everything is deterministic keyword / phrase logic (no network, no model): it recognises the common ways of asking to copy, it cannot judge intent in general.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from app.captions.styles import PRESETS as CAPTION_PRESETS
from app.editing.overrides import PARAMETERS, EditingStrategyOverrides
from app.reference.style_model import ReferenceStyleProfile

if TYPE_CHECKING:  # pragma: no cover
    from app.project.project import Project

# kind -> (label shown to the user, the abstract instruction that replaces the request)
KINDS: dict[str, tuple[str, str]] = {
    "FOOTAGE": ("the reference's footage", "Use only your own footage and visuals; take over only the editing rhythm."),
    "OPENING": ("the reference's exact opening", "Use a fast, high-information opening with strong text emphasis."),
    "SEQUENCE": ("the reference's exact shot sequence", "Use a similar overall rhythm - shot lengths and how often the picture changes - but build your own sequence from your own scenes."),
    "TEXT": ("the reference's exact words, script or captions", "Write your own text, captions and script; match only how much text appears and how fast it changes."),
    "TYPOGRAPHY": ("the reference's exact typography", "Pick a caption style from your own presets with a similar character (size, contrast, position, highlighting); exact fonts and designs are not copied."),
    "GRAPHICS": ("the reference's graphics or animations", "Create your own graphics; match only how often they appear and how strongly they are animated."),
    "BRANDING": ("the reference's logo or branding", "Use your own branding; no logo, watermark or brand element of the reference is used."),
    "COMPOSITION": ("the reference's exact composition", "Frame your own visuals freely; match only the pacing and the amount of camera movement."),
    "MUSIC": ("the reference's music or sound effects", "Use music and effects you have the right to use; match only how present the music is, how it ducks under the voice and how often effects occur."),
    "THUMBNAIL": ("the reference's thumbnail", "Design your own thumbnail; nothing is taken from the reference."),
    "TRANSITIONS": ("the reference's transitions in the same order", "Use transitions about as often as the reference does, chosen by your own story, not in the same sequence."),
}
# what the request is about
OBJECTS: dict[str, tuple[str, ...]] = {
    "OPENING": ("opening", "intro", "introduction", "hook", "first seconds", "first scene", "beginning", "start of the video", "ending", "outro", "closing"),
    "FOOTAGE": ("footage", "clip", "clips", "video", "videos", "scene", "scenes", "b-roll", "broll", "b roll", "recording", "screen recording", "screencast", "shots"),
    "SEQUENCE": ("sequence", "order of", "shot order", "shot-by-shot", "shot by shot", "cut-for-cut", "cut for cut", "storyboard", "edit decision", "timeline", "same cuts", "same shots", "frame by frame",
                 "frame-by-frame"),
    "TEXT": ("script", "text", "wording", "words", "narration", "voiceover", "voice-over", "transcript", "captions", "subtitles", "quote", "quotes", "lines", "sentences", "headline", "headlines",
             "title", "titles", "slogan", "catchphrase", "ad copy", "sales copy", "copywriting"),
    "GRAPHICS": ("graphic", "graphics", "animation", "animations", "lower third", "lower-third", "lower thirds", "overlay", "overlays", "chart", "charts", "infographic", "infographics",
                 "motion graphics", "template", "templates", "icon", "icons", "design", "visuals", "artwork", "illustration", "illustrations"),
    "BRANDING": ("logo", "logos", "watermark", "brand", "branding", "mascot", "channel art", "banner", "trademark", "intro sting", "sting", "bumper", "endcard", "end card"),
    "TYPOGRAPHY": ("font", "fonts", "typography", "typeface", "typefaces", "lettering"),
    "COMPOSITION": ("composition", "framing", "layout", "camera angle", "camera angles", "angles", "shot design"),
    "MUSIC": ("music", "song", "songs", "track", "soundtrack", "beat", "audio", "sound effects", "sfx", "jingle", "melody", "sound design"),
    "THUMBNAIL": ("thumbnail", "thumbnails", "cover image", "cover art"),
    "TRANSITIONS": ("transition", "transitions", "wipe", "wipes"),
}
# words that turn a mention into a request to reproduce it: an explicit act of taking ...
HARD_COPY = (
    "copy", "copies", "copying", "duplicate", "duplicating", "replicate", "replicating", "reproduce", "reproducing", "clone", "cloning", "steal", "stealing", "rip", "ripping", "lift", "lifting",
    "reuse", "re-use", "recreate", "re-create", "remake", "trace", "tracing", "plagiarize", "plagiarise", "download", "grab", "paste", "take their", "use their", "use his", "use her",
    "use the same", "use that same", "carbon copy", "straight from",
)
# ... or a demand for sameness
SOFT_COPY = (
    "same as", "exactly like", "exactly as", "exact", "exactly", "identical", "verbatim", "word for word", "word-for-word", "1:1", "one to one", "one-to-one", "pixel for pixel",
    "pixel-for-pixel", "as is", "unchanged", "the very same", "same exact", "just like theirs", "match theirs", "the same", "same",
)
WHOLE = ("whole video", "entire video", "the video", "their video", "whole thing", "everything", "it all", "all of it", "the whole")
OWN = re.compile(r"\b(?:my|our|mine)\s+(?:own\s+)?(?:(?:intro|outro|opening|ending|closing|old|new|previous|last|latest|usual|regular|raw|original|b-roll|broll|screen)\s+)*[\w-]+", re.IGNORECASE)  # "my video", "my script": the user's own material is not a reference
ADJECTIVE = re.compile(r"\b(?:intro|outro|opening|ending|closing)\s+(?=(?:music|sound|sfx|song|track|sting|jingle|animation|graphics?|logo|card|title|text|captions?)\b)", re.IGNORECASE)
LOOK = ("look", "looks", "style", "styled", "design", "designed", "appearance", "font", "fonts", "typography")
SPLIT = re.compile(r"[.!?;\n]+|\s+(?:but|and then|then|also|plus)\s+|\s+and\s+(?=(?:copy|use|reuse|take|steal|duplicate|replicate|reproduce|recreate|download)\b)", re.IGNORECASE)
QUOTED = re.compile(r"[\"“”„«»`]([^\"“”„«»`]{3,})[\"“”„«»`]|(?<![\w])['‘]([^'‘’]{3,})['’](?![\w])")  # "..." or '...' (an apostrophe inside a word is not a quote)
URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
# abstract wishes worth keeping a note of (nothing is applied from them: the sliders and the style plan do that)
STYLE_WORDS = {
    "pacing": ("faster", "quicker", "punchy", "snappy", "fast-paced", "fast paced", "slower", "calmer", "relaxed", "slow-paced", "pacing", "rhythm", "tempo", "shorter shots", "longer shots"),
    "motion": ("zoom", "zooms", "pan", "pans", "movement", "motion", "punch-in", "punch in", "static", "ken burns"),
    "captions": ("caption", "captions", "subtitles", "highlight", "highlighted", "keyword", "keywords"),
    "text": ("text", "headline", "headlines", "lower third", "number cards", "titles"),
    "transitions": ("transition", "transitions", "dissolve", "fade", "fades", "cuts only", "hard cuts"),
    "audio": ("music", "ducking", "sound effects", "sfx", "silence", "pauses", "voice"),
    "density": ("busy", "dense", "minimal", "clean", "sparse", "information", "high-energy", "high energy", "dynamic"),
}


@dataclass
class OriginalityFinding:
    kind: str  # a key of KINDS
    label: str  # what would have been reproduced (generic wording, never the user's text)
    replacement: str  # the abstract instruction that took its place


@dataclass
class OriginalityResult:
    flagged: bool = False
    abstract_instruction: str = ""
    findings: list[OriginalityFinding] = field(default_factory=list)
    neutralised: list[str] = field(default_factory=list)  # human-readable: what was removed from the request
    kept: list[str] = field(default_factory=list)  # abstract wishes that passed unchanged
    hints: list[str] = field(default_factory=list)  # which style areas the wish touches ("pacing", "captions"...)
    notes: list[str] = field(default_factory=list)

    @property
    def kinds(self) -> list[str]:
        return sorted({f.kind for f in self.findings})


@dataclass
class OriginalityAudit:
    """Isolation check of a project: ``errors`` are violations of the no-reuse rule, ``warnings`` are things worth a look."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _has(text: str, words: tuple[str, ...]) -> bool:
    return any(re.search(r"(?<![\w-])" + re.escape(w) + r"(?![\w-])", text) for w in words)


class OriginalityGuard:
    # ------------------------------------------------------------------ requests
    def convert_request(self, text: str) -> OriginalityResult:
        """A free-text style wish -> abstract instructions. A clause that asks to reproduce specific content is replaced by the matching abstract instruction."""
        res = OriginalityResult()
        raw = (text or "").strip()
        if not raw:
            return res
        if URL.search(raw):
            res.notes.append("Links are never opened or downloaded: only a video file you import yourself is analysed.")
        replaced: list[str] = []
        kept: list[str] = []
        raw = URL.sub(" ", raw)  # a link is never followed or kept
        for clause in [c.strip(" ,:-") for c in SPLIT.split(raw) if c and c.strip(" ,:-")]:
            kinds = self._copy_kinds(clause)
            if not kinds:
                kept.append(self._clean(clause))
                res.hints += [area for area, words in STYLE_WORDS.items() if _has(clause.lower(), words) and area not in res.hints]
                continue
            for kind in kinds:
                label, replacement = KINDS[kind]
                if kind not in {f.kind for f in res.findings}:
                    res.findings.append(OriginalityFinding(kind, label, replacement))
                    res.neutralised.append(f"A request to reproduce {label} was not followed.")
                if replacement not in replaced:
                    replaced.append(replacement)
        res.flagged = bool(res.findings)
        res.kept = [k for k in kept if k and (not res.flagged or len(k.split()) >= 3)]  # a left-over "Use it." next to a neutralised request means nothing
        if res.flagged:
            res.abstract_instruction = " ".join([*res.kept, *replaced]).strip()
            res.notes.append("The reference is used for editing principles only; its footage, words, graphics, branding, music and exact sequence are never copied.")
        else:
            res.abstract_instruction = raw
        return res

    def _copy_kinds(self, clause: str) -> list[str]:
        """The kinds of content a clause asks to reproduce (empty when it is an abstract wish)."""
        low = ADJECTIVE.sub("", OWN.sub(" ", clause.lower()))  # "my video" is the user's own; "intro music" is music, not an opening
        quoted = bool(QUOTED.search(clause))
        hard, soft = _has(low, HARD_COPY), _has(low, SOFT_COPY)
        if not (hard or soft or quoted):
            return []
        found = [kind for kind, words in OBJECTS.items() if _has(low, words)]
        if "TEXT" in found and _has(low, LOOK) and not quoted:
            found[found.index("TEXT")] = "TYPOGRAPHY"  # "make the captions look the same" is about typography, not about the words
        if "TYPOGRAPHY" in found and "TEXT" in found:
            found.remove("TEXT")
        if quoted and not found:
            found = ["TEXT"]  # a quoted passage that is to be used is text
        elif quoted and "TEXT" not in found and "TYPOGRAPHY" not in found:
            found.append("TEXT")
        if not found and hard and _has(low, WHOLE):
            found = ["FOOTAGE"]  # "copy the whole thing"
        if "OPENING" in found and "SEQUENCE" in found:
            found.remove("SEQUENCE")  # an opening is a sequence of shots; the opening instruction covers it
        if "OPENING" in found and "FOOTAGE" in found and not _has(low, ("clip", "clips", "footage", "recording", "b-roll", "broll")):
            found.remove("FOOTAGE")  # "the opening scene": the opening instruction is enough
        return found

    @staticmethod
    def _clean(clause: str) -> str:
        """A kept clause: whitespace-normalised, quotes dropped (a quoted word is a request to use those exact words)."""
        out = QUOTED.sub("", clause)
        out = re.sub(r"\s+", " ", out).strip(" ,:-")
        return (out[0].upper() + out[1:] if out else out) + ("." if out and out[-1] not in ".!?" else "")

    # ------------------------------------------------------------------ isolation of the reference
    def audit_project(self, project: "Project") -> OriginalityAudit:
        """No reference video may be in the media library or on the timeline (spec 38). Path-based: an asset stored in ``references/`` or pointing at a reference file."""
        out = OriginalityAudit()
        if project.root is None:
            return out
        ref_files: set[Path] = set()
        sizes: dict[int, str] = {}
        for rid, a in project.reference_assets.items():
            try:
                ref_files.add(project.paths.resolve(a.path).resolve())
            except OSError:
                pass
            if a.size_bytes:
                sizes[a.size_bytes] = a.name
        refs_dir = (project.root / "references").resolve()
        bad_assets: set[str] = set()
        for asset in project.assets.all():
            try:
                p = project.asset_path(asset).resolve()
            except OSError:
                continue
            inside = refs_dir == p or refs_dir in p.parents
            if inside or p in ref_files:
                bad_assets.add(asset.id)
                out.errors.append(f"The media library asset “{asset.name}” is a reference video; references must stay separate from your production media.")
            elif asset.size_bytes and asset.size_bytes in sizes:
                out.warnings.append(f"“{asset.name}” has the same size as the reference “{sizes[asset.size_bytes]}”. If it is the same file, make sure you are allowed to use it.")
        for c in project.timeline.all_clips():
            if c.asset_id in bad_assets:
                out.errors.append(f"Timeline clip {c.id} uses a reference video.")
        return out

    def is_reference_path(self, project: "Project", path: Path) -> bool:
        """True when ``path`` is inside the project's ``references/`` folder or is one of the linked reference files."""
        if project.root is None:
            return False
        try:
            p = Path(path).resolve()
            refs = (project.root / "references").resolve()
            if refs == p or refs in p.parents:
                return True
            return any(project.paths.resolve(a.path).resolve() == p for a in project.reference_assets.values())
        except OSError:
            return False

    # ------------------------------------------------------------------ what leaves the analysis is abstract
    def audit_overrides(self, ov: EditingStrategyOverrides) -> list[str]:
        """Problems found in the overrides: every value must be a number or one of the known style ids / positions."""
        problems = []
        for name in PARAMETERS:
            v = getattr(ov, name)
            if v is None or isinstance(v, (int, float)):
                continue
            if name == "caption_style" and v in CAPTION_PRESETS:
                continue
            if name == "caption_position" and v in ("bottom", "center", "top"):
                continue
            problems.append(f"{name} holds a value that is not an abstract parameter.")
        for n in ov.notes:
            if len(n) > 400 or re.search(r"[\"“”„«»`]", n):
                problems.append("A note contains quoted text.")
        return problems

    def audit_profile(self, profile: ReferenceStyleProfile) -> list[str]:
        """Problems found in a style profile: it may hold measurements, classes and labels - no text of the reference, no per-event timeline, no file names."""
        problems = []
        data = profile.to_dict()
        banned = {"text", "words", "transcript", "caption_text", "content", "ocr", "frames", "frame_sample", "shots", "events", "path", "file", "filename"}

        def walk(node, trail: str) -> None:
            if isinstance(node, dict):
                for k, v in node.items():
                    if str(k).lower() in banned and ((isinstance(v, str) and v) or (isinstance(v, list) and v)):  # a number (a count of events) or a dict of aggregates is fine
                        problems.append(f"The profile contains a '{k}' field with content ({trail or 'root'}).")
                    walk(v, f"{trail}.{k}" if trail else str(k))
            elif isinstance(node, list):
                if len(node) > 200:
                    problems.append(f"The profile contains a long list ({trail}); it should hold aggregates only.")
                for v in node:
                    walk(v, trail)
            elif isinstance(node, str) and len(node) > 160:
                problems.append(f"The profile contains a long text value ({trail}).")

        walk(json.loads(json.dumps(data, default=str)), "")
        return problems
