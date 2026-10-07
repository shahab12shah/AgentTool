"""Synthesised sound-effects (no asset downloads needed) + user sfx folder support."""
import glob
import os
import random

from .util import run

RECIPES = {
    "whoosh": ["-f", "lavfi", "-i", "anoisesrc=d=0.9:c=pink:a=0.8", "-af",
               "bandpass=f=900:w=1.2,tremolo=f=3:d=0.3,afade=t=in:d=0.35,afade=t=out:st=0.35:d=0.55,volume=2.2"],
    "hit": ["-f", "lavfi", "-i", "sine=f=55:d=0.7", "-af",
            "afade=t=out:st=0.05:d=0.65,volume=3,aecho=0.7:0.6:60:0.3"],
    "click": ["-f", "lavfi", "-i", "anoisesrc=d=0.12:c=white:a=0.9", "-af",
              "highpass=f=2500,afade=t=out:st=0.01:d=0.1,volume=1.5"],
    "riser": ["-f", "lavfi", "-i", "anoisesrc=d=1.6:c=pink:a=0.7", "-af",
              "bandpass=f=1500:w=2,afade=t=in:d=1.5,volume=2.2,afade=t=out:st=1.5:d=0.1"],
}


def ensure_sfx(dest_dir, user_folder=""):
    os.makedirs(dest_dir, exist_ok=True)
    paths = {}
    for name, args in RECIPES.items():
        p = os.path.join(dest_dir, name + ".wav")
        if not os.path.exists(p):
            run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args, "-ar", "44100", "-ac", "2", p])
        paths[name] = [p]
    if user_folder and os.path.isdir(user_folder):
        for f in glob.glob(os.path.join(user_folder, "**", "*.*"), recursive=True):
            if f.lower().endswith((".wav", ".mp3", ".ogg", ".flac", ".m4a")):
                base = os.path.basename(f).lower()
                for name in RECIPES:
                    if name in base:
                        paths[name].append(f)
    return paths


def pick(paths, name, rng=random):
    opts = paths.get(name) or []
    return rng.choice(opts) if opts else None
