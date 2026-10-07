from .util import deep_merge

DEFAULT_SETTINGS = {
    "keys": {
        "anthropic": "",
        "pexels": "",
        "pixabay": "",
        "storyblocks": "",
        "storyblocks_secret": "",
    },
    "ai_model": "claude-sonnet-5-5",
    # Every source can be switched on/off and given a share (percent) of the scenes.
    "sources": {
        "pexels_video": {"enabled": True, "percent": 35},
        "pixabay_video": {"enabled": True, "percent": 15},
        "pexels_photo": {"enabled": True, "percent": 15},
        "pixabay_photo": {"enabled": False, "percent": 0},
        "youtube": {"enabled": False, "percent": 20, "clip_seconds": 6},
        "storyblocks": {"enabled": False, "percent": 15},
        "local": {"enabled": False, "percent": 0, "folder": ""},
    },
    "captions": {
        "enabled": True,
        "style": "pop",          # classic | karaoke | pop
        "position": "bottom",    # bottom | center | top
        "font": "Inter",
        "size": 64,
        "color": "#FFFFFF",
        "highlight": "#FFD400",
        "uppercase": False,
        "words_per_caption": 3,
    },
    "editing": {
        "scene_seconds": 3.5,     # target scene length (shorter = faster cutting)
        "zoom": True,
        "transitions": True,
        "overlays": True,
        "sfx": True,
        "grade": "cinematic",     # none | cinematic | warm | cool | bw
        "music": "",
        "music_volume": 0.12,
        "sfx_volume": 0.55,
        "sfx_folder": "",
    },
    "width": 1920,
    "height": 1080,
    "fps": 30,
}


def load_settings(overrides=None):
    return deep_merge(DEFAULT_SETTINGS, overrides or {})
