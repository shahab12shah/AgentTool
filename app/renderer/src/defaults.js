// Mirrors engine/agenttool/config.py (the engine merges whatever we send over its own defaults).
export const DEFAULTS = {
  keys: { anthropic: '', pexels: '', pixabay: '', storyblocks: '', storyblocks_secret: '' },
  sources: {
    pexels_video: { enabled: true, percent: 35 },
    pixabay_video: { enabled: true, percent: 15 },
    pexels_photo: { enabled: true, percent: 15 },
    pixabay_photo: { enabled: false, percent: 0 },
    youtube: { enabled: false, percent: 20, clip_seconds: 6 },
    storyblocks: { enabled: false, percent: 15 },
    local: { enabled: false, percent: 0, folder: '' },
  },
  captions: { enabled: true, style: 'pop', position: 'bottom', font: 'Inter', size: 64, color: '#FFFFFF', highlight: '#FFD400', uppercase: false, words_per_caption: 3 },
  editing: { scene_seconds: 3.5, zoom: true, transitions: true, overlays: true, sfx: true, grade: 'cinematic', music: '', music_volume: 0.12, sfx_volume: 0.55, sfx_folder: '' },
};

export const SOURCE_LABELS = {
  pexels_video: 'Pexels videos', pixabay_video: 'Pixabay videos', pexels_photo: 'Pexels photos',
  pixabay_photo: 'Pixabay images', youtube: 'YouTube clips (yt-dlp)', storyblocks: 'Storyblocks', local: 'My own library folder',
};
export const ZOOMS = ['zoom_in', 'zoom_out', 'punch_in', 'pan_left', 'pan_right', 'drift', 'shake'];
export const TRANSITIONS = ['hardcut', 'fade', 'dissolve', 'fadeblack', 'slideleft', 'slideright', 'wipeleft', 'circleopen', 'zoomin', 'smoothleft', 'radial', 'pixelize'];
export const SFX = ['', 'whoosh', 'hit', 'click', 'riser'];
export const OVERLAYS = ['', 'grain', 'vignette', 'flash'];

export function merge(base, over) {
  const out = { ...base };
  for (const [k, v] of Object.entries(over || {})) {
    out[k] = v && typeof v === 'object' && !Array.isArray(v) && typeof base[k] === 'object' ? merge(base[k], v) : v;
  }
  return out;
}
