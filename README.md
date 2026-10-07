# AgentTool

Desktop AI video editor. Give it a **script** and a **voiceover**; it analyses both, finds visuals,
lets you review the picks, then edits the video (zoom/pan/shake, transitions, overlays, colour grade,
sound effects, music ducking, animated captions) and renders it with FFmpeg.

```
app/      Electron + React desktop UI (talks to the engine over stdio)
engine/   Python engine (stdlib only): transcribe -> plan -> find visuals -> rank -> render
```

## Flow
1. Paste script, choose voiceover (+ optional description of the video).
2. **Analyze**: word timing (faster-whisper if installed, else even timing), scene split, AI edit plan
   (Claude if an Anthropic key is set, else rule-based), visuals searched per scene and ranked
   (Claude looks at the thumbnails; else a heuristic).
3. **Review**: swap candidates, change zoom/transition/SFX/overlay per scene, search more.
4. **OK -> render**: effect segments -> xfade chain -> captions -> audio mix -> `output.mp4`.

## Settings
- Every visual source (Pexels/Pixabay video+photo, YouTube via yt-dlp, Storyblocks, your own folder)
  has an on/off switch and a percent share of the scenes.
- Captions: on/off, style (pop / karaoke / classic), position, size, colours.
- Editing: scene length, zoom, transitions, overlays, SFX, grade, music.

## Run
Requirements: Node 18+, Python 3.9+, FFmpeg (with libass). Optional: `pip install faster-whisper`, `yt-dlp`.
```
cd app && npm install && npm start      # desktop app
python -m agenttool analyze --script s.txt --audio v.mp3 --project out/p1   # CLI (from engine/)
python -m agenttool render --project out/p1
```
API keys (Pexels, Pixabay, Anthropic, Storyblocks) are entered in Settings and stored locally.
Storyblocks support is best-effort and untested (needs an API plan). Use YouTube clips only where you
have the rights or fair use applies.
