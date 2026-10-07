"""ASS caption generation (classic / karaoke / pop styles)."""


def _ass_color(hex_):
    h = hex_.lstrip("#")
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H00{b}{g}{r}".upper()


def _t(sec):
    sec = max(sec, 0)
    h, m = int(sec // 3600), int(sec % 3600 // 60)
    return f"{h}:{m:02d}:{sec % 60:05.2f}"


def _chunks(words, max_words):
    out, cur = [], []
    for w in words:
        cur.append(w)
        end_punct = w["w"].endswith((".", "!", "?", ",", ";", ":"))
        if len(cur) >= max_words or (end_punct and len(cur) >= 2):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def build_ass(words, cfg, width, height):
    style = cfg["style"]
    size = int(cfg["size"] * height / 1080)
    align = {"bottom": 2, "center": 5, "top": 8}[cfg["position"]]
    margin_v = int(height * (0.09 if cfg["position"] != "center" else 0))
    prim, hi = _ass_color(cfg["color"]), _ass_color(cfg["highlight"])
    head = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {width}\nPlayResY: {height}\nWrapStyle: 2\n\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Main,{cfg['font']},{size},{prim},{hi},&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,"
        f"{max(3, size // 14)},{max(1, size // 30)},{align},{int(width * 0.08)},{int(width * 0.08)},{margin_v},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    def txt(w):
        t = w["w"].upper() if cfg.get("uppercase") or style == "pop" else w["w"]
        return t.replace("{", "(").replace("}", ")")

    lines = []
    n = 2 if style == "pop" and cfg["words_per_caption"] < 2 else cfg["words_per_caption"]
    if style == "classic":
        n = max(n, 6)
    elif style == "karaoke":
        n = max(n, 5)
    chunks = _chunks(words, n)
    for ci, ch in enumerate(chunks):
        start = ch[0]["start"]
        nxt = chunks[ci + 1][0]["start"] if ci + 1 < len(chunks) else ch[-1]["end"] + 0.4
        end = min(max(ch[-1]["end"], start + 0.3) + 0.15, nxt)
        if style == "classic":
            lines.append(f"Dialogue: 0,{_t(start)},{_t(end)},Main,,0,0,0,,{' '.join(txt(w) for w in ch)}")
        elif style == "karaoke":
            parts = []
            for w in ch:
                cs = max(int(round((w["end"] - w["start"]) * 100)), 1)
                parts.append(f"{{\\kf{cs}}}{txt(w)}")
            lines.append(f"Dialogue: 0,{_t(start)},{_t(end)},Main,,0,0,0,,{' '.join(parts)}")
        else:  # pop: whole chunk shown, active word highlighted + bumped, chunk pops in
            for wi, w in enumerate(ch):
                s = w["start"]
                e = ch[wi + 1]["start"] if wi + 1 < len(ch) else end
                parts = []
                for wj, x in enumerate(ch):
                    if wj == wi:
                        parts.append(f"{{\\c{hi}\\fscx118\\fscy118}}{txt(x)}{{\\r}}")
                    else:
                        parts.append(txt(x))
                pop = "{\\fscx80\\fscy80\\t(0,90,\\fscx100\\fscy100)}" if wi == 0 else ""
                lines.append(f"Dialogue: 0,{_t(s)},{_t(max(e, s + 0.05))},Main,,0,0,0,,{pop}{' '.join(parts)}")
    return head + "\n".join(lines) + "\n"
