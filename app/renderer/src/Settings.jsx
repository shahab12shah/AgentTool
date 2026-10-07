import React from 'react';
import { SOURCE_LABELS } from './defaults.js';

const Row = ({ label, children }) => <label className="row"><span>{label}</span>{children}</label>;

export default function Settings({ s, setS, api, onClose }) {
  const set = (path, value) => {
    const next = structuredClone(s);
    let o = next;
    path.slice(0, -1).forEach((k) => (o = o[k]));
    o[path.at(-1)] = value;
    setS(next);
  };
  const pickFile = async (path, kind) => { const p = await api.pick(kind); if (p) set(path, p); };
  const enabled = Object.entries(s.sources).filter(([, v]) => v.enabled);
  const total = enabled.reduce((a, [, v]) => a + (v.percent || 0), 0);

  return (
    <div className="drawer">
      <div className="drawer-head"><h2>Settings</h2><button onClick={onClose}>Done</button></div>

      <h3>Visual sources <small>on/off + share of scenes</small></h3>
      {Object.entries(s.sources).map(([id, v]) => (
        <div className="source" key={id}>
          <label className="switch"><input type="checkbox" checked={v.enabled} onChange={(e) => set(['sources', id, 'enabled'], e.target.checked)} />{SOURCE_LABELS[id]}</label>
          <input type="range" min="0" max="100" step="5" disabled={!v.enabled} value={v.percent} onChange={(e) => set(['sources', id, 'percent'], +e.target.value)} />
          <span className="pct">{v.enabled && total ? Math.round((100 * v.percent) / total) : 0}%</span>
          {id === 'local' && v.enabled && <button onClick={() => pickFile(['sources', id, 'folder'], 'folder')}>{v.folder || 'Choose folder'}</button>}
          {id === 'youtube' && v.enabled && <Row label="clip sec"><input type="number" min="2" max="20" value={v.clip_seconds} onChange={(e) => set(['sources', id, 'clip_seconds'], +e.target.value)} /></Row>}
        </div>
      ))}
      <p className="hint">Percentages are shares among the enabled sources. If a source finds nothing for a scene, the others fill in. Only use YouTube clips you have the rights to (or that fall under fair use).</p>

      <h3>Captions</h3>
      <Row label="Add captions"><input type="checkbox" checked={s.captions.enabled} onChange={(e) => set(['captions', 'enabled'], e.target.checked)} /></Row>
      <Row label="Style">
        <select value={s.captions.style} onChange={(e) => set(['captions', 'style'], e.target.value)}>
          <option value="pop">Pop (word highlight, 2-3 words)</option><option value="karaoke">Karaoke (sweep)</option><option value="classic">Classic subtitles</option>
        </select>
      </Row>
      <Row label="Position">
        <select value={s.captions.position} onChange={(e) => set(['captions', 'position'], e.target.value)}>
          <option value="bottom">Bottom</option><option value="center">Center</option><option value="top">Top</option>
        </select>
      </Row>
      <Row label="Font size"><input type="number" min="24" max="140" value={s.captions.size} onChange={(e) => set(['captions', 'size'], +e.target.value)} /></Row>
      <Row label="Words per caption"><input type="number" min="1" max="8" value={s.captions.words_per_caption} onChange={(e) => set(['captions', 'words_per_caption'], +e.target.value)} /></Row>
      <Row label="Text colour"><input type="color" value={s.captions.color} onChange={(e) => set(['captions', 'color'], e.target.value)} /></Row>
      <Row label="Highlight colour"><input type="color" value={s.captions.highlight} onChange={(e) => set(['captions', 'highlight'], e.target.value)} /></Row>
      <Row label="UPPERCASE"><input type="checkbox" checked={s.captions.uppercase} onChange={(e) => set(['captions', 'uppercase'], e.target.checked)} /></Row>

      <h3>Editing</h3>
      <Row label="Scene length (sec) - lower = faster cuts"><input type="number" step="0.5" min="1.5" max="10" value={s.editing.scene_seconds} onChange={(e) => set(['editing', 'scene_seconds'], +e.target.value)} /></Row>
      {['zoom', 'transitions', 'overlays', 'sfx'].map((k) => (
        <Row key={k} label={{ zoom: 'Zoom / pan / shake', transitions: 'Transitions', overlays: 'Overlays (grain, vignette, flash)', sfx: 'Sound effects' }[k]}>
          <input type="checkbox" checked={s.editing[k]} onChange={(e) => set(['editing', k], e.target.checked)} />
        </Row>
      ))}
      <Row label="Colour grade">
        <select value={s.editing.grade} onChange={(e) => set(['editing', 'grade'], e.target.value)}>
          {['none', 'cinematic', 'warm', 'cool', 'bw'].map((g) => <option key={g}>{g}</option>)}
        </select>
      </Row>
      <Row label="SFX volume"><input type="range" min="0" max="1" step="0.05" value={s.editing.sfx_volume} onChange={(e) => set(['editing', 'sfx_volume'], +e.target.value)} /></Row>
      <Row label="Background music"><button onClick={() => pickFile(['editing', 'music'], 'audio')}>{s.editing.music || 'Choose file (optional)'}</button></Row>
      <Row label="Music volume"><input type="range" min="0" max="0.6" step="0.02" value={s.editing.music_volume} onChange={(e) => set(['editing', 'music_volume'], +e.target.value)} /></Row>
      <Row label="Own SFX folder"><button onClick={() => pickFile(['editing', 'sfx_folder'], 'folder')}>{s.editing.sfx_folder || 'Choose folder (optional)'}</button></Row>
      <p className="hint">Name files whoosh*, hit*, click*, riser* to have them used in place of the built-in sounds.</p>

      <h3>API keys <small>stored only on this computer</small></h3>
      {[['anthropic', 'Anthropic (AI planning + picking)'], ['pexels', 'Pexels'], ['pixabay', 'Pixabay'], ['storyblocks', 'Storyblocks API key'], ['storyblocks_secret', 'Storyblocks secret']].map(([k, label]) => (
        <Row key={k} label={label}><input type="password" value={s.keys[k]} onChange={(e) => set(['keys', k], e.target.value)} /></Row>
      ))}
    </div>
  );
}
