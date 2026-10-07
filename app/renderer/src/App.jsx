import React, { useEffect, useRef, useState } from 'react';
import Settings from './Settings.jsx';
import { DEFAULTS, merge, ZOOMS, TRANSITIONS, SFX, OVERLAYS, SOURCE_LABELS } from './defaults.js';

const api = window.api;
const fmt = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`;
const fileUrl = (p) => (p.startsWith('file://') ? p : `file://${p}`);

function Preview({ cand }) {
  if (!cand) return <div className="noprev">No visual found - a placeholder will be used. Try "Find more".</div>;
  if (cand.source === 'local' && cand.kind === 'video') return <video className="preview" src={fileUrl(cand.url)} muted loop autoPlay />;
  if (cand.thumb) return <img className="preview" src={cand.thumb} alt="" />;
  return <div className="noprev">{cand.kind}: {cand.title}</div>;
}

function Scene({ sc, onChange, onMore, busy }) {
  const cand = sc.candidates[sc.chosen];
  const fx = (k, v) => onChange({ ...sc, effects: { ...sc.effects, [k]: v || null } });
  const [q, setQ] = useState('');
  const [src, setSrc] = useState('');
  return (
    <div className="panel scene">
      <Preview cand={cand} />
      <div>
        <h4>Scene {sc.id + 1} <span className="meta">{fmt(sc.start)} - {fmt(sc.end)} ({(sc.end - sc.start).toFixed(1)}s)</span></h4>
        <div className="line">{sc.text}</div>
        <div className="meta">
          {cand && <><span className="chip">{SOURCE_LABELS[cand.source]}</span><span className="chip">{cand.kind}</span>{cand.credit}</>}
          {sc.reason && <div>AI pick: {sc.reason}</div>}
        </div>
        <div className="cands">
          {sc.candidates.map((c, i) => (
            <div key={c.cid} className={'cand' + (i === sc.chosen ? ' sel' : '')} title={c.title} onClick={() => onChange({ ...sc, chosen: i })}>
              {c.thumb ? <img src={c.thumb} alt="" /> : <span>{c.title}</span>}<b>{c.kind[0].toUpperCase()}</b>
            </div>
          ))}
        </div>
        <div className="fx">
          <label>zoom <select value={sc.effects.zoom} onChange={(e) => fx('zoom', e.target.value)}>{ZOOMS.map((z) => <option key={z}>{z}</option>)}</select></label>
          <label>transition <select value={sc.effects.transition} onChange={(e) => fx('transition', e.target.value)}>{TRANSITIONS.map((z) => <option key={z}>{z}</option>)}</select></label>
          <label>sfx <select value={sc.effects.sfx || ''} onChange={(e) => fx('sfx', e.target.value)}>{SFX.map((z) => <option key={z} value={z}>{z || 'none'}</option>)}</select></label>
          <label>overlay <select value={sc.effects.overlay || ''} onChange={(e) => fx('overlay', e.target.value)}>{OVERLAYS.map((z) => <option key={z} value={z}>{z || 'none'}</option>)}</select></label>
          <input placeholder="search words…" value={q} onChange={(e) => setQ(e.target.value)} style={{ width: 150 }} />
          <select value={src} onChange={(e) => setSrc(e.target.value)}>
            <option value="">source…</option>{Object.entries(SOURCE_LABELS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
          </select>
          <button disabled={busy || !src} onClick={() => onMore(sc.id, src, q)}>Find more</button>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [settings, setSettingsState] = useState(DEFAULTS);
  const [showSettings, setShowSettings] = useState(false);
  const [step, setStep] = useState('input'); // input | working | review | done
  const [script, setScript] = useState('');
  const [audio, setAudio] = useState('');
  const [context, setContext] = useState('');
  const [project, setProject] = useState('');
  const [plan, setPlan] = useState(null);
  const [prog, setProg] = useState({ pct: 0, msg: '' });
  const [err, setErr] = useState('');
  const [output, setOutput] = useState('');
  const [deps, setDeps] = useState(null);
  const [projects, setProjects] = useState([]);
  const saveTimer = useRef();

  useEffect(() => {
    api.getSettings().then((s) => setSettingsState(merge(DEFAULTS, s)));
    api.rpc('deps', {}).then(setDeps).catch((e) => setErr(String(e)));
    api.listProjects().then(setProjects);
    return api.onProgress((m) => setProg(m));
  }, []);

  const setSettings = (s) => { setSettingsState(s); clearTimeout(saveTimer.current); saveTimer.current = setTimeout(() => api.setSettings(s), 300); };
  const wrap = async (fn) => { setErr(''); try { return await fn(); } catch (e) { setErr(e.message); setStep(plan ? 'review' : 'input'); } };

  const analyze = () => wrap(async () => {
    if (!script.trim() || !audio) throw new Error('Add a script and a voiceover file first.');
    setStep('working'); setProg({ pct: 0, msg: 'Starting…' });
    const dir = await api.newProject();
    setProject(dir);
    const p = await api.rpc('analyze', { script, audio, project_dir: dir, context });
    setPlan(p); setStep('review');
  });
  const openProject = (dir) => wrap(async () => { const p = await api.rpc('load', { project_dir: dir }); setProject(dir); setPlan(p); setScript(p.script); setAudio(p.audio); setStep('review'); });
  const updateScene = (sc) => setPlan((p) => ({ ...p, scenes: p.scenes.map((s) => (s.id === sc.id ? sc : s)) }));
  const more = (id, source, q) => wrap(async () => {
    await api.rpc('save', { project_dir: project, plan });
    const r = await api.rpc('research', { project_dir: project, scene_id: id, source, queries: q ? [q] : null });
    if (r.errors.length) setErr(r.errors.join('\n'));
    updateScene({ ...r.scene, chosen: plan.scenes[id].chosen });
  });
  const render = () => wrap(async () => {
    setStep('working'); setProg({ pct: 0, msg: 'Preparing render…' });
    await api.rpc('save', { project_dir: project, plan });
    const r = await api.rpc('render', { project_dir: project });
    setOutput(r.output); setStep('done');
  });

  const missing = deps && (!deps.ffmpeg ? 'FFmpeg is not installed (required).' : '');
  return (
    <>
      <header>
        <h1>AgentTool</h1>
        <div className="steps">{[['input', '1 Script + voice'], ['working', '2 Analysis'], ['review', '3 Review visuals'], ['done', '4 Final video']].map(([k, l]) => <span key={k} className={step === k ? 'on' : ''}>{l}</span>)}</div>
        <div className="grow" />
        <button onClick={() => { setPlan(null); setStep('input'); }}>New</button>
        <button onClick={() => setShowSettings(true)}>Settings</button>
      </header>
      {showSettings && <Settings s={settings} setS={setSettings} api={api} onClose={() => setShowSettings(false)} />}
      <main>
        {missing && <div className="panel err">{missing}</div>}
        {err && <div className="panel err">{err}</div>}

        {step === 'input' && (
          <div className="grid2">
            <div className="panel">
              <h3>Script</h3>
              <textarea className="script" placeholder="Paste your script here…" value={script} onChange={(e) => setScript(e.target.value)} />
              <button onClick={async () => { const p = await api.pick('script'); if (p) setScript(await api.readText(p)); }}>Load .txt</button>
            </div>
            <div>
              <div className="panel">
                <h3>Voiceover</h3>
                <button onClick={async () => { const p = await api.pick('audio'); if (p) setAudio(p); }}>{audio || 'Choose audio file'}</button>
                <h3>About the video <small className="meta">(optional, helps the AI choose visuals)</small></h3>
                <textarea style={{ width: '100%', minHeight: 70 }} placeholder="e.g. Documentary about oceans, serious cinematic tone" value={context} onChange={(e) => setContext(e.target.value)} />
                <p className="meta">Tools found: ffmpeg {deps?.ffmpeg ? '✓' : '✗'} · yt-dlp {deps?.yt_dlp ? '✓' : '✗'} · whisper {deps?.faster_whisper ? '✓' : '✗ (even timing fallback)'} · AI {settings.keys.anthropic ? '✓' : '✗ (rule-based)'}</p>
                <button className="primary" onClick={analyze}>Analyze &amp; find visuals</button>
              </div>
              {projects.length > 0 && <div className="panel"><h3>Recent projects</h3>{projects.slice(0, 6).map((p) => <div key={p}><button onClick={() => openProject(p)}>{p.split(/[\\/]/).pop()}</button></div>)}</div>}
            </div>
          </div>
        )}

        {step === 'working' && <div className="panel"><h3>{prog.msg || 'Working…'}</h3><div className="progress"><i style={{ width: prog.pct + '%' }} /></div><span className="meta">{prog.stage} · {prog.pct}%</span></div>}

        {step === 'review' && plan && (
          <>
            <div className="panel">
              <b>{plan.scenes.length} scenes</b> · {fmt(plan.duration)} · timing: {plan.timing}. Check the picks, swap anything you don't like, then approve.
              <div style={{ marginTop: 10 }}><button className="primary" onClick={render}>OK - start editing &amp; render</button></div>
            </div>
            {plan.scenes.map((sc) => <Scene key={sc.id} sc={sc} onChange={updateScene} onMore={more} busy={false} />)}
          </>
        )}

        {step === 'done' && (
          <div className="panel">
            <h3 className="ok">Video ready</h3>
            <video src={fileUrl(output)} controls style={{ width: '100%', maxWidth: 960, borderRadius: 8 }} />
            <p><button onClick={() => api.reveal(output)}>Show in folder</button> <button onClick={() => setStep('review')}>Back to review</button></p>
            <div className="meta">{output}</div>
          </div>
        )}
      </main>
    </>
  );
}
