const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const readline = require('readline');

const userData = () => app.getPath('userData');
const settingsFile = () => path.join(userData(), 'settings.json');
const projectsDir = () => path.join(userData(), 'projects');
const engineDir = () => app.isPackaged ? path.join(process.resourcesPath, 'engine') : path.join(__dirname, '..', 'engine');

let win, py, nextId = 1;
const pending = new Map();

function startEngine() {
  const exe = process.env.AGENTTOOL_PYTHON || (process.platform === 'win32' ? 'python' : 'python3');
  py = spawn(exe, ['-m', 'agenttool', 'serve'], { cwd: engineDir(), env: { ...process.env, PYTHONUNBUFFERED: '1' } });
  readline.createInterface({ input: py.stdout }).on('line', (line) => {
    let msg;
    try { msg = JSON.parse(line); } catch { return; }
    if (msg.event === 'progress') return win && win.webContents.send('progress', msg);
    const p = pending.get(msg.id);
    if (!p) return;
    pending.delete(msg.id);
    msg.event === 'error' ? p.reject(new Error(msg.message)) : p.resolve(msg.data);
  });
  py.stderr.on('data', (d) => console.error('[engine]', d.toString()));
  py.on('exit', () => { for (const p of pending.values()) p.reject(new Error('engine stopped')); pending.clear(); py = null; });
  py.on('error', (e) => win && win.webContents.send('progress', { stage: 'error', pct: 0, msg: `Cannot start Python (${exe}): ${e.message}` }));
}

function rpc(cmd, args) {
  if (!py) startEngine();
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    py.stdin.write(JSON.stringify({ id, cmd, args }) + '\n');
  });
}

function readSettings() {
  try { return JSON.parse(fs.readFileSync(settingsFile(), 'utf8')); } catch { return {}; }
}

function createWindow() {
  win = new BrowserWindow({
    width: 1400, height: 900, backgroundColor: '#0f1117', title: 'AgentTool',
    webPreferences: { preload: path.join(__dirname, 'preload.js'), contextIsolation: true, webSecurity: false },
  });
  if (process.env.VITE_DEV) win.loadURL('http://localhost:5173');
  else win.loadFile(path.join(__dirname, 'renderer', 'dist', 'index.html'));
}

ipcMain.handle('settings:get', () => readSettings());
ipcMain.handle('settings:set', (_, s) => { fs.mkdirSync(userData(), { recursive: true }); fs.writeFileSync(settingsFile(), JSON.stringify(s, null, 2)); return true; });
ipcMain.handle('pick', async (_, kind) => {
  const filters = {
    audio: [{ name: 'Audio', extensions: ['mp3', 'wav', 'm4a', 'aac', 'ogg', 'flac'] }],
    script: [{ name: 'Text', extensions: ['txt', 'md'] }],
  }[kind];
  const r = await dialog.showOpenDialog(win, kind === 'folder' ? { properties: ['openDirectory'] } : { properties: ['openFile'], filters });
  return r.canceled ? null : r.filePaths[0];
});
ipcMain.handle('readText', (_, p) => fs.readFileSync(p, 'utf8'));
ipcMain.handle('project:new', () => {
  const dir = path.join(projectsDir(), new Date().toISOString().replace(/[:.]/g, '-'));
  fs.mkdirSync(dir, { recursive: true });
  return dir;
});
ipcMain.handle('project:list', () => {
  try { return fs.readdirSync(projectsDir()).filter((d) => fs.existsSync(path.join(projectsDir(), d, 'plan.json'))).sort().reverse().map((d) => path.join(projectsDir(), d)); } catch { return []; }
});
ipcMain.handle('reveal', (_, p) => shell.showItemInFolder(p));
ipcMain.handle('rpc', (_, cmd, args) => rpc(cmd, { ...args, settings: readSettings() }));

app.whenReady().then(createWindow);
app.on('window-all-closed', () => { if (py) py.kill(); app.quit(); });
