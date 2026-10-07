const { contextBridge, ipcRenderer } = require('electron');
contextBridge.exposeInMainWorld('api', {
  getSettings: () => ipcRenderer.invoke('settings:get'),
  setSettings: (s) => ipcRenderer.invoke('settings:set', s),
  pick: (kind) => ipcRenderer.invoke('pick', kind),
  readText: (p) => ipcRenderer.invoke('readText', p),
  newProject: () => ipcRenderer.invoke('project:new'),
  listProjects: () => ipcRenderer.invoke('project:list'),
  reveal: (p) => ipcRenderer.invoke('reveal', p),
  rpc: (cmd, args) => ipcRenderer.invoke('rpc', cmd, args),
  onProgress: (fn) => { const h = (_, m) => fn(m); ipcRenderer.on('progress', h); return () => ipcRenderer.removeListener('progress', h); },
});
