'use strict'

/*
 * Preload bridge. Exposes a minimal, audited surface (window.viratra) to the
 * renderer with contextIsolation on - no direct Node or Python access leaks in.
 */

const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('viratra', {
  // Engine commands
  ping: () => ipcRenderer.invoke('viratra:request', 'ping', {}),
  analyze: (path, k) => ipcRenderer.invoke('viratra:request', 'analyze', { path, k }),
  catalog: () => ipcRenderer.invoke('viratra:request', 'catalog', {}),
  saree: (id, k) => ipcRenderer.invoke('viratra:request', 'saree', { id, k }),
  stats: () => ipcRenderer.invoke('viratra:request', 'stats', {}),
  predict: (saree) => ipcRenderer.invoke('viratra:request', 'predict', { saree }),

  // Desktop helpers
  pickImage: () => ipcRenderer.invoke('viratra:pickImage'),
  resolveImage: (relPath) => ipcRenderer.invoke('viratra:resolveImage', relPath),
})
