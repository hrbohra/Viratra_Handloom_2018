'use strict'

/*
 * Electron main process (Layer 1).
 *
 * Owns the application window and brokers every request from the renderer to
 * the Python engine over the JSON bridge. The renderer never touches Node or
 * Python directly - it calls window.viratra.* which is forwarded here via IPC.
 */

const path = require('path')
const { app, BrowserWindow, ipcMain, dialog } = require('electron')
const { PythonBridge, ROOT } = require('./ipc/python-bridge')

const bridge = new PythonBridge()
let win = null

function createWindow () {
  win = new BrowserWindow({
    width: 1280,
    height: 860,
    minWidth: 1024,
    minHeight: 700,
    backgroundColor: '#1a1220',
    title: 'Viratra Handloom',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  })
  win.setMenuBarVisibility(false)
  win.loadFile(path.join(__dirname, 'renderer', 'index.html'))

  // Surface renderer problems in the main-process log for easier debugging.
  win.webContents.on('console-message', (_e, level, message) => {
    if (level >= 2) console.log('[renderer]', message)
  })
  win.webContents.on('did-fail-load', (_e, code, desc) =>
    console.error('[renderer:load-fail]', code, desc))

  // Headless screenshot pass: VH_SHOT=1 drives each view and captures a PNG.
  if (process.env.VH_SHOT) {
    const fs = require('fs')
    const shotDir = path.join(ROOT, 'docs', 'screenshots')
    fs.mkdirSync(shotDir, { recursive: true })
    const wait = (ms) => new Promise(r => setTimeout(r, ms))
    const capture = async (name) => {
      const img = await win.capturePage()
      fs.writeFileSync(path.join(shotDir, name + '.png'), img.toPNG())
      console.log('[shot]', name)
    }
    win.webContents.on('did-finish-load', async () => {
      try {
        await wait(800)
        await win.webContents.executeJavaScript(
          'window.__vh.analyzePath("' + path.join(ROOT, 'data/samples/temple/temple_04.png').replace(/\\/g, '/') + '")')
        await wait(2500)
        await capture('analyse')
        await win.webContents.executeJavaScript('window.__vh.switchView("catalog")')
        await wait(2000)
        await capture('catalog')
        await win.webContents.executeJavaScript('window.__vh.switchView("dashboard")')
        await wait(1800)
        await capture('dashboard')
      } catch (err) {
        console.error('[shot] FAIL', err && err.message)
      } finally {
        setTimeout(() => app.quit(), 400)
      }
    })
  }

  // Headless smoke check: VH_SMOKE=1 loads the UI, exercises a call, quits.
  if (process.env.VH_SMOKE) {
    win.webContents.on('did-finish-load', async () => {
      try {
        const out = await win.webContents.executeJavaScript(
          '(async () => { const s = await window.viratra.stats();' +
          ' const a = await window.viratra.analyze("data/samples/paisley/paisley_01.png", 3);' +
          ' return { sarees: s.counts.sarees, pattern: a.pattern.label,' +
          ' similar: a.similar.length }; })()')
        console.log('[smoke] OK', JSON.stringify(out))
      } catch (err) {
        console.error('[smoke] FAIL', err && err.message)
      } finally {
        setTimeout(() => app.quit(), 400)
      }
    })
  }
}

// ---- IPC: renderer -> main -> Python engine -------------------------------
ipcMain.handle('viratra:request', async (_evt, cmd, params) => {
  return bridge.request(cmd, params)
})

ipcMain.handle('viratra:pickImage', async () => {
  const res = await dialog.showOpenDialog(win, {
    title: 'Choose a saree photograph',
    properties: ['openFile'],
    filters: [{ name: 'Images', extensions: ['png', 'jpg', 'jpeg', 'bmp', 'webp'] }],
  })
  return res.canceled ? null : res.filePaths[0]
})

// Resolve a catalog-relative image path to a file:// URL for the renderer.
ipcMain.handle('viratra:resolveImage', async (_evt, relPath) => {
  const abs = path.isAbsolute(relPath) ? relPath : path.join(ROOT, relPath)
  return 'file://' + abs.replace(/\\/g, '/')
})

app.whenReady().then(createWindow)

app.on('window-all-closed', () => {
  bridge.stop()
  if (process.platform !== 'darwin') app.quit()
})

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow()
})
