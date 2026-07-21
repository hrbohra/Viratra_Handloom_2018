'use strict'

/*
 * Layer 10 - the Node <-> Python bridge.
 *
 * Spawns the Python engine (scripts/serve.py) once and speaks the JSON
 * line-protocol over stdio via python-shell. Each request gets a monotonic
 * `rid`; responses are matched back to their pending promise by that id.
 */

const path = require('path')
const fs = require('fs')
const { PythonShell } = require('python-shell')

const ROOT = path.resolve(__dirname, '..', '..')

function resolvePython () {
  // Prefer the project virtualenv, fall back to a system interpreter.
  const candidates = [
    path.join(ROOT, '.venv', 'Scripts', 'python.exe'), // Windows
    path.join(ROOT, '.venv', 'bin', 'python'),          // macOS / Linux
  ]
  for (const c of candidates) {
    if (fs.existsSync(c)) return c
  }
  return process.platform === 'win32' ? 'python' : 'python3'
}

class PythonBridge {
  constructor () {
    this.rid = 0
    this.pending = new Map()
    this.ready = false
    this.shell = null
  }

  start () {
    if (this.shell) return
    this.shell = new PythonShell(path.join('scripts', 'serve.py'), {
      mode: 'json',
      pythonPath: resolvePython(),
      pythonOptions: ['-u'],
      cwd: ROOT,
      // stdout carries protocol JSON; stderr is human-readable diagnostics, so
      // keep it as raw text rather than letting json-mode try to parse it.
      stderrParser: (line) => line,
    })

    this.shell.on('message', (msg) => {
      const entry = this.pending.get(msg.rid)
      if (!entry) return
      this.pending.delete(msg.rid)
      if (msg.ok) entry.resolve(msg.data)
      else entry.reject(new Error(msg.error || 'engine error'))
    })

    this.shell.on('stderr', (line) => console.log('[engine]', line))
    this.shell.on('error', (err) => console.error('[engine:error]', err))
    this.shell.on('close', () => {
      this.shell = null
      for (const { reject } of this.pending.values()) {
        reject(new Error('engine closed'))
      }
      this.pending.clear()
    })
  }

  request (cmd, params = {}) {
    this.start()
    const rid = ++this.rid
    return new Promise((resolve, reject) => {
      this.pending.set(rid, { resolve, reject })
      this.shell.send(Object.assign({ rid, cmd }, params))
      setTimeout(() => {
        if (this.pending.has(rid)) {
          this.pending.delete(rid)
          reject(new Error('engine timeout for command: ' + cmd))
        }
      }, 60000)
    })
  }

  stop () {
    if (this.shell) {
      try { this.shell.end(() => {}) } catch (e) { /* ignore */ }
      this.shell = null
    }
  }
}

module.exports = { PythonBridge, ROOT }
