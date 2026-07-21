'use strict'

/* Renderer logic. Talks to the engine exclusively through window.viratra
   (exposed by preload) and never touches Node or Python directly. */

const vh = window.viratra
const $ = (sel) => document.querySelector(sel)
const $$ = (sel) => Array.from(document.querySelectorAll(sel))

let CATALOG = []
let ACTIVE_FILTER = 'all'

// ---- Navigation -----------------------------------------------------------
function switchView (name) {
  $$('.nav-item').forEach(b => b.classList.toggle('active', b.dataset.view === name))
  $$('.view').forEach(v => v.classList.toggle('active', v.id === 'view-' + name))
  if (name === 'catalog' && CATALOG.length === 0) loadCatalog()
  if (name === 'dashboard') loadDashboard()
}
$$('.nav-item').forEach(b => b.addEventListener('click', () => switchView(b.dataset.view)))

// ---- Engine status --------------------------------------------------------
async function checkEngine () {
  try {
    const s = await vh.ping()
    $('#statusDot').className = 'dot ok'
    $('#statusText').textContent = `engine · ${s.search_backend}`
    $('#statusText').title = `embeddings: ${s.embedding_backend}\nclassifier: ${s.classifier}\nindex: ${s.index}`
  } catch (e) {
    $('#statusDot').className = 'dot err'
    $('#statusText').textContent = 'engine offline'
  }
}

// ---- Analyse --------------------------------------------------------------
const dropZone = $('#dropZone')
const previewImg = $('#previewImg')

async function analyzePath (absPath) {
  previewImg.hidden = false
  $('#dropInner').style.display = 'none'
  previewImg.src = await vh.resolveImage(absPath)

  $('#resultEmpty').innerHTML = '<span class="loader"></span>'
  $('#resultBody').hidden = true
  try {
    const res = await vh.analyze(absPath, 6)
    renderAnalysis(res)
  } catch (e) {
    $('#resultEmpty').textContent = 'Analysis failed: ' + e.message
  }
}

function renderAnalysis (res) {
  $('#resultEmpty').style.display = 'none'
  $('#resultBody').hidden = false

  // pattern
  if (res.pattern) {
    $('#patternName').textContent = res.pattern.label
    $('#patternConf').textContent = (res.pattern.confidence * 100).toFixed(1) + '% confidence'
    const ranked = Object.entries(res.pattern.scores).sort((a, b) => b[1] - a[1]).slice(0, 5)
    $('#patternScores').innerHTML = ranked.map(([k, v]) => `
      <div class="score">
        <span class="lbl">${k}</span>
        <span class="bar"><i style="width:${(v * 100).toFixed(0)}%"></i></span>
        <span class="val">${(v * 100).toFixed(0)}%</span>
      </div>`).join('')
  } else {
    $('#patternName').textContent = '-'
    $('#patternConf').textContent = 'classifier not built'
    $('#patternScores').innerHTML = ''
  }

  // palette
  $('#palette').innerHTML = res.colors.map(c => `
    <div class="swatch">
      <span class="chip" style="background:${c.hex}"></span>
      <span class="meta"><span class="hex">${c.hex}</span> · <span class="pct">${(c.fraction * 100).toFixed(0)}%</span></span>
    </div>`).join('')

  // similar
  const panel = $('#similarPanel')
  if (res.similar && res.similar.length) {
    panel.hidden = false
    $('#similarBackend').textContent = `${res.search_backend} · ${res.embedding_backend} embeddings`
    renderCards($('#similarGrid'), res.similar, true)
  } else {
    panel.hidden = true
  }
}

$('#pickBtn').addEventListener('click', async () => {
  const p = await vh.pickImage()
  if (p) analyzePath(p)
})
;['dragenter', 'dragover'].forEach(ev =>
  dropZone.addEventListener(ev, (e) => { e.preventDefault(); dropZone.classList.add('hot') }))
;['dragleave', 'drop'].forEach(ev =>
  dropZone.addEventListener(ev, (e) => { e.preventDefault(); dropZone.classList.remove('hot') }))
dropZone.addEventListener('drop', (e) => {
  const f = e.dataTransfer.files[0]
  if (f && f.path) analyzePath(f.path)
})

// ---- Cards ----------------------------------------------------------------
async function renderCards (container, items, withSim) {
  container.innerHTML = ''
  for (const s of items) {
    const url = await vh.resolveImage(s.image_path)
    const card = document.createElement('div')
    card.className = 'card'
    const sim = (withSim && s.similarity != null)
      ? `<span class="sim-tag">${(s.similarity * 100).toFixed(0)}% match</span>` : ''
    card.innerHTML = `
      ${sim}
      <img class="thumb" src="${url}" alt="${s.title}" />
      <div class="cap">
        <div class="t">${s.title}</div>
        <div class="s"><span class="badge">${s.pattern || '-'}</span><span>₹${Math.round(s.price || 0).toLocaleString()}</span></div>
      </div>`
    card.addEventListener('click', () => openSaree(s.id))
    container.appendChild(card)
  }
}

// ---- Catalog --------------------------------------------------------------
async function loadCatalog () {
  const res = await vh.catalog()
  CATALOG = res.sarees
  buildFilters()
  applyFilter()
}

function buildFilters () {
  const patterns = ['all', ...Array.from(new Set(CATALOG.map(s => s.pattern))).sort()]
  $('#catalogFilters').innerHTML = patterns.map(p =>
    `<button class="chipbtn ${p === ACTIVE_FILTER ? 'active' : ''}" data-f="${p}">${p}</button>`).join('')
  $$('#catalogFilters .chipbtn').forEach(b =>
    b.addEventListener('click', () => { ACTIVE_FILTER = b.dataset.f; buildFilters(); applyFilter() }))
}

function applyFilter () {
  const items = ACTIVE_FILTER === 'all'
    ? CATALOG : CATALOG.filter(s => s.pattern === ACTIVE_FILTER)
  renderCards($('#catalogGrid'), items, false)
}

// ---- Saree modal ----------------------------------------------------------
async function openSaree (id) {
  const modal = $('#modal')
  modal.hidden = false
  $('#modalBody').innerHTML = '<div style="height:200px;display:grid;place-items:center"><span class="loader"></span></div>'
  const s = await vh.saree(id, 6)
  const url = await vh.resolveImage(s.image_path)
  const colors = (s.dominant_colors || []).map(c =>
    `<span class="chip" title="${c.hex}" style="display:inline-block;width:24px;height:24px;border-radius:6px;border:1px solid #ffffff22;background:${c.hex}"></span>`).join(' ')

  $('#modalBody').innerHTML = `
    <div class="modal-head">
      <img src="${url}" alt="${s.title}" />
      <div>
        <span class="badge">${s.pattern}</span>
        <div class="modal-title">${s.title}</div>
        <dl class="kv">
          <dt>SKU</dt><dd>${s.sku}</dd>
          <dt>Region</dt><dd>${s.region}</dd>
          <dt>Fabric</dt><dd>${s.fabric}</dd>
          <dt>Price</dt><dd>₹${Math.round(s.price || 0).toLocaleString()}</dd>
        </dl>
        <div style="margin-top:12px"><div class="kicker">Palette</div>${colors}</div>
      </div>
    </div>
    <div class="panel-head"><h2>Visually similar</h2></div>
    <div class="grid cards" id="modalSimilar"></div>`
  renderCards($('#modalSimilar'), s.similar || [], true)
}
$('#modalClose').addEventListener('click', () => { $('#modal').hidden = true })
$('#modal').addEventListener('click', (e) => { if (e.target.id === 'modal') $('#modal').hidden = true })

// ---- Dashboard ------------------------------------------------------------
let dashLoaded = false
async function loadDashboard () {
  const st = await vh.stats()
  const totalUnits = st.sales_by_pattern.reduce((a, r) => a + r.units, 0)
  const totalRev = st.sales_by_pattern.reduce((a, r) => a + r.revenue, 0)
  const top = st.sales_by_pattern[0]

  $('#kpis').innerHTML = `
    <div class="kpi"><div class="n">${st.counts.sarees}</div><div class="l">Sarees indexed</div></div>
    <div class="kpi"><div class="n">${totalUnits.toLocaleString()}</div><div class="l">Units sold · 2018</div></div>
    <div class="kpi"><div class="n">₹${(totalRev / 100000).toFixed(1)}L</div><div class="l">Total revenue</div></div>
    <div class="kpi"><div class="n" style="text-transform:capitalize">${top ? top.pattern : '-'}</div><div class="l">Best-selling pattern</div></div>`

  VH_CHARTS.renderPattern($('#patternChart'), st.sales_by_pattern)
  VH_CHARTS.renderRevenue($('#revenueChart'), st.sales_timeseries)
  dashLoaded = true
}

// ---- Boot -----------------------------------------------------------------
checkEngine()

// Debug/automation hook (used by the headless screenshot pass).
window.__vh = { analyzePath, switchView, loadCatalog, loadDashboard }
