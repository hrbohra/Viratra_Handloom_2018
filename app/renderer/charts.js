'use strict'

/* Chart.js dashboards (Layer 10). Two charts driven by the engine's `stats`
   response: units sold per pattern, and monthly revenue for 2018. */

const VH_CHARTS = (() => {
  const GOLD = '#d4af37'
  const GRID = 'rgba(255,255,255,0.06)'
  const TICK = '#a892b8'
  let patternChart = null
  let revenueChart = null

  Chart.defaults.global.defaultFontColor = TICK
  Chart.defaults.global.defaultFontFamily =
    '-apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'

  function barGradient (ctx) {
    const g = ctx.createLinearGradient(0, 0, 0, 260)
    g.addColorStop(0, '#e8cf83')
    g.addColorStop(1, '#7a1f3d')
    return g
  }

  function renderPattern (canvas, rows) {
    const ctx = canvas.getContext('2d')
    if (patternChart) patternChart.destroy()
    patternChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: rows.map(r => r.pattern),
        datasets: [{
          label: 'Units sold',
          data: rows.map(r => r.units),
          backgroundColor: barGradient(ctx),
          borderColor: GOLD,
          borderWidth: 1,
          borderRadius: 6,
        }],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        legend: { display: false },
        scales: {
          xAxes: [{ gridLines: { color: 'transparent' }, ticks: { fontColor: TICK } }],
          yAxes: [{ gridLines: { color: GRID }, ticks: { beginAtZero: true, fontColor: TICK } }],
        },
      },
    })
  }

  function renderRevenue (canvas, series) {
    const ctx = canvas.getContext('2d')
    if (revenueChart) revenueChart.destroy()
    const grad = ctx.createLinearGradient(0, 0, 0, 260)
    grad.addColorStop(0, 'rgba(212,175,55,0.35)')
    grad.addColorStop(1, 'rgba(212,175,55,0.02)')
    revenueChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: series.map(s => s.month),
        datasets: [{
          label: 'Revenue (₹)',
          data: series.map(s => Math.round(s.revenue)),
          borderColor: GOLD,
          backgroundColor: grad,
          pointBackgroundColor: GOLD,
          pointRadius: 3,
          borderWidth: 2,
          fill: true,
          lineTension: 0.35,
        }],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        legend: { display: false },
        scales: {
          xAxes: [{ gridLines: { color: 'transparent' }, ticks: { fontColor: TICK } }],
          yAxes: [{
            gridLines: { color: GRID },
            ticks: {
              fontColor: TICK,
              callback: (v) => '₹' + (v / 1000).toFixed(0) + 'k',
            },
          }],
        },
      },
    })
  }

  return { renderPattern, renderRevenue }
})()
