/* Utilitários compartilhados do CreditAI HUD */
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];

async function api(path, opts = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  let body = null;
  try { body = await res.json(); } catch (_) { body = null; }
  return { status: res.status, ok: res.ok, body };
}

function toast(msg, color = "var(--cyan)") {
  const t = document.createElement("div");
  t.className = "toast";
  t.style.borderColor = color;
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 2400);
}

function fmtK(v) { return v >= 1e6 ? (v / 1e6).toFixed(2) + "M" : v >= 1e3 ? (v / 1e3).toFixed(1) + "k" : (+v).toFixed(0); }

/** Sparkline em área com gradiente (estilo onda do HUD) */
function drawSpark(canvas, values, color) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  const vals = values.filter((v) => Number.isFinite(v));
  if (vals.length < 2) {
    // onda ociosa decorativa
    ctx.beginPath();
    for (let x = 0; x <= w; x += 4) ctx.lineTo(x, h * 0.62 + Math.sin(x / 26) * 5);
    ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
    ctx.fillStyle = color + "22"; ctx.fill();
    return;
  }
  const mn = Math.min(...vals), mx = Math.max(...vals), rg = mx - mn || 1;
  const pts = vals.map((v, i) => [(i / (vals.length - 1)) * w, h - 6 - ((v - mn) / rg) * (h - 14)]);
  const g = ctx.createLinearGradient(0, 0, 0, h);
  g.addColorStop(0, color + "66"); g.addColorStop(1, color + "05");
  ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) {
    const [x0, y0] = pts[i - 1], [x1, y1] = pts[i], cx = (x0 + x1) / 2;
    ctx.bezierCurveTo(cx, y0, cx, y1, x1, y1);
  }
  ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
  ctx.fillStyle = g; ctx.fill();
  ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) {
    const [x0, y0] = pts[i - 1], [x1, y1] = pts[i], cx = (x0 + x1) / 2;
    ctx.bezierCurveTo(cx, y0, cx, y1, x1, y1);
  }
  ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.shadowColor = color; ctx.shadowBlur = 8; ctx.stroke();
  const [lx, ly] = pts[pts.length - 1];
  ctx.beginPath(); ctx.arc(lx, ly, 2.6, 0, Math.PI * 2); ctx.fillStyle = "#fff"; ctx.fill();
}

/* Tema padrão dos gráficos Chart.js */
function chartDefaults() {
  if (!window.Chart) return;
  Chart.defaults.font.family = "'JetBrains Mono', monospace";
  Chart.defaults.font.size = 10;
  Chart.defaults.color = "#7f90b5";
  Chart.defaults.borderColor = "rgba(38,54,90,.45)";
  Chart.defaults.animation.duration = 250;
  Chart.defaults.plugins.legend.labels.boxWidth = 10;
  Chart.defaults.plugins.legend.labels.boxHeight = 10;
}

function lineChart(canvas, datasets, opts = {}) {
  return new Chart(canvas, {
    type: "line",
    data: { labels: [], datasets: datasets.map((d) => ({
      borderWidth: 2, pointRadius: 0, tension: 0.35, fill: d.fill ?? false, ...d,
    })) },
    options: {
      responsive: true, maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
      plugins: { legend: { position: "top" } },
      scales: { x: { grid: { display: false } }, y: { ...(opts.y || {}) } },
      ...opts.extra,
    },
  });
}

/** Painel flutuante arrastável */
function makeDraggable(el) {
  let sx, sy, ox, oy, drag = false;
  el.addEventListener("mousedown", (e) => { drag = true; sx = e.clientX; sy = e.clientY; const r = el.getBoundingClientRect(); ox = r.left; oy = r.top; });
  window.addEventListener("mousemove", (e) => {
    if (!drag) return;
    el.style.left = ox + e.clientX - sx + "px"; el.style.top = oy + e.clientY - sy + "px";
    el.style.right = "auto"; el.style.bottom = "auto";
  });
  window.addEventListener("mouseup", () => (drag = false));
}
