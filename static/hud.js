/* =====================================================================
   CreditAI — HUD de Treinamento (WebSocket /api/ws + controles REST)
   ===================================================================== */
chartDefaults();

const CARDS = [
  { id: "throughput", key: "throughput", title: "THROUGHPUT", ic: "⚡", cls: "c-teal", color: "#14b8a6", unit: "amostras/s",
    fmt: (v) => fmtK(v), bar: (v, mx) => v / (mx || 1) },
  { id: "auc", key: "auc_val", title: "ROC-AUC", ic: "🎓", cls: "c-amber", color: "#f59e0b", unit: "validação",
    fmt: (v) => v.toFixed(4), bar: (v) => (v - 0.5) / 0.5, best: "max" },
  { id: "ks", key: "ks", title: "KS", ic: "🎯", cls: "c-green", color: "#10b981", unit: "separação",
    fmt: (v) => v.toFixed(3), bar: (v) => v / 0.7, best: "max" },
  { id: "loss", key: "loss_val", title: "PERDA", ic: "📉", cls: "c-violet", color: "#a855f7", unit: "LogLoss",
    fmt: (v) => v.toFixed(4), bar: (v, mx) => v / (mx || 1), best: "min" },
  { id: "ece", key: "ece", title: "CALIBRAÇÃO", ic: "🧠", cls: "c-rose", color: "#f43f5e", unit: "ECE",
    fmt: (v) => v.toFixed(3), bar: (v) => Math.max(0, 1 - v * 6), best: "min" },
  { id: "gini", key: "gini", title: "GINI", ic: "⚖️", cls: "c-indigo", color: "#818cf8", unit: "2·AUC−1",
    fmt: (v) => v.toFixed(3), bar: (v) => v, best: "max" },
];

$("#cards").innerHTML = CARDS.map((c) => `
  <div class="mcard ${c.cls}" id="card-${c.id}">
    <canvas class="spark"></canvas>
    <div class="head">
      <span class="chip"><span class="ic">${c.ic}</span>${c.title}</span>
      <div class="maxtag">${c.best === "min" ? "MIN ▾" : "MAX ▴"} <b data-r="max">—</b><br><span class="tag" data-r="tag">—</span></div>
    </div>
    <div class="big"><span data-r="val">—</span><small>${c.unit}</small></div>
    <div class="bar"><i data-r="bar"></i></div>
    <div class="foot"><span>Atual: <b data-r="cur">—</b></span><span>Média: <b data-r="avg">—</b></span></div>
  </div>`).join("");

/* ---------------- gráficos ---------------- */
const chAuc = lineChart($("#chAuc"), [
  { label: "AUC treino", borderColor: "#f59e0b", backgroundColor: "#f59e0b" },
  { label: "AUC validação", borderColor: "#22d3ee", backgroundColor: "rgba(34,211,238,.12)", fill: true },
]);
const chLoss = lineChart($("#chLoss"), [
  { label: "LogLoss treino", borderColor: "#c084fc", backgroundColor: "#c084fc" },
  { label: "LogLoss validação", borderColor: "#f472b6", backgroundColor: "rgba(244,114,182,.12)", fill: true },
]);
const chCal = lineChart($("#chCal"), [
  { label: "ECE", borderColor: "#f43f5e", backgroundColor: "#f43f5e", yAxisID: "y" },
  { label: "KS", borderColor: "#10b981", backgroundColor: "#10b981", yAxisID: "y1", borderDash: [5, 4] },
], { extra: { scales: { x: { grid: { display: false } }, y: { position: "left" }, y1: { position: "right", grid: { display: false } } } } });
const chRel = new Chart($("#chRel"), {
  type: "scatter",
  data: { datasets: [
    { label: "Ideal", data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], showLine: true, borderColor: "#4f5f82", borderDash: [4, 4], pointRadius: 0 },
    { label: "Modelo", data: [], showLine: true, borderColor: "#f59e0b", backgroundColor: "#facc15", pointRadius: 3, tension: 0.2 },
  ] },
  options: { responsive: true, maintainAspectRatio: false,
    scales: { x: { min: 0, max: 1, title: { display: true, text: "previsto" } }, y: { min: 0, max: 1, title: { display: true, text: "observado" } } } },
});

/* ---------------- estado ---------------- */
let histLen = -1, lastStatus = "", filtro = "ALL", allLogs = [], cursor = 0, lastSnap = null;

function renderCards(hist) {
  for (const c of CARDS) {
    const el = $(`#card-${c.id}`);
    const vals = hist.map((h) => h[c.key]);
    drawSpark($("canvas", el), vals, c.color);
    if (!vals.length) continue;
    const cur = vals[vals.length - 1];
    const mx = Math.max(...vals), mn = Math.min(...vals);
    const best = c.best === "min" ? mn : mx;
    const bestIdx = vals.indexOf(best);
    const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
    $('[data-r="val"]', el).textContent = c.fmt(cur);
    $('[data-r="cur"]', el).textContent = c.fmt(cur);
    $('[data-r="avg"]', el).textContent = c.fmt(avg);
    $('[data-r="max"]', el).textContent = c.fmt(c.best ? best : mx);
    $('[data-r="tag"]', el).textContent = `iter ${hist[bestIdx >= 0 ? bestIdx : 0].passo}`;
    $('[data-r="bar"]', el).style.width = Math.max(0, Math.min(1, c.bar(cur, mx))) * 100 + "%";
  }
}

function renderCharts(hist) {
  const labels = hist.map((h) => h.passo);
  chAuc.data.labels = chLoss.data.labels = chCal.data.labels = labels;
  chAuc.data.datasets[0].data = hist.map((h) => h.auc_treino);
  chAuc.data.datasets[1].data = hist.map((h) => h.auc_val);
  chLoss.data.datasets[0].data = hist.map((h) => h.loss_treino);
  chLoss.data.datasets[1].data = hist.map((h) => h.loss_val);
  chCal.data.datasets[0].data = hist.map((h) => h.ece);
  chCal.data.datasets[1].data = hist.map((h) => h.ks);
  chAuc.update(); chLoss.update(); chCal.update();
}

function escapeHtml(s) { return s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]); }
const ICON = { TREINO: "", VALID: "🔬 ", CHECKPOINT: "💾 ", SISTEMA: "🌐 ", ERRO: "⛔ " };

function logHtml(l) {
  let m = escapeHtml(l.msg)
    .replace(/(AUC val: |AUC: |KS: |Gini: )([\d.]+)/g, '$1<span class="hl">$2</span>')
    .replace(/(Novo pico[^→]*)/, '<span class="hl">🏆 $1</span>');
  return `<div class="logline ${l.canal}"><span class="ts">[${l.ts}]</span>${ICON[l.canal] || ""}${m}</div>`;
}

function passa(l) {
  const q = $("#logFilter").value.trim().toLowerCase();
  return (filtro === "ALL" || l.canal === filtro) && (!q || l.msg.toLowerCase().includes(q));
}

function renderLogsFull() {
  const vis = allLogs.filter(passa);
  $("#logbox").innerHTML = vis.map(logHtml).join("");
  $("#lineCount").textContent = `${vis.length} / ${allLogs.length} linhas`;
  if ($("#autoScroll").checked) $("#logbox").scrollTop = 1e9;
}

function appendLogs(novos) {
  if (!novos.length) return;
  allLogs.push(...novos);
  if (allLogs.length > 2000) allLogs = allLogs.slice(-2000);
  const box = $("#logbox");
  box.insertAdjacentHTML("beforeend", novos.filter(passa).map(logHtml).join(""));
  $("#lineCount").textContent = `${allLogs.filter(passa).length} / ${allLogs.length} linhas`;
  if ($("#autoScroll").checked) box.scrollTop = 1e9;
}

function renderStatus(s) {
  const pill = $("#statusPill");
  pill.className = "pill " + s.status;
  $("#statusTxt").textContent = s.status;
  const running = s.status === "RUNNING" || s.status === "STOPPING";
  $("#btnStart").disabled = running;
  $("#btnStop").disabled = !running;
  $("#termState").textContent = running ? "TREINANDO" : "PRONTO";
  $("#progBar").style.width = (s.progresso * 100).toFixed(1) + "%";
  $("#hDur").textContent = `duração: ${s.duracao_s.toFixed(1)}s • ${s.checkpoints.length} checkpoints`;
  $("#hData").textContent = fmtK(s.config.n_amostras);
  const h = s.historico[s.historico.length - 1];
  $("#fStatus").textContent = s.status;
  $("#fIter").textContent = h ? `${h.passo}/${h.max_iter}` : "—";
  $("#fAuc").textContent = h ? h.auc_val.toFixed(4) : "—";
  $("#fKs").textContent = h ? h.ks.toFixed(3) : "—";
  $("#fPico").textContent = s.pico.auc_val ? `${s.pico.auc_val} @${s.pico.passo}` : "—";
}

function onSnapshot(s) {
  lastSnap = s;
  renderStatus(s);
  if (s.historico.length !== histLen) {
    histLen = s.historico.length;
    renderCards(s.historico);
    renderCharts(s.historico);
  }
  if (s.status !== lastStatus) {
    if (lastStatus === "RUNNING" && ["COMPLETED", "STOPPED"].includes(s.status)) {
      toast("🏁 Treino concluído — modelo hot-swap na API", "var(--green)");
      loadReport();
    }
    lastStatus = s.status;
  }
}

/* ---------------- relatório ---------------- */
async function loadReport() {
  const r = await api("/api/report");
  if (!r.ok) return;
  const rep = r.body, M = rep.metricas;
  $("#repMeta").textContent = `${rep.modelo.nome} v${rep.modelo.versao} • iter ${rep.modelo.melhor_iteracao} • ${rep.modelo.treinado_em}`;
  const cols = [["roc_auc", "ROC-AUC"], ["ks", "KS"], ["gini", "Gini"], ["log_loss", "LogLoss"], ["brier", "Brier"], ["ece", "ECE"], ["acuracia", "Acurácia"], ["f1_score", "F1"], ["taxa_aprovacao", "Aprovação"]];
  const parts = [["treino", "Treino (70%)"], ["validacao", "Validação (15%)"], ["teste_sem_calibracao", "Teste — bruto"], ["teste", "Teste — calibrado"]];
  $("#repTable").innerHTML = `<tr><th>Partição</th><th>n</th>${cols.map((c) => `<th>${c[1]}</th>`).join("")}</tr>` +
    parts.map(([k, nome]) => `<tr class="${k === "teste" ? "best" : ""}"><td>${nome}</td><td>${M[k].amostras}</td>${cols.map((c) => `<td>${M[k][c[0]]}</td>`).join("")}</tr>`).join("");
  const cm = M.teste.matriz_confusao;
  $("#cmBox").innerHTML = `
    <div></div><div class="lab">Prev. Reprov.</div><div class="lab">Prev. Aprov.</div>
    <div class="lab">Real Reprov.</div><div class="cell" style="background:rgba(16,185,129,.22);color:#6ee7b7">${cm[0][0]}</div><div class="cell" style="background:rgba(244,63,94,.2);color:#fda4af">${cm[0][1]}</div>
    <div class="lab">Real Aprov.</div><div class="cell" style="background:rgba(245,158,11,.18);color:#fcd34d">${cm[1][0]}</div><div class="cell" style="background:rgba(16,185,129,.32);color:#6ee7b7">${cm[1][1]}</div>`;
  chRel.data.datasets[1].data = rep.curva_calibracao_teste.map((b) => ({ x: b.previsto, y: b.observado }));
  chRel.update();
  $("#hModel").textContent = rep.modelo.nome;
}

/* ---------------- WebSocket com reconexão + fallback ---------------- */
function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/ws?desde=${cursor}`);
  ws.onmessage = (ev) => {
    const d = JSON.parse(ev.data);
    if (d.logs.length) { cursor = d.logs[d.logs.length - 1].seq; appendLogs(d.logs); }
    onSnapshot(d.estado);
  };
  ws.onclose = () => setTimeout(connect, 1200);
}

/* ---------------- controles ---------------- */
async function iniciar(cfg = {}) {
  const r = await api("/api/train/start", { method: "POST", body: JSON.stringify(cfg) });
  if (r.ok) { histLen = -1; toast("▶ Treino iniciado"); } else toast(r.body?.detail || "Falha ao iniciar", "var(--rose)");
}
$("#btnStart").onclick = () => iniciar();
$("#btnStop").onclick = async () => { await api("/api/train/stop", { method: "POST" }); toast("■ Interrompendo...", "var(--amber)"); };
$("#btnSync").onclick = async () => {
  const r = await api("/api/model/reload", { method: "POST" });
  toast(r.ok ? `🔄 Modelo ${r.body.meta.nome} sincronizado (disco → RAM)` : r.body?.detail, r.ok ? "var(--cyan)" : "var(--rose)");
};
$("#btnFloat").onclick = () => $("#floatBox").classList.toggle("hidden");
makeDraggable($("#floatBox"));

const openCfg = () => {
  const c = lastSnap?.config || {};
  for (const k of ["n_amostras", "max_iter", "passo_iter", "learning_rate", "max_depth", "paciencia", "limiar_decisao", "pausa_visual_s"])
    $("#cfg_" + k).value = c[k] ?? "";
  $("#cfg_regerar_dataset").checked = false;
  $("#cfgModal").classList.remove("hidden");
};
$("#btnCfg").onclick = openCfg; $("#btnCfg2").onclick = openCfg;
$("#cfgSave").onclick = () => {
  const cfg = {};
  for (const k of ["n_amostras", "max_iter", "passo_iter", "max_depth", "paciencia"]) cfg[k] = parseInt($("#cfg_" + k).value);
  for (const k of ["learning_rate", "limiar_decisao", "pausa_visual_s"]) cfg[k] = parseFloat($("#cfg_" + k).value);
  cfg.regerar_dataset = $("#cfg_regerar_dataset").checked;
  $("#cfgModal").classList.add("hidden");
  iniciar(cfg);
};
$("#btnCkpt").onclick = () => {
  const ck = lastSnap?.checkpoints || [];
  const pico = lastSnap?.pico?.arquivo;
  $("#ckTable").innerHTML = `<tr><th>Arquivo</th><th>Iter</th><th>AUC val</th><th>KS</th><th>Hora</th></tr>` +
    (ck.length ? ck.slice().reverse().map((c) => `<tr class="${c.arquivo === pico ? "best" : ""}"><td>${c.arquivo === pico ? "🏆 " : ""}${c.arquivo}</td><td>${c.passo}</td><td>${c.auc_val}</td><td>${c.ks}</td><td>${c.ts}</td></tr>`).join("")
      : `<tr><td colspan="5" class="muted">Nenhum checkpoint nesta sessão — clique em INICIAR.</td></tr>`);
  $("#ckModal").classList.remove("hidden");
};
$$("[data-close]").forEach((b) => (b.onclick = () => $$(".modal-bg").forEach((m) => m.classList.add("hidden"))));
$$(".modal-bg").forEach((m) => m.addEventListener("click", (e) => { if (e.target === m) m.classList.add("hidden"); }));

$("#filters").onclick = (e) => {
  const b = e.target.closest("button"); if (!b) return;
  $$("#filters button").forEach((x) => x.classList.toggle("on", x === b));
  filtro = b.dataset.f; renderLogsFull();
};
$("#logFilter").oninput = renderLogsFull;
$("#btnCopy").onclick = async () => {
  await navigator.clipboard.writeText(allLogs.filter(passa).map((l) => `[${l.ts}] [${l.canal}] ${l.msg}`).join("\n"));
  toast("📋 Logs copiados");
};
$("#btnClear").onclick = async () => { await api("/api/logs", { method: "DELETE" }); allLogs = []; renderLogsFull(); };
window.addEventListener("resize", () => lastSnap && renderCards(lastSnap.historico));

renderCards([]);
loadReport();
connect();
