/* =====================================================================
   C.R.E.D.I.T — Avaliação ao vivo (POST /predict, /api/bench)
   ===================================================================== */
chartDefaults();

const FIELDS = ["nome_cliente", "idade", "pontuacao_credito", "renda_mensal", "valor_emprestimo", "prazo_meses",
  "tempo_emprego_anos", "utilizacao_cartao", "consultas_6m", "inadimplencia_previa", "tipo_moradia", "finalidade_emprestimo"];
const INTS = ["idade", "pontuacao_credito", "prazo_meses", "consultas_6m", "inadimplencia_previa"];
const FLOATS = ["renda_mensal", "valor_emprestimo", "tempo_emprego_anos", "utilizacao_cartao"];

const PRESETS = {
  A: { nome_cliente: "Ana Beatriz Souza", idade: 38, pontuacao_credito: 785, renda_mensal: 12500, valor_emprestimo: 18000, prazo_meses: 24,
       tempo_emprego_anos: 9, utilizacao_cartao: 0.18, consultas_6m: 0, inadimplencia_previa: 0, tipo_moradia: "PROPRIA", finalidade_emprestimo: "NEGOCIOS" },
  B: { nome_cliente: "Marcos Vinícius Lima", idade: 29, pontuacao_credito: 615, renda_mensal: 4200, valor_emprestimo: 19000, prazo_meses: 36,
       tempo_emprego_anos: 2.5, utilizacao_cartao: 0.55, consultas_6m: 2, inadimplencia_previa: 0, tipo_moradia: "FINANCIADA", finalidade_emprestimo: "PESSOAL" },
  C: { nome_cliente: "Roberto Almeida", idade: 22, pontuacao_credito: 410, renda_mensal: 1900, valor_emprestimo: 38000, prazo_meses: 48,
       tempo_emprego_anos: 0.5, utilizacao_cartao: 0.92, consultas_6m: 6, inadimplencia_previa: 1, tipo_moradia: "ALUGADA", finalidade_emprestimo: "QUITACAO_DIVIDAS" },
};
const LABELS = { idade: "Idade", pontuacao_credito: "Score bureau", renda_mensal: "Renda mensal", valor_emprestimo: "Valor solicitado",
  prazo_meses: "Prazo", tempo_emprego_anos: "Tempo de emprego", inadimplencia_previa: "Inadimplência prévia", consultas_6m: "Consultas 6m",
  utilizacao_cartao: "Uso do cartão", tipo_moradia: "Moradia", finalidade_emprestimo: "Finalidade" };

/* ---------------- gauge SVG ---------------- */
const CX = 150, CY = 150, R = 112, A0 = 135, SWEEP = 270;
const polar = (deg, r = R) => [CX + r * Math.cos((deg * Math.PI) / 180), CY + r * Math.sin((deg * Math.PI) / 180)];
function arcPath(frac) {
  frac = Math.max(0.0001, Math.min(1, frac));
  const [x0, y0] = polar(A0), [x1, y1] = polar(A0 + SWEEP * frac);
  return `M ${x0} ${y0} A ${R} ${R} 0 ${SWEEP * frac > 180 ? 1 : 0} 1 ${x1} ${y1}`;
}
$("#arcBg").setAttribute("d", arcPath(1));
$("#ticks").innerHTML = Array.from({ length: 11 }, (_, i) => {
  const d = A0 + (SWEEP * i) / 10, [x0, y0] = polar(d, 128), [x1, y1] = polar(d, 136), [tx, ty] = polar(d, 147);
  return `<line x1="${x0}" y1="${y0}" x2="${x1}" y2="${y1}" stroke="#33466e" stroke-width="2"/>` +
    `<text x="${tx}" y="${ty + 3}" fill="#4f5f82" font-size="9" text-anchor="middle" font-family="JetBrains Mono">${i * 100}</text>`;
}).join("");
let gaugeNow = 0;
function setGauge(score) {
  const from = gaugeNow, to = score, t0 = performance.now();
  const step = (t) => {
    const k = Math.min(1, (t - t0) / 650), e = 1 - Math.pow(1 - k, 3), v = from + (to - from) * e;
    $("#arcVal").setAttribute("d", arcPath(v / 1000));
    const [nx, ny] = polar(A0 + SWEEP * (v / 1000));
    $("#needle").setAttribute("cx", nx); $("#needle").setAttribute("cy", ny);
    $("#gScore").textContent = Math.round(v);
    if (k < 1) requestAnimationFrame(step); else gaugeNow = to;
  };
  requestAnimationFrame(step);
}
function setLimMark(lim) {
  // marca a posição do score correspondente ao limiar (aprox. pela escala PDO do backend)
  const fator = 60 / Math.log(2), off = 600 - fator * Math.log(3);
  const s = Math.max(0, Math.min(1000, off + fator * Math.log(lim / (1 - lim))));
  const d = A0 + SWEEP * (s / 1000), [x0, y0] = polar(d, R - 14), [x1, y1] = polar(d, R + 14);
  Object.entries({ x1: x0, y1: y0, x2: x1, y2: y1 }).forEach(([k, v]) => $("#limMark").setAttribute(k, v));
}
const RCOLOR = { AA: "#10b981", A: "#34d399", B: "#facc15", C: "#f59e0b", D: "#f43f5e" };

/* ---------------- charts ---------------- */
const chLat = lineChart($("#chLat"), [
  { label: "Latência (ms)", borderColor: "#22d3ee", backgroundColor: "rgba(34,211,238,.12)", fill: true },
  { label: "Budget 50 ms", borderColor: "#f43f5e", borderDash: [5, 5], borderWidth: 1.2 },
], { y: { beginAtZero: true } });
const chScore = lineChart($("#chScore"), [
  { label: "Score", borderColor: "#facc15", backgroundColor: "rgba(250,204,21,.15)", fill: true, yAxisID: "y" },
  { label: "P(adimp.)", borderColor: "#a855f7", backgroundColor: "#a855f7", yAxisID: "y1", borderDash: [4, 3] },
], { extra: { scales: { x: { grid: { display: false } }, y: { min: 0, max: 1000 }, y1: { min: 0, max: 1, position: "right", grid: { display: false } } } } });
const chHist = new Chart($("#chHist"), {
  type: "bar",
  data: { labels: Array.from({ length: 20 }, (_, i) => i * 50), datasets: [{ label: "Proponentes", data: [],
    backgroundColor: Array.from({ length: 20 }, (_, i) => i < 9 ? "rgba(244,63,94,.65)" : i < 12 ? "rgba(245,158,11,.7)" : "rgba(16,185,129,.7)"), borderRadius: 3 }] },
  options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { grid: { display: false } } } },
});

/* ---------------- console ---------------- */
function clog(msg, cls = "i") {
  const c = $("#console"), ts = new Date().toLocaleTimeString("pt-BR");
  c.insertAdjacentHTML("beforeend", `<div><span class="t">[${ts}]</span> <span class="${cls}">${msg}</span></div>`);
  if (c.children.length > 300) c.firstElementChild.remove();
  c.scrollTop = 1e9;
}
$("#clrLog").onclick = () => ($("#console").innerHTML = "");

/* ---------------- formulário ---------------- */
function setForm(p) {
  for (const k of FIELDS) $("#" + k).value = p[k];
  syncLabels();
}
function getForm() {
  const o = {};
  for (const k of FIELDS) {
    const v = $("#" + k).value;
    o[k] = INTS.includes(k) ? parseInt(v) : FLOATS.includes(k) ? parseFloat(v) : v;
  }
  return o;
}
function syncLabels() {
  $("#v_idade").textContent = $("#idade").value + " anos";
  $("#v_pontuacao_credito").textContent = $("#pontuacao_credito").value;
  $("#v_utilizacao_cartao").textContent = Math.round($("#utilizacao_cartao").value * 100) + "%";
  $("#v_limiar").textContent = (+$("#limiar").value).toFixed(2);
  $("#tLim").textContent = (+$("#limiar").value).toFixed(2);
  setLimMark(+$("#limiar").value);
}
function randomProfile() {
  const r = (a, b) => a + Math.random() * (b - a), pick = (a) => a[Math.floor(Math.random() * a.length)];
  const idade = Math.round(r(19, 70)), renda = Math.round(r(1500, 18000) / 50) * 50;
  return { nome_cliente: pick(["Júlia Martins", "Pedro Henrique", "Larissa Costa", "Felipe Rocha", "Camila Duarte", "Rafael Nunes"]),
    idade, pontuacao_credito: Math.round(r(380, 840)), renda_mensal: renda, valor_emprestimo: Math.round(renda * r(0.8, 8) / 500) * 500,
    prazo_meses: pick([12, 18, 24, 36, 48, 60]), tempo_emprego_anos: Math.round(r(0, Math.min(25, idade - 16)) * 2) / 2,
    utilizacao_cartao: +r(0.05, 0.95).toFixed(2), consultas_6m: Math.round(r(0, 6)), inadimplencia_previa: Math.random() < 0.2 ? 1 : 0,
    tipo_moradia: pick(["PROPRIA", "ALUGADA", "FINANCIADA", "OUTROS"]), finalidade_emprestimo: pick(["EDUCACAO", "SAUDE", "NEGOCIOS", "PESSOAL", "QUITACAO_DIVIDAS", "IMOVEL"]) };
}

/* ---------------- estado de sessão ---------------- */
const sess = { n: 0, aprov: 0, lat: [], scores: [], probs: [] };
const BRL = (v) => v.toLocaleString("pt-BR", { style: "currency", currency: "BRL" });
function setPill(s, cls) { $("#evalState").textContent = s; $("#evalPill").className = "pill " + (cls || ""); }

function renderResult(r) {
  setGauge(r.score_creditai);
  const col = RCOLOR[r.rating] || "#fff";
  $("#gScore").style.color = col;
  $("#gRating").innerHTML = `<span style="color:${col}">RATING ${r.rating}</span>`;
  $("#gFaixa").textContent = r.faixa_risco.toUpperCase();
  const ok = r.codigo_decisao === 1;
  $("#decision").className = "decision " + (ok ? "ok" : "no");
  $("#decision").innerHTML = `${ok ? "✅ CRÉDITO APROVADO" : "❌ CRÉDITO REPROVADO"}<small>${r.nome_cliente} • P(adimplência) ${(r.probabilidade_aprovacao * 100).toFixed(1)}% vs limiar ${(r.limiar_politica * 100).toFixed(0)}% • ${r.latencia_ms.toFixed(1)} ms • modelo v${r.versao_modelo}</small>`;
  const mx = Math.max(0.02, ...r.fatores.map((f) => Math.abs(f.impacto)));
  $("#factors").innerHTML = r.fatores.slice(0, 8).map((f) => {
    const w = (Math.abs(f.impacto) / mx) * 50, pos = f.impacto >= 0;
    const val = typeof f.valor === "number" ? (f.valor % 1 ? f.valor.toFixed(2) : f.valor) : f.valor;
    return `<div class="fac"><span class="nm" title="${LABELS[f.atributo]} = ${val}">${LABELS[f.atributo]} <span class="mute">${val}</span></span>
      <span class="trk"><i style="left:${pos ? 50 : 50 - w}%;width:${w}%;background:${pos ? "#10b981" : "#f43f5e"};box-shadow:0 0 8px ${pos ? "#10b981" : "#f43f5e"}"></i></span>
      <span class="v" style="color:${pos ? "#6ee7b7" : "#fda4af"}">${pos ? "+" : ""}${(f.impacto * 100).toFixed(1)}pp</span></div>`;
  }).join("");
  $("#parcela").textContent = BRL(r.parcela_estimada);
  $("#comp").textContent = (r.comprometimento_renda * 100).toFixed(1) + "%";
  $("#reflex").textContent = `${r.decisao} • score ${r.score_creditai} (passo ${sess.n})`;

  const p = r.probabilidade_aprovacao;
  const H = -(p * Math.log2(p) + (1 - p) * Math.log2(1 - p));
  $("#sConf").textContent = (p * 100).toFixed(1) + "%"; $("#bConf").style.width = p * 100 + "%";
  $("#sRisk").textContent = ((1 - p) * 100).toFixed(1) + "%";
  $("#sEnt").textContent = (H * 100).toFixed(0) + "%"; $("#bEnt").style.width = H * 100 + "%";
  $("#sComp").textContent = (r.comprometimento_renda * 100).toFixed(1) + "%"; $("#bComp").style.width = Math.min(100, r.comprometimento_renda * 100) + "%";
  $("#tScore").textContent = `${r.score_creditai} / ${r.rating}`;
  $("#tConf").textContent = (p * 100).toFixed(1) + "%";
}

function pushTelemetry(lat, r) {
  sess.lat.push(lat); if (r) { sess.n++; sess.aprov += r.codigo_decisao; sess.scores.push(r.score_creditai); sess.probs.push(r.probabilidade_aprovacao); }
  const N = 40, lab = sess.lat.slice(-N).map((_, i) => i + 1);
  chLat.data.labels = lab; chLat.data.datasets[0].data = sess.lat.slice(-N); chLat.data.datasets[1].data = lab.map(() => 50); chLat.update();
  chScore.data.labels = sess.scores.slice(-N).map((_, i) => i + 1);
  chScore.data.datasets[0].data = sess.scores.slice(-N); chScore.data.datasets[1].data = sess.probs.slice(-N); chScore.update();
  const sorted = [...sess.lat].sort((a, b) => a - b), p50 = sorted[Math.floor(sorted.length / 2)];
  $("#tLat").textContent = p50.toFixed(1) + " ms"; $("#sLat").textContent = p50.toFixed(1) + " ms"; $("#sLatA").textContent = lat.toFixed(1) + " ms";
  $("#bLat").style.width = Math.min(100, (p50 / 50) * 100) + "%";
  $("#tN").textContent = sess.n;
  $("#tAprov").textContent = sess.n ? ((sess.aprov / sess.n) * 100).toFixed(0) + "%" : "--%";
}

async function avaliar(payload = getForm(), origem = "FORM") {
  setPill("AVALIANDO", "RUNNING");
  const t0 = performance.now();
  const r = await api(`/predict?limiar=${(+$("#limiar").value).toFixed(2)}`, { method: "POST", body: JSON.stringify(payload) });
  const lat = performance.now() - t0;
  if (r.status === 200) {
    renderResult(r.body); pushTelemetry(lat, r.body);
    clog(`[${origem}] ${r.body.decisao.toUpperCase()} • ${r.body.nome_cliente} • score ${r.body.score_creditai} (${r.body.rating}) • P=${r.body.probabilidade_aprovacao} • ${lat.toFixed(1)} ms ida-e-volta`, r.body.codigo_decisao ? "a" : "r");
    setPill("IDLE");
  } else if (r.status === 422) {
    const errs = (r.body.detail || []).map((e) => `${e.loc.slice(-1)[0]}: ${e.msg}`).join(" | ");
    clog(`[HTTP 422] Pydantic bloqueou a entrada → ${errs}`, "w"); setPill("422", "STOPPED");
    $("#decision").className = "decision warn"; $("#decision").innerHTML = `⚠ HTTP 422 — VALIDAÇÃO PYDANTIC<small>${errs}</small>`;
  } else if (r.status === 400) {
    clog(`[HTTP 400] Regra de negócio → ${r.body.detail}`, "r"); setPill("400", "ERROR");
    $("#decision").className = "decision no"; $("#decision").innerHTML = `🚫 HTTP 400 — REGRA DE NEGÓCIO<small>${r.body.detail}</small>`;
  } else {
    clog(`[HTTP ${r.status}] ${r.body?.detail || "erro de conexão"}`, "r"); setPill("ERRO", "ERROR");
  }
  return r;
}

async function bench() {
  setPill("BENCH", "RUNNING"); clog("[BENCH] Disparando 500 perfis sintéticos inéditos contra o modelo em RAM...", "i");
  const r = await api(`/api/bench?n=500&limiar=${(+$("#limiar").value).toFixed(2)}`);
  if (!r.ok) { clog(`[BENCH] falhou: ${r.body?.detail}`, "r"); setPill("ERRO", "ERROR"); return; }
  const b = r.body;
  chHist.data.datasets[0].data = b.histograma_score.contagens; chHist.update();
  $("#benchSub").textContent = `n=${b.n} • aprovação ${(b.taxa_aprovacao * 100).toFixed(1)}% • score médio ${b.score_medio}`;
  clog(`[BENCH] p50 ${b.latencia_ms.p50} ms | p95 ${b.latencia_ms.p95} ms | p99 ${b.latencia_ms.p99} ms | lote vetorizado ${b.lote_vetorizado_ms} ms | aprovação ${(b.taxa_aprovacao * 100).toFixed(1)}% | bloqueados por regra ${b.bloqueados_regra_negocio}`, "a");
  setPill("IDLE");
}

/* ---------------- eventos ---------------- */
let debounce;
FIELDS.concat(["limiar"]).forEach((k) => $("#" + k).addEventListener("input", () => {
  syncLabels();
  if ($("#live").checked) { clearTimeout(debounce); debounce = setTimeout(() => avaliar(getForm(), "LIVE"), 220); }
}));
$$("[data-p]").forEach((b) => (b.onclick = () => {
  const p = b.dataset.p === "R" ? randomProfile() : PRESETS[b.dataset.p];
  setForm(p); avaliar(p, `PERFIL ${b.dataset.p}`);
}));
$$("[data-lim]").forEach((b) => (b.onclick = () => { $("#limiar").value = b.dataset.lim; syncLabels(); avaliar(getForm(), "POLÍTICA"); }));
$("#btnRun").onclick = () => avaliar();
$("#btnBench").onclick = bench;
$("#btnReset").onclick = () => { setForm(PRESETS.A); $("#limiar").value = 0.5; syncLabels(); gaugeNow = 0; setGauge(0); $("#factors").innerHTML = ""; clog("[VIEW] Estado restaurado para o perfil padrão.", "t"); };
$("#mAval").onclick = () => { $("#mAval").classList.add("on"); $("#mBench").classList.remove("on"); };
$("#mBench").onclick = () => { $("#mBench").classList.add("on"); $("#mAval").classList.remove("on"); bench(); };
$("#t200").onclick = () => avaliar({ idade: 40, pontuacao_credito: 760 }, "CONTRATO MÍNIMO");
$("#t422").onclick = () => avaliar({ idade: 15, pontuacao_credito: 990 }, "TESTE 422");
$("#t400").onclick = () => avaliar({ idade: 20, pontuacao_credito: 710, renda_mensal: 4500, valor_emprestimo: 10000, tempo_emprego_anos: 14 }, "TESTE 400");
$("#btnSync").onclick = async () => {
  const r = await api("/api/model/reload", { method: "POST" });
  clog(r.ok ? `[SYNC] Artefato recarregado: ${r.body.meta.nome} (${r.body.meta.treinado_em})` : `[SYNC] ${r.body?.detail}`, r.ok ? "a" : "r");
  loadHealth();
};
document.addEventListener("keydown", (e) => {
  if (["INPUT", "SELECT"].includes(document.activeElement.tagName) && document.activeElement.type !== "range") return;
  if (e.code === "Space") { e.preventDefault(); avaliar(); }
  else if (e.key === "1") $('[data-p="A"]').click();
  else if (e.key === "2") $('[data-p="B"]').click();
  else if (e.key === "3") $('[data-p="C"]').click();
  else if (e.key === "4") $('[data-p="R"]').click();
  else if (e.key.toLowerCase() === "b") bench();
  else if (e.key.toLowerCase() === "r") $("#btnReset").click();
});

async function loadHealth() {
  const r = await api("/health");
  if (!r.ok) return;
  const h = r.body;
  $("#hModel").textContent = h.modelo || "—"; $("#hPort").textContent = location.port || "";
  $("#rAlg").textContent = (h.algoritmo || "—").replace(" + ", " + ");
  $("#rTs").textContent = (h.treinado_em || "—").replace("T", " ");
  $("#tAuc").textContent = h.metricas_referencia_teste?.roc_auc ?? "—";
  const rep = await api("/api/report");
  if (rep.ok) $("#rIter").textContent = rep.body.modelo.melhor_iteracao;
}

setForm(PRESETS.A);
loadHealth().then(() => avaliar(PRESETS.A, "BOOT"));
clog("[VIEW] Viewport inicializado • atalhos: Espaço avaliar • 1-4 perfis • B bench • R reset", "t");
