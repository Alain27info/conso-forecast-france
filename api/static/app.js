import { barChart, dataTable, fmt, hBarChart, hideTip, lineChart } from "/static/charts.js";

const $ = (selector) => document.querySelector(selector);
const TZ = "Europe/Paris";
const MINUS = "−";

// One entity = one color, fixed order. Order = legend order (gray sits next to blue: validated adjacency).
const MODEL = {
  baseline_day_before: { color: "--s-base", label: "Baseline « veille »" },
  gru_residual_ensemble: { color: "--s-final", label: "Modèle final (GRU « écart »)" },
  gbm: { color: "--s-gbm", label: "Gradient boosting" },
  sarima: { color: "--s-sarima", label: "SARIMA" },
  "RTE_J-1": { color: "--s-rte", label: "RTE J-1 (référence)" },
};
const state = { results: null, models: new Set(Object.keys(MODEL)), tab: "hour", day: null, dayHidden: new Set(), tables: {}, token: 0 };

const store = {
  get: (k) => { try { return localStorage.getItem(k); } catch { return null; } },
  set: (k, v) => { try { v === null ? localStorage.removeItem(k) : localStorage.setItem(k, v); } catch { /* storage unavailable */ } },
};

// ---------------------------------------------------------------- helpers
const clock = new Intl.DateTimeFormat("fr-FR", { timeZone: TZ, hour: "2-digit", minute: "2-digit", hourCycle: "h23" });
const parisDate = new Intl.DateTimeFormat("sv-SE", { timeZone: TZ });
const localHM = (ms) => { const p = Object.fromEntries(clock.formatToParts(ms).map((x) => [x.type, x.value])); return { h: +p.hour, m: +p.minute }; };
const el = (tag, text, parent, className) => {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  if (parent) parent.appendChild(node);
  return node;
};
const fetchJson = async (url, options) => {
  const response = await fetch(url, options);
  let body = null;
  try { body = await response.json(); } catch { /* not JSON */ }
  return { ok: response.ok, status: response.status, body };
};
const detailText = (body) => {
  const d = body && body.detail;
  if (Array.isArray(d)) return d.map((x) => x.msg || JSON.stringify(x)).join(" ; ");
  return typeof d === "string" ? d : "Réponse inattendue du serveur.";
};
const show = (id, text) => { const box = $(id); box.hidden = !text; if (text) box.querySelector("span:last-child").textContent = text; };

function chips(container, items, selected, onToggle, line = false) {
  container.replaceChildren();
  for (const item of items) {
    const button = el("button", undefined, container, `chip${line ? " line" : ""}`);
    button.type = "button";
    button.setAttribute("aria-pressed", String(selected.has(item.id)));
    button.style.setProperty("--c", `var(${item.color})`);
    el("span", undefined, button, "key");
    button.append(item.label);
    button.addEventListener("click", () => onToggle(item.id));
  }
}

function toggleTable(key, render) {
  const on = !state.tables[key];
  state.tables[key] = on;
  const button = $(`#${key}-table-toggle`);
  button.setAttribute("aria-pressed", String(on));
  button.textContent = on ? "Vue graphique" : "Vue tableau";
  $(`#${key}-chart`).hidden = on;
  $(`#${key}-table`).hidden = !on;
  render();
}

// ---------------------------------------------------------------- status + KPIs
async function loadStatus() {
  const dot = $("#status-dot"), text = $("#status-text");
  try {
    const { ok, body } = await fetchJson("/health");
    if (!ok) throw new Error();
    dot.className = "dot ok";
    const parsed = /(\d+)seeds-trained-until-(.+)$/.exec(body.model_version);
    text.textContent = "Modèle chargé";
    el("span", parsed ? ` · GRU, ${parsed[1]} graines · données jusqu'au ${parsed[2]}` : ` · ${body.model_version}`, text, "long");
    $("#status").title = body.model_version;
  } catch {
    dot.className = "dot err";
    text.textContent = "API indisponible";
  }
}

function renderKpis(results) {
  const final = results.mean.gru_residual_ensemble, base = results.mean.baseline_day_before;
  const folds = results.by_fold.folds;
  const finalByFold = results.by_fold["MAE (MW)"].gru_residual_ensemble, baseByFold = results.by_fold["MAE (MW)"].baseline_day_before;
  const wins = folds.filter((_, i) => finalByFold[i] < baseByFold[i]).length;
  const box = $("#kpis");
  box.replaceChildren();
  const tile = (className, label, value, sub, unit) => {
    const t = el("div", undefined, box, `tile ${className}`);
    el("div", label, t, "label");
    const v = el("div", value, t, "value");
    if (unit) el("small", unit, v);
    el("div", sub, t, "sub");
    return t;
  };
  tile("hero-figure", "Erreur moyenne du modèle déployé (MAE)", fmt(final.mae), "Moyenne de 5 années de test, données historiques consolidées. Pas une garantie en production.", "MW");
  tile("", "Par rapport à la baseline « veille »", `${MINUS}${fmt((1 - final.mae / base.mae) * 100)} %`, `MAE de ${fmt(base.mae)} MW pour la baseline qui recopie la veille`);
  tile("", "Années de test gagnées", `${wins} / ${folds.length}`, "Le modèle final bat la baseline « veille » chaque année");
  const live = tile("", "Latence de /predict", "—", "Mesurée à chaque prévision lancée ci-dessus");
  live.id = "latency";
}

// ---------------------------------------------------------------- forecast panel
function ticksFor(xs) {
  const out = [];
  xs.forEach((x) => { const { h, m } = localHM(x); if (m === 0 && h % 3 === 0) out.push({ x, label: `${h} h` }); });
  return out;
}

function daySeries(day) {
  const defs = [
    ["actual", "Consommation réelle", "--s-actual", day.actual],
    ["forecast", "Prévision du modèle (API)", "--s-final", day.forecast],
    ["rte", "Prévision RTE J-1", "--s-rte", day.rte],
    ["base", "Baseline « veille »", "--s-base", day.base],
  ];
  return defs.filter(([, , , values]) => values.some((v) => v !== null)).map(([id, label, color, values]) => ({ id, label, color, values }));
}

function renderDay() {
  const day = state.day;
  if (!day) return;
  const all = daySeries(day);
  chips($("#day-chips"), all, new Set(all.filter((s) => !state.dayHidden.has(s.id)).map((s) => s.id)), (id) => {
    state.dayHidden.has(id) ? state.dayHidden.delete(id) : state.dayHidden.add(id);
    if (all.every((s) => state.dayHidden.has(s.id))) state.dayHidden.delete(id);
    renderDay();
  }, true);
  const visible = all.filter((s) => !state.dayHidden.has(s.id));
  const chart = $("#day-chart");
  if (!state.tables.day) {
    lineChart(chart, {
      xs: day.xs, series: visible, xTicks: ticksFor(day.xs), yTitle: "MW (l'axe ne part pas de zéro)", tight: true,
      xTitle: (x) => { const { h, m } = localHM(x); return `${String(h).padStart(2, "0")}h${String(m).padStart(2, "0")} (heure de Paris)`; },
      ariaLabel: `Consommation du ${day.date} par demi-heure : ${visible.map((s) => s.label).join(", ")}`,
    });
  } else {
    dataTable($("#day-table"), ["Heure (Paris)", ...visible.map((s) => `${s.label} (MW)`)], day.xs.map((x, i) => {
      const { h, m } = localHM(x);
      return [`${String(h).padStart(2, "0")}h${String(m).padStart(2, "0")}`, ...visible.map((s) => (s.values[i] === null ? "—" : fmt(s.values[i], 1)))];
    }));
  }
}

function renderDayStats(day, ms) {
  const box = $("#day-stats");
  box.replaceChildren();
  const stat = (label, value, unit, color) => {
    const s = el("div", undefined, box, "stat");
    const k = el("div", undefined, s, "k");
    if (color) { const i = el("i", undefined, k); i.style.setProperty("--c", `var(${color})`); }
    k.append(label);
    const v = el("div", value, s, "v");
    if (unit) el("small", unit, v);
  };
  const both = day.xs.map((_, i) => [day.actual[i], day.forecast[i], day.rte[i], day.base[i]].every((v) => v !== null));
  if (day.actual.some((v) => v !== null) && both.some(Boolean)) {
    const mae = (vals) => { let sum = 0, n = 0; vals.forEach((v, i) => { if (both[i]) { sum += Math.abs(v - day.actual[i]); n++; } }); return sum / n; };
    stat("Erreur du modèle (MAE du jour)", fmt(mae(day.forecast)), "MW", "--s-final");
    stat("Erreur de RTE J-1", fmt(mae(day.rte)), "MW", "--s-rte");
    stat("Erreur de la baseline « veille »", fmt(mae(day.base)), "MW", "--s-base");
  } else {
    const values = day.forecast;
    stat("Moyenne prévue", fmt(values.reduce((a, b) => a + b, 0) / values.length), "MW", "--s-final");
    stat("Creux prévu", fmt(Math.min(...values)), "MW", "--s-final");
  }
  const peak = Math.max(...day.forecast), at = localHM(day.xs[day.forecast.indexOf(peak)]);
  stat("Pic prévu", fmt(peak), `MW à ${at.h} h${String(at.m).padStart(2, "0")}`, "--s-final");
  stat("Latence de l'appel", fmt(ms), "ms");
  const live = $("#latency");
  if (live) { live.querySelector(".value").textContent = `${fmt(ms)} ms`; }
}

function abridge(list, head, tail = 1) {
  return list.length <= head + tail + 1 ? list : [...list.slice(0, head), `… (${list.length - head - tail} autres éléments)`, ...list.slice(-tail)];
}

async function runForecast() {
  const date = $("#date").value;
  if (!date) return;
  const token = ++state.token;
  const chart = $("#day-chart");
  chart.classList.add("loading");
  ["#error", "#rule", "#insample", "#dst"].forEach((id) => show(id, ""));
  const demo = await fetchJson(`/demo/day?date=${date}`);
  if (token !== state.token) return;
  if (!demo.ok) { chart.classList.remove("loading"); show("#error", detailText(demo.body)); if (demo.status === 503) $("#demo-missing").hidden = false; return; }
  const started = performance.now();
  const response = await fetchJson("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(demo.body.request) });
  const elapsed = performance.now() - started;
  if (token !== state.token) return;
  chart.classList.remove("loading");
  if (!response.ok) { show("#error", `Le serveur a refusé la requête (HTTP ${response.status}) : ${detailText(response.body)}`); return; }
  const predictions = response.body.predictions;
  const xs = predictions.map((p) => Date.parse(p.timestamp));
  const align = (points) => { const byTime = new Map(points.map((p) => [Date.parse(p.timestamp), p.consumption_mw])); return xs.map((x) => (byTime.has(x) ? byTime.get(x) : null)); };
  state.day = { date, xs, forecast: predictions.map((p) => p.consumption_mw), actual: align(demo.body.actual), rte: align(demo.body.rte_j1), base: align(demo.body.day_before), request: demo.body.request };
  $("#day-title").textContent = `Prévision du ${new Date(`${date}T12:00:00`).toLocaleDateString("fr-FR", { weekday: "long", day: "numeric", month: "long", year: "numeric" })}`;
  $("#day-sub").textContent = state.day.actual.some((v) => v !== null) ? "Consommation nationale par demi-heure, en MW : réalité et prévisions comparées" : "Consommation nationale par demi-heure, en MW : vraie prévision, la réalité n'est pas encore connue";
  $("#insample").hidden = !(demo.body.in_sample && state.day.actual.some((v) => v !== null));
  if (xs.length !== 48) show("#dst", xs.length === 46 ? "Jour de changement d'heure de printemps : il ne dure que 23 heures, donc 46 créneaux de 30 minutes. L'API s'adapte." : "Jour de changement d'heure d'automne : il dure 25 heures, donc 50 créneaux de 30 minutes. L'API s'adapte. Le trou dans la courbe réelle est une anomalie connue des données RTE : une des deux heures répétées n'a pas de mesure.");
  const req = state.day.request;
  $("#req-json").textContent = JSON.stringify({ date: req.date, history: abridge(req.history, 2) }, null, 2);
  $("#res-json").textContent = JSON.stringify({ date: response.body.date, model_version: response.body.model_version, timezone: response.body.timezone, predictions: abridge(response.body.predictions, 3) }, null, 2);
  renderDay();
  renderDayStats(state.day, elapsed);
}

async function testRule() {
  if (!state.day) { await runForecast(); if (!state.day) return; }
  const last = state.day.request.history.at(-1).timestamp;
  const body = { ...state.day.request, date: parisDate.format(new Date(last)) };
  const response = await fetchJson("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  show("#rule", `Prévision demandée pour ${body.date}, alors que l'historique contient déjà ce jour. Réponse du serveur : HTTP ${response.status} — ${detailText(response.body)}`);
  $("#rule").scrollIntoView({ behavior: "smooth", block: "center" });
}

async function initPanel() {
  const { ok, status, body } = await fetchJson("/demo/range");
  if (!ok) { $("#demo-missing").hidden = status !== 503; if (status !== 503) show("#error", detailText(body)); ["#run", "#prev", "#next", "#date", "#examples"].forEach((id) => { $(id).disabled = true; }); return; }
  const input = $("#date");
  input.min = body.first_date; input.max = body.last_date; input.value = body.last_date;
  // Typical and worst days come from the backtest itself (ranked by the model's daily error), not from hand-picked days.
  const { body: results } = await resultsPromise;
  const days = results && results.days;
  const longDate = (d) => new Date(`${d}T12:00:00`).toLocaleDateString("fr-FR", { day: "numeric", month: "long", year: "numeric" });
  const examples = [
    [body.last_date, `Le lendemain des données (${longDate(body.last_date)}) : vraie prévision`],
    ...(days ? [[days.typical.date, `Jour typique du backtest : ${longDate(days.typical.date)} (erreur médiane, ${fmt(days.typical.mae)} MW)`]] : []),
    ...(days ? days.worst.map((w) => [w.date, `Pire jour n°${w.rank} sur ${fmt(days.count)} : ${longDate(w.date)} (erreur du backtest ${fmt(w.mae)} MW)`]) : []),
    ["2026-03-29", "Changement d'heure de printemps (jour de 23 h)"],
    ["2025-10-26", "Changement d'heure d'automne (jour de 25 h)"],
    ["2026-07-14", "Jour férié : 14 juillet 2026"],
    ["2025-12-25", "Jour férié : Noël 2025"],
  ].filter(([d]) => d >= body.first_date && d <= body.last_date);
  const select = $("#examples");
  examples.forEach(([d, label]) => { const o = el("option", label, select); o.value = d; });
  const shift = (days) => { const d = new Date(`${input.value}T12:00:00Z`); d.setUTCDate(d.getUTCDate() + days); const v = d.toISOString().slice(0, 10); if (v >= input.min && v <= input.max) { input.value = v; runForecast(); } };
  $("#run").addEventListener("click", runForecast);
  $("#prev").addEventListener("click", () => shift(-1));
  $("#next").addEventListener("click", () => shift(1));
  input.addEventListener("change", runForecast);
  select.addEventListener("change", () => { if (select.value) { input.value = select.value; runForecast(); } });
  $("#refuse").addEventListener("click", testRule);
  $("#copy").addEventListener("click", async () => {
    if (!state.day) return;
    try { await navigator.clipboard.writeText(JSON.stringify(state.day.request)); $("#copy").textContent = "Copié ✓"; } catch { $("#copy").textContent = "Copie impossible"; }
    setTimeout(() => { $("#copy").textContent = "Copier la requête complète"; }, 1800);
  });
  $("#day-table-toggle").addEventListener("click", () => toggleTable("day", renderDay));
  await runForecast();
}

// ---------------------------------------------------------------- results dashboards
const visibleModels = () => Object.keys(MODEL).filter((id) => state.models.has(id));
const modelSeries = (id, values) => ({ id, label: MODEL[id].label, color: MODEL[id].color, values });

function renderFold() {
  const r = state.results, ids = visibleModels(), folds = r.by_fold.folds;
  const categories = [...folds.map((f) => f.replace("-", "–")), "Moyenne"];
  const series = ids.map((id) => modelSeries(id, [...r.by_fold["MAE (MW)"][id], r.mean[id].mae]));
  if (!state.tables.fold) {
    const shortLabels = [...folds.map((f) => `${f.slice(2).replace("-", "–")}`), "Moy."];
    barChart($("#fold-chart"), { categories, shortLabels, series, yTitle: "MAE (MW), plus bas = mieux", separatorBefore: folds.length, ariaLabel: `MAE par année de test pour ${ids.map((i) => MODEL[i].label).join(", ")}` });
  } else {
    dataTable($("#fold-table"), ["Modèle", ...categories.map((c) => `${c} (MW)`)], series.map((s) => [s.label, ...s.values.map((v) => fmt(v, 1))]), ids.indexOf("gru_residual_ensemble"));
  }
}

function renderBreak() {
  const r = state.results, ids = visibleModels(), key = { hour: "by_hour", weekday: "by_weekday", season: "by_season" }[state.tab];
  const data = r[key];
  const series = ids.map((id) => modelSeries(id, data.values[id]));
  const names = { hour: "heure", weekday: "jour de la semaine", season: "saison" }[state.tab];
  if (state.tables.break) {
    dataTable($("#break-table"), [state.tab === "hour" ? "Heure (Paris)" : "Catégorie", ...series.map((s) => `${s.label} (MW)`)],
      data.labels.map((label, i) => [state.tab === "hour" ? `${label} h` : label, ...series.map((s) => fmt(s.values[i], 1))]));
    return;
  }
  if (state.tab === "hour") {
    lineChart($("#break-chart"), {
      xs: data.labels, series, yTitle: "MAE (MW)", xTicks: data.labels.filter((h) => h % 3 === 0).map((h) => ({ x: h, label: `${h} h` })),
      xTitle: (x) => `${x} h – ${x + 1} h (heure de Paris)`, ariaLabel: `MAE par ${names}`,
    });
  } else {
    const shortLabels = state.tab === "season" ? ["Hiver", "Print.", "Été", "Autom."] : data.labels;
    barChart($("#break-chart"), { categories: data.labels, shortLabels, series, yTitle: "MAE (MW)", ariaLabel: `MAE par ${names}` });
  }
}

function renderParity() {
  const m = state.results.mean;
  const rows = [
    ["gbm", "Gradient boosting (étapes initiales)", "--s-base"],
    ["gbm_residual", "Gradient boosting « écart à la veille »", "--s-base"],
    ["gbm_profile", "Gradient boosting + profil de la veille", "--s-base"],
    ["gbm_profile_residual", "Gradient boosting « écart » + profil de la veille", "--s-base"],
    ["gru_ensemble", "GRU (direct)", "--s-final"],
    ["gru_residual_ensemble", "GRU « écart à la veille » (modèle final)", "--s-final"],
  ].map(([id, label, color]) => ({ label, color, value: m[id].mae, bold: id === "gru_residual_ensemble" }));
  hBarChart($("#parity"), rows.sort((a, b) => b.value - a.value), (v) => `${fmt(v)} MW`);
}

function renderFull() {
  const r = state.results;
  const ids = r.models.map((x) => x.id).sort((a, b) => r.mean[a].mae - r.mean[b].mae);
  const label = Object.fromEntries(r.models.map((x) => [x.id, x.label]));
  const rows = ids.map((id) => [label[id], ...r.by_fold["MAE (MW)"][id].map((v) => fmt(v)), fmt(r.mean[id].mae), fmt(r.mean[id].mae_std), fmt(r.mean[id].rmse), r.mean[id].mape.toFixed(2).replace(".", ",")]);
  dataTable($("#full-table"), ["Modèle", ...r.by_fold.folds.map((f) => `MAE ${f}`), "MAE moyenne", "Écart-type", "RMSE moyen", "MAPE moyen (%)"], rows, ids.indexOf("gru_residual_ensemble"));
}

function renderResults() {
  if (!state.results) return;
  renderFold(); renderBreak(); renderParity(); renderFull();
}

function initResults(results) {
  state.results = results;
  renderKpis(results);
  const items = Object.entries(MODEL).map(([id, m]) => ({ id, ...m }));
  const draw = () => chips($("#model-chips"), items, state.models, (id) => {
    state.models.has(id) ? state.models.delete(id) : state.models.add(id);
    if (state.models.size === 0) state.models.add(id);
    draw(); renderFold(); renderBreak();
  });
  draw();
  for (const tab of ["hour", "weekday", "season"]) {
    $(`#tab-${tab}`).addEventListener("click", () => {
      state.tab = tab;
      ["hour", "weekday", "season"].forEach((t) => $(`#tab-${t}`).setAttribute("aria-selected", String(t === tab)));
      renderBreak();
    });
  }
  $("#fold-table-toggle").addEventListener("click", () => toggleTable("fold", renderFold));
  $("#break-table-toggle").addEventListener("click", () => toggleTable("break", renderBreak));
  renderResults();
}

// ---------------------------------------------------------------- theme + resize
function applyTheme() {
  const saved = store.get("theme");
  if (saved) document.documentElement.dataset.theme = saved; else delete document.documentElement.dataset.theme;
}
function redraw() { hideTip(); renderResults(); renderDay(); }
$("#theme").addEventListener("click", () => {
  const dark = document.documentElement.dataset.theme === "dark" || (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
  store.set("theme", dark ? "light" : "dark");
  applyTheme(); redraw();
});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (!store.get("theme")) redraw(); });
let resizeTimer;
addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(redraw, 120); });

const resultsPromise = fetchJson("/static/results.json");
applyTheme();
loadStatus();
resultsPromise.then(({ ok, body }) => {
  if (!ok) { $("#kpis").replaceChildren(); el("p", "Les résultats du backtest sont introuvables (api/static/results.json). Générez-les avec python -m src.make_dashboard_data.", $("#kpis")); return; }
  initResults(body);
});
initPanel();
