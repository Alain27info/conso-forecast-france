// Small dependency-free SVG charts: line (crosshair + tooltip), grouped bars, ranked horizontal bars, data table.
// Every text coming from data goes through textContent / setAttribute, never innerHTML.
const NS = "http://www.w3.org/2000/svg";
const nf0 = new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 0 });
const nf1 = new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 1 });
export const fmt = (v, digits = 0) => (digits ? nf1 : nf0).format(v);

const svg = (tag, attrs = {}, parent) => {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
};
const h = (tag, className, text, parent) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  if (parent) parent.appendChild(node);
  return node;
};
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const colorOf = (series) => cssVar(series.color);

/** Round the maximum up to a clean axis top and return the ticks (0, step, 2 step, ...). */
export function niceScale(max, target = 5, min = 0) {
  if (!(max > min)) return { min: 0, top: 1, ticks: [0, 1] };
  const rough = (max - min) / target;
  const pow = 10 ** Math.floor(Math.log10(rough));
  const step = [1, 2, 2.5, 5, 10].map((s) => s * pow).find((s) => s >= rough);
  const start = Math.floor(min / step) * step;
  const ticks = [];
  for (let v = start; v <= Math.ceil(max / step) * step + 1e-9; v += step) ticks.push(v);
  return { min: start, top: ticks[ticks.length - 1], ticks };
}

// ---------------------------------------------------------------- tooltip
let tip;
function tooltip() {
  if (!tip) {
    tip = h("div", "tooltip");
    tip.hidden = true;
    tip.setAttribute("role", "status");
    document.body.appendChild(tip);
  }
  return tip;
}
export function showTip(clientX, clientY, title, rows) {
  const t = tooltip();
  t.replaceChildren();
  if (title) h("div", "tip-title", title, t);
  for (const row of rows) {
    const line = h("div", "tip-row", undefined, t);
    const key = h("span", "tip-key", undefined, line);
    key.style.background = row.color;
    h("span", "tip-label", row.label, line);
    h("span", "tip-val", row.value, line);
  }
  t.hidden = false;
  const box = t.getBoundingClientRect();
  const x = clientX + 16 + box.width > innerWidth ? clientX - 16 - box.width : clientX + 16;
  const y = Math.min(Math.max(8, clientY - box.height / 2), innerHeight - box.height - 8);
  t.style.left = `${Math.max(8, x)}px`;
  t.style.top = `${y}px`;
}
export const hideTip = () => { if (tip) tip.hidden = true; };

function frame(container, height, label) {
  container.replaceChildren();
  const width = Math.max(320, container.clientWidth || 640);
  const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": label });
  container.appendChild(root);
  return { root, width };
}

function yAxis(root, scale, m, width, height, format, title) {
  for (const v of scale.ticks) {
    const y = m.top + (1 - (v - scale.min) / (scale.top - scale.min)) * (height - m.top - m.bottom);
    svg("line", { x1: m.left, x2: width - m.right, y1: y, y2: y, class: v === scale.min ? "axis" : "grid" }, root);
    const label = svg("text", { x: m.left - 8, y: y + 4, "text-anchor": "end" }, root);
    label.textContent = format(v);
  }
  if (title) {
    const t = svg("text", { x: m.left, y: 10, class: "ytitle" }, root);
    t.textContent = title;
  }
}

// ---------------------------------------------------------------- line chart
/**
 * opts: { xs:number[], series:[{id,label,color(css var),values:(number|null)[],width?}], xTicks:[{x,label}],
 *         xTitle:(x)=>string, yFormat, yTitle, height, ariaLabel }
 */
export function lineChart(container, opts) {
  const { xs, series, xTicks, xTitle, yFormat = fmt, yTitle, height = 360, ariaLabel, tight = false } = opts;
  const { root, width } = frame(container, height, ariaLabel);
  const m = { top: yTitle ? 22 : 12, right: 16, bottom: 28, left: 58 };
  const all = series.flatMap((s) => s.values).filter((v) => v !== null && v !== undefined);
  const lo = Math.min(...all), hi = Math.max(...all, 1);
  // `tight`: a line may leave the zero baseline (the title says so); magnitudes (errors) always start at 0
  const scale = niceScale(hi, 5, tight ? Math.max(0, lo - (hi - lo) * 0.1) : 0);
  const x0 = xs[0], x1 = xs[xs.length - 1];
  const sx = (x) => m.left + ((x - x0) / (x1 - x0 || 1)) * (width - m.left - m.right);
  const sy = (v) => m.top + (1 - (v - scale.min) / (scale.top - scale.min)) * (height - m.top - m.bottom);
  yAxis(root, scale, m, width, height, yFormat, yTitle);
  for (const t of xTicks) {
    svg("line", { x1: sx(t.x), x2: sx(t.x), y1: height - m.bottom, y2: height - m.bottom + 4, class: "axis" }, root);
    const label = svg("text", { x: sx(t.x), y: height - 8, "text-anchor": "middle" }, root);
    label.textContent = t.label;
  }
  for (const s of series) {
    let d = "";
    let pen = false;
    s.values.forEach((v, i) => {
      if (v === null || v === undefined) { pen = false; return; }
      d += `${pen ? "L" : "M"}${sx(xs[i]).toFixed(1)},${sy(v).toFixed(1)}`;
      pen = true;
    });
    svg("path", { d, fill: "none", stroke: colorOf(s), "stroke-width": s.width || 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, root);
  }
  // crosshair + one tooltip listing every series at the nearest x
  const guide = svg("line", { y1: m.top, y2: height - m.bottom, stroke: cssVar("--axis"), "stroke-width": 1, visibility: "hidden" }, root);
  const dots = series.map((s) => svg("circle", { r: 4.5, fill: colorOf(s), stroke: cssVar("--surface"), "stroke-width": 2, visibility: "hidden" }, root));
  let current = -1;
  const show = (i, clientX, clientY) => {
    current = i;
    guide.setAttribute("x1", sx(xs[i])); guide.setAttribute("x2", sx(xs[i])); guide.setAttribute("visibility", "visible");
    const rows = [];
    series.forEach((s, k) => {
      const v = s.values[i];
      const visible = v !== null && v !== undefined;
      dots[k].setAttribute("visibility", visible ? "visible" : "hidden");
      if (visible) {
        dots[k].setAttribute("cx", sx(xs[i])); dots[k].setAttribute("cy", sy(v));
        rows.push({ color: colorOf(s), label: s.label, value: `${yFormat(v)} MW` });
      }
    });
    showTip(clientX, clientY, xTitle(xs[i]), rows);
  };
  const hide = () => {
    current = -1; guide.setAttribute("visibility", "hidden"); dots.forEach((d) => d.setAttribute("visibility", "hidden")); hideTip();
  };
  const nearest = (clientX) => {
    const box = root.getBoundingClientRect();
    const x = ((clientX - box.left) / box.width) * width;
    const target = x0 + ((x - m.left) / (width - m.left - m.right)) * (x1 - x0);
    let best = 0;
    for (let i = 1; i < xs.length; i++) if (Math.abs(xs[i] - target) < Math.abs(xs[best] - target)) best = i;
    return best;
  };
  const overlay = svg("rect", { x: m.left, y: m.top, width: width - m.left - m.right, height: height - m.top - m.bottom, fill: "transparent" }, root);
  overlay.addEventListener("pointermove", (e) => show(nearest(e.clientX), e.clientX, e.clientY));
  overlay.addEventListener("pointerleave", hide);
  root.setAttribute("tabindex", "0");
  root.addEventListener("keydown", (e) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) return;
    e.preventDefault();
    const i = e.key === "Home" ? 0 : e.key === "End" ? xs.length - 1 : Math.min(xs.length - 1, Math.max(0, (current < 0 ? 0 : current) + (e.key === "ArrowRight" ? 1 : -1)));
    const box = root.getBoundingClientRect();
    show(i, box.left + (sx(xs[i]) / width) * box.width, box.top + box.height / 2);
  });
  root.addEventListener("blur", hide);
}

// ---------------------------------------------------------------- grouped bars
/** opts: { categories:string[], series:[{id,label,color,values:number[]}], yFormat, yTitle, height, ariaLabel, separatorBefore? } */
export function barChart(container, opts) {
  const { categories, series, yFormat = fmt, yTitle, height = 340, ariaLabel, separatorBefore, shortLabels } = opts;
  const { root, width } = frame(container, height, ariaLabel);
  const m = { top: yTitle ? 22 : 12, right: 12, bottom: 30, left: 58 };
  const scale = niceScale(Math.max(...series.flatMap((s) => s.values), 1));
  yAxis(root, scale, m, width, height, yFormat, yTitle);
  const plotW = width - m.left - m.right, plotH = height - m.top - m.bottom;
  const groupW = plotW / categories.length;
  const gap = 2;
  const barW = Math.max(4, Math.min(24, (groupW - 18) / series.length - gap));
  const used = series.length * barW + (series.length - 1) * gap;
  categories.forEach((cat, c) => {
    const left = m.left + c * groupW + (groupW - used) / 2;
    const label = svg("text", { x: m.left + c * groupW + groupW / 2, y: height - 9, "text-anchor": "middle" }, root);
    label.textContent = shortLabels && groupW < 72 ? shortLabels[c] : cat;     // short labels when the group is narrow
    series.forEach((s, k) => {
      const value = s.values[c];
      if (value === null || value === undefined) return;
      const bh = Math.max(1, (value / scale.top) * plotH);
      const x = left + k * (barW + gap), y = m.top + plotH - bh, r = Math.min(4, barW / 2, bh);
      const path = svg("path", {
        d: `M${x},${y + bh} L${x},${y + r} Q${x},${y} ${x + r},${y} L${x + barW - r},${y} Q${x + barW},${y} ${x + barW},${y + r} L${x + barW},${y + bh} Z`,
        fill: colorOf(s), class: "bar", tabindex: 0, role: "img", "aria-label": `${cat}, ${s.label} : ${yFormat(value)} MW`,
      }, root);
      const tipRows = [{ color: colorOf(s), label: s.label, value: `${yFormat(value)} MW` }];
      path.addEventListener("pointermove", (e) => showTip(e.clientX, e.clientY, cat, tipRows));
      path.addEventListener("pointerleave", hideTip);
      path.addEventListener("focus", () => { const b = path.getBoundingClientRect(); showTip(b.right, b.top, cat, tipRows); });
      path.addEventListener("blur", hideTip);
    });
  });
  if (separatorBefore !== undefined) {
    const x = m.left + separatorBefore * groupW;
    svg("line", { x1: x, x2: x, y1: m.top, y2: m.top + plotH, class: "grid" }, root);
  }
}

// ---------------------------------------------------------------- ranked horizontal bars (HTML)
/** rows: [{label, value, color(css var), bold?}] ; bars start at 0, value printed at the tip. */
export function hBarChart(container, rows, format = (v) => fmt(v)) {
  container.replaceChildren();
  const max = Math.max(...rows.map((r) => r.value));
  const list = h("div", "hbars", undefined, container);
  list.setAttribute("role", "list");
  for (const r of rows) {
    const row = h("div", "hrow", undefined, list);
    row.setAttribute("role", "listitem");
    h("div", "name", r.label, row);
    const track = h("div", "track", undefined, row);
    const fill = h("div", "fill", undefined, track);
    fill.style.setProperty("--c", cssVar(r.color));
    fill.style.width = `${(r.value / max) * 78}%`;
    const val = h("span", "val", format(r.value), track);
    if (r.bold) val.style.color = cssVar("--accent");
  }
}

// ---------------------------------------------------------------- table view (the accessible twin of a chart)
/** columns:string[], rows:(string|number)[][], highlightRow? index */
export function dataTable(container, columns, rows, highlightRow = -1) {
  container.replaceChildren();
  const wrap = h("div", "table-wrap", undefined, container);
  const table = h("table", "data", undefined, wrap);
  const head = h("tr", undefined, undefined, h("thead", undefined, undefined, table));
  columns.forEach((c) => h("th", undefined, c, head));
  const body = h("tbody", undefined, undefined, table);
  rows.forEach((row, i) => {
    const tr = h("tr", i === highlightRow ? "final" : undefined, undefined, body);
    row.forEach((cell) => h("td", undefined, String(cell), tr));
  });
}
