/* Seasonal Policy Studio: interface. Computation lives in engine.js; verified numbers in data.js. */
(function () {
"use strict";
const E = Engine, DATA = window.STUDIO_DATA, V = DATA.verified, T = 8;
const $ = (s, r) => (r || document).querySelector(s), $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
const fmt = (v, d = 0) => (isFinite(v) ? v : 0).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const clone = o => JSON.parse(JSON.stringify(o));
const SPL = E.SPLITS, SPS = ["Sp D", "Sp N", "Su D", "Su N", "Fa D", "Fa N", "Wi D", "Wi N"];
const ACT = ["no maintenance", "1 block out", "2 blocks out"];
const COL = { design: "var(--s1)", heat: "var(--s2)", lull: "var(--s3)" }, LET = { design: "D", heat: "H", lull: "L", heat_mild: "h", lull_mild: "l", compound: "HL", custom: "C", unknown: "?", robust: "R", observed: "obs" };
const col = k => COL[k] || "var(--neutral)", letter = k => LET[k] || "•";
const TABS = [["data", "System & data"], ["track", "Track"], ["design", "Design & check"], ["optimise", "Optimise"], ["compare", "Compare"], ["method", "Method"]];
const S = { tab: "data", env: "design", seed: 20261122, simEnv: "heat", simTau: 0, mode: "robust", win: 4, backend: false, opt: { obj: "heat", depth: 2, restarts: 30 }, msg: {} };

/* ---------- tooltips, small helpers ---------- */
const tip = $("#tip");
function showTip(ev, html) { tip.innerHTML = html; tip.style.display = "block"; const w = tip.offsetWidth, h = tip.offsetHeight; let x = ev.clientX + 14, y = ev.clientY + 14; if (x + w > innerWidth - 8) x = ev.clientX - w - 14; if (y + h > innerHeight - 8) y = ev.clientY - h - 14; tip.style.left = Math.max(8, x) + "px"; tip.style.top = Math.max(8, y) + "px"; }
function bindTips(root) { $$("[data-tip]", root).forEach(el => { el.addEventListener("mousemove", ev => showTip(ev, el.dataset.tip)); el.addEventListener("mouseleave", () => tip.style.display = "none"); el.addEventListener("focus", () => { const r = el.getBoundingClientRect(); showTip({ clientX: r.right, clientY: r.top }, el.dataset.tip); }); el.addEventListener("blur", () => tip.style.display = "none"); }); }
function download(name, text, type) { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text], { type: type || "text/plain" })); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000); }
const envLabel = k => k === "unknown" ? "Unknown" : k === "observed" ? "Uploaded observations" : k === "robust" ? "Robust" : (S.case.envs[k] ? S.case.envs[k].label : k);
const polName = k => k === "robust" ? "Robust tree" : envLabel(k) + " tree";
const weightsFor = k => k === "robust" ? Object.fromEntries(S.lib.map(e => [e, 1 / S.lib.length])) : { [k]: 1 };
const valueFor = (k, pol) => Object.entries(weightsFor(k)).reduce((a, [e, w]) => a + w * E.fastValue(S.m, e, pol), 0);
const gapBadge = g => g <= 1e-6 ? `<span class="badge good">✓ optimal</span>` : g <= 0.05 ? `<span class="badge mid">▲ ${fmt(g * 100, 1)}% above best</span>` : `<span class="badge bad">▲ ${fmt(g * 100, 1)}% above best</span>`;

/* ---------- case, trees, monitor ---------- */
function searchBest(key, maxDepth, restarts, seed) {
  const w = weightsFor(key), bound = E.boundOf(S.m, w), u = E.rng(seed || 3); let best = null;
  for (let d = 1; d <= maxDepth; d++) { for (let r = 0; r < restarts; r++) { const x = E.searchRestart(S.m, S.pr, w, d, u); if (!best || x.value < best.value - 1e-9) best = x; } if (best.value - bound <= 1e-6 * Math.max(1, bound)) break; }
  return { tree: E.fromArrays(best.nodes, best.leaves, best.depth), value: best.value, bound, proven: best.value - bound <= 1e-6 * Math.max(1, bound), source: "search in the browser" };
}
function setCase(c, builtin) {
  S.case = c; S.builtin = builtin; S.m = E.makeModel(c); S.pr = E.predicates(S.m);
  S.lib = Object.keys(c.envs).filter(k => c.envs[k].library); if (!c.envs[S.env]) S.env = S.lib[0]; if (!c.envs[S.simEnv]) S.simEnv = Object.keys(c.envs).find(k => k !== S.lib[0]);
  S.trees = {};
  for (const k of S.lib.concat(["robust"])) {
    const v = builtin && V.trees[k];
    if (v && JSON.stringify(S.pr.labels) === JSON.stringify(V.predicate_labels)) S.trees[k] = { tree: E.fromArrays(v.nodes, v.leaves, v.depth), value: v.obj, bound: v.bound, proven: v.proven, gapLimit: v.gap, time: v.time, mip: v, source: "mixed-integer program (verified run)" };
    else S.trees[k] = searchBest(k, 3, k === "robust" ? 24 : 12);
  }
  S.policy = clone(S.trees[S.env] ? S.trees[S.env].tree : S.trees.robust.tree); S.better = null; S.optRun = null; S.cmp = null; S.sim = null; S.proof = null;
  S.mon = null; S.conf = null; setTimeout(trainMon, 20);
}
function trainMon() { S.mon = E.trainMonitor(S.m, S.lib, { win: S.win }); S.conf = E.testMonitor(S.m, S.mon, Object.keys(S.case.envs), { paths: 200 }); S.sim = null; S.cmp = null; render(); }
function refreshEnvs() { S.m = E.makeModel(S.case); S.sim = null; S.cmp = null; S.better = null; }
const tableFor = k => E.tableOf(S.m, S.pr, S.trees[k].tree);

/* ---------- read-only trees ---------- */
function treeHTML(n) { if (n.a !== undefined) return `<span class="leaf">${ACT[n.a]}</span>`; return `<span class="test">${esc(S.pr.labels[n.j])}?</span><ul><li><span class="br">yes</span>${treeHTML(n.yes)}</li><li><span class="br">no</span>${treeHTML(n.no)}</li></ul>`; }
function monHTML(n) { if (n.cls !== undefined) return `<span class="leaf"><span class="sq" style="background:${col(S.mon.classes[n.cls])}"></span>${esc(envLabel(S.mon.classes[n.cls]))} <span class="note">${fmt(n.share * 100)}% pure</span></span>`; return `<span class="test">${E.FEATS[n.f]} ≤ ${fmt(n.th, 3)}?</span><ul><li><span class="br">yes</span>${monHTML(n.yes)}</li><li><span class="br">no</span>${monHTML(n.no)}</li></ul>`; }

/* ---------- charts ---------- */
function niceMax(v) { const p = Math.pow(10, Math.floor(Math.log10(Math.max(v, 1e-9)))), f = v / p; return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * p; }
function caseChart(env) {
  const e = S.m.dv.envs[env], W = 760, H = 320, L = 52, R = 138, Tp = 14, Bm = 34, pw = W - L - R, ph = H - Tp - Bm;
  const ymax = niceMax(Math.max(...e.NL.flat(), S.m.caps[0]) * 1.02), y = v => Tp + ph - Math.max(0, v) / ymax * ph, x = t => L + (t + .5) * pw / T;
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Net load of the three states of each split in ${esc(e.label)}, with the capacity left for 0, 1 and 2 blocks out">`;
  for (let i = 0; i <= 5; i++) { const v = ymax * i / 5; s += `<line x1="${L}" x2="${L + pw}" y1="${y(v)}" y2="${y(v)}" stroke="var(--grid)"/><text x="${L - 8}" y="${y(v) + 4}" text-anchor="end">${fmt(v / 1000, ymax < 8000 ? 1 : 0)} GW</text>`; }
  S.m.caps.forEach((c, a) => { s += `<line x1="${L}" x2="${L + pw}" y1="${y(c)}" y2="${y(c)}" stroke="var(--axis)" stroke-width="1.5"/><text x="${L + pw + 8}" y="${y(c) + 4}">${a} out: ${fmt(c / 1000, 1)} GW</text>`; });
  for (let t = 0; t < T; t++) { s += `<text x="${x(t)}" y="${H - 12}" text-anchor="middle">${SPS[t]}</text>`;
    for (let j = 0; j < 3; j++) { const nl = e.NL[t][j], p0 = S.m.dv.P0[t][j], r = 4 + 9 * Math.sqrt(p0), short = Math.max(0, nl - S.m.caps[0]);
      const tp = `<b>${SPL[t]}, state ${j + 1}</b><br>Net load ${fmt(nl)} MW<br>Demand ${fmt(S.m.dv.D[t][j] * e.d[t])} MW<br>Wind ${fmt(S.m.dv.W[t][j] * e.w[t] * 100)}% of capacity, solar ${fmt(S.m.dv.S[t][j] * 100)}%<br>Design probability ${fmt(p0 * 100)}%` + (short > 0 ? `<br><b>Short by ${fmt(short)} MW even with no block out</b>` : "");
      s += `<circle cx="${x(t) + (j - 1) * 15}" cy="${y(nl)}" r="${r.toFixed(1)}" fill="${col(env)}" fill-opacity=".85" stroke="var(--panel)" stroke-width="2" tabindex="0" data-tip="${esc(tp)}"/>`; } }
  return s + `<line x1="${L}" x2="${L + pw}" y1="${y(0)}" y2="${y(0)}" stroke="var(--axis)"/></svg>`;
}
function bars(vals, o) {
  const W = o.W || 520, H = o.H || 170, L = 46, Tp = 18, Bm = 28, pw = W - L - 8, ph = H - Tp - Bm, mx = o.max || niceMax(Math.max(...vals, 1e-9)), y = v => Tp + ph - v / mx * ph, bw = Math.min(24, pw / vals.length - 12);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(o.aria)}">`;
  for (let i = 0; i <= 2; i++) { const v = mx * i / 2; s += `<line x1="${L}" x2="${L + pw}" y1="${y(v)}" y2="${y(v)}" stroke="var(--grid)"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${fmt(v, o.dec || 0)}</text>`; }
  vals.forEach((v, i) => { const cx = L + (i + .5) * pw / vals.length, h = Math.max(0, y(0) - y(v)), r = Math.min(4, h);
    if (h > 0.5) s += `<path d="M${cx - bw / 2},${y(0)} v${-(h - r)} q0,${-r} ${r},${-r} h${bw - 2 * r} q${r},0 ${r},${r} v${h - r} z" fill="${o.color || "var(--bar)"}" tabindex="0" data-tip="${esc(`<b>${SPL[i]}</b><br>${fmt(v, o.tipDec === undefined ? 1 : o.tipDec)} ${o.unit}`)}"/>`;
    if (v > mx * 0.004) s += `<text class="val" x="${cx}" y="${y(v) - 5}" text-anchor="middle">${fmt(v, o.dec || 0)}</text>`;
    s += `<text x="${cx}" y="${H - 9}" text-anchor="middle">${SPS[i]}</text>`; });
  return s + `<line x1="${L}" x2="${L + pw}" y1="${y(0)}" y2="${y(0)}" stroke="var(--axis)"/></svg>`;
}
function simChart(sim) {
  const st = sim.st, K = st.length, W = 1100, L = 104, Tp = 12, ph = 190, pw = W - L - 10, rowH = 20, gap = 6, rows = sim.rows, H = Tp + ph + 22 + rows.length * (rowH + gap) + 6;
  const ymax = niceMax(Math.max(...st.map(r => r.nl), S.m.caps[0]) * 1.02), y = v => Tp + ph - Math.max(0, v) / ymax * ph, cw = pw / K, x = k => L + (k + .5) * cw;
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Simulated or observed years: net load, capacity left by the operating policy, monitor label, active policy and blocks out per split">`;
  for (let i = 0; i <= 5; i++) { const v = ymax * i / 5; s += `<line x1="${L}" x2="${L + pw}" y1="${y(v)}" y2="${y(v)}" stroke="var(--grid)"/><text x="${L - 8}" y="${y(v) + 4}" text-anchor="end">${fmt(v / 1000)} GW</text>`; }
  if (sim.tau !== null) s += `<line x1="${L + sim.tau * cw}" x2="${L + sim.tau * cw}" y1="${Tp - 4}" y2="${Tp + ph}" stroke="var(--axis)" stroke-width="1.5"/><text x="${L + sim.tau * cw + 5}" y="${Tp + 6}">change</text>`;
  for (let k = T; k < K; k += T) s += `<line x1="${L + k * cw}" x2="${L + k * cw}" y1="${Tp + ph}" y2="${H - 6}" stroke="var(--line)"/>`;
  let path = ""; sim.steps.forEach((r, k) => { path += `${k ? "L" : "M"}${L + k * cw},${y(r.cap)} L${L + (k + 1) * cw},${y(r.cap)} `; if (r.shed > 0) s += `<rect x="${x(k) - 5}" y="${y(st[k].nl)}" width="10" height="${Math.max(2, y(r.cap) - y(st[k].nl))}" fill="var(--crit)" rx="2"/>`; });
  s += `<path d="${path}" fill="none" stroke="var(--ink)" stroke-width="2" stroke-linejoin="round"/>`;
  const y0 = Tp + ph + 22;
  st.forEach((r, k) => { const stp = sim.steps[k], lb = sim.labels[k];
    const tp = `<b>Year ${Math.floor(k / T) + 1}, ${SPL[r.t]}</b><br>${sim.tau === null ? "" : "True environment: " + esc(envLabel(r.env)) + "<br>"}Monitor label: ${esc(envLabel(lb.env))}${lb.warm ? " (warming up)" : ""}<br>Active policy: ${esc(polName(sim.pols[k]))}<br>Blocks out ${stp.a}, backlog left ${stp.left}<br>Net load ${fmt(r.nl)} MW, capacity ${fmt(stp.cap)} MW` + (stp.shed > 0 ? `<br><b>Unserved ${fmt(stp.shed)} MW</b>` : "");
    s += `<circle cx="${x(k)}" cy="${y(r.nl)}" r="4" fill="var(--ink)" stroke="var(--panel)" stroke-width="2"/><text x="${x(k)}" y="${Tp + ph + 14}" text-anchor="middle">${K > 32 ? SPS[r.t].slice(0, 2) : SPS[r.t]}</text>`;
    rows.forEach((row, i) => { const v = row.f(k), yy = y0 + i * (rowH + gap);
      if (row.kind === "env") s += `<rect x="${L + k * cw + 1}" y="${yy}" width="${cw - 2}" height="${rowH}" rx="3" fill="${col(v)}"/><text x="${x(k)}" y="${yy + 14}" text-anchor="middle" style="fill:#fff;font-weight:600">${letter(v)}</text>`;
      else s += `<rect x="${L + k * cw + 1}" y="${yy}" width="${cw - 2}" height="${rowH}" rx="3" fill="var(--sunk)" stroke="var(--line)"/><text class="val" x="${x(k)}" y="${yy + 14}" text-anchor="middle">${v}</text>`; });
    s += `<rect x="${L + k * cw}" y="${Tp}" width="${cw}" height="${H - Tp - 6}" fill="transparent" tabindex="0" data-tip="${esc(tp)}"/>`; });
  rows.forEach((row, i) => s += `<text x="${L - 8}" y="${y0 + i * (rowH + gap) + 14}" text-anchor="end">${row.name}</text>`);
  return s + `</svg>`;
}
function hbars(groups, strat, winKey) {
  const W = 1100, colW = W / 3, rowH = 22, blockH = 42 + strat.length * rowH, H = Math.ceil(groups.length / 3) * blockH;
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Unserved energy per year for each strategy and scenario">`;
  groups.forEach((g, i) => { const cx = (i % 3) * colW, cy = Math.floor(i / 3) * blockH, L = 190, pw = colW - L - 60, mx = Math.max(...strat.map(([k]) => g.vals[k].mean), 1e-9);
    s += `<text class="val" x="${cx + 4}" y="${cy + 16}" style="font-weight:600">${esc(g.title)}</text>`;
    strat.forEach(([k, lab], r) => { const v = g.vals[k], yy = cy + 28 + r * rowH, w = Math.max(1, v.mean / mx * pw), win = k === winKey, ref = k === "bound" || k === "nominal";
      s += `<text x="${cx + L - 8}" y="${yy + 12}" text-anchor="end" ${win ? 'class="val" style="font-weight:600"' : ""}>${lab}</text><path d="M${cx + L},${yy + 2} h${Math.max(0, w - 4)} q4,0 4,4 v6 q0,4 -4,4 h${-Math.max(0, w - 4)} z" fill="${win ? "var(--bar-strong)" : ref ? "var(--neutral)" : "var(--bar)"}" tabindex="0" data-tip="${esc(`<b>${lab}</b><br>${fmt(v.mean)} GWh per year (95% interval ± ${fmt(v.ci)})`)}"/><text class="val" x="${cx + L + w + 6}" y="${yy + 13}">${fmt(v.mean)}</text>`; }); });
  return s + `</svg>`;
}
function anytimeChart(hist, bound) {
  const W = 560, H = 220, L = 60, Tp = 14, Bm = 32, pw = W - L - 70, ph = H - Tp - Bm, n = hist.length, mx = niceMax(Math.max(...hist, bound) * 1.02), mn = Math.min(bound, ...hist) * 0.98, y = v => Tp + ph - (v - mn) / (mx - mn) * ph, x = i => L + (n > 1 ? i / (n - 1) : 0.5) * pw;
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Best tree found so far after each restart, against the bound">`;
  for (let i = 0; i <= 3; i++) { const v = mn + (mx - mn) * i / 3; s += `<line x1="${L}" x2="${L + pw}" y1="${y(v)}" y2="${y(v)}" stroke="var(--grid)"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${fmt(v)}</text>`; }
  s += `<line x1="${L}" x2="${L + pw}" y1="${y(bound)}" y2="${y(bound)}" stroke="var(--good)" stroke-width="2"/><text class="val" x="${L + pw + 6}" y="${y(bound) + 4}">bound ${fmt(bound)}</text>`;
  let d = ""; hist.forEach((v, i) => { d += (i ? `L${x(i)},${y(hist[i - 1])} L${x(i)},${y(v)}` : `M${x(0)},${y(v)}`); });
  s += `<path d="${d}" fill="none" stroke="var(--s1)" stroke-width="2" stroke-linejoin="round"/><circle cx="${x(n - 1)}" cy="${y(hist[n - 1])}" r="4.5" fill="var(--s1)" stroke="var(--panel)" stroke-width="2"/>`;
  if (Math.abs(y(hist[n - 1]) - y(bound)) > 14) s += `<text class="val" x="${L + pw + 6}" y="${y(hist[n - 1]) + 4}">tree ${fmt(hist[n - 1])}</text>`;
  hist.forEach((v, i) => s += `<rect x="${x(i) - Math.max(3, pw / n / 2)}" y="${Tp}" width="${Math.max(6, pw / n)}" height="${ph}" fill="transparent" data-tip="${esc(`<b>After restart ${i + 1}</b><br>Best tree ${fmt(v, 1)} GWh<br>${fmt((v - bound) / bound * 100, 2)}% above the bound`)}"/>`);
  return s + `<text x="${L + pw / 2}" y="${H - 8}" text-anchor="middle">restarts of the search</text></svg>`;
}

/* ---------- hero ---------- */
function policyEval() { const pol = E.tableOf(S.m, S.pr, S.policy); return Object.assign(E.evaluate(S.m, S.env, pol), { pol }); }
function renderHero() {
  const ev = policyEval(), g = ev.gap, pct = Math.max(2, Math.min(100, 100 - g * 400));
  $("#chipCase").textContent = S.builtin ? "Built-in case" : "Case: " + S.case.name;
  $("#hero").innerHTML = `
   <div class="tile"><div class="l">Environment in focus</div><select id="envSel" aria-label="Environment in focus">${Object.keys(S.case.envs).map(k => `<option value="${k}"${k === S.env ? " selected" : ""}>${esc(S.case.envs[k].label)}</option>`).join("")}</select><div class="s">${S.case.envs[S.env].library ? "has a library policy" : "test environment, no library policy"}</div></div>
   <div class="tile"><div class="l">Best possible</div><div class="v">${fmt(ev.bound)} <small>GWh / year</small></div><div class="s">any policy, by dynamic programming</div></div>
   <div class="tile"><div class="l">Policy in the designer</div><div class="v">${fmt(ev.value)} <small>GWh / year</small></div><div class="s">${E.leavesOf(S.policy)} rule${E.leavesOf(S.policy) === 1 ? "" : "s"}, depth ${E.depthOf(S.policy)}</div></div>
   <div class="tile"><div class="l">Optimality gap</div><div class="v">${fmt(g * 100, g < 0.1 ? 2 : 1)}%</div><div class="meter" role="img" aria-label="Closeness to the best possible policy"><div style="width:${pct}%;background:${g <= 1e-6 ? "var(--good)" : g <= 0.05 ? "var(--warn)" : "var(--crit)"}"></div></div><div class="s" style="margin-top:4px">${gapBadge(g)}</div></div>`;
  $("#envSel").onchange = e => { S.env = e.target.value; S.better = null; render(); };
}

/* ---------- tab 1: system & data ---------- */
function renderData() {
  const f = S.case.fleet, cu = S.case.envs.custom, e = S.m.dv.envs[S.env];
  const sl = (id, lab, v, mn, mx) => `<div class="sl"><span>${lab}</span><input type="range" id="${id}" min="${mn}" max="${mx}" step="1" value="${Math.round(v * 100)}" aria-label="${lab}"><span class="num" id="${id}v">${Math.round(v * 100)}%</span></div>`;
  $("#view").innerHTML = `<div class="wide">
  <section class="panel"><h2>The stochastic system: eight splits, three states each</h2>
    <p class="note">Each dot is one joint state of demand, wind and solar; its area follows its probability. Lines show the capacity left when 0, 1 or 2 maintenance blocks are out. Environment shown: <b>${esc(e.label)}</b> (change it in the tile above).</p>
    <div id="cCase">${caseChart(S.env)}</div>
    <div class="tscroll"><table><tr><th>Split</th><th class="num">State</th><th class="num">Probability</th><th class="num">Demand MW</th><th class="num">Wind</th><th class="num">Solar</th><th class="num">Net load MW</th></tr>
    ${S.case.splits.map((s, t) => s.states.map((x, j) => `<tr><td>${j === 0 ? SPL[t] : ""}</td><td class="num">${j + 1}</td><td class="num">${fmt(S.m.dv.P0[t][j] * 100)}%</td><td class="num">${fmt(x.demand * e.d[t])}</td><td class="num">${fmt(x.wind_cf * e.w[t] * 100)}%</td><td class="num">${fmt(x.solar_cf * 100)}%</td><td class="num">${fmt(e.NL[t][j])}</td></tr>`).join("")).join("")}</table></div>
  </section>
  <div style="display:grid;gap:16px;align-content:start">
  <section class="panel"><h2>Data source</h2>
    <p class="note">Now loaded: <b>${esc(S.case.name)}</b>.</p>
    <div class="drop"><b>Upload hourly observations (CSV)</b><span class="note">Columns: <code>timestamp</code> (or <code>month</code>, <code>hour</code>), <code>demand_MW</code>, <code>wind_cf</code> or <code>wind_MW</code>, optional <code>solar_cf</code> or <code>solar_MW</code>. The studio splits the rows by season and day or night and groups each split into three joint states.</span><input type="file" id="upHourly" accept=".csv,.txt" aria-label="Upload hourly observations"></div>
    <div class="drop"><b>Upload a state table (CSV)</b><span class="note">24 rows: <code>split</code> (1 to 8), <code>state</code> (1 to 3), <code>p</code>, <code>demand_MW</code>, <code>wind_cf</code>, <code>solar_cf</code>.</span><input type="file" id="upTable" accept=".csv,.txt" aria-label="Upload a state table"></div>
    ${S.msg.data ? `<div class="msg${S.msg.dataErr ? " err" : ""}">${esc(S.msg.data)}</div>` : ""}
    <div class="controls"><button class="btn" id="dlTable">Download current state table</button><button class="btn" id="dlHourly">Hourly template</button>${S.builtin ? "" : `<button class="btn" id="resetCase">Back to the built-in case</button>`}</div>
  </section>
  <section class="panel"><h2>Fleet and maintenance</h2>
    <div class="form">
      <label>Conventional MW<input type="number" id="fConv" value="${Math.round(f.conv)}"></label><label>Wind MW<input type="number" id="fWind" value="${Math.round(f.wind)}"></label><label>Solar MW<input type="number" id="fSolar" value="${Math.round(f.solar)}"></label>
      <label>Availability %<input type="number" id="fAvail" value="${Math.round(f.avail * 100)}" min="50" max="100"></label><label>Block size %<input type="number" id="fBlock" value="${Math.round(f.blockFrac * 100)}" min="2" max="25"></label><label>Block-splits due<input type="number" id="fBack" value="${f.backlog}" min="2" max="14"></label>
    </div>
    <div class="controls"><button class="btn primary" id="applyFleet">Apply and recompute</button><span class="note">${fmt(S.m.nReach)} reachable states, ${S.pr.labels.length} candidate tests.</span></div>
  </section>
  <section class="panel"><h2>Scenario generator</h2>
    <p class="note">Build a mock-up environment and test every policy against it. It appears as “Custom scenario” in all tabs.</p>
    <h3>Demand, share of the design value</h3>${["Spring", "Summer", "Fall", "Winter"].map((n, i) => sl("cd" + i, n, cu.d[i], 85, 125)).join("")}
    <h3>Wind output, share of the design value</h3>${["Spring", "Summer", "Fall", "Winter"].map((n, i) => sl("cw" + i, n, cu.w[i], 10, 120)).join("")}
    <label style="display:flex;gap:8px;align-items:center"><input type="checkbox" id="cPers"${cu.kind ? " checked" : ""}> Stressed states are more likely and persist</label>
    <label style="display:flex;gap:8px;align-items:center"><input type="checkbox" id="cLib"${cu.library ? " checked" : ""}> Add to the policy library (builds a tree for it and retrains the monitor)</label>
    <div class="controls"><button class="btn primary" id="applyCustom">Apply scenario</button><button class="btn" id="focusCustom">Show it</button></div>
  </section></div></div>`;
  bindTips($("#view"));
  for (let i = 0; i < 4; i++) for (const p of ["cd", "cw"]) $("#" + p + i).oninput = ev => $("#" + p + i + "v").textContent = ev.target.value + "%";
  $("#applyCustom").onclick = () => { const c = S.case.envs.custom, wasLib = c.library; for (let i = 0; i < 4; i++) { c.d[i] = +$("#cd" + i).value / 100; c.w[i] = +$("#cw" + i).value / 100; }
    const hot = c.d.some(v => v > 1.001), calm = c.w.some(v => v < 0.999); c.kind = $("#cPers").checked ? (hot && calm ? "both" : hot ? "heat" : calm ? "drought" : null) : null; c.library = $("#cLib").checked;
    if (c.library || wasLib) setCase(S.case, S.builtin); else refreshEnvs(); S.env = "custom"; render(); };
  $("#focusCustom").onclick = () => { S.env = "custom"; render(); };
  $("#applyFleet").onclick = () => { const c = clone(S.case); c.fleet = { conv: +$("#fConv").value, wind: +$("#fWind").value, solar: +$("#fSolar").value, avail: Math.min(1, Math.max(.5, +$("#fAvail").value / 100)), blockFrac: Math.min(.25, Math.max(.02, +$("#fBlock").value / 100)), backlog: Math.min(14, Math.max(2, Math.round(+$("#fBack").value))), rho: f.rho };
    const same = JSON.stringify(c.fleet) === JSON.stringify(DATA.case.fleet) && S.builtin; setCase(c, same); S.builtin = same; if (!same && S.case.name === DATA.case.name) S.case.name = "Built-in data, edited fleet"; S.builtin = same; render(); };
  const up = (id, fn) => $("#" + id).onchange = ev => { const file = ev.target.files[0]; if (!file) return; const rd = new FileReader(); rd.onload = () => { try { const r = fn(rd.result, DATA.case.splits.map(s => s.hours)), c = clone(S.case); c.splits = r.splits; c.name = file.name;
        const peak = Math.max(...r.splits.flatMap(s => s.states.map(x => x.demand))), b = DATA.case.fleet, bp = Math.max(...DATA.case.splits.flatMap(s => s.states.map(x => x.demand)));
        c.fleet = Object.assign({}, c.fleet, { conv: Math.round(b.conv / bp * peak), wind: Math.round(b.wind / bp * peak), solar: Math.round(b.solar / bp * peak) });
        setCase(c, false); S.msg = { data: `Loaded ${fmt(r.rows)} rows from ${file.name}. Fleet rescaled to the built-in ratios (peak demand ${fmt(peak)} MW); edit it below if needed. Library trees were rebuilt by search in the browser.` }; }
      catch (err) { S.msg = { data: "Could not read the file: " + err.message, dataErr: true }; } render(); }; rd.readAsText(file); };
  up("upHourly", E.statesFromHourly); up("upTable", E.statesFromTable);
  $("#dlTable").onclick = () => download("state_table.csv", "split,state,p,demand_MW,wind_cf,solar_cf\n" + S.case.splits.map((s, t) => s.states.map((x, j) => [t + 1, j + 1, S.m.dv.P0[t][j].toFixed(4), x.demand.toFixed(1), x.wind_cf.toFixed(4), x.solar_cf.toFixed(4)].join(",")).join("\n")).join("\n"), "text/csv");
  $("#dlHourly").onclick = () => download("hourly_template.csv", "timestamp,demand_MW,wind_cf,solar_cf\n2025-01-01 00:00,25100,0.62,0\n2025-01-01 01:00,24350,0.66,0\n2025-01-01 12:00,27800,0.31,0.44\n", "text/csv");
  if ($("#resetCase")) $("#resetCase").onclick = () => { S.msg = {}; setCase(clone(DATA.case), true); render(); };
}

/* ---------- tab 2: track ---------- */
function runSim(obs) {
  if (!S.mon) return; const m = S.m; let st, tau = null;
  if (obs) { st = []; let o = 1; obs.forEach((r, k) => { const t = k % T, g = E.gaps(m, t, r.demand, r.wind); let near = 0, bd = Infinity; for (let j = 0; j < 3; j++) { const d = Math.pow((r.demand - m.dv.D[t][j]) / (Math.max(...m.dv.D[t]) - Math.min(...m.dv.D[t]) || 1), 2) + Math.pow(r.wind - m.dv.W[t][j], 2); if (d < bd) { bd = d; near = j; } }
      st.push({ k, t, env: "observed", o, demand: r.demand, wind: r.wind, gD: g.gD, gW: g.gW, j: near, nl: r.demand - r.wind * S.case.fleet.wind - m.dv.S[t][near] * S.case.fleet.solar }); o = m.dv.RANK[t][near]; }); }
  else { tau = T + S.simTau; const seq = []; for (let k = 0; k < 3 * T; k++) seq.push(k < tau || S.simEnv === "none" ? S.lib[0] : S.simEnv); st = E.stream(m, seq, E.rng(S.seed), S.mon.noise); if (S.simEnv === "none") tau = null; S.simSeq = seq; }
  const labels = E.labelStream(S.mon, st), tabs = {}; for (const k of Object.keys(S.trees)) tabs[k] = tableFor(k);
  const libPol = k => labels[k].env === "unknown" ? "robust" : labels[k].env, runs = { nominal: E.follow(m, st, () => tabs[S.lib[0]]), robust: E.follow(m, st, () => tabs.robust), monitor: E.follow(m, st, k => tabs[libPol(k)]) };
  if (!obs) { const seqTabs = E.dpSequence(m, S.simSeq); runs.bound = E.follow(m, st, k => seqTabs[k]); }
  const shown = S.mode === "monitor" ? "monitor" : "robust", pols = st.map((_, k) => shown === "monitor" ? libPol(k) : "robust");
  let delay = null; if (tau !== null) { const want = S.lib.includes(S.simEnv) ? S.simEnv : null; for (let k = tau; k < st.length; k++) if (want ? labels[k].env === want : labels[k].env !== S.lib[0]) { delay = k - tau; break; } }
  const rows = [{ name: "Monitor label", kind: "env", f: k => labels[k].env }, { name: "Active policy", kind: "env", f: k => pols[k] }, { name: "Blocks out", kind: "num", f: k => runs[shown].steps[k].a }];
  if (!obs) rows.unshift({ name: "True environment", kind: "env", f: k => st[k].env });
  S.sim = { st, labels, steps: runs[shown].steps, pols, tau, runs, delay, rows, obs: !!obs, years: st.length / T };
}
function renderTrack() {
  if (!S.mon) { $("#view").innerHTML = `<section class="panel"><p>Training the monitor…</p></section>`; return; }
  if (!S.sim) runSim();
  const sim = S.sim, names = { nominal: `${envLabel(S.lib[0])} tree, no monitor`, robust: "Robust tree (F1)", monitor: "Monitor and library (F2)", bound: "Lower bound (knows the change)" };
  const alarm = sim.obs ? (() => { const n = sim.labels.filter((l, k) => k >= S.mon.win && l.env !== S.lib[0]).length; return n ? `<div class="verdict mid"><b>Alarm.</b> ${n} of ${sim.st.length - S.mon.win} windows do not look like the design year. The guarantees computed for the design year should not be relied on for these splits.</div>` : `<div class="verdict good"><b>No alarm.</b> Every window is consistent with the design year.</div>`; })()
    : sim.tau === null ? "" : sim.delay === null ? `<div class="verdict bad"><b>Not detected.</b> The monitor never gave the expected label on this path.</div>` : `<div class="verdict ${sim.delay <= 3 ? "good" : "mid"}"><b>Change ${S.lib.includes(S.simEnv) ? "identified" : "flagged"} after ${sim.delay} split${sim.delay === 1 ? "" : "s"}.</b> The monitor sees the last ${S.mon.win} splits, so it cannot react at once.</div>`;
  $("#view").innerHTML = `<section class="panel"><h2>Track the system: which environment is it in?</h2>
    <p class="note">Year 1 follows the design year. In year 2 the environment changes at the split you choose. The monitor reads demand and wind of the last ${S.mon.win} splits and names the environment; every strategy gets the same random path.</p>
    <div class="controls">
      <label>Environment after the change<select id="tEnv">${Object.keys(S.case.envs).filter(k => k !== S.lib[0]).map(k => `<option value="${k}"${k === S.simEnv ? " selected" : ""}>${esc(S.case.envs[k].label)}</option>`).join("")}<option value="none"${S.simEnv === "none" ? " selected" : ""}>No change</option></select></label>
      <label>Change at<select id="tTau">${SPL.map((n, i) => `<option value="${i}"${i === S.simTau ? " selected" : ""}>Year 2, ${n.toLowerCase()}</option>`).join("")}</select></label>
      <label>Policy shown below<select id="tMode"><option value="robust"${S.mode === "robust" ? " selected" : ""}>Robust tree, monitor as alarm (F1)</option><option value="monitor"${S.mode === "monitor" ? " selected" : ""}>Monitor selects from the library (F2)</option></select></label>
      <button class="btn primary" id="tNew">New random path</button>
      <label>Or upload observations (CSV: demand_MW, wind_cf per split)<input type="file" id="tUp" accept=".csv,.txt"></label>
    </div>
    ${S.msg.track ? `<div class="msg${S.msg.trackErr ? " err" : ""}">${esc(S.msg.track)}</div>` : ""}${alarm}
    <div class="hero">${Object.keys(sim.runs).map(k => `<div class="tile"${(k === "monitor") === (S.mode === "monitor") && k !== "nominal" && k !== "bound" ? ' style="border-color:var(--accent)"' : ""}><div class="l">${names[k]}</div><div class="v">${fmt(sim.runs[k].cost / sim.years)} <small>GWh / year</small></div><div class="s">on this path</div></div>`).join("")}</div>
    <div class="legend"><span><span class="dot" style="background:var(--ink)"></span>net load</span><span><span class="key-line"></span>capacity left</span><span><span class="sq" style="background:var(--crit)"></span>unserved</span>${S.lib.map(k => `<span><span class="sq" style="background:${col(k)}"></span>${letter(k)} ${esc(envLabel(k))}</span>`).join("")}<span><span class="sq" style="background:var(--neutral)"></span>other, unknown or robust</span></div>
    <div id="cSim">${simChart(sim)}</div></section>
  <div class="two"><section class="panel"><h2>The monitor tree</h2>
    <p class="note">Trained in the browser on simulated windows of the library environments (accuracy on training windows ${fmt(S.mon.trainAcc * 100)}%). A window outside the range seen for its class is labelled unknown.</p>
    <div class="tree"><ul><li>${monHTML(S.mon.tree)}</li></ul></div>
    <div class="controls"><label>Window (splits)<select id="tWin">${[2, 3, 4, 6].map(w => `<option${w === S.win ? " selected" : ""}>${w}</option>`).join("")}</select></label></div></section>
  <section class="panel"><h2>How often each label is given</h2><p class="note">Fresh simulations of every environment, including those the monitor never saw.</p>
    <div class="tscroll"><table><tr><th>True environment</th>${S.mon.classes.concat(["unknown"]).map(c => `<th class="num">${esc(envLabel(c))}</th>`).join("")}</tr>
    ${Object.keys(S.conf).map(e => `<tr><td>${esc(envLabel(e))}${S.case.envs[e].library ? "" : ' <span class="note">(not in library)</span>'}</td>${S.mon.classes.concat(["unknown"]).map(c => `<td class="num">${fmt(S.conf[e][c] * 100)}%</td>`).join("")}</tr>`).join("")}</table></div></section></div>`;
  bindTips($("#view"));
  $("#tEnv").onchange = e => { S.simEnv = e.target.value; S.msg = {}; S.sim = null; render(); }; $("#tTau").onchange = e => { S.simTau = +e.target.value; S.sim = null; render(); };
  $("#tMode").onchange = e => { S.mode = e.target.value; const o = S.sim.obs ? S.obsRows : null; S.sim = null; runSim(o); render(); };
  $("#tNew").onclick = () => { S.seed = (S.seed * 1664525 + 1013904223) >>> 0; S.msg = {}; S.sim = null; render(); };
  $("#tWin").onchange = e => { S.win = +e.target.value; S.mon = null; render(); setTimeout(trainMon, 20); };
  $("#tUp").onchange = ev => { const file = ev.target.files[0]; if (!file) return; const rd = new FileReader(); rd.onload = () => { try { const { head, rows } = E.parseCSV(rd.result), di = head.findIndex(h => /demand|load/.test(h)), wi = head.findIndex(h => /wind/.test(h)); if (di < 0 || wi < 0) throw new Error("columns demand_MW and wind_cf are needed");
        const obs = rows.map(r => ({ demand: +r[di], wind: +r[wi] })).filter(r => isFinite(r.demand) && isFinite(r.wind)); if (obs.length < S.mon.win + 1) throw new Error("at least " + (S.mon.win + 1) + " rows are needed"); S.obsRows = obs.slice(0, 48); runSim(S.obsRows); S.msg = { track: `Showing ${S.obsRows.length} uploaded splits from ${file.name}, starting at spring day. Net load uses the solar share of the nearest design state.` }; }
      catch (err) { S.msg = { track: "Could not read the file: " + err.message, trackErr: true }; } render(); }; rd.readAsText(file); };
}

/* ---------- tab 3: design & check ---------- */
const nodeAt = p => { let n = S.policy; for (const c of p) n = c === "y" ? n.yes : n.no; return n; };
function setAt(p, v) { if (!p) { S.policy = v; return; } const par = nodeAt(p.slice(0, -1)); if (p.slice(-1) === "y") par.yes = v; else par.no = v; }
function editHTML(n, p) {
  if (n.a !== undefined) return `<span class="node"><span class="act" role="group" aria-label="Action">${ACT.map((t, a) => `<button data-do="act" data-p="${p}" data-v="${a}" aria-pressed="${n.a === a}">${a}</button>`).join("")}</span><span class="note">${ACT[n.a]}</span>${p.length < 4 ? `<button class="mini" data-do="split" data-p="${p}">+ add a test</button>` : ""}</span>`;
  return `<span class="node q"><span class="note">if</span><select data-do="pred" data-p="${p}" aria-label="Test">${S.pr.labels.map((l, j) => `<option value="${j}"${j === n.j ? " selected" : ""}>${esc(l)}</option>`).join("")}</select><button class="mini" data-do="drop" data-p="${p}">remove</button></span><ul><li><span class="br">yes</span>${editHTML(n.yes, p + "y")}</li><li><span class="br">no</span>${editHTML(n.no, p + "n")}</li></ul>`;
}
function verdictHTML(ev) {
  const g = ev.gap, worst = ev.lossBySplit.indexOf(Math.max(...ev.lossBySplit));
  if (g <= 1e-6) return `<div class="verdict good"><b>This policy is optimal.</b> In ${esc(envLabel(S.env))} it costs ${fmt(ev.value, 1)} GWh per year, exactly the best possible value. No policy of any form, tree or not, does better.</div>`;
  return `<div class="verdict ${g <= 0.05 ? "mid" : "bad"}"><b>This policy is not optimal.</b> In ${esc(envLabel(S.env))} it costs ${fmt(ev.value, 1)} GWh per year, ${fmt(ev.value - ev.bound, 1)} GWh (${fmt(g * 100, 1)}%) above the best possible ${fmt(ev.bound, 1)}. The largest avoidable loss is in <b>${SPL[worst].toLowerCase()}</b> (${fmt(ev.lossBySplit[worst], 1)} GWh).</div>`;
}
function renderDesign() {
  const ev = policyEval(), d = Math.max(1, Math.min(3, E.depthOf(S.policy)));
  const cross = Object.keys(S.case.envs).map(k => { const x = E.evaluate(S.m, k, ev.pol); return `<tr${k === S.env ? ' class="win"' : ""}><td>${esc(envLabel(k))}</td><td class="num">${fmt(x.value, 1)}</td><td class="num">${fmt(x.bound, 1)}</td><td class="num">${fmt(x.value - x.bound, 1)}</td><td>${gapBadge(x.gap)}</td></tr>`; }).join("");
  const b = S.better;
  $("#view").innerHTML = `<div class="wide">
  <section class="panel"><h2>Your policy</h2>
    <p class="note">A policy is a decision tree: tests on the split, the maintenance backlog, the slack and the last net load; each leaf sets the blocks out (0, 1 or 2). Start from a library tree or build your own.</p>
    <div class="seg" id="presets">${Object.keys(S.trees).map(k => `<button data-k="${k}">${esc(polName(k))}</button>`).join("")}<button data-k="_zero">Never maintain</button><button data-k="_two">Always 2 blocks</button></div>
    <div class="tree" id="editor"><ul><li>${editHTML(S.policy, "")}</li></ul></div>
    <div class="controls"><button class="btn" id="dlPol">Download policy (JSON)</button></div>
  </section>
  <section class="panel"><h2>Is it optimal?</h2>
    ${verdictHTML(ev)}
    <p class="note">The bound is the optimum of the unrestricted Markov decision process, computed exactly by dynamic programming. It holds for every policy, so the gap is a certificate, not an estimate.</p>
    <h3>Is it the best tree of its size?</h3>
    <div class="controls"><button class="btn primary" id="findBetter">Search for a better tree of depth ${d}</button>${S.backend ? `<button class="btn" id="proveCur">Prove with the solver</button>` : ""}</div>
    ${b ? (b.found ? `<div class="verdict mid"><b>A better tree of depth ${b.depth} exists.</b> Cost ${fmt(b.value, 1)} GWh (${fmt((b.value - ev.bound) / ev.bound * 100, 1)}% above the best possible). <button class="btn" id="adopt">Load it into the designer</button></div>` : `<div class="verdict good"><b>No better tree of depth ${b.depth} found</b> in ${b.restarts} restarts. This is a search result, not a proof${S.backend ? "; use “Prove with the solver”." : "; start the solver back end to prove it."}</div>`) : ""}
    ${S.proof ? `<div class="verdict ${S.proof.err ? "bad" : S.proof.proven ? "good" : "mid"}">${S.proof.html}</div>` : ""}
  </section></div>
  <div class="two">
    <section class="panel"><h2>Where the policy loses</h2><p class="note">Avoidable loss per split in ${esc(envLabel(S.env))}, GWh per year. The bars add up to the gap (${fmt(ev.value - ev.bound, 1)} GWh).</p>${bars(ev.lossBySplit, { aria: "Avoidable loss per split", unit: "GWh avoidable", color: "var(--crit)", dec: ev.value - ev.bound < 20 ? 1 : 0 })}</section>
    <section class="panel"><h2>What the policy does</h2><p class="note">Expected maintenance blocks out per split in ${esc(envLabel(S.env))}.</p>${bars(ev.blocks, { aria: "Expected blocks out per split", unit: "blocks out on average", color: col(S.env), max: 2, dec: 1, tipDec: 2 })}</section>
  </div>
  <section class="panel"><h2>The same policy in every environment</h2><div class="tscroll"><table><tr><th>Environment</th><th class="num">Policy cost</th><th class="num">Best possible</th><th class="num">Avoidable</th><th>Verdict</th></tr>${cross}</table></div></section>`;
  bindTips($("#view"));
  $("#presets").onclick = e => { const k = e.target.dataset.k; if (!k) return; S.policy = k === "_zero" ? { a: 0 } : k === "_two" ? { a: 2 } : clone(S.trees[k].tree); S.better = null; S.proof = null; render(); };
  $("#editor").onclick = e => { const el = e.target.closest("[data-do]"); if (!el || el.tagName === "SELECT") return; const p = el.dataset.p, n = nodeAt(p);
    if (el.dataset.do === "act") n.a = +el.dataset.v; else if (el.dataset.do === "split") setAt(p, { j: 0, yes: { a: n.a }, no: { a: n.a } }); else if (el.dataset.do === "drop") setAt(p, { a: (function f(x) { return x.a !== undefined ? x.a : f(x.yes); })(n) });
    S.better = null; S.proof = null; render(); };
  $("#editor").onchange = e => { if (e.target.dataset.do === "pred") { nodeAt(e.target.dataset.p).j = +e.target.value; S.better = null; S.proof = null; render(); } };
  $("#dlPol").onclick = () => download("policy.json", JSON.stringify({ environment: S.env, cost_GWh: ev.value, bound_GWh: ev.bound, tests: S.pr.labels, tree: S.policy }, null, 1), "application/json");
  $("#findBetter").onclick = () => { const u = E.rng(S.seed + 5), w = { [S.env]: 1 }; let best = null; const R = 30; for (let r = 0; r < R; r++) { const x = E.searchRestart(S.m, S.pr, w, d, u); if (!best || x.value < best.value - 1e-9) best = x; }
    S.better = { found: best.value < ev.value - 1e-6, depth: d, restarts: R, value: best.value, tree: E.fromArrays(best.nodes, best.leaves, d) }; render(); };
  if ($("#adopt")) $("#adopt").onclick = () => { S.policy = clone(S.better.tree); S.better = null; S.proof = null; render(); };
  if ($("#proveCur")) $("#proveCur").onclick = () => prove({ [S.env]: 1 }, d, ev.value, r => { S.proof = r; render(); });
}

/* ---------- solver back end ---------- */
function prove(weights, depth, cutoff, done) {
  const m = S.m, envs = Object.keys(weights).map(e => { const b = E.build(m, e); return { weight: weights[e], cost: Array.from(b.cost), nxt: Array.from(b.nxt), pr: Array.from(b.pr), mu: Array.from(b.mu) }; });
  done({ html: "<b>Solver running…</b> The mixed-integer program is being solved on the back end (time limit 120 s)." });
  fetch("api/solve_tree", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ nS: m.nS, A: 3, reach: Array.from(m.reach), preds: S.pr.cols.map(c => Array.from(c)), depth, cutoff, time_limit: 120, envs }) })
    .then(r => r.json()).then(r => { if (r.error) throw new Error(r.error); const better = r.obj !== null && r.obj < cutoff - 1e-6;
      const html = r.proven ? (better ? `<b>Proved: the best tree of depth ${depth} costs ${fmt(r.obj, 1)} GWh</b>, lower than the tree you asked about (${fmt(cutoff, 1)}). Solved in ${fmt(r.time, 1)} s.` : `<b>Proved optimal for depth ${depth}.</b> No tree of this depth over these tests costs less than ${fmt(cutoff, 1)} GWh. Solved in ${fmt(r.time, 1)} s (${fmt(r.n_vars)} variables, ${fmt(r.n_bin)} binary).`)
        : `<b>Not proved within the time limit.</b> Solver bound ${fmt(r.bound, 1)} GWh; the tree is at most ${fmt((cutoff - r.bound) / cutoff * 100, 1)}% above the best tree of depth ${depth}.`;
      done({ html, proven: r.proven, result: r, tree: r.nodes ? E.fromArrays(r.nodes, r.leaves, depth) : null }); })
    .catch(err => done({ err: true, html: "<b>Solver call failed.</b> " + esc(err.message) }));
}

/* ---------- tab 4: optimise ---------- */
function renderOptimise() {
  const o = S.opt, run = S.optRun;
  $("#view").innerHTML = `<div class="wide"><section class="panel"><h2>Search for the best tree policy</h2>
    <p class="note">The search changes one test or one leaf at a time and keeps every improvement, with exact evaluation of each candidate. The green line is the bound no policy can beat; when the blue line reaches it, the tree is proven optimal without any solver.</p>
    <div class="controls">
      <label>Objective<select id="oObj">${Object.keys(S.case.envs).map(k => `<option value="${k}"${k === o.obj ? " selected" : ""}>${esc(envLabel(k))}</option>`).join("")}<option value="robust"${o.obj === "robust" ? " selected" : ""}>Robust: mean of the library environments</option></select></label>
      <label>Largest depth<select id="oDepth">${[1, 2, 3].map(d => `<option${d === o.depth ? " selected" : ""}>${d}</option>`).join("")}</select></label>
      <label>Restarts<select id="oRes">${[10, 30, 100].map(d => `<option${d === o.restarts ? " selected" : ""}>${d}</option>`).join("")}</select></label>
      <button class="btn primary" id="oRun"${run && run.busy ? " disabled" : ""}>${run && run.busy ? "Searching…" : "Run the search"}</button>
    </div>
    ${run ? `<div id="cAny">${anytimeChart(run.hist, run.bound)}</div>` : `<p class="note">No search run yet.</p>`}
  </section>
  <section class="panel"><h2>Result</h2>
    ${run && run.best ? (() => { const g = (run.best.value - run.bound) / run.bound, tr = E.fromArrays(run.best.nodes, run.best.leaves, run.best.depth);
      return `<div class="verdict ${g <= 1e-6 ? "good" : "mid"}">${g <= 1e-6 ? `<b>Optimal.</b> The tree meets the bound (${fmt(run.bound, 1)} GWh): no policy of any form does better.` : `<b>Best tree found: ${fmt(run.best.value, 1)} GWh</b>, ${fmt(g * 100, 1)}% above the bound ${fmt(run.bound, 1)}. A tree this small may not be able to reach the bound; the solver can prove whether a better tree of this depth exists.`}</div>
      <div class="tree"><ul><li>${treeHTML(tr)}</li></ul></div><p class="note">${E.leavesOf(tr)} rules, depth ${E.depthOf(tr)}; ${fmt(run.evals)} candidate trees evaluated${run.ms >= 50 ? " in " + fmt(run.ms / 1000, 1) + " s" : ""}.</p>
      <div class="controls"><button class="btn" id="oOpen">Open in the designer</button>${S.trees[o.obj] ? `<button class="btn" id="oLib">Use as the ${esc(polName(o.obj)).toLowerCase()} in the library</button>` : ""}${S.backend && !run.busy ? `<button class="btn" id="oProve">Prove with the solver</button>` : ""}</div>
      ${run.proof ? `<div class="verdict ${run.proof.err ? "bad" : run.proof.proven ? "good" : "mid"}">${run.proof.html}</div>` : ""}`; })() : `<p class="note">Run the search to see the tree.</p>`}
    ${S.backend ? "" : `<p class="note">Solver back end not connected. Start it with <code>python server.py</code> in the project folder to prove optimality for a given depth with mixed-integer programming.</p>`}
  </section></div>
  <section class="panel"><h2>Library trees now in use</h2><div class="tscroll"><table><tr><th>Tree</th><th class="num">Cost</th><th class="num">Bound</th><th>Status</th><th class="num">Rules</th><th>Found by</th></tr>
  ${Object.keys(S.trees).map(k => { const t = S.trees[k], g = (t.value - t.bound) / t.bound; return `<tr><td>${esc(polName(k))}</td><td class="num">${fmt(t.value, 1)}</td><td class="num">${fmt(t.bound, 1)}</td><td>${t.proven ? `<span class="badge good">✓ proven optimal</span>` : `<span class="badge mid">▲ ${fmt(g * 100, 1)}% above bound${t.gapLimit ? ", not proven" : ""}</span>`}</td><td class="num">${E.leavesOf(t.tree)}</td><td class="note">${esc(t.source)}${t.time ? ", " + fmt(t.time) + " s" : ""}</td></tr>`; }).join("")}</table></div></section>`;
  bindTips($("#view"));
  $("#oObj").onchange = e => { o.obj = e.target.value; S.optRun = null; render(); }; $("#oDepth").onchange = e => { o.depth = +e.target.value; }; $("#oRes").onchange = e => { o.restarts = +e.target.value; };
  $("#oRun").onclick = () => { const w = weightsFor(o.obj), bound = E.boundOf(S.m, w), u = E.rng(S.seed + 11), t0 = performance.now(); S.optRun = { hist: [], bound, busy: true, evals: 0, best: null, ms: 0 }; let r = 0, d = 1;
    const stepFn = () => { const run = S.optRun; if (!run || !run.busy) return; const until = performance.now() + 40;
      while (performance.now() < until && d <= o.depth) { const x = E.searchRestart(S.m, S.pr, w, d, u); run.evals += x.evals; if (!run.best || x.value < run.best.value - 1e-9) run.best = x; run.hist.push(run.best.value); r++;
        if (run.best.value - bound <= 1e-6 * Math.max(1, bound)) { d = 99; break; } if (r >= o.restarts) { r = 0; d++; } }
      run.ms = performance.now() - t0; if (d > o.depth) run.busy = false; if (S.tab === "optimise") render(); if (run.busy) setTimeout(stepFn, 0); };
    render(); setTimeout(stepFn, 10); };
  if ($("#oOpen")) $("#oOpen").onclick = () => { S.policy = E.fromArrays(run.best.nodes, run.best.leaves, run.best.depth); if (S.case.envs[o.obj]) S.env = o.obj; S.tab = "design"; S.better = null; S.proof = null; render(); };
  if ($("#oLib")) $("#oLib").onclick = () => { S.trees[o.obj] = { tree: E.fromArrays(run.best.nodes, run.best.leaves, run.best.depth), value: run.best.value, bound: run.bound, proven: run.best.value - run.bound <= 1e-6 * Math.max(1, run.bound), source: "search in the browser" }; S.sim = null; S.cmp = null; render(); };
  if ($("#oProve")) $("#oProve").onclick = () => prove(weightsFor(o.obj), run.best.depth, run.best.value, p => { run.proof = p; if (S.tab === "optimise") render(); });
}

/* ---------- tab 5: compare ---------- */
function runCompare(nPer) {
  const m = S.m, u = E.rng(S.seed + 101), tabs = {}; for (const k of Object.keys(S.trees)) tabs[k] = tableFor(k);
  const scen = Object.keys(S.case.envs).filter(k => k !== S.lib[0]).concat([S.lib[0]]), out = [];
  for (const e of scen) { const acc = { nominal: [], robust: [], monitor: [], bound: [] }; let dl = 0, nd = 0;
    for (let tau = T; tau < 2 * T; tau++) { const seq = []; for (let k = 0; k < 3 * T; k++) seq.push(k < tau ? S.lib[0] : e); const seqTabs = E.dpSequence(m, seq);
      for (let i = 0; i < nPer; i++) { const st = E.stream(m, seq, u, S.mon.noise), lb = E.labelStream(S.mon, st);
        acc.nominal.push(E.follow(m, st, () => tabs[S.lib[0]]).cost / 3); acc.robust.push(E.follow(m, st, () => tabs.robust).cost / 3); acc.monitor.push(E.follow(m, st, k => tabs[lb[k].env === "unknown" ? "robust" : lb[k].env]).cost / 3); acc.bound.push(E.follow(m, st, k => seqTabs[k]).cost / 3);
        if (S.lib.includes(e) && e !== S.lib[0]) { let f = 3 * T - tau; for (let k = tau; k < st.length; k++) if (lb[k].env === e) { f = k - tau; break; } dl += f; nd++; } } }
    const stat = a => { const n = a.length, mu = a.reduce((x, y) => x + y, 0) / n, sd = Math.sqrt(a.reduce((x, y) => x + (y - mu) * (y - mu), 0) / (n - 1)); return { mean: mu, ci: 1.96 * sd / Math.sqrt(n) }; };
    out.push({ key: e, title: e === S.lib[0] ? "No change (" + envLabel(e).toLowerCase() + ")" : "Change to " + envLabel(e).toLowerCase(), vals: Object.fromEntries(Object.entries(acc).map(([k, a]) => [k, stat(a)])), delay: nd ? dl / nd : null }); }
  const reg = k => out.reduce((a, g) => a + g.vals[k].mean - g.vals.bound.mean, 0) / out.length;
  S.cmp = { groups: out, n: nPer * T, regret: { nominal: reg("nominal"), robust: reg("robust"), monitor: reg("monitor") } };
}
function renderCompare() {
  if (!S.mon) { $("#view").innerHTML = `<section class="panel"><p>Training the monitor…</p></section>`; return; }
  if (!S.cmp) runCompare(40);
  const c = S.cmp, best = c.regret.robust <= c.regret.monitor ? "robust" : "monitor", sc = V.scores;
  const strat = [["nominal", envLabel(S.lib[0]) + " tree, no monitor"], ["robust", "F1 robust tree"], ["monitor", "F2 monitor and library"], ["bound", "Lower bound (knows the change)"]];
  $("#view").innerHTML = `<section class="panel"><h2>Which framework copes best with a change of environment?</h2>
    <p class="note">Three-year paths; year 1 is the design year, the change comes at a random split of year 2. Mean unserved energy per year over ${fmt(c.n)} paths per scenario, computed in your browser for the case now loaded. The darker bar is the framework with the lowest mean extra cost.</p>
    <div class="hero"><div class="tile"><div class="l">No monitor, ${esc(envLabel(S.lib[0]).toLowerCase())} tree</div><div class="v">+${fmt(c.regret.nominal)} <small>GWh / year</small></div><div class="s">mean extra cost over the lower bound</div></div>
      <div class="tile"${best === "robust" ? ' style="border-color:var(--accent)"' : ""}><div class="l">F1: one robust tree</div><div class="v">+${fmt(c.regret.robust)} <small>GWh / year</small></div><div class="s">no monitor needed to act</div></div>
      <div class="tile"${best === "monitor" ? ' style="border-color:var(--accent)"' : ""}><div class="l">F2: monitor and library</div><div class="v">+${fmt(c.regret.monitor)} <small>GWh / year</small></div><div class="s">switches on the monitor label</div></div></div>
    <div id="cLoop">${hbars(c.groups, strat, best)}</div>
    <div class="controls"><button class="btn" id="cMore">Run again with four times the paths</button><button class="btn" id="cTbl">Show as table</button></div>
    <div class="tscroll" id="cTblWrap" hidden><table><tr><th>Scenario</th>${strat.map(s => `<th class="num">${s[1]}</th>`).join("")}<th class="num">Mean detection delay (splits)</th></tr>${c.groups.map(g => `<tr><td>${esc(g.title)}</td>${strat.map(s => `<td class="num">${fmt(g.vals[s[0]].mean)} ± ${fmt(g.vals[s[0]].ci)}</td>`).join("")}<td class="num">${g.delay === null ? "" : fmt(g.delay, 1)}</td></tr>`).join("")}</table></div>
  </section>
  ${S.builtin ? `<section class="panel"><h2>Verified run and scores (built-in case)</h2>
    <p>${esc(sc.summary)}</p>
    <div class="tscroll"><table><tr><th>Criterion</th><th class="num">Weight</th>${sc.frameworks.map(f => `<th class="num">${f.short}</th>`).join("")}<th>Basis</th></tr>
    ${sc.criteria.map((cr, i) => `<tr><td>${esc(cr.name)}</td><td class="num">${cr.weight}</td>${sc.frameworks.map(f => `<td class="num">${fmt(f.scores[i], 1)}</td>`).join("")}<td class="note">${esc(cr.basis)}</td></tr>`).join("")}
    <tr class="win"><td>Weighted total (0 to 100)</td><td class="num">100</td>${sc.frameworks.map(f => `<td class="num">${fmt(f.total, 1)}</td>`).join("")}<td>Highest: ${esc(sc.best_name)}</td></tr></table></div>
    <p class="note">${sc.frameworks.map(f => `<b>${f.short}</b>: ${esc(f.name)}`).join(" · ")}. Python run of ${esc(V.meta.date)} with ${fmt(sc.episodes_per_scenario)} paths per scenario and mixed-integer proofs; scores run from 0 to 5.</p>
    <div class="verdict"><b>After the review.</b> ${esc(sc.final.text)}</div>
    <h3>When does selecting a policy pay off?</h3>
    <div class="tscroll"><table><tr><th>Block-splits due</th><th>Environment</th><th class="num">Bound</th><th class="num">Own tree</th><th class="num">Robust tree</th><th class="num">Cost of not selecting</th></tr>
    ${Object.keys(V.sensitivity).map(b => ["design", "heat", "lull"].map((e, i) => { const r = V.sensitivity[b][e]; return `<tr><td>${i === 0 ? b : ""}</td><td>${esc(DATA.case.envs[e].label)}</td><td class="num">${fmt(r.dp)}</td><td class="num">${fmt(r.own_tree)}</td><td class="num">${fmt(r.robust_tree)}</td><td class="num">${fmt(r.robust_tree - r.own_tree)}</td></tr>`; }).join("")).join("")}</table></div>
    <p class="note">${esc(sc.sensitivity_note)}</p></section>` : `<section class="panel"><p class="note">The verified scoring table refers to the built-in case and is hidden while another case is loaded.</p></section>`}`;
  bindTips($("#view"));
  $("#cMore").onclick = () => { $("#cMore").textContent = "Running…"; setTimeout(() => { runCompare(160); render(); }, 20); };
  $("#cTbl").onclick = () => { const w = $("#cTblWrap"); w.hidden = !w.hidden; $("#cTbl").textContent = w.hidden ? "Show as table" : "Hide table"; };
}

/* ---------- tab 6: method ---------- */
function renderMethod() {
  $("#view").innerHTML = `<section class="panel"><h2>How the pieces connect</h2><div class="pipe">
    <div><b>1. Stochastic system</b>Eight seasonal splits, three joint states each. Demand, wind and solar set the net load; the planner sets the maintenance blocks out.</div>
    <div><b>2. Runtime monitor</b>A depth-2 tree over the last splits names the environment. A box test flags windows it has not seen.</div>
    <div><b>3. Tree policies</b>One readable tree per environment and one robust tree, found by search and proven by mixed-integer programming.</div>
    <div><b>4. Certificate</b>Dynamic programming gives the best possible cost. Every tree is reported with its gap to that bound.</div>
    <div><b>5. Regime change</b>Strategies are compared on paths where the environment changes, against a bound that knows the change.</div></div></section>
  <div class="two"><section class="panel"><h2>Model</h2>
    <p><b>State.</b> Split (1 to 8), maintenance backlog, rank of the last net load. <b>Action.</b> Blocks out: 0, 1 or 2. <b>Cost.</b> Hours of the split times the shortfall of capacity against net load, plus a charge for each block left undone at the end of the year.</p>
    <p><b>Tree search as a mixed-integer program.</b> Binary variables choose one test per inner node and one action per leaf. Continuous variables route every state to its leaf and carry the visit probabilities of each state and action, so the objective is linear. The relaxation without the tree is the Markov decision process itself, which gives the bound.</p>
    <p><b>Monitor features.</b> ${E.FEATS.join("; ")}. Gaps are measured to the nearest design value of the split.</p></section>
  <section class="panel"><h2>Solver results of the verified run</h2><div class="tscroll"><table><tr><th>Tree</th><th class="num">Depth</th><th class="num">Cost</th><th class="num">Bound</th><th>Proof</th><th class="num">Time s</th><th class="num">Variables / binary</th></tr>
    ${Object.keys(V.trees).map(k => { const t = V.trees[k]; return `<tr><td>${k}</td><td class="num">${t.depth}</td><td class="num">${fmt(t.obj, 1)}</td><td class="num">${fmt(t.bound, 1)}</td><td>${t.proven ? "optimal" : "gap ≤ " + fmt(t.gap * 100, 1) + "%"}</td><td class="num">${fmt(t.time)}</td><td class="num">${fmt(t.n_vars)} / ${t.n_bin}</td></tr>`; }).join("")}</table></div>
    <h3>Scaling with the state space</h3><div class="tscroll"><table><tr><th class="num">Memory (splits)</th><th class="num">States</th><th class="num">MIP variables</th><th class="num">MIP time s</th><th class="num">MIP gap</th><th class="num">Search time s</th></tr>
    ${V.scaling.map(r => `<tr><td class="num">${r.memory}</td><td class="num">${fmt(r.states)}</td><td class="num">${fmt(r.milp_vars)}</td><td class="num">${fmt(r.milp_time)}</td><td class="num">${fmt(r.milp_gap * 100, 2)}%</td><td class="num">${fmt(r.local_search_time, 1)}</td></tr>`).join("")}</table></div>
    <p class="note">${esc(V.scores.scaling_note)}</p></section></div>
  <section class="panel"><h2>Use the case as a benchmark</h2>
    <p>The case is a family of finite-horizon Markov decision processes with seasonal structure and a hidden environment. Three knobs change its difficulty: the maintenance requirement, the observation memory and the set of environments. The exact bound is always available, so the gap of any tree policy is known.</p>
    <div class="controls"><button class="btn" id="dlCase">Download the case (JSON)</button><button class="btn" id="dlMdp">Download the decision process of ${esc(envLabel(S.env))} (JSON)</button></div></section>
  <section class="panel"><h2>Limits</h2><p class="note">Single node: the network, unit commitment and zonal limits of the expansion model are not represented. Block size, availability, penalty, measurement noise and the stressed environments are assumptions. The monitor is a heuristic tree trained on simulated windows. The in-browser search proves optimality only when it reaches the bound; otherwise the solver back end is needed. Uploaded hourly data are grouped by k-means, a simpler method than the copula and scenario reduction of the manuscript.</p></section>`;
  $("#dlCase").onclick = () => download("case.json", JSON.stringify(S.case, null, 1), "application/json");
  $("#dlMdp").onclick = () => { const b = E.build(S.m, S.env); download("mdp_" + S.env + ".json", JSON.stringify({ description: "Finite-horizon MDP; state index = (split*(B+1)+backlog)*3+rank; next = -1 marks the end of the year", environment: S.env, n_states: S.m.nS, n_actions: 3, horizon: 8, backlog: S.m.B, reachable: Array.from(S.m.reach), cost: Array.from(b.cost), next: Array.from(b.nxt), prob: Array.from(b.pr), initial: Array.from(b.mu), tests: S.pr.labels, bound: E.dp(S.m, S.env).value }), "application/json"); };
}

/* ---------- shell ---------- */
function render() {
  $("#tabs").innerHTML = TABS.map(([k, n], i) => `<button role="tab" aria-selected="${k === S.tab}" data-k="${k}"><span class="n">${i + 1}</span>${n}</button>`).join("");
  renderHero(); ({ data: renderData, track: renderTrack, design: renderDesign, optimise: renderOptimise, compare: renderCompare, method: renderMethod })[S.tab]();
}
$("#tabs").onclick = e => { const b = e.target.closest("button"); if (!b) return; S.tab = b.dataset.k; render(); window.scrollTo({ top: 0 }); };
const c0 = clone(DATA.case); c0.envs.custom = { label: "Custom scenario", d: [1.05, 1.1, 1.05, 1.05], w: [0.9, 0.9, 0.7, 0.6], kind: "both", tau: 1.4, taup: 2.2, library: false };
DATA.case.envs.custom = clone(c0.envs.custom);
setCase(c0, true);
/* direct links: #tab,environment,policy,run  e.g. #design,heat,design  or  #optimise,robust,,run */
(function fromHash() { const h = decodeURIComponent(location.hash.slice(1)).split(","); if (!h[0]) return;
  if (TABS.some(t => t[0] === h[0])) S.tab = h[0];
  if (h[1] && S.case.envs[h[1]]) { S.env = h[1]; S.simEnv = h[1] === S.lib[0] ? S.simEnv : h[1]; }
  if (h[1] === "robust" || (h[1] && S.case.envs[h[1]])) S.opt.obj = h[1];
  if (h[2] && S.trees[h[2]]) S.policy = clone(S.trees[h[2]].tree);
  if (h[2] === "monitor") S.mode = "monitor";
  if (h[3] === "run" && S.tab === "optimise") { S.opt.depth = 3; setTimeout(() => { const b = $("#oRun"); if (b) b.click(); }, 300); } })();
render();
fetch("api/health").then(r => r.ok ? r.json() : Promise.reject()).then(j => { S.backend = !!j.ok; const c = $("#chipBackend"); c.classList.add("on"); c.lastElementChild.textContent = "Solver back end: connected"; render(); })
  .catch(() => { $("#chipBackend").lastElementChild.textContent = "Solver back end: not connected"; });
window.Studio = S;
})();
