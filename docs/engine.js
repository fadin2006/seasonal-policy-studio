/* Seasonal Policy Studio: computation engine (browser and Node).
   Mirrors code/lab_core.py and code/lab_sim.py. All costs: expected unserved energy, GWh per year. */
(function (root, factory) { if (typeof module === "object" && module.exports) module.exports = factory(); else root.Engine = factory(); })(this, function () {
  "use strict";
  const T = 8, A = 3;
  const SPLITS = ["Spring day", "Spring night", "Summer day", "Summer night", "Fall day", "Fall night", "Winter day", "Winter night"];
  const sea = v => [v[0], v[0], v[1], v[1], v[2], v[2], v[3], v[3]];
  const rankOf = row => { const idx = row.map((v, i) => i).sort((a, b) => row[a] - row[b]); const r = new Array(row.length); idx.forEach((i, k) => r[i] = k); return { r, idx }; };
  function rng(seed) { let a = seed >>> 0; return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  function gauss(u) { return Math.sqrt(-2 * Math.log(Math.max(1e-12, u()))) * Math.cos(2 * Math.PI * u()); }

  /* ---------------- case -> environments ---------------- */
  function derive(c) {
    const f = c.fleet, D = c.splits.map(s => s.states.map(x => x.demand)), W = c.splits.map(s => s.states.map(x => x.wind_cf)), S = c.splits.map(s => s.states.map(x => x.solar_cf));
    const P0 = c.splits.map(s => { const t = s.states.reduce((a, x) => a + x.p, 0); return s.states.map(x => x.p / t); });
    const H = c.splits.map(s => s.hours);
    const nl = (d, w) => D.map((row, t) => row.map((v, j) => v * d[t] - W[t][j] * w[t] * f.wind - S[t][j] * f.solar));
    const NL0 = nl(sea([1, 1, 1, 1]), sea([1, 1, 1, 1]));
    const RANK = NL0.map(r => rankOf(r).r), INV = NL0.map(r => rankOf(r).idx), DEMRANK = D.map(r => rankOf(r).r);
    const LOWWIND = W.map(r => r.map(v => v < 0.1 ? 2 : v < 0.5 ? 1 : 0));
    const envs = {};
    for (const [k, e] of Object.entries(c.envs)) {
      const d = sea(e.d), w = sea(e.w), P = [];
      for (let t = 0; t < T; t++) { const tp = (t + 7) % T; P.push([]);
        for (let o = 0; o < 3; o++) { const jp = INV[tp][o]; let q = P0[t].slice();
          if (e.kind === "heat" || e.kind === "both") q = q.map((v, j) => v * Math.pow(DEMRANK[tp][jp] === 2 ? e.taup : e.tau, DEMRANK[t][j]));
          if (e.kind === "drought" || e.kind === "both") q = q.map((v, j) => v * Math.pow(LOWWIND[tp][jp] === 2 ? e.taup : e.tau, LOWWIND[t][j]));
          const s = q.reduce((a, b) => a + b, 0); P[t].push(q.map(v => v / s)); } }
      let mu = [1 / 3, 1 / 3, 1 / 3];
      for (let it = 0; it < 20; it++) for (let t = 0; t < T; t++) { const nx = [0, 0, 0]; for (let o = 0; o < 3; o++) for (let j = 0; j < 3; j++) nx[RANK[t][j]] += mu[o] * P[t][o][j]; mu = nx; }
      envs[k] = { key: k, label: e.label, d, w, kind: e.kind, library: !!e.library, NL: nl(d, w), P, mu0: mu };
    }
    return { c, D, W, S, P0, H, NL0, RANK, INV, envs };
  }

  /* ---------------- Markov decision process ---------------- */
  function makeModel(c) {
    const dv = derive(c), f = c.fleet, B = f.backlog, nS = T * (B + 1) * 3, block = f.blockFrac * f.conv;
    const m = { dv, B, nS, block, T, A, caps: [0, 1, 2].map(a => f.avail * (f.conv - a * block)), pen: f.rho * block * (dv.H.reduce((a, b) => a + b, 0) / T) / 1000, cache: {} };
    m.tOf = new Int32Array(nS); m.bOf = new Int32Array(nS); m.oOf = new Int32Array(nS); m.reach = new Uint8Array(nS);
    for (let s = 0; s < nS; s++) { const t = Math.floor(s / ((B + 1) * 3)), r = s % ((B + 1) * 3); m.tOf[s] = t; m.bOf[s] = Math.floor(r / 3); m.oOf[s] = r % 3; m.reach[s] = m.bOf[s] >= Math.max(0, B - 2 * t) ? 1 : 0; }
    m.sid = (t, b, o) => (t * (B + 1) + b) * 3 + o;
    m.nReach = m.reach.reduce((a, b) => a + b, 0);
    return m;
  }
  function build(m, name) {
    if (m.cache[name]) return m.cache[name];
    const e = m.dv.envs[name], nS = m.nS, cost = new Float64Array(nS * A), nxt = new Int32Array(nS * A * 3).fill(-1), pr = new Float64Array(nS * A * 3);
    for (let s = 0; s < nS; s++) { const t = m.tOf[s], b = m.bOf[s], o = m.oOf[s];
      for (let a = 0; a < A; a++) { const ae = Math.min(a, b); let cst = 0;
        for (let j = 0; j < 3; j++) { const p = e.P[t][o][j]; cst += p * m.dv.H[t] * Math.max(0, e.NL[t][j] - m.caps[ae]) / 1000;
          if (t < T - 1) { nxt[(s * A + a) * 3 + j] = m.sid(t + 1, b - ae, m.dv.RANK[t][j]); pr[(s * A + a) * 3 + j] = p; } }
        if (t === T - 1) cst += m.pen * (b - ae);
        cost[s * A + a] = cst; } }
    const mu = new Float64Array(nS); for (let o = 0; o < 3; o++) mu[m.sid(0, m.B, o)] = e.mu0[o];
    return m.cache[name] = { cost, nxt, pr, mu };
  }
  function dp(m, name) {
    const k = "dp:" + name; if (m.cache[k]) return m.cache[k];
    const { cost, nxt, pr, mu } = build(m, name), nS = m.nS, V = new Float64Array(nS), Q = new Float64Array(nS * A), pol = new Int8Array(nS);
    for (let s = nS - 1; s >= 0; s--) { let best = Infinity, ba = 0;
      for (let a = 0; a < A; a++) { let q = cost[s * A + a]; for (let j = 0; j < 3; j++) { const n = nxt[(s * A + a) * 3 + j]; if (n >= 0) q += pr[(s * A + a) * 3 + j] * V[n]; } Q[s * A + a] = q; if (q < best - 1e-12) { best = q; ba = a; } }
      V[s] = best; pol[s] = ba; }
    let value = 0; for (let s = 0; s < nS; s++) value += mu[s] * V[s];
    return m.cache[k] = { value, V, Q, pol };
  }
  function fastValue(m, name, pol) {
    const { cost, nxt, pr, mu } = build(m, name), nS = m.nS, V = new Float64Array(nS);
    for (let s = nS - 1; s >= 0; s--) { if (!m.reach[s]) continue; const a = pol[s]; let q = cost[s * A + a]; for (let j = 0; j < 3; j++) { const n = nxt[(s * A + a) * 3 + j]; if (n >= 0) q += pr[(s * A + a) * 3 + j] * V[n]; } V[s] = q; }
    let v = 0; for (let s = 0; s < nS; s++) v += mu[s] * V[s]; return v;
  }
  /* exact value, expected blocks out, cost and avoidable loss per split (performance-difference identity) */
  function evaluate(m, name, pol) {
    const { cost, nxt, pr, mu } = build(m, name), opt = dp(m, name), nS = m.nS;
    let d = Float64Array.from(mu), value = 0; const bySplit = new Array(T).fill(0), lossBySplit = new Array(T).fill(0), blocks = new Array(T).fill(0);
    for (let t = 0; t < T; t++) { const nx = new Float64Array(nS);
      for (let s = t * (m.B + 1) * 3; s < (t + 1) * (m.B + 1) * 3; s++) { const w = d[s]; if (!w) continue; const a = pol[s];
        value += w * cost[s * A + a]; bySplit[t] += w * cost[s * A + a]; lossBySplit[t] += w * (opt.Q[s * A + a] - opt.V[s]); blocks[t] += w * Math.min(a, m.bOf[s]);
        for (let j = 0; j < 3; j++) { const n = nxt[(s * A + a) * 3 + j]; if (n >= 0) nx[n] += w * pr[(s * A + a) * 3 + j]; } }
      d = nx; }
    return { value, bound: opt.value, gap: opt.value > 0 ? Math.max(0, (value - opt.value) / opt.value) : 0, bySplit, lossBySplit, blocks };
  }

  /* ---------------- tree policies ---------------- */
  function predicates(m) {
    const cols = [], labels = [], nS = m.nS, B = m.B, mk = fn => { const c = new Uint8Array(nS); for (let s = 0; s < nS; s++) c[s] = fn(s) ? 1 : 0; return c; };
    for (let k = 1; k < T; k++) { cols.push(mk(s => m.tOf[s] + 1 <= k)); labels.push(`split is ${SPLITS[k - 1].toLowerCase()} or earlier`); }
    cols.push(mk(s => m.tOf[s] % 2 === 0)); labels.push("it is a day split");
    for (let k = 0; k < B; k++) { cols.push(mk(s => m.bOf[s] <= k)); labels.push(`backlog ≤ ${k} blocks`); }
    cols.push(mk(s => m.oOf[s] <= 0)); labels.push("last net load was low");
    cols.push(mk(s => m.oOf[s] <= 1)); labels.push("last net load was low or middle");
    for (let k = 0; k <= 2 * T - B; k++) { cols.push(mk(s => 2 * (T - m.tOf[s]) - m.bOf[s] <= k)); labels.push(`slack ≤ ${k} blocks`); }
    const keep = [], seen = new Set();
    cols.forEach((c, j) => { let key = "", any = 0, all = 1; for (let s = 0; s < nS; s++) if (m.reach[s]) { key += c[s]; any |= c[s]; all &= c[s]; } if (!any || all || seen.has(key)) return; seen.add(key); keep.push(j); });
    return { cols: keep.map(j => cols[j]), labels: keep.map(j => labels[j]) };
  }
  /* nested tree: {j, yes, no} or {a} */
  function tableOf(m, pr, tree) { const pol = new Int8Array(m.nS); for (let s = 0; s < m.nS; s++) { let n = tree; while (n.a === undefined) n = pr.cols[n.j][s] ? n.yes : n.no; pol[s] = n.a; } return pol; }
  function fromArrays(nodes, leaves, depth) { const nI = (1 << depth) - 1; const rec = i => i >= nI ? { a: leaves[i - nI] } : { j: nodes[i], yes: rec(2 * i + 1), no: rec(2 * i + 2) }; return prune(rec(0)); }
  function prune(n) { if (n.a !== undefined) return n; const y = prune(n.yes), no = prune(n.no); if (y.a !== undefined && no.a !== undefined && y.a === no.a) return { a: y.a }; return { j: n.j, yes: y, no }; }
  const leavesOf = n => n.a !== undefined ? 1 : leavesOf(n.yes) + leavesOf(n.no);
  const depthOf = n => n.a !== undefined ? 0 : 1 + Math.max(depthOf(n.yes), depthOf(n.no));
  function tableArr(m, pr, nodes, leaves, depth) { const nI = (1 << depth) - 1, pol = new Int8Array(m.nS); for (let s = 0; s < m.nS; s++) { let p = 0; for (let k = 0; k < depth; k++) p = 2 * p + (pr.cols[nodes[p]][s] ? 1 : 2); pol[s] = leaves[p - nI]; } return pol; }
  /* best-improvement local search with exact evaluation; one restart per call so the page stays responsive */
  function searchRestart(m, pr, weights, depth, u, start) {
    const nI = (1 << depth) - 1, nL = 1 << depth, nJ = pr.cols.length, envs = Object.keys(weights);
    const val = (n, l) => { const pol = tableArr(m, pr, n, l, depth); let v = 0; for (const e of envs) v += weights[e] * fastValue(m, e, pol); return v; };
    let nodes = start ? start.nodes.slice() : Array.from({ length: nI }, () => Math.floor(u() * nJ)), leaves = start ? start.leaves.slice() : Array.from({ length: nL }, () => Math.floor(u() * A));
    let v = val(nodes, leaves), evals = 1;
    for (;;) { let best = v, mv = null;
      for (let i = 0; i < nI; i++) for (let j = 0; j < nJ; j++) { if (j === nodes[i]) continue; const o = nodes[i]; nodes[i] = j; const v2 = val(nodes, leaves); evals++; nodes[i] = o; if (v2 < best - 1e-9) { best = v2; mv = [0, i, j]; } }
      for (let i = 0; i < nL; i++) for (let a = 0; a < A; a++) { if (a === leaves[i]) continue; const o = leaves[i]; leaves[i] = a; const v2 = val(nodes, leaves); evals++; leaves[i] = o; if (v2 < best - 1e-9) { best = v2; mv = [1, i, a]; } }
      if (!mv) break; if (mv[0] === 0) nodes[mv[1]] = mv[2]; else leaves[mv[1]] = mv[2]; v = best; }
    return { nodes, leaves, depth, value: v, evals };
  }
  function boundOf(m, weights) { let b = 0; for (const e of Object.keys(weights)) b += weights[e] * dp(m, e).value; return b; }

  /* ---------------- observations and the runtime monitor ---------------- */
  const FEATS = ["mean demand gap", "largest demand gap", "mean wind gap", "largest wind gap", "share outside design box"];
  function observe(m, envName, t, o, u, noise) { const e = m.dv.envs[envName], r = u(); let j = 2, c = 0; for (let i = 0; i < 3; i++) { c += e.P[t][o][i]; if (r <= c) { j = i; break; } }
    return { j, demand: m.dv.D[t][j] * e.d[t] * (1 + noise.d * gauss(u)), wind: Math.min(1, Math.max(0, m.dv.W[t][j] * e.w[t] + noise.w * gauss(u))), nl: e.NL[t][j] }; }
  function gaps(m, t, demand, wind) { let nd = 0; for (let i = 1; i < 3; i++) if (Math.abs(demand - m.dv.D[t][i]) < Math.abs(demand - m.dv.D[t][nd])) nd = i;
    return { gD: demand / m.dv.D[t][nd] - 1, gW: Math.min(...m.dv.W[t].map(v => Math.abs(wind - v))) }; }
  function features(gD, gW, k, win, noise) { let sD = 0, mD = -Infinity, sW = 0, mW = -Infinity, out = 0; for (let i = k - win; i < k; i++) { sD += gD[i]; mD = Math.max(mD, gD[i]); sW += gW[i]; mW = Math.max(mW, gW[i]); if (Math.abs(gD[i]) > 3 * noise.d || gW[i] > 3 * noise.w) out++; } return [sD / win, mD, sW / win, mW, out / win]; }
  function gini(cnt) { const n = cnt.reduce((a, b) => a + b, 0); if (!n) return 0; return 1 - cnt.reduce((a, c) => a + (c / n) * (c / n), 0); }
  function fitNode(X, y, idx, nC, depth, minLeaf) {
    const cnt = new Array(nC).fill(0); idx.forEach(i => cnt[y[i]]++); const cls = cnt.indexOf(Math.max(...cnt)), leaf = { cls, share: cnt[cls] / idx.length, n: idx.length };
    if (depth === 0 || gini(cnt) < 1e-9) return leaf;
    let best = null;
    for (let f = 0; f < X[0].length; f++) { const vals = idx.map(i => X[i][f]).sort((a, b) => a - b);
      for (let q = 1; q < 40; q++) { const th = vals[Math.floor(q * vals.length / 40)]; const l = new Array(nC).fill(0), r = new Array(nC).fill(0); let nl = 0;
        for (const i of idx) { if (X[i][f] <= th) { l[y[i]]++; nl++; } else r[y[i]]++; }
        if (nl < minLeaf || idx.length - nl < minLeaf) continue; const g = (nl * gini(l) + (idx.length - nl) * gini(r)) / idx.length; if (!best || g < best.g - 1e-12) best = { g, f, th }; } }
    if (!best || best.g >= gini(cnt) - 1e-9) return leaf;
    const L = idx.filter(i => X[i][best.f] <= best.th), R = idx.filter(i => X[i][best.f] > best.th);
    return { f: best.f, th: best.th, yes: fitNode(X, y, L, nC, depth - 1, minLeaf), no: fitNode(X, y, R, nC, depth - 1, minLeaf) };
  }
  function stream(m, seq, u, noise, o0) { let o = o0 === undefined ? (() => { const r = u(), mu = m.dv.envs[seq[0]].mu0; let c = 0; for (let i = 0; i < 3; i++) { c += mu[i]; if (r <= c) return i; } return 2; })() : o0;
    const out = []; seq.forEach((en, k) => { const t = k % T, ob = observe(m, en, t, o, u, noise), g = gaps(m, t, ob.demand, ob.wind); out.push(Object.assign(ob, g, { k, t, env: en, o })); o = m.dv.RANK[t][ob.j]; }); return out; }
  function trainMonitor(m, classes, opt) {
    const o = Object.assign({ win: 4, paths: 500, years: 5, depth: 2, seed: 1, noise: { d: 0.015, w: 0.03 } }, opt || {}), u = rng(o.seed), X = [], y = [];
    classes.forEach((en, ci) => { for (let p = 0; p < o.paths; p++) { const st = stream(m, new Array(T * o.years).fill(en), u, o.noise), gD = st.map(r => r.gD), gW = st.map(r => r.gW); for (let k = o.win; k < st.length; k++) { X.push(features(gD, gW, k, o.win, o.noise)); y.push(ci); } } });
    const idx = X.map((_, i) => i), tree = fitNode(X, y, idx, classes.length, o.depth, Math.max(50, Math.floor(X.length / 400)));
    const boxes = classes.map((_, ci) => { const rows = idx.filter(i => y[i] === ci); return [0, 1, 2, 3, 4].map(f => { const v = rows.map(i => X[i][f]).sort((a, b) => a - b); return [v[Math.floor(0.002 * (v.length - 1))], v[Math.ceil(0.998 * (v.length - 1))]]; }); });
    const mon = { tree, boxes, classes, win: o.win, noise: o.noise };
    let ok = 0; for (const i of idx) if (label(mon, X[i]).cls === y[i]) ok++; mon.trainAcc = ok / idx.length; return mon;
  }
  function label(mon, F) { let n = mon.tree; while (n.cls === undefined) n = F[n.f] <= n.th ? n.yes : n.no; const bx = mon.boxes[n.cls];
    for (let i = 0; i < F.length; i++) if (F[i] < bx[i][0] - 1e-9 || F[i] > bx[i][1] + 1e-9) return { cls: -1, env: "unknown", share: n.share }; return { cls: n.cls, env: mon.classes[n.cls], share: n.share }; }
  function labelStream(mon, st) { const gD = st.map(r => r.gD), gW = st.map(r => r.gW); return st.map((r, k) => k < mon.win ? { cls: 0, env: mon.classes[0], share: 1, warm: true } : label(mon, features(gD, gW, k, mon.win, mon.noise))); }
  function testMonitor(m, mon, envNames, opt) { const o = Object.assign({ paths: 300, years: 5, seed: 77 }, opt || {}), u = rng(o.seed), res = {};
    for (const en of envNames) { const cnt = {}; let n = 0; for (let p = 0; p < o.paths; p++) { const st = stream(m, new Array(T * o.years).fill(en), u, mon.noise), lb = labelStream(mon, st); for (let k = mon.win; k < st.length; k++) { cnt[lb[k].env] = (cnt[lb[k].env] || 0) + 1; n++; } } res[en] = {}; for (const c of mon.classes.concat(["unknown"])) res[en][c] = (cnt[c] || 0) / n; } return res; }

  /* ---------------- strategies in the loop ---------------- */
  function dpSequence(m, seq) { const K = seq.length, nS = m.nS; let V = new Float64Array(nS); const tabs = new Array(K);
    for (let k = K - 1; k >= 0; k--) { const t = k % T, { cost, nxt, pr } = build(m, seq[k]), e = m.dv.envs[seq[k]], pol = new Int8Array(nS), Vn = Float64Array.from(V);
      for (let s = t * (m.B + 1) * 3; s < (t + 1) * (m.B + 1) * 3; s++) { let best = Infinity, ba = 0;
        for (let a = 0; a < A; a++) { let q = cost[s * A + a];
          if (t === T - 1) { if (k < K - 1) for (let j = 0; j < 3; j++) q += e.P[t][m.oOf[s]][j] * V[m.sid(0, m.B, m.dv.RANK[t][j])]; }
          else for (let j = 0; j < 3; j++) q += pr[(s * A + a) * 3 + j] * V[nxt[(s * A + a) * 3 + j]];
          if (q < best - 1e-12) { best = q; ba = a; } }
        Vn[s] = best; pol[s] = ba; }
      V = Vn; tabs[k] = pol; }
    return tabs; }
  /* cost of following tableAt(k) along one observed stream */
  function follow(m, st, tableAt) { let b = m.B, cost = 0; const steps = [];
    st.forEach((r, k) => { if (r.t === 0) b = m.B; const a = Math.min(tableAt(k)[m.sid(r.t, b, r.o)], b), shed = Math.max(0, r.nl - m.caps[a]); let c = m.dv.H[r.t] * shed / 1000; b -= a; if (r.t === T - 1) c += m.pen * b; cost += c; steps.push({ a, cap: m.caps[a], shed, left: b, cost: c }); });
    return { cost, steps }; }

  /* ---------------- data import ---------------- */
  function parseCSV(text) { const rows = text.trim().split(/\r?\n/).map(l => l.split(/[,;\t]/).map(x => x.trim())); const head = rows.shift().map(h => h.toLowerCase()); return { head, rows }; }
  function kmeans3(pts, u) { const n = pts.length, dim = pts[0].length, sorted = pts.map((p, i) => i).sort((a, b) => pts[a][0] - pts[b][0]);
    let cen = [0.17, 0.5, 0.83].map(q => pts[sorted[Math.floor(q * (n - 1))]].slice()), asg = new Int8Array(n);
    for (let it = 0; it < 40; it++) { let ch = 0; for (let i = 0; i < n; i++) { let b = 0, bd = Infinity; for (let c = 0; c < 3; c++) { let d = 0; for (let k = 0; k < dim; k++) d += (pts[i][k] - cen[c][k]) ** 2; if (d < bd) { bd = d; b = c; } } if (asg[i] !== b) { asg[i] = b; ch++; } }
      const sum = [0, 1, 2].map(() => new Array(dim).fill(0)), cnt = [0, 0, 0]; for (let i = 0; i < n; i++) { cnt[asg[i]]++; for (let k = 0; k < dim; k++) sum[asg[i]][k] += pts[i][k]; }
      for (let c = 0; c < 3; c++) if (cnt[c]) cen[c] = sum[c].map(v => v / cnt[c]); if (!ch && it) break; }
    return { asg, cen }; }
  /* hourly rows -> eight splits, three joint states each. Columns: month (1-12) or timestamp, hour (0-23), demand_MW, wind_cf or wind_MW, solar_cf or solar_MW */
  function statesFromHourly(text, baseHours) {
    const { head, rows } = parseCSV(text), col = n => head.findIndex(h => h.replace(/[^a-z_]/g, "") === n), ci = { ts: head.findIndex(h => /time|date/.test(h)), month: col("month"), hour: col("hour"), dem: head.findIndex(h => /demand|load/.test(h)), wcf: col("wind_cf"), wmw: head.findIndex(h => /wind.*mw/.test(h)), scf: col("solar_cf"), smw: head.findIndex(h => /solar.*mw|pv.*mw/.test(h)) };
    if (ci.dem < 0) throw new Error("No demand column found (name it demand_MW).");
    if (ci.wcf < 0 && ci.wmw < 0) throw new Error("No wind column found (wind_cf or wind_MW).");
    const rec = [];
    for (const r of rows) { let mo, hr; if (ci.month >= 0) { mo = +r[ci.month]; hr = ci.hour >= 0 ? +r[ci.hour] : 12; } else if (ci.ts >= 0) { const d = new Date(r[ci.ts]); if (isNaN(d)) continue; mo = d.getMonth() + 1; hr = d.getHours(); } else throw new Error("Need a month column or a timestamp column.");
      const dem = +r[ci.dem], w = +r[ci.wcf >= 0 ? ci.wcf : ci.wmw], s = ci.scf >= 0 ? +r[ci.scf] : ci.smw >= 0 ? +r[ci.smw] : null; if (!isFinite(dem) || !isFinite(w)) continue; rec.push({ mo, hr, dem, w, s }); }
    if (rec.length < 200) throw new Error("Too few valid rows (" + rec.length + "); at least 200 hourly rows are needed.");
    const wMax = ci.wcf >= 0 ? 1 : Math.max(...rec.map(r => r.w)), sMax = ci.scf >= 0 ? 1 : Math.max(1e-9, ...rec.map(r => r.s || 0)), hasS = rec.some(r => r.s !== null);
    const season = mo => mo >= 3 && mo <= 5 ? 0 : mo >= 6 && mo <= 8 ? 1 : mo >= 9 && mo <= 11 ? 2 : 3, isDay = r => hasS ? (r.s / sMax) > 0.005 : (r.hr >= 6 && r.hr < 18), u = rng(5), splits = [];
    for (let t = 0; t < T; t++) { const part = rec.filter(r => season(r.mo) === (t >> 1) && isDay(r) === (t % 2 === 0)); if (part.length < 9) throw new Error("Split '" + SPLITS[t] + "' has only " + part.length + " rows.");
      const dMin = Math.min(...part.map(r => r.dem)), dR = Math.max(1e-9, Math.max(...part.map(r => r.dem)) - dMin);
      const pts = part.map(r => [(r.dem - dMin) / dR, r.w / wMax, hasS ? (r.s || 0) / sMax : 0]), km = kmeans3(pts, u), states = [];
      for (let c = 0; c < 3; c++) { const mem = part.filter((_, i) => km.asg[i] === c); if (!mem.length) continue; const mean = f => mem.reduce((a, r) => a + f(r), 0) / mem.length;
        states.push({ p: mem.length / part.length, demand: mean(r => r.dem), wind_cf: mean(r => r.w / wMax), solar_cf: hasS ? mean(r => (r.s || 0) / sMax) : (t % 2 === 0 ? 0.5 : 0) }); }
      while (states.length < 3) states.push(Object.assign({}, states[0], { p: 1e-6 }));
      splits.push({ name: SPLITS[t], hours: baseHours[t], states }); }
    return { splits, rows: rec.length };
  }
  /* state table: split(1-8), state(1-3), p, demand_MW, wind_cf, solar_cf */
  function statesFromTable(text, baseHours) { const { head, rows } = parseCSV(text), g = n => head.findIndex(h => h.includes(n)), ix = { sp: g("split"), p: head.findIndex(h => h === "p" || h.includes("prob")), d: g("demand"), w: g("wind"), s: g("solar") };
    if (Object.values(ix).some(v => v < 0)) throw new Error("Columns needed: split, state, p, demand_MW, wind_cf, solar_cf.");
    const splits = SPLITS.map((n, t) => ({ name: n, hours: baseHours[t], states: [] }));
    for (const r of rows) { const t = +r[ix.sp] - 1; if (!(t >= 0 && t < T)) continue; splits[t].states.push({ p: +r[ix.p], demand: +r[ix.d], wind_cf: +r[ix.w], solar_cf: +r[ix.s] }); }
    splits.forEach(s => { if (s.states.length !== 3) throw new Error("Split '" + s.name + "' needs exactly 3 states, found " + s.states.length + "."); });
    return { splits, rows: rows.length }; }

  return { T, A, SPLITS, FEATS, sea, rng, gauss, derive, makeModel, build, dp, fastValue, evaluate, predicates, tableOf, tableArr, fromArrays, prune, leavesOf, depthOf, searchRestart, boundOf,
    stream, gaps, features, trainMonitor, label, labelStream, testMonitor, dpSequence, follow, parseCSV, statesFromHourly, statesFromTable };
});
