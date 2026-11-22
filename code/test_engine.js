/* Check the browser engine against the verified Python run. Usage: node code/test_engine.js */
global.window = {};
const path = require("path");
const D = require(path.join(__dirname, "..", "docs", "data.js")), E = require(path.join(__dirname, "..", "docs", "engine.js"));
const m = E.makeModel(D.case), pr = E.predicates(m), V = D.verified;
let bad = 0;
const chk = (n, a, b, tol = 0.05) => { const ok = Math.abs(a - b) <= tol; if (!ok) bad++; console.log((ok ? "ok  " : "FAIL") + " " + n + ": js " + a.toFixed(2) + " python " + b.toFixed(2)); };
console.log("states", m.nReach, "predicates", pr.labels.length, "labels equal", JSON.stringify(pr.labels) === JSON.stringify(V.predicate_labels));
for (const e of Object.keys(D.case.envs)) chk("DP bound " + e, E.dp(m, e).value, V.dp_bound[e]);
for (const k of Object.keys(V.trees)) { const t = V.trees[k], pol = E.tableArr(m, pr, t.nodes, t.leaves, t.depth);
  for (const e of Object.keys(D.case.envs)) chk("tree " + k + " in " + e, E.evaluate(m, e, pol).value, V.cross_eval["milp_" + k][e]); }
const t = V.trees.design, pol = E.tableArr(m, pr, t.nodes, t.leaves, t.depth), ev = E.evaluate(m, "heat", pol);
chk("loss by split sums to gap", ev.lossBySplit.reduce((a, b) => a + b, 0), ev.value - ev.bound, 1e-6);
const u = E.rng(3); let best = Infinity; const t0 = Date.now();
for (let r = 0; r < 12; r++) best = Math.min(best, E.searchRestart(m, pr, { heat: 1 }, 2, u).value);
chk("local search heat depth 2 (12 restarts, " + (Date.now() - t0) + " ms)", best, V.trees.heat.obj, 20);
const t1 = Date.now(), mon = E.trainMonitor(m, ["design", "heat", "lull"]), conf = E.testMonitor(m, mon, Object.keys(D.case.envs));
console.log("monitor trained and tested in", Date.now() - t1, "ms; train accuracy", mon.trainAcc.toFixed(3));
for (const e of Object.keys(conf)) console.log("  " + e.padEnd(10), Object.entries(conf[e]).map(([k, v]) => k + " " + (v * 100).toFixed(0) + "%").join("  "), "| python", Object.entries(V.monitor.confusion[e]).map(([k, v]) => (v * 100).toFixed(0) + "%").join(" "));
const seq = new Array(24).fill("heat"), tabs = E.dpSequence(m, seq), u2 = E.rng(9); let s = 0; const N = 4000;
for (let i = 0; i < N; i++) s += E.follow(m, E.stream(m, seq, u2, mon.noise), k => tabs[k]).cost / 3;
chk("sequence DP, heat, simulated mean per year", s / N, V.dp_bound.heat, 60);
console.log(bad ? bad + " CHECKS FAILED" : "ALL CHECKS PASSED");
