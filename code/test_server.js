/* Ask the back end to prove the verified heat tree optimal for depth 2. Usage: node code/test_server.js */
global.window = {};
const path = require("path"), D = require(path.join(__dirname, "..", "docs", "data.js")), E = require(path.join(__dirname, "..", "docs", "engine.js"));
const m = E.makeModel(D.case), pr = E.predicates(m);
async function ask(env, depth, cutoff) { const b = E.build(m, env);
  const r = await fetch("http://127.0.0.1:8765/api/solve_tree", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ nS: m.nS, A: 3, reach: Array.from(m.reach), preds: pr.cols.map(c => Array.from(c)), depth, cutoff, time_limit: 120, envs: [{ weight: 1, cost: Array.from(b.cost), nxt: Array.from(b.nxt), pr: Array.from(b.pr), mu: Array.from(b.mu) }] }) });
  return r.json(); }
(async () => {
  const v = D.verified.trees.heat; let r = await ask("heat", 2, v.obj);
  console.log("heat depth 2, cutoff = verified optimum", v.obj.toFixed(2), "->", { proven: r.proven, obj: r.obj, bound: r.bound, time: +r.time.toFixed(1), status: r.status });
  r = await ask("heat", 1, 1e9); const t = E.tableArr(m, pr, r.nodes, r.leaves, 1);
  console.log("heat depth 1, no cutoff ->", { proven: r.proven, obj: +r.obj.toFixed(2), check: +E.fastValue(m, "heat", t).toFixed(2), time: +r.time.toFixed(1) });
})();
