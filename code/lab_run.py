"""Run the whole lab and write results/results.json (also used by the web mock-up).
Usage: python -X utf8 lab_run.py [--fast]"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
from lab_core import (ROOT, T, HOURS, SPLITS, DEM, WCF, SCF, P0, RANK, INV, ENVS, LIBRARY, ACTIONS, WIND_MW, SOLAR_MW, CONV_MW,
                      AVAIL, BLOCK_FRAC, RHO, Model, optimal_tree, solve_tree, local_search, fast_value, dp_sequence, tree_table, describe, count_leaves)
from lab_sim import (Monitor, BlackBox, exo, residuals, window_features, rollout, WINDOW, FEATS, SIG_D, SIG_W)

RES = ROOT / "results"; RES.mkdir(exist_ok=True)
FAST = "--fast" in sys.argv
B_BASE, DEPTH, TL = 10, 3, (30 if FAST else 240)
THETA = 0.9          # leaf purity needed before the improved model leaves the robust tree
OBJ = {"design": {"design": 1.0}, "heat": {"heat": 1.0}, "lull": {"lull": 1.0}, "robust": {e: 1 / 3 for e in LIBRARY}}
ALL = list(ENVS)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def cached(path, fn):
    p = RES / path
    if p.exists():
        return json.loads(p.read_text())
    out = fn(); p.write_text(json.dumps(out, indent=1)); return out


def solve_trees(B, mem=1, names=("design", "heat", "lull", "robust"), tl=TL):
    m = Model(B=B, mem=mem); M, labels = m.predicates()
    out = {}
    for nm in names:
        tr = cached(f"tree_B{B}_m{mem}_{nm}.json", lambda: solve_tree(m, M, OBJ[nm], DEPTH, time_limit=tl))
        log(f"tree B={B} {nm}: depth {tr['depth']} cost {tr['obj']:.1f} bound {tr['bound_dp']:.1f} proven {tr['proven']} gap {tr['mip_gap']:.4f} time {tr['time']:.0f}s")
        out[nm] = tr
    return m, M, labels, out


def table_of(M, tr):
    return tree_table(M, tr["nodes"], tr["leaves"], tr["depth"])


def schedule(m, pol, envs):
    out = np.zeros(T)
    for e in envs:
        _, visit = m.evaluate(e, pol)
        np.add.at(out, m.t_of, visit * np.minimum(pol, m.b_of) / len(envs))
    return out.tolist()


def scores(R, trees, bb, labels):
    L = R["in_loop"]; sc = list(L)
    reg = lambda k: float(np.mean([L[e][k]["mean"] - L[e]["clairvoyant_dp"]["mean"] for e in sc]))
    r_nom = reg("nominal")
    perf = lambda k: 5 * max(0.0, 1 - reg(k) / r_nom) if r_nom > 0 else 0.0
    lv = lambda t: count_leaves(describe(t, labels))
    n1 = lv(trees["robust"]); n2 = sum(lv(trees[k]) for k in trees) + 4; n3 = sum(lv(bb[k]) for k in bb) + 4
    interp = lambda n: 5 if n <= 6 else 4 if n <= 12 else 3 if n <= 20 else 2 if n <= 30 else 1
    S = R["scaling"]; solved = sum(1 for r in S if r["milp_gap"] is not None and r["milp_gap"] <= 1e-4)
    mip_scal = {len(S): 5.0, len(S) - 1: 3.5}.get(solved, 2.0 if solved else 1.0)
    bbg = (S[-1]["blackbox_value"] - S[-1]["dp"]) / S[-1]["dp"]
    bb_scal = 5.0 - (1.0 if bbg > 0.05 else 0.0) - (1.0 if S[-1]["blackbox_time"] > 2 * S[0]["blackbox_time"] + 5 else 0.0)
    lib_proven = sum(bool(trees[k]["proven"]) for k in LIBRARY)
    crit = [
        dict(name="Cost when the environment changes", weight=25, basis="Measured: share of the avoidable extra cost removed, between the design-year tree without a monitor (0) and the lower bound that knows the change in advance (5); mean of six scenarios."),
        dict(name="Optimality certificate", weight=20, basis=f"F1: robust tree {'proven optimal' if trees['robust']['proven'] else 'not proven, gap reported'}. F2: {lib_proven} of 3 library trees proven optimal; the monitor is tested, not proven. F3: confidence intervals only."),
        dict(name="Readability", weight=15, basis=f"Measured: rules an operator must read (F1 {n1}, F2 {n2}, F3 {n3}); 5 for at most 6, 4 for at most 12, 3 for at most 20."),
        dict(name="Works without a model", weight=15, basis="Judgement: F1 and F2 need the transition model of each environment; F3 needs only a simulator."),
        dict(name="Scaling with the state space", weight=15, basis=f"Measured: the plain mixed-integer program was solved to optimality in {solved} of {len(S)} sizes within {R['meta']['time_limit_s']} s; the simulator search took under {max(1, round(S[-1]['blackbox_time'] + 0.5)):.0f} s at the largest size and ended {bbg*100:.0f}% above the bound."),
        dict(name="Reuse of the existing optimisation model", weight=10, basis="Judgement: F1 is one mixed-integer program; F2 adds one per environment and a monitor; F3 is new code."),
    ]
    fw = [dict(short="F1", name="one robust tree by mixed-integer programming, no monitor", key="robust",
               scores=[perf("robust"), 5.0 if trees["robust"]["proven"] else 3.0, interp(n1), 1.0, mip_scal, 5.0]),
          dict(short="F2", name="monitor tree with a library of trees by mixed-integer programming", key="monitor_milp",
               scores=[perf("monitor_milp"), 4.0 if lib_proven == 3 else 3.0, interp(n2), 2.0, mip_scal, 4.0]),
          dict(short="F3", name="monitor tree with trees found by simulator-only search", key="monitor_blackbox",
               scores=[perf("monitor_blackbox"), 2.0, interp(n3), 5.0, bb_scal, 2.0])]
    for f in fw:
        f["total"] = float(sum(s * c["weight"] for s, c in zip(f["scores"], crit)) / 5)
        f["regret"] = reg(f["key"])
    best = max(fw, key=lambda f: f["total"])
    order = sorted(fw, key=lambda f: -f["total"])
    hyb = dict(regret=reg("hybrid"), perf=perf("hybrid"), mean={e: L[e]["hybrid"]["mean"] for e in sc})
    adopt = bool(hyb["regret"] < reg(best["key"]) - 1e-9)
    final = dict(key="hybrid" if adopt else best["key"], adopted_improvement=adopt, theta=THETA,
                 text=("After the review the best-scoring framework was combined with the monitor: the robust tree runs by default and a specialised tree takes over only on a heat or wind-lull label from a leaf that is at least %d%% pure. " % round(THETA * 100)
                       + ("This lowers the mean extra cost over the lower bound from %.0f to %.0f GWh per year, so it is the model applied on this page." % (reg(best["key"]), hyb["regret"]) if adopt else
                          "This does not lower the mean extra cost over the lower bound (%.0f against %.0f GWh per year), so the page applies the best-scoring framework unchanged and uses the monitor as an alarm only." % (hyb["regret"], reg(best["key"])))))
    return dict(criteria=crit, frameworks=fw, best=best["short"], best_name=f"{best['short']}, {best['name']}", best_strategy=best["key"],
                regret=dict(nominal=r_nom, hybrid=hyb["regret"], **{f["short"]: f["regret"] for f in fw}), final=final,
                summary=(f"Three ways to extend the model were built and scored on this case. {order[0]['short']} scores highest ({order[0]['total']:.0f} of 100), "
                         f"ahead of {order[1]['short']} ({order[1]['total']:.0f}) and {order[2]['short']} ({order[2]['total']:.0f}). Two criteria are measured on the simulations, "
                         f"two on the solver runs, and two are judgements, marked as such. Weights are a choice; the table lets a reader change them."))


def main():
    t_start = time.time()
    R = {"meta": {"date": time.strftime("%Y-%m-%d"), "B": B_BASE, "depth": DEPTH, "time_limit_s": TL}}
    m, M, labels, trees = solve_trees(B_BASE)
    dp = {e: m.dp(e) for e in ALL}
    R["dp_bound"] = {e: dp[e][0] for e in ALL}
    tab = {k: table_of(M, v) for k, v in trees.items()}

    # ---- black-box search (simulator calls only) --------------------------------
    bb = {}
    for nm in OBJ:
        def run(nm=nm):
            t0 = time.time()
            s = BlackBox(m, M, list(OBJ[nm]), DEPTH, n=200 if FAST else 400).search(budget=1000 if FAST else 6000)
            s["time"] = time.time() - t0; return s
        bb[nm] = cached(f"bb_B{B_BASE}_d{DEPTH}_{nm}.json", run)
        log(f"black-box {nm}: SAA {bb[nm]['saa_mean']:.1f} ± {bb[nm]['saa_ci']:.1f}, {bb[nm]['candidates']} candidates, {bb[nm]['time']:.0f}s")
    tabb = {k: table_of(M, v) for k, v in bb.items()}

    # ---- exact cross-evaluation: every policy in every environment ----------------
    cross = {}
    for nm in OBJ:
        cross["milp_" + nm] = {e: m.evaluate(e, tab[nm])[0] for e in ALL}
        cross["bb_" + nm] = {e: m.evaluate(e, tabb[nm])[0] for e in ALL}
    R["cross_eval"] = cross
    R["trees"] = {nm: dict(milp=trees[nm], blackbox=bb[nm], rules=describe(trees[nm], labels), rules_blackbox=describe(bb[nm], labels),
                           leaves=count_leaves(describe(trees[nm], labels)), leaves_blackbox=count_leaves(describe(bb[nm], labels))) for nm in OBJ}
    R["schedule"] = {nm: schedule(m, tab[nm], list(OBJ[nm])) for nm in OBJ}

    # ---- runtime monitor ---------------------------------------------------------
    mon = Monitor(m, depth=2)
    rng = np.random.default_rng(2026)
    conf = {}
    for e in ALL:
        _, dob, wob, _ = exo(m, [e] * (T * 6), 4000, rng)
        gD, gW = residuals(dob, wob)
        labs = np.concatenate([mon.label(window_features(gD, gW, k)) for k in range(WINDOW, T * 6)])
        conf[e] = {n: float((labs == i).mean()) for i, n in enumerate(LIBRARY + ["unknown"])}
    R["monitor"] = dict(rules=mon.rules(), features=FEATS, window=WINDOW, train_accuracy=mon.train_acc, confusion=conf,
                        boxes={LIBRARY[c]: {"low": mon.boxes[c, 0].tolist(), "high": mon.boxes[c, 1].tolist()} for c in range(3)},
                        noise=dict(demand_rel=SIG_D, wind_cf=SIG_W))
    log("monitor confusion", {e: {k: round(v, 2) for k, v in c.items()} for e, c in conf.items()})

    # ---- monitor in the loop: three years, change of environment in year 2 --------
    K, n_per = 3 * T, (250 if FAST else 1000)
    truth = {"design": 0, "heat": 1, "lull": 2, "heat_mild": 1, "lull_mild": 2, "compound": 3}
    dp_tab = {e: dp[e][1] for e in ALL}
    loop = {}
    for e in ALL:
        costs = {s: [] for s in ["nominal", "robust", "monitor_milp", "monitor_blackbox", "hybrid", "clairvoyant_library", "clairvoyant_dp"]}
        delays, false_alarm, lab_share = [], [], np.zeros(4)
        for tau in range(T, 2 * T):
            seq = ["design"] * tau + [e] * (K - tau)
            j, dob, wob, obef = exo(m, seq, n_per, rng)
            gD, gW = residuals(dob, wob)
            lab = np.zeros((n_per, K), int); hyb = np.full((n_per, K), 3)
            for k in range(WINDOW, K):
                lab[:, k], conf = mon.label_conf(window_features(gD, gW, k))
                hyb[:, k] = np.where((lab[:, k] > 0) & (lab[:, k] < 3) & (conf >= THETA), lab[:, k], 3)   # robust unless a confident heat or lull label
            true_lab = np.zeros((n_per, K), int); true_lab[:, tau:] = truth[e]
            step_lab = np.tile(np.arange(K), (n_per, 1))
            z = np.zeros((n_per, K), int)
            costs["nominal"].append(rollout(m, [tab["design"]], z, j, seq, obef))
            costs["robust"].append(rollout(m, [tab["robust"]], z, j, seq, obef))
            lib = [tab["design"], tab["heat"], tab["lull"], tab["robust"]]
            costs["monitor_milp"].append(rollout(m, lib, lab, j, seq, obef))
            costs["monitor_blackbox"].append(rollout(m, [tabb["design"], tabb["heat"], tabb["lull"], tabb["robust"]], lab, j, seq, obef))
            costs["clairvoyant_library"].append(rollout(m, lib, true_lab, j, seq, obef))
            costs["hybrid"].append(rollout(m, lib, hyb, j, seq, obef))
            costs["clairvoyant_dp"].append(rollout(m, dp_sequence(m, seq), step_lab, j, seq, obef))
            false_alarm.append((lab[:, WINDOW:tau] != 0).mean())
            if e != "design":
                hit = lab[:, tau:] == truth[e]
                first = np.where(hit.any(1), hit.argmax(1), K - tau)
                delays.append(first)
            for c in range(4):
                lab_share[c] += (lab[:, tau:] == c).mean() / T
        out = {}
        for s, v in costs.items():
            v = np.concatenate(v) / 3.0                                   # GWh per year
            out[s] = dict(mean=float(v.mean()), ci=float(1.96 * v.std(ddof=1) / np.sqrt(len(v))))
        d = np.concatenate(delays) if delays else np.array([0])
        out["detection"] = dict(mean_delay_splits=float(d.mean()), median_delay_splits=float(np.median(d)),
                                share_within_4=float((d <= 4).mean()), false_alarm_before_change=float(np.mean(false_alarm)),
                                label_share_after_change={n: float(lab_share[i]) for i, n in enumerate(LIBRARY + ["unknown"])})
        loop[e] = out
        log("in-loop", e, {s: round(out[s]["mean"], 1) for s in costs}, "delay", round(out["detection"]["mean_delay_splits"], 2))
    R["in_loop"] = loop

    # ---- sensitivity to the maintenance requirement (exact) -----------------------
    sens = {}
    for B in sorted(set(([] if FAST else [8, 12]) + [B_BASE])):
        mB, MB, lB, tB = (m, M, labels, trees) if B == B_BASE else solve_trees(B, tl=min(TL, 90))
        row = {}
        for e in LIBRARY:
            row[e] = dict(dp=mB.dp(e)[0], own_tree=fast_value(mB, e, table_of(MB, tB[e])), robust_tree=fast_value(mB, e, table_of(MB, tB["robust"])),
                          design_tree=fast_value(mB, e, table_of(MB, tB["design"])), robust_proven=bool(tB["robust"]["proven"]))
        sens[str(B)] = row
    R["sensitivity_backlog"] = sens

    # ---- scaling: same case with a longer observation memory (heat objective, depth 2) ----
    scal = []
    for mem in ([1, 2] if FAST else [1, 2, 3]):
        mm = Model(B=B_BASE, mem=mem); MM, _ = mm.predicates()
        def run(mm=mm, MM=MM):
            t0 = time.time(); bound = mm.dp("heat")[0]; t_dp = time.time() - t0
            tr = optimal_tree(mm, OBJ["heat"], 2, MM, time_limit=TL)
            t0 = time.time(); ls = local_search(mm, MM, OBJ["heat"], 2, restarts=6); t_ls = time.time() - t0
            t0 = time.time(); s = BlackBox(mm, MM, ["heat"], 2, n=200).search(budget=1500); tb = time.time() - t0
            return dict(memory=mem, states=int(mm.reach.sum()), milp_vars=tr.get("n_vars"), milp_cons=tr.get("n_cons"), milp_time=tr["time"],
                        milp_gap=tr.get("mip_gap"), milp_obj=tr.get("obj"), dp=bound, dp_time=t_dp, local_search_time=t_ls, local_search_value=ls["obj"],
                        blackbox_time=tb, blackbox_value=fast_value(mm, "heat", tree_table(MM, s["nodes"], s["leaves"], 2)))
        scal.append(cached(f"scaling_B{B_BASE}_m{mem}.json", run))
        log("scaling", scal[-1])
    R["scaling"] = scal
    R["scores"] = scores(R, trees, bb, labels)
    R["scores"]["episodes_per_scenario"] = n_per * T
    own = lambda b, e: sens[b][e]["own_tree"]
    R["scores"]["sensitivity_note"] = ("Exact expected cost in GWh per year. 'Cost of not selecting' is the robust tree minus the tree built for that environment. Sum of the three differences: "
        + "; ".join(f"with {b} block-splits the three differences sum to {sum(sens[b][e]['robust_tree'] - own(b, e) for e in LIBRARY):,.0f} GWh" for b in sens) + ".")
    R["scores"]["scaling_note"] = ("Heat objective, depth-2 tree, plain mixed-integer program without a starting solution, time limit %d s. Longer memory multiplies the states by three each step; "
        "the tests of the tree stay the same. The local search with exact evaluation took %s s and the dynamic-programming bound %s s at the three sizes." % (
            TL, ", ".join(f"{r['local_search_time']:.0f}" for r in scal), ", ".join(f"{r['dp_time']:.2f}" for r in scal)))
    log("scores", {f["short"]: round(f["total"], 1) for f in R["scores"]["frameworks"]}, "best", R["scores"]["best"])

    # ---- model description for the page -------------------------------------------
    R["model"] = dict(splits=SPLITS, hours=HOURS.tolist(), demand=DEM.tolist(), wind_cf=WCF.tolist(), solar_cf=SCF.tolist(), p_design=P0.tolist(),
                      rank=RANK.tolist(), wind_MW=WIND_MW, solar_MW=SOLAR_MW, conv_MW=CONV_MW, avail=AVAIL, block_MW=BLOCK_FRAC * CONV_MW,
                      caps=[m.cap(a) for a in ACTIONS], penalty_GWh_per_block=m.pen, rho=RHO, n_states=int(m.reach.sum()), n_predicates=int(M.shape[1]),
                      predicate_labels=labels,
                      envs={e: dict(label=ENVS[e]["label"], d=list(map(float, ENVS[e]["d"])), w=list(map(float, ENVS[e]["w"])), NL=m.envs[e].NL.tolist(),
                                    P=m.envs[e].P.tolist(), mu0=m.envs[e].mu0.tolist(), in_library=e in LIBRARY) for e in ALL})
    R["tables"] = {"milp": {k: v.tolist() for k, v in tab.items()}, "blackbox": {k: v.tolist() for k, v in tabb.items()},
                   "dp": {e: dp_tab[e].tolist() for e in ALL}}
    R["meta"]["runtime_s"] = time.time() - t_start
    (RES / ("results_fast.json" if FAST else "results.json")).write_text(json.dumps(R, indent=1))
    log("done", round(time.time() - t_start), "s")


if __name__ == "__main__":
    main()
