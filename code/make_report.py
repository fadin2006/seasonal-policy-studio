"""Write REPORT.md from results/results.json (numbers only from the run)."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "results" / "results.json").read_text())
M, S, L = R["model"], R["scores"], R["in_loop"]
LIB = ["design", "heat", "lull"]
lab = {e: M["envs"][e]["label"] for e in M["envs"]}
f0 = lambda v: f"{v:,.0f}"
f1 = lambda v: f"{v:,.1f}"


def tree_text(n, ind=0):
    pad = "  " * ind
    if "test" not in n:
        return f"{pad}- {n['text']}\n"
    return f"{pad}- {n['test']}?\n{pad}  - yes:\n{tree_text(n['yes'], ind + 2)}{pad}  - no:\n{tree_text(n['no'], ind + 2)}"


def mon_text(n, ind=0):
    pad = "  " * ind
    if "test" not in n:
        return f"{pad}- {lab[n['env']]} ({n['share']*100:.0f}% of training windows in this leaf)\n"
    return f"{pad}- {n['test']}?\n{pad}  - yes:\n{mon_text(n['yes'], ind + 2)}{pad}  - no:\n{mon_text(n['no'], ind + 2)}"


o = []
o.append(f"# Seasonal Tree-Policy Lab: results of the run of {R['meta']['date']}\n")
o.append("Stylised single-node study case built on the scenario data of the generation-expansion manuscript (Muna, Malik and Syed, submitted to Smart Cities, 3 November 2026). "
         "The maintenance decision, block size, availability, penalty and the stressed environments are assumptions; nothing here is a result of the expansion model. "
         "All costs are expected unserved energy in GWh per year plus a charge for unfinished maintenance.\n")
o.append("## 1. The case\n")
o.append(f"- Eight seasonal-diurnal splits, three joint demand-wind-solar states per split, probabilities from the 156 reduced scenarios.\n"
         f"- Fleet: conventional {f0(M['conv_MW'])} MW, wind {f0(M['wind_MW'])} MW, solar {f0(M['solar_MW'])} MW (manuscript tables). Availability of units not in maintenance {M['avail']*100:.0f}% (assumed).\n"
         f"- Decision per split: 0, 1 or 2 maintenance blocks out; one block = {f0(M['block_MW'])} MW (assumed 10% of the conventional fleet). {R['meta']['B']} block-splits are due per year; an unfinished block costs {f0(M['penalty_GWh_per_block'])} GWh.\n"
         f"- State: split, backlog, rank of the last net load. {M['n_states']} reachable states, 3 actions, {M['n_predicates']} candidate tests for a tree.\n")
o.append("| Environment | In library | Demand factor (spring, summer, fall, winter) | Wind factor | Bound (GWh) |\n|---|---|---|---|---|")
for e, v in M["envs"].items():
    dd = ", ".join("%.2f" % v["d"][i] for i in (0, 2, 4, 6)); ww = ", ".join("%.2f" % v["w"][i] for i in (0, 2, 4, 6))
    o.append("| %s | %s | %s | %s | %s |" % (v["label"], "yes" if v["in_library"] else "no", dd, ww, f1(R["dp_bound"][e])))
o.append("\nThe bound is the optimum of the unrestricted Markov decision process (dynamic programming); no tree can do better.\n")
o.append("## 2. Tree policies found by search plus mixed-integer programming\n")
o.append("| Tree | Depth | Rules | Cost (GWh) | Bound (GWh) | Above bound | Proven optimal | Time (s) | MIP size at last depth (variables / constraints / binary) |\n|---|---|---|---|---|---|---|---|---|")
for k, v in R["trees"].items():
    m = v["milp"]
    o.append(f"| {k} | {m['depth']} | {v['leaves']} | {f1(m['obj'])} | {f1(m['bound_dp'])} | {(m['obj']/m['bound_dp']-1)*100:.1f}% | {'yes' if m['proven'] else 'no (gap ≤ %.1f%%)' % (m['mip_gap']*100)} | {m['time']:.0f} | {m['n_vars']:,} / {m['n_cons']:,} / {m['n_bin']} |")
o.append("\nMethod: smallest depth first; at each depth a local search with exact evaluation gives a starting tree, and the mixed-integer program (occupancy-measure formulation, only the tree choices binary) runs with that value as a cutoff. "
         "The search stops as soon as a tree meets the bound. The robust tree minimises the mean cost over the three library environments.\n")
for k, v in R["trees"].items():
    o.append(f"**{k} tree**\n\n{tree_text(v['rules'])}")
o.append("### Exact cost of every tree in every environment (GWh per year)\n")
o.append("| Policy | " + " | ".join(lab[e] for e in M["envs"]) + " |\n|---|" + "---|" * len(M["envs"]))
for k, row in R["cross_eval"].items():
    o.append(f"| {k.replace('milp_', 'MIP tree: ').replace('bb_', 'simulator-only tree: ')} | " + " | ".join(f1(row[e]) for e in M["envs"]) + " |")
o.append("| bound (dynamic programming) | " + " | ".join(f1(R["dp_bound"][e]) for e in M["envs"]) + " |\n")
o.append("### Simulator-only search (framework 3)\n")
o.append("| Tree | Sample-average cost ± 95% | Exact cost | Candidates | Simulated years | Time (s) |\n|---|---|---|---|---|---|")
for k, v in R["trees"].items():
    b = v["blackbox"]; own = [k] if k != "robust" else LIB
    ex = sum(R["cross_eval"]["bb_" + k][e] for e in own) / len(own)
    o.append(f"| {k} | {f1(b['saa_mean'])} ± {f1(b['saa_ci'])} | {f1(ex)} | {b['candidates']:,} | {b['simulator_episodes']:,} | {b['time']:.0f} |")
mo = R["monitor"]
o.append("\n## 3. Runtime monitor\n")
o.append(f"Depth-2 tree (CART) over features of the last {mo['window']} splits; training accuracy {mo['train_accuracy']*100:.1f}%. Features: {', '.join(mo['features'])}. "
         f"A window outside the 0.2%-99.8% range of its predicted class is labelled unknown. Assumed measurement noise: demand {mo['noise']['demand_rel']*100:.1f}% relative, wind capacity factor {mo['noise']['wind_cf']:.2f} absolute.\n")
o.append(mon_text(mo["rules"]))
o.append("Share of windows given each label on fresh simulations:\n")
o.append("| True environment | Design year | Extended heat | Cold wind lull | Unknown |\n|---|---|---|---|---|")
for e, c in mo["confusion"].items():
    o.append(f"| {lab[e]} | {c['design']*100:.0f}% | {c['heat']*100:.0f}% | {c['lull']*100:.0f}% | {c['unknown']*100:.0f}% |")
o.append("\n## 4. Three frameworks in the loop\n")
o.append(f"Three-year paths; year 1 design year; change at a uniformly random split of year 2; {S['episodes_per_scenario']:,} paths per scenario; mean GWh per year ± 95% interval.\n")
cols = [("nominal", "Design-year tree, no monitor"), ("robust", "F1 robust tree"), ("monitor_milp", "F2 monitor + library"), ("monitor_blackbox", "F3 simulator-only trees"),
        ("hybrid", "After review: confident switching"), ("clairvoyant_library", "Library with true label"), ("clairvoyant_dp", "Lower bound")]
o.append("| Change to | " + " | ".join(c[1] for c in cols) + " | Mean detection delay (splits) | Wrong label before change |\n|---|" + "---|" * (len(cols) + 2))
for e, r in L.items():
    d = r["detection"]
    o.append(f"| {lab[e]} | " + " | ".join(f"{f0(r[c[0]]['mean'])} ± {f0(r[c[0]]['ci'])}" for c in cols) + f" | {d['mean_delay_splits']:.1f} | {d['false_alarm_before_change']*100:.1f}% |")
o.append("\n### Scores\n")
o.append("| Criterion | Weight | " + " | ".join(f["short"] for f in S["frameworks"]) + " | Basis |\n|---|---|" + "---|" * (len(S["frameworks"]) + 1))
for i, c in enumerate(S["criteria"]):
    o.append(f"| {c['name']} | {c['weight']} | " + " | ".join(f"{f['scores'][i]:.1f}" for f in S["frameworks"]) + f" | {c['basis']} |")
o.append("| **Weighted total (0-100)** | 100 | " + " | ".join(f"**{f['total']:.1f}**" for f in S["frameworks"]) + f" | Highest: {S['best_name']} |\n")
o.append("Mean extra cost over the lower bound (GWh per year): " + ", ".join(f"{k} {v:,.0f}" for k, v in S["regret"].items()) + ".\n")
o.append(S["summary"] + "\n")
o.append("### Improved after review\n\n" + S["final"]["text"] + "\n")
o.append("## 5. When does selection pay off? Sensitivity to the maintenance requirement (exact)\n")
o.append("| Block-splits due | Environment | Bound | Own tree | Robust tree | Design-year tree | Cost of not selecting |\n|---|---|---|---|---|---|---|")
for b, rows in R["sensitivity_backlog"].items():
    for e in LIB:
        r = rows[e]
        o.append(f"| {b} | {lab[e]} | {f1(r['dp'])} | {f1(r['own_tree'])} | {f1(r['robust_tree'])} | {f1(r['design_tree'])} | {f1(r['robust_tree'] - r['own_tree'])} |")
o.append("\n" + S["sensitivity_note"] + "\n")
o.append("## 6. Scaling (benchmark view)\n")
o.append("| Memory | Reachable states | MIP variables | MIP constraints | MIP time (s) | MIP gap | MIP tree cost | Bound | Local search (s) / cost | Simulator search (s) / cost |\n|---|---|---|---|---|---|---|---|---|---|")
for r in R["scaling"]:
    gap = "no tree found" if r["milp_gap"] is None else f"{r['milp_gap']*100:.2f}%"
    obj = "" if r["milp_obj"] is None else f1(r["milp_obj"])
    o.append(f"| {r['memory']} | {r['states']:,} | {r['milp_vars']:,} | {r['milp_cons']:,} | {r['milp_time']:.0f} | {gap} | {obj} | {f1(r['dp'])} | {r['local_search_time']:.0f} / {f1(r['local_search_value'])} | {r['blackbox_time']:.0f} / {f1(r['blackbox_value'])} |")
o.append("\n" + S["scaling_note"] + "\n")
o.append("## 7. Limits\n")
o.append("- Single node; the network, unit commitment and zonal limits of the expansion model are not represented.\n- The environments, noise levels, block size, availability and penalty are assumptions chosen to make a test case; the maintenance requirement was set where the three environments need different schedules, and section 5 shows the result for other values.\n"
         "- The monitor is a CART tree, not an optimal tree, and its thresholds were learned on simulated data from the same generators it is tested on (fresh seeds; two milder variants and one compound environment were held out).\n"
         "- 'Proven optimal' refers to the tree class used here (complete trees up to depth 3 over the listed tests).\n- The simulator-only search has no optimality proof; its exact cost is known here only because the model is available.\n")
(ROOT / "REPORT.md").write_text("\n".join(o), encoding="utf-8")
print("wrote REPORT.md", len("\n".join(o)))
