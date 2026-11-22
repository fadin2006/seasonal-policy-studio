"""Write docs/data.js: the built-in case and the verified Python results (results/results.json)."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "results" / "results.json").read_text())
D = json.loads((ROOT / "data" / "split_states.json").read_text())
M = R["model"]
def rnd(o, nd=5):
    if isinstance(o, float): return round(o, nd)
    if isinstance(o, list): return [rnd(v, nd) for v in o]
    if isinstance(o, dict): return {k: rnd(v, nd) for k, v in o.items()}
    return o
kind = {"design": None, "heat": "heat", "lull": "drought", "heat_mild": "heat", "lull_mild": "drought", "compound": "both"}
tau = {"design": (1.0, 1.0), "heat": (1.4, 2.2), "lull": (1.4, 2.2), "heat_mild": (1.25, 1.8), "lull_mild": (1.25, 1.8), "compound": (1.4, 2.2)}
case = dict(name="Built-in case (generation-expansion manuscript, 156 scenarios)",
            splits=[dict(name=s["name"], hours=s["hours"], states=[dict(p=x["p"], demand=x["demand_MW"], wind_cf=x["wind_cf"], solar_cf=x["solar_cf"]) for x in s["states"]]) for s in D["splits"]],
            fleet=dict(conv=M["conv_MW"], wind=M["wind_MW"], solar=M["solar_MW"], avail=M["avail"], blockFrac=M["block_MW"] / M["conv_MW"], backlog=R["meta"]["B"], rho=M["rho"]),
            envs={k: dict(label=v["label"], d=[v["d"][i] for i in (0, 2, 4, 6)], w=[v["w"][i] for i in (0, 2, 4, 6)], kind=kind[k], tau=tau[k][0], taup=tau[k][1], library=v["in_library"]) for k, v in M["envs"].items()})
ver = dict(meta=R["meta"], dp_bound=R["dp_bound"], cross_eval=R["cross_eval"], in_loop=R["in_loop"], scores=R["scores"], scaling=R["scaling"], sensitivity=R["sensitivity_backlog"],
           monitor=dict(confusion=R["monitor"]["confusion"], train_accuracy=R["monitor"]["train_accuracy"]),
           trees={k: dict(nodes=v["milp"]["nodes"], leaves=v["milp"]["leaves"], depth=v["milp"]["depth"], obj=v["milp"]["obj"], bound=v["milp"]["bound_dp"], proven=v["milp"]["proven"],
                          gap=v["milp"]["mip_gap"], time=v["milp"]["time"], n_vars=v["milp"]["n_vars"], n_cons=v["milp"]["n_cons"], n_bin=v["milp"]["n_bin"]) for k, v in R["trees"].items()},
           predicate_labels=M["predicate_labels"])
js = "window.STUDIO_DATA = " + json.dumps(rnd(dict(case=case, verified=ver), 6), separators=(",", ":")) + ";\nif (typeof module === 'object') module.exports = window.STUDIO_DATA;\n"
(ROOT / "docs" / "data.js").write_text(js, encoding="utf-8")
print("wrote docs/data.js", round(len(js) / 1024), "KB")
