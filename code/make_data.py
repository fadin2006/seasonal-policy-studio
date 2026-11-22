"""Derive the 8 x 3 seasonal-diurnal joint states from the reduced 156-scenario files of the
generation-expansion model (read-only source). Output: data/split_states.json"""
import json, os, sys
from pathlib import Path
import pandas as pd

SRC = Path(os.environ.get("EXPANSION_DATA", "."))   # folder with reduced_load/wind/solar_8stage CSV files of the expansion model
OUT = Path(__file__).resolve().parents[1] / "data" / "split_states.json"
SPLITS = ["spring_day", "spring_night", "summer_day", "summer_night", "fall_day", "fall_night", "winter_day", "winter_night"]
HOURS = [1104, 1104, 1104, 1104, 1092, 1092, 1080, 1080]   # manuscript, per-stage hours
L = pd.read_csv(SRC / "reduced_load_8stage.csv", dtype={"Pair_ID8": str})
W = pd.read_csv(SRC / "reduced_wind_8stage.csv", dtype={"Pair_ID8": str})
S = pd.read_csv(SRC / "reduced_solar_8stage_110MW.csv", dtype={"Pair_ID8": str})
ids = L.Pair_ID8.str.zfill(8)
assert (ids == W.Pair_ID8.str.zfill(8)).all() and (ids == S.Pair_ID8.str.zfill(8)).all() and len(L) == 156
assert (L.GlobalIdx == W.GlobalIdx).all() and (L.GlobalIdx == S.GlobalIdx).all()
out = {"source": "reduced_load/wind/solar_8stage CSVs (156 scenarios), read %s" % pd.Timestamp.now().date(),
       "wind_plant_MW": 200, "solar_plant_MW": 110, "splits": []}
for i, name in enumerate(SPLITS):
    j = ids.str[i]
    rows = []
    for k in "012":
        m = j == k
        assert L.loc[m, name].nunique() == 1 and W.loc[m, name].nunique() == 1 and S.loc[m, name].nunique() == 1
        rows.append({"j": int(k), "p": float(L.loc[m, "Prob"].sum()), "demand_MW": float(L.loc[m, name].iloc[0]),
                     "wind_cf": float(W.loc[m, name].iloc[0]) / 200.0, "solar_cf": float(S.loc[m, name].iloc[0]) / 110.0})
    out["splits"].append({"name": name, "hours": HOURS[i], "states": rows})
OUT.write_text(json.dumps(out, indent=1))
print("wrote", OUT, "n scenarios", len(L), "prob sum", L.Prob.sum())
for s in out["splits"]:
    print(s["name"], [(round(r["p"], 3), round(r["demand_MW"]), round(r["wind_cf"], 3), round(r["solar_cf"], 3)) for r in s["states"]])
