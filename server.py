"""Seasonal Policy Studio: local back end.

Serves the front end in ./docs and proves tree policies optimal with a mixed-integer program (SciPy HiGHS).
Needs only Python with numpy and scipy.   Run:  python server.py   then open http://localhost:8765
The front end also works without this server (GitHub Pages); then only the "Prove with the solver" buttons are missing.
"""
from __future__ import annotations
import json, sys, time
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.optimize import milp, LinearConstraint, Bounds

WEB = Path(__file__).resolve().parent / "docs"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765


def solve_tree(req: dict) -> dict:
    """Best complete tree of the given depth for a finite-horizon decision process sent by the front end.
    req: nS, A, reach[nS], preds[nJ][nS] (0/1), depth, cutoff, time_limit, envs[{weight, cost[nS*A], nxt[nS*A*3], pr[nS*A*3], mu[nS]}].
    Binary: one test per inner node, one action per leaf. Continuous: leaf reached by each state, state-leaf-action
    links, policy, and visit probabilities per environment (occupancy measure)."""
    nS_all, nA, depth = int(req["nS"]), int(req["A"]), int(req["depth"])
    if not (1 <= depth <= 3):
        raise ValueError("depth must be 1, 2 or 3")
    reach = np.array(req["reach"], bool)
    S = np.where(reach)[0]; nS = len(S); pos = {int(s): i for i, s in enumerate(S)}
    if nS > 4000:
        raise ValueError("model too large for the interactive solver (%d reachable states)" % nS)
    M = np.array(req["preds"], bool).T                      # [state, test]
    nJ = M.shape[1]; nI, nL = 2 ** depth - 1, 2 ** depth
    envs = req["envs"]
    oz, oy = 0, nI * nJ
    orr = oy + nL * nA; ow = orr + nS * nL; opi = ow + nS * nL * nA; ox = opi + nS * nA
    nV = ox + len(envs) * nS * nA
    z = lambda m, j: oz + m * nJ + j
    y = lambda l, a: oy + l * nA + a
    r = lambda i, l: orr + i * nL + l
    w = lambda i, l, a: ow + (i * nL + l) * nA + a
    pi = lambda i, a: opi + i * nA + a
    x = lambda e, i, a: ox + (e * nS + i) * nA + a
    rows, cols, vals, lo, hi = [], [], [], [], []

    def add(cv, l, h):
        k = len(lo)
        for c, v in cv:
            rows.append(k); cols.append(c); vals.append(v)
        lo.append(l); hi.append(h)

    for m in range(nI):
        add([(z(m, j), 1) for j in range(nJ)], 1, 1)
    for l in range(nL):
        add([(y(l, a), 1) for a in range(nA)], 1, 1)
    paths = []
    for l in range(nL):
        node, p = l + nI, []
        while node > 0:
            par = (node - 1) // 2; p.append((par, node == 2 * par + 1)); node = par
        paths.append(p)
    for i, s in enumerate(S):
        tj = np.nonzero(M[s])[0]
        for l in range(nL):
            for (m, left) in paths[l]:
                if left:
                    add([(r(i, l), 1)] + [(z(m, j), -1) for j in tj], -np.inf, 0)
                else:
                    add([(r(i, l), 1)] + [(z(m, j), 1) for j in tj], -np.inf, 1)
            add([(w(i, l, a), 1) for a in range(nA)] + [(r(i, l), -1)], 0, 0)
            for a in range(nA):
                add([(w(i, l, a), 1), (y(l, a), -1)], -np.inf, 0)
        add([(r(i, l), 1) for l in range(nL)], 1, 1)
        for a in range(nA):
            add([(pi(i, a), 1)] + [(w(i, l, a), -1) for l in range(nL)], 0, 0)
    c = np.zeros(nV)
    for ei, e in enumerate(envs):
        cost = np.array(e["cost"], float).reshape(nS_all, nA); nxt = np.array(e["nxt"], int).reshape(nS_all, nA, 3)
        pr = np.array(e["pr"], float).reshape(nS_all, nA, 3); mu = np.array(e["mu"], float); wt = float(e["weight"])
        inflow = {i: [] for i in range(nS)}
        for i, s in enumerate(S):
            for a in range(nA):
                c[x(ei, i, a)] = wt * cost[s, a]
                add([(x(ei, i, a), 1), (pi(i, a), -1)], -np.inf, 0)
                for k in range(3):
                    n = int(nxt[s, a, k])
                    if n >= 0 and n in pos and pr[s, a, k] > 0:
                        inflow[pos[n]].append((x(ei, i, a), -float(pr[s, a, k])))
        for i, s in enumerate(S):
            add([(x(ei, i, a), 1.0) for a in range(nA)] + inflow[i], float(mu[s]), float(mu[s]))
    if req.get("cutoff") is not None:
        cut = float(req["cutoff"])
        add([(k, float(v)) for k, v in enumerate(c) if v != 0.0], -np.inf, cut * (1 + 1e-9) + 1e-9)
    Amat = sparse.csr_matrix((vals, (rows, cols)), shape=(len(lo), nV))
    integrality = np.zeros(nV); integrality[:orr] = 1
    t0 = time.time()
    res = milp(c, constraints=LinearConstraint(Amat, lo, hi), integrality=integrality, bounds=Bounds(np.zeros(nV), np.ones(nV)),
               options={"time_limit": float(min(300, req.get("time_limit", 120))), "mip_rel_gap": 1e-6, "disp": False})
    out = dict(time=time.time() - t0, status=str(res.message), n_vars=int(nV), n_cons=int(len(lo)), n_bin=int(orr), n_states=int(nS), obj=None, nodes=None, leaves=None)
    bound = getattr(res, "mip_dual_bound", None)
    out["bound"] = float(bound) if bound is not None and np.isfinite(bound) else None
    if res.x is not None:
        out["nodes"] = [int(v) for v in res.x[oz:oy].reshape(nI, nJ).argmax(1)]
        out["leaves"] = [int(v) for v in res.x[oy:orr].reshape(nL, nA).argmax(1)]
        out["obj"] = float(res.fun)
    out["proven"] = bool(res.status == 0) or bool(res.status == 2 and req.get("cutoff") is not None)   # optimal, or no tree below the cutoff
    if res.status == 2 and req.get("cutoff") is not None and out["bound"] is None:
        out["bound"] = float(req["cutoff"])
    return out


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(WEB), **k)

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/health"):
            return self._json(200, {"ok": True, "solver": "SciPy HiGHS"})
        return super().do_GET()

    def do_POST(self):
        if not self.path.startswith("/api/solve_tree"):
            return self._json(404, {"error": "unknown endpoint"})
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n > 30_000_000:
                raise ValueError("request too large")
            self._json(200, solve_tree(json.loads(self.rfile.read(n))))
        except Exception as exc:                       # report the reason to the page instead of a broken response
            self._json(400, {"error": str(exc)})

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        if "/api/" in (args[0] if args else ""):
            super().log_message(fmt, *args)


if __name__ == "__main__":
    print(f"Seasonal Policy Studio: http://localhost:{PORT}   (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
