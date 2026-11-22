"""Seasonal tree-policy and monitor lab: core model.

A stylised single-node maintenance-scheduling MDP built on the eight seasonal-diurnal
splits and the three joint demand-wind-solar states per split of the generation-expansion
model (Muna, Malik & Syed, submitted to Smart Cities, 3 Nov 2026).

What is taken from the model data: split order and hours, the 24 joint states and their
probabilities (156 reduced scenarios), fleet sizes (wind 8,000 MW, solar 4,700 MW,
conventional 36,638 MW = existing 23,300 MW + 13,338 MW built in the baseline Design C).
What is ASSUMED here (not in the manuscript): the maintenance decision, block size,
availability factor, backlog, penalty for unfinished maintenance, and the three stressed
environments. The network is ignored (single node), so numbers are not results of the
expansion model.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.optimize import milp, LinearConstraint, Bounds

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "data" / "split_states.json").read_text())
SPLITS = [s["name"] for s in DATA["splits"]]
HOURS = np.array([s["hours"] for s in DATA["splits"]], float)
T = 8
P0 = np.array([[r["p"] for r in s["states"]] for s in DATA["splits"]]); P0 /= P0.sum(1, keepdims=True)
DEM = np.array([[r["demand_MW"] for r in s["states"]] for s in DATA["splits"]])
WCF = np.array([[r["wind_cf"] for r in s["states"]] for s in DATA["splits"]])
SCF = np.array([[r["solar_cf"] for r in s["states"]] for s in DATA["splits"]])

WIND_MW, SOLAR_MW, CONV_MW = 8000.0, 4700.0, 36638.0      # from the manuscript tables
AVAIL = 0.92            # ASSUMED availability of conventional units not in maintenance
BLOCK_FRAC = 0.10       # ASSUMED size of one maintenance block (share of conventional fleet)
RHO = 0.25              # ASSUMED risk factor for an unfinished block
ACTIONS = [0, 1, 2]     # blocks taken out for maintenance in the split


def netload(dscale=1.0, wscale=1.0):
    d = np.asarray(dscale, float).reshape(-1, 1) if np.ndim(dscale) else dscale
    w = np.asarray(wscale, float).reshape(-1, 1) if np.ndim(wscale) else wscale
    return DEM * d - WCF * w * WIND_MW - SCF * SOLAR_MW


NL0 = netload()
RANK = np.argsort(np.argsort(NL0, axis=1), axis=1)          # design net-load rank of state j in split t (0 low, 2 high)
INV = np.argsort(NL0, axis=1)                                # INV[t, r] = j with rank r
DEM_RANK = np.argsort(np.argsort(DEM, axis=1), axis=1)
LOWWIND = np.where(WCF < 0.1, 2, np.where(WCF < 0.5, 1, 0))  # stress score for a wind drought

def _sea(spring, summer, fall, winter):
    return np.array([spring, spring, summer, summer, fall, fall, winter, winter], float)


ENVS = {
    "design":  dict(label="Design year", d=_sea(1, 1, 1, 1), w=_sea(1, 1, 1, 1), kind=None, tau=1.0, taup=1.0),
    "heat":    dict(label="Extended heat", d=_sea(1.15, 1.07, 1.15, 1.00), w=_sea(1, 1, 1, 1), kind="heat", tau=1.4, taup=2.2),
    "lull":    dict(label="Cold wind lull", d=_sea(1.00, 1.00, 1.02, 1.10), w=_sea(0.8, 0.8, 0.5, 0.3), kind="drought", tau=1.4, taup=2.2),
    # held-out variants, never used for training or policy synthesis
    "heat_mild": dict(label="Mild heat (held out)", d=_sea(1.10, 1.05, 1.10, 1.00), w=_sea(1, 1, 1, 1), kind="heat", tau=1.25, taup=1.8),
    "lull_mild": dict(label="Mild wind lull (held out)", d=_sea(1.00, 1.00, 1.01, 1.06), w=_sea(0.9, 0.9, 0.65, 0.45), kind="drought", tau=1.25, taup=1.8),
    "compound":  dict(label="Heat and wind lull (unseen)", d=_sea(1.12, 1.05, 1.12, 1.07), w=_sea(0.8, 0.8, 0.5, 0.4), kind="both", tau=1.4, taup=2.2),
}
LIBRARY = ["design", "heat", "lull"]


class Env:
    """One environment: net loads NL[t, j] and transition P[t, o, j], o = design net-load rank of the previous state."""

    def __init__(self, name):
        e = ENVS[name]
        self.name, self.label = name, e["label"]
        self.dscale, self.wscale = e["d"], e["w"]
        self.NL = netload(e["d"], e["w"])
        self.P = np.zeros((T, 3, 3))
        for t in range(T):
            tp = (t - 1) % T
            for o in range(3):
                jp = INV[tp, o]
                w = P0[t].copy()
                if e["kind"] in ("heat", "both"):
                    w = w * (e["taup"] if DEM_RANK[tp, jp] == 2 else e["tau"]) ** DEM_RANK[t]
                if e["kind"] in ("drought", "both"):
                    w = w * (e["taup"] if LOWWIND[tp, jp] == 2 else e["tau"]) ** LOWWIND[t]
                self.P[t, o] = w / w.sum()
        mu = np.ones(3) / 3                       # rank distribution of the last split, iterated over years
        for _ in range(20):
            for t in range(T):
                nxt = np.zeros(3)
                for o in range(3):
                    for j in range(3):
                        nxt[RANK[t, j]] += mu[o] * self.P[t, o, j]
                mu = nxt
        self.mu0 = mu                              # distribution of o at the start of a year


class Model:
    """Finite-horizon MDP over states (t, b, o): split, maintenance backlog, last net-load rank."""

    def __init__(self, B=10, mem=1):
        self.B, self.mem = B, mem
        self.block = BLOCK_FRAC * CONV_MW
        self.pen = RHO * self.block * HOURS.mean() / 1000.0          # GWh-equivalent per unfinished block
        self.nO = 3 ** mem
        self.nS = T * (B + 1) * self.nO
        self.nA = len(ACTIONS)
        self.envs = {n: Env(n) for n in ENVS}
        idx = np.arange(self.nS)
        self.t_of, rem = idx // ((B + 1) * self.nO), idx % ((B + 1) * self.nO)
        self.b_of, self.o_of = rem // self.nO, rem % self.nO
        self.reach = (self.b_of >= np.maximum(0, B - 2 * self.t_of))
        self._cache = {}

    def sid(self, t, b, o):
        return (t * (self.B + 1) + b) * self.nO + o

    def cap(self, a):
        return AVAIL * (CONV_MW - a * self.block)

    def build(self, name):
        """cost[s, a] (GWh of expected unserved energy, plus terminal penalty) and sparse transitions."""
        if name in self._cache:
            return self._cache[name]
        e = self.envs[name]
        cost = np.zeros((self.nS, self.nA))
        rows, cols, vals = [], [], []                 # P[(s, a) -> s2]
        for s in range(self.nS):
            t, b, o = self.t_of[s], self.b_of[s], self.o_of[s]
            last = o % 3
            for a in ACTIONS:
                ae = min(a, b)
                for j in range(3):
                    p = e.P[t, last, j]
                    shed = max(0.0, e.NL[t, j] - self.cap(ae))
                    cost[s, a] += p * HOURS[t] * shed / 1000.0
                    if t < T - 1:
                        o2 = (o * 3 + RANK[t, j]) % self.nO
                        rows.append(s * self.nA + a); cols.append(self.sid(t + 1, b - ae, o2)); vals.append(p)
                if t == T - 1:
                    cost[s, a] += self.pen * (b - ae)
        Pm = sparse.csr_matrix((vals, (rows, cols)), shape=(self.nS * self.nA, self.nS))
        mu = np.zeros(self.nS)
        if self.mem == 1:
            for o in range(3):
                mu[self.sid(0, self.B, o)] = e.mu0[o]
        else:                                          # longer memory: older ranks start at the middle rank
            for o in range(3):
                code = sum(1 * 3 ** k for k in range(1, self.mem)) + o
                mu[self.sid(0, self.B, code)] = e.mu0[o]
        self._cache[name] = (cost, Pm, mu)
        return self._cache[name]

    def dp(self, name):
        cost, Pm, mu = self.build(name)
        V = np.zeros(self.nS); pol = np.zeros(self.nS, int)
        for t in range(T - 1, -1, -1):
            S = np.where(self.t_of == t)[0]
            Q = cost[S] + (Pm @ V).reshape(self.nS, self.nA)[S]
            pol[S] = Q.argmin(1); V[S] = Q.min(1)
        return float(mu @ V), pol, V

    def evaluate(self, name, pol):
        """Exact expected cost of a deterministic policy table pol[s] in an environment; also state visit probabilities."""
        cost, Pm, mu = self.build(name)
        d, tot, visit = mu.copy(), 0.0, np.zeros(self.nS)
        for t in range(T):
            visit += d
            sa = np.nonzero(d)[0]
            tot += float((d[sa] * cost[sa, pol[sa]]).sum())
            nxt = np.zeros(self.nS)
            for s in sa:
                r = Pm.getrow(s * self.nA + pol[s])
                nxt[r.indices] += d[s] * r.data
            d = nxt
        return tot, visit

    # ---- tree predicates -------------------------------------------------
    def predicates(self, slack=True):
        """Boolean matrix b[s, j] and readable labels. A state goes LEFT when the predicate is true."""
        cols, labels = [], []
        t1 = self.t_of + 1
        names = ["spring day", "spring night", "summer day", "summer night", "fall day", "fall night", "winter day", "winter night"]
        for k in range(1, T):
            cols.append(t1 <= k); labels.append(f"split is {names[k-1]} or earlier")
        cols.append(self.t_of % 2 == 0); labels.append("it is a day split")
        for k in range(self.B):
            cols.append(self.b_of <= k); labels.append(f"backlog ≤ {k} blocks")
        for k in range(2):
            cols.append((self.o_of % 3) <= k); labels.append("last net load was low" if k == 0 else "last net load was low or middle")
        if slack:
            sl = 2 * (T - self.t_of) - self.b_of       # blocks that can still be scheduled minus backlog
            for k in range(0, 2 * T - self.B + 1):
                cols.append(sl <= k); labels.append(f"slack ≤ {k} blocks")
        M = np.array(cols).T
        R = np.where(self.reach)[0]
        keep, seen = [], set()
        for j in range(M.shape[1]):
            key = M[R, j].tobytes()
            if M[R, j].all() or not M[R, j].any() or key in seen:
                continue
            seen.add(key); keep.append(j)
        return M[:, keep], [labels[j] for j in keep]


def tree_table(M, nodes, leaves, depth):
    """Policy table from a complete binary tree: nodes[m] = predicate index, leaves[l] = action."""
    pos = np.zeros(M.shape[0], int)
    for _ in range(depth):
        go_left = M[np.arange(M.shape[0]), np.asarray(nodes)[pos]]
        pos = 2 * pos + np.where(go_left, 1, 2)
    return np.asarray(leaves)[pos - (2 ** depth - 1)]


def optimal_tree(model: Model, env_weights: dict, depth: int, M, time_limit=120.0, gap=1e-4, cutoff=None):
    """Optimal depth-limited tree policy by mixed-integer programming (occupancy-measure formulation).

    Variables: z[m, j] predicate chosen at internal node m; y[l, a] action at leaf l; r[s, l] state s reaches
    leaf l; w[s, l, a] state s reaches leaf l and the leaf plays a; pi[s, a] policy; x_e[s, a] visit probability
    of (s, a) in environment e. Only z and y are binary.
    """
    S = np.where(model.reach)[0]; nS = len(S); pos = {s: i for i, s in enumerate(S)}
    nJ, nA = M.shape[1], model.nA
    nI, nL = 2 ** depth - 1, 2 ** depth
    envs = list(env_weights)
    oz, oy = 0, nI * nJ
    orr = oy + nL * nA
    opi = orr + nS * nL
    ow = opi + nS * nA
    ox = ow + nS * nL * nA
    nV = ox + len(envs) * nS * nA
    w = lambda i, l, a: ow + (i * nL + l) * nA + a
    z = lambda m, j: oz + m * nJ + j
    y = lambda l, a: oy + l * nA + a
    r = lambda i, l: orr + i * nL + l
    pi = lambda i, a: opi + i * nA + a
    x = lambda e, i, a: ox + (e * nS + i) * nA + a
    rows, cols, vals, lo, hi = [], [], [], [], []
    nrow = 0

    def add(cv, l, h):
        nonlocal nrow
        for c, v in cv:
            rows.append(nrow); cols.append(c); vals.append(v)
        lo.append(l); hi.append(h); nrow += 1

    for m in range(nI):
        add([(z(m, j), 1) for j in range(nJ)], 1, 1)
    for l in range(nL):
        add([(y(l, a), 1) for a in range(nA)], 1, 1)
    paths = []                                    # per leaf: list of (internal node, went_left)
    for l in range(nL):
        node, p = l + nI, []
        while node > 0:
            par = (node - 1) // 2
            p.append((par, node == 2 * par + 1)); node = par
        paths.append(p)
    for i, s in enumerate(S):
        tj = np.nonzero(M[s])[0]
        for l in range(nL):
            for (m, left) in paths[l]:
                if left:      # r <= sum_j z[m, j] b[s, j]
                    add([(r(i, l), 1)] + [(z(m, j), -1) for j in tj], -np.inf, 0)
                else:         # r <= 1 - sum_j z[m, j] b[s, j]
                    add([(r(i, l), 1)] + [(z(m, j), 1) for j in tj], -np.inf, 1)
            add([(w(i, l, a), 1) for a in range(nA)] + [(r(i, l), -1)], 0, 0)          # sum_a w = r
            for a in range(nA):
                add([(w(i, l, a), 1), (y(l, a), -1)], -np.inf, 0)                      # w <= y
        add([(r(i, l), 1) for l in range(nL)], 1, 1)
        for a in range(nA):
            add([(pi(i, a), 1)] + [(w(i, l, a), -1) for l in range(nL)], 0, 0)         # pi = sum_l w
    c = np.zeros(nV)
    for ei, en in enumerate(envs):
        cost, Pm, mu = model.build(en)
        Pc = Pm.tocsc()
        for i, s in enumerate(S):
            for a in range(nA):
                c[x(ei, i, a)] = env_weights[en] * cost[s, a]
                add([(x(ei, i, a), 1), (pi(i, a), -1)], -np.inf, 0)
            cv = [(x(ei, i, a), 1.0) for a in range(nA)]
            col = Pc.getcol(s)
            for sa, p in zip(col.indices, col.data):
                s2, a2 = divmod(sa, nA)
                if s2 in pos:
                    cv.append((x(ei, pos[s2], a2), -p))
            add(cv, mu[s], mu[s])
    if cutoff is not None:                       # objective cutoff from a known tree (search gives it, the MIP proves)
        add([(k, float(v)) for k, v in enumerate(c) if v != 0.0], -np.inf, float(cutoff))
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(nrow, nV))
    integrality = np.zeros(nV); integrality[:orr] = 1
    ub = np.ones(nV)
    t0 = time.time()
    res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=integrality, bounds=Bounds(np.zeros(nV), ub),
               options={"time_limit": time_limit, "mip_rel_gap": gap, "disp": False})
    dt = time.time() - t0
    if res.x is None:
        return dict(status=res.message, time=dt)
    zz = res.x[oz:oy].reshape(nI, nJ).argmax(1); yy = res.x[oy:orr].reshape(nL, nA).argmax(1)
    return dict(nodes=[int(v) for v in zz], leaves=[int(v) for v in yy], depth=depth, obj=float(res.fun),
                bound=float(getattr(res, "mip_dual_bound", np.nan)), mip_gap=float(getattr(res, "mip_gap", np.nan)),
                time=dt, status=res.message, n_vars=int(nV), n_cons=int(nrow), n_bin=int(orr), n_states=int(nS))


def describe(tree, labels, depth=None):
    """Readable nested description of a tree."""
    depth = depth or tree["depth"]; nI = 2 ** depth - 1
    act = {0: "no maintenance", 1: "1 block out", 2: "2 blocks out"}

    def rec(node):
        if node >= nI:
            return {"action": int(tree["leaves"][node - nI]), "text": act[int(tree["leaves"][node - nI])]}
        return {"test": labels[tree["nodes"][node]], "yes": rec(2 * node + 1), "no": rec(2 * node + 2)}

    def prune(n):
        if "test" not in n:
            return n
        n["yes"], n["no"] = prune(n["yes"]), prune(n["no"])
        if "action" in n["yes"] and "action" in n["no"] and n["yes"]["action"] == n["no"]["action"]:
            return n["yes"]
        return n
    return prune(rec(0))


def count_leaves(d):
    return 1 if "test" not in d else count_leaves(d["yes"]) + count_leaves(d["no"])


# ---- exact fast evaluation, local search, and the staged search-plus-MIP solve -----------------
def fast_value(model: Model, name: str, pol):
    """Exact expected cost of policy table pol in one environment (backward recursion)."""
    cost, Pm, mu = model.build(name)
    V = np.zeros(model.nS)
    for t in range(T - 1, -1, -1):
        S = np.where((model.t_of == t) & model.reach)[0]
        V[S] = cost[S, pol[S]] + Pm[S * model.nA + pol[S]] @ V
    return float(mu @ V)


def local_search(model: Model, M, env_weights: dict, depth: int, restarts=30, seed=3, start=None):
    """Best-improvement local search over tree predicates and leaf actions, with exact evaluation.
    Gives the MIP a starting solution; it proves nothing by itself."""
    rng = np.random.default_rng(seed)
    nI, nL, nJ, nA = 2 ** depth - 1, 2 ** depth, M.shape[1], model.nA
    val = lambda n, l: sum(w * fast_value(model, e, tree_table(M, n, l, depth)) for e, w in env_weights.items())
    best = (np.inf, None, None); evals = 0
    starts = list(start or [])
    for r in range(restarts):
        nodes, leaves = (np.array(starts[r][0]), np.array(starts[r][1])) if r < len(starts) else (rng.integers(0, nJ, nI), rng.integers(0, nA, nL))
        v = val(nodes, leaves); evals += 1
        while True:
            cand = (v, None)
            for m_ in range(nI):
                for j in rng.permutation(nJ)[:12] if nI > 3 else range(nJ):
                    if j == nodes[m_]:
                        continue
                    n2 = nodes.copy(); n2[m_] = j; v2 = val(n2, leaves); evals += 1
                    if v2 < cand[0] - 1e-9:
                        cand = (v2, (n2, leaves))
            for l_ in range(nL):
                for a in range(nA):
                    if a == leaves[l_]:
                        continue
                    l2 = leaves.copy(); l2[l_] = a; v2 = val(nodes, l2); evals += 1
                    if v2 < cand[0] - 1e-9:
                        cand = (v2, (nodes, l2))
            if cand[1] is None:
                break
            v, (nodes, leaves) = cand[0], cand[1]
        if v < best[0] - 1e-9:
            best = (v, nodes.copy(), leaves.copy())
    return dict(nodes=[int(x) for x in best[1]], leaves=[int(x) for x in best[2]], depth=depth, obj=float(best[0]), evaluations=int(evals))


def pad_tree(tree, depth):
    """Write a shallower tree as an equivalent complete tree of the given depth."""
    nodes, leaves, d = list(tree["nodes"]), list(tree["leaves"]), tree["depth"]
    while d < depth:
        nI = 2 ** d - 1
        nodes = nodes + [0] * (2 ** d)
        leaves = [a for a in leaves for _ in (0, 1)]
        d += 1
    return nodes, leaves


def solve_tree(model: Model, M, env_weights: dict, max_depth=3, time_limit=300.0, tol=1e-4, seed=3):
    """Smallest-depth tree first. At each depth: local search for a starting value, then the MIP with that value
    as a cutoff. Stops as soon as a tree meets the dynamic-programming bound (no deeper tree can do better)."""
    bound_dp = sum(w * model.dp(e)[0] for e, w in env_weights.items())
    best, hist, t0 = None, [], time.time()
    for depth in range(1, max_depth + 1):
        start = [pad_tree(best, depth)] if best else None
        ls = local_search(model, M, env_weights, depth, restarts=12 if depth < 3 else 20, seed=seed, start=start)
        mip = optimal_tree(model, env_weights, depth, M, time_limit=time_limit, cutoff=ls["obj"] * (1 + 1e-9) + 1e-9)
        cand = ls
        if "nodes" in mip:
            v = sum(w * fast_value(model, e, tree_table(M, mip["nodes"], mip["leaves"], depth)) for e, w in env_weights.items())
            if v < ls["obj"] - 1e-9:
                cand = dict(nodes=mip["nodes"], leaves=mip["leaves"], depth=depth, obj=v)
        lb = mip.get("bound", np.nan)
        proven = ("nodes" in mip and mip.get("mip_gap", 1) <= tol) or (np.isfinite(lb) and cand["obj"] - lb <= tol * max(1.0, cand["obj"]))
        hist.append(dict(depth=depth, local_search=ls["obj"], mip_obj=mip.get("obj"), mip_bound=lb, mip_time=mip["time"], mip_status=mip["status"],
                         proven_optimal_for_depth=bool(proven), n_vars=mip.get("n_vars"), n_cons=mip.get("n_cons"), n_bin=mip.get("n_bin")))
        if best is None or cand["obj"] < best["obj"] - 1e-9:
            best = dict(cand)
        if best["obj"] - bound_dp <= tol * max(1.0, bound_dp):
            break
    last = hist[-1]
    best.update(bound_dp=float(bound_dp), gap_to_dp=float((best["obj"] - bound_dp) / max(1e-9, best["obj"])),
                proven=bool(best["obj"] - bound_dp <= tol * max(1.0, bound_dp) or (last["proven_optimal_for_depth"] and last["depth"] == max_depth)),
                mip_bound=float(last["mip_bound"]) if last["mip_bound"] is not None and np.isfinite(last["mip_bound"]) else None,
                time=time.time() - t0, history=hist, n_vars=last["n_vars"], n_cons=last["n_cons"], n_bin=last["n_bin"], max_depth=max_depth)
    best["mip_gap"] = 0.0 if best["proven"] else float((best["obj"] - (best["mip_bound"] or bound_dp)) / max(1e-9, best["obj"]))
    return best


def dp_sequence(model: Model, seq):
    """Optimal policy when the whole sequence of environments is known in advance (lower bound for any strategy).
    Returns one policy table per step. Years are linked only through the last observed rank."""
    K = len(seq); V = np.zeros(model.nS); tables = [None] * K
    for k in range(K - 1, -1, -1):
        t = k % T
        cost, Pm, _ = model.build(seq[k])
        S = np.where(model.t_of == t)[0]
        if t == T - 1 and k < K - 1:                       # value of the next year's start, by the rank observed in this split
            e = model.envs[seq[k]]
            start = np.array([V[model.sid(0, model.B, r)] for r in range(3)])
            cont = np.array([sum(e.P[t, model.o_of[s] % 3, j] * start[RANK[t, j]] for j in range(3)) for s in S])
            Q = cost[S] + cont[:, None]
        else:
            Q = cost[S] + (Pm @ V).reshape(model.nS, model.nA)[S]
        pol = np.zeros(model.nS, int); pol[S] = Q.argmin(1)
        Vn = np.zeros(model.nS); Vn[S] = Q.min(1)
        tables[k] = pol
        if t == 0:
            V = Vn                                           # values of the year's first split feed the previous year
        else:
            V = V.copy(); V[S] = Vn[S]
    return tables
