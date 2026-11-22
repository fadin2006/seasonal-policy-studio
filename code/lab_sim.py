"""Simulation, runtime monitor and black-box tree search for the seasonal lab."""
from __future__ import annotations
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from lab_core import (T, HOURS, DEM, WCF, P0, RANK, ENVS, LIBRARY, ACTIONS, Model, tree_table)

SIG_D, SIG_W = 0.015, 0.03        # ASSUMED measurement noise: demand (relative), wind capacity factor (absolute)
WINDOW = 4                        # splits in the monitor window (half a year)
FEATS = ["mean demand gap", "largest demand gap", "mean wind gap", "largest wind gap", "share outside design box"]
D_RANGE = DEM.max(1) - DEM.min(1)


def exo(model: Model, env_seq, n, rng, o0=None):
    """Exogenous process for n episodes; env_seq[k] = environment name at step k.
    Returns the state index j[n, K], demand and wind observations, and the rank o seen before each step."""
    K = len(env_seq)
    j = np.zeros((n, K), int); dob = np.zeros((n, K)); wob = np.zeros((n, K)); obef = np.zeros((n, K), int)
    o = rng.choice(3, size=n, p=model.envs[env_seq[0]].mu0) if o0 is None else o0.copy()
    for k, en in enumerate(env_seq):
        e, t = model.envs[en], k % T
        cum = e.P[t][o].cumsum(1)
        u = rng.random(n)
        jj = (u[:, None] > cum).sum(1).clip(0, 2)
        obef[:, k] = o; j[:, k] = jj
        dob[:, k] = DEM[t, jj] * e.dscale[t] * (1 + SIG_D * rng.standard_normal(n))
        wob[:, k] = np.clip(WCF[t, jj] * e.wscale[t] + SIG_W * rng.standard_normal(n), 0, 1)
        o = RANK[t, jj]
    return j, dob, wob, obef


def residuals(dob, wob):
    """Gap of each observation to the nearest design value of its split, variable by variable (the 'box' abstraction):
    gD = relative demand gap (signed), gW = absolute wind capacity-factor gap (unsigned)."""
    n, K = dob.shape
    gD = np.zeros((n, K)); gW = np.zeros((n, K))
    for k in range(K):
        t = k % T
        nd = np.abs(dob[:, k, None] - DEM[t][None]).argmin(1)
        gD[:, k] = dob[:, k] / DEM[t, nd] - 1
        gW[:, k] = np.abs(wob[:, k, None] - WCF[t][None]).min(1)
    return gD, gW


def window_features(gD, gW, k):
    """Features available BEFORE the decision at step k: the WINDOW observations k-WINDOW .. k-1."""
    a, b = gD[:, k - WINDOW:k], gW[:, k - WINDOW:k]
    out = (np.abs(a) > 3 * SIG_D) | (b > 3 * SIG_W)
    return np.column_stack([a.mean(1), a.max(1), b.mean(1), b.max(1), out.mean(1)])


class Monitor:
    """Depth-limited tree that names the environment, plus a per-class box that flags windows it has not seen."""

    def __init__(self, model, depth=3, years=6, n=3000, seed=1):
        rng = np.random.default_rng(seed)
        X, y = [], []
        for ci, en in enumerate(LIBRARY):
            _, dob, wob, _ = exo(model, [en] * (T * years), n, rng)
            rD, rW = residuals(dob, wob)
            for k in range(WINDOW, T * years):
                X.append(window_features(rD, rW, k)); y.append(np.full(n, ci))
        X, y = np.vstack(X), np.concatenate(y)
        self.tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=500, random_state=0).fit(X, y)
        self.boxes = np.array([[np.quantile(X[y == c], 0.002, axis=0), np.quantile(X[y == c], 0.998, axis=0)] for c in range(len(LIBRARY))])
        self.train_acc = float((self.tree.predict(X) == y).mean())

    def label(self, F):
        """0..2 = library environment, 3 = unknown (outside the box of the predicted class)."""
        c = self.tree.predict(F)
        lo, hi = self.boxes[c, 0], self.boxes[c, 1]
        unknown = ((F < lo - 1e-9) | (F > hi + 1e-9)).any(1)
        return np.where(unknown, 3, c)

    def label_conf(self, F):
        """Label as in label(), plus the purity of the leaf (share of training windows of the winning class)."""
        leaf = self.tree.apply(F)
        val = self.tree.tree_.value[leaf][:, 0, :]
        return self.label(F), val.max(1) / val.sum(1)

    def rules(self):
        tr = self.tree.tree_

        def rec(i):
            if tr.children_left[i] == -1:
                return {"env": LIBRARY[int(tr.value[i][0].argmax())], "share": float(tr.value[i][0].max() / tr.value[i][0].sum())}
            return {"feature": int(tr.feature[i]), "test": f"{FEATS[tr.feature[i]]} ≤ {tr.threshold[i]:.3f}", "threshold": float(tr.threshold[i]),
                    "yes": rec(tr.children_left[i]), "no": rec(tr.children_right[i])}
        return rec(0)


def rollout(model: Model, tables, lab, j, env_of_step, obef):
    """Cost of following tables[lab[n, k]] over K steps for exogenous states j. Returns cost per episode (GWh)."""
    n, K = j.shape
    b = np.full(n, model.B); cost = np.zeros(n)
    tabs = np.stack(tables)
    for k in range(K):
        t = k % T
        if t == 0:
            b[:] = model.B
        s = (t * (model.B + 1) + b) * model.nO + obef[:, k]
        a = np.minimum(tabs[lab[:, k], s], b)
        NL = model.envs[env_of_step[k]].NL[t, j[:, k]]
        cost += HOURS[t] * np.maximum(0.0, NL - model.cap(a)) / 1000.0
        b = b - a
        if t == T - 1:
            cost += model.pen * b
    return cost


class BlackBox:
    """Tree search that only calls a simulator: sample-average evaluation with common random numbers and local search."""

    def __init__(self, model, M, envs, depth, n=400, seed=7):
        self.model, self.M, self.depth, self.envs = model, M, depth, envs
        rng = np.random.default_rng(seed)
        self.samples = [(en,) + exo(model, [en] * T, n, rng) for en in envs]
        self.calls = 0
        self.nI, self.nL, self.nJ = 2 ** depth - 1, 2 ** depth, M.shape[1]

    def value(self, nodes, leaves, samples=None):
        tab = tree_table(self.M, nodes, leaves, self.depth)
        tot = []
        for (en, j, dob, wob, obef) in (samples or self.samples):
            lab = np.zeros(j.shape, int)
            tot.append(rollout(self.model, [tab], lab, j, [en] * T, obef))
            self.calls += j.shape[0]
        return np.concatenate(tot)

    def search(self, budget=4000, restarts=8, seed=11):
        rng = np.random.default_rng(seed)
        best, best_v, evals, pool = None, np.inf, 0, []
        per = budget // restarts
        for _ in range(restarts):
            nodes = rng.integers(0, self.nJ, self.nI); leaves = rng.integers(0, len(ACTIONS), self.nL)
            v = self.value(nodes, leaves).mean(); evals += 1; used = 1
            while used < per:
                n2, l2 = nodes.copy(), leaves.copy()
                if rng.random() < 0.5:
                    n2[rng.integers(self.nI)] = rng.integers(self.nJ)
                else:
                    l2[rng.integers(self.nL)] = rng.integers(len(ACTIONS))
                v2 = self.value(n2, l2).mean(); used += 1; evals += 1
                if v2 <= v:
                    nodes, leaves, v = n2, l2, v2
            pool.append((v, nodes, leaves))
            if v < best_v:
                best, best_v = (nodes, leaves), v
        # final selection on fresh samples
        rng2 = np.random.default_rng(seed + 100)
        fresh = [(en,) + exo(self.model, [en] * T, 5000, rng2) for en in self.envs]
        pool.sort(key=lambda r: r[0])
        scored = []
        for v, nodes, leaves in pool[:5]:
            c = self.value(nodes, leaves, fresh)
            scored.append((c.mean(), 1.96 * c.std(ddof=1) / np.sqrt(len(c)), nodes, leaves))
        scored.sort(key=lambda r: r[0])
        mean, ci, nodes, leaves = scored[0]
        return dict(nodes=[int(x) for x in nodes], leaves=[int(x) for x in leaves], depth=self.depth, saa_mean=float(mean), saa_ci=float(ci),
                    candidates=int(evals), simulator_episodes=int(self.calls))
