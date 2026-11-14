"""Evaluate trained D3QN arms and tuned baselines on held-out test days (same days for every policy)."""
import argparse, itertools, re
import numpy as np
import pandas as pd
from config import *
from env import World, CourierEnv
from agent import D3QN
from baselines import Centroid, Threshold, run_policy

THETAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
LAMS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]   # lam = 0 included, so B2 can fall back to B1
PAT = re.compile(r"^(?P<level>[a-z]+)_(?P<pattern>[a-z_]+?)_(?P<arm>A\d_[a-z_0-9]+?)_c(?P<c>\d+)_p(?P<p>[\d.]+)_s(?P<seed>\d+)$")
FEAT = {"A0_d3qn": "base", "A1_d3qn_dhat": "dhat", "A2_d3qn_dos": "dos", "A3_d3qn_oracle": "oracle", "A4_d3qn_noise": "noise"}


class Greedy:
    def __init__(self, agent):
        self.agent = agent

    def act(self, obs, mask, info, env):
        return self.agent.act(obs, mask, 0.0)


def tune(w, level, pattern, n_comp, n_days=40):
    env = CourierEnv(w, "base", n_comp=n_comp, seed=123)
    days = w.day_ids(level, pattern, "train")[:n_days]
    score = lambda p: np.mean([run_policy(env, p, d)["net"] for d in days])
    best1 = max(THETAS, key=lambda th: score(Threshold(th)))
    best2 = max(itertools.product(THETAS, LAMS), key=lambda tl: score(Threshold(*tl)))
    return best1, best2


def evaluate(level, pattern, n_comp=N_COMPETITORS, penalty=0.0, world=None):
    w = world or World()
    test = w.day_ids(level, pattern, "test")
    rows = []

    def rec(policy_name, seed, env, pol, extra=""):
        for d in test:
            st = run_policy(env, pol, d)
            rows.append(dict(level=level, pattern=pattern, n_comp=n_comp, penalty=penalty, policy=policy_name,
                             seed=seed, day_id=d, net=st["net"], orders=st["n"], offers=st["offers"],
                             reject_rate=st["rejects"] / max(1, st["offers"]), idle_min=st["idle_min"],
                             km=st["km"], params=extra))

    env = CourierEnv(w, "base", n_comp=n_comp, seed=999)
    th, (th2, lam) = tune(w, level, pattern, n_comp)
    rec("B0_centroid", -1, env, Centroid())
    rec("B1_threshold", -1, env, Threshold(th), f"theta={th}")
    rec("B2_threshold_dos", -1, env, Threshold(th2, lam), f"theta={th2};lam={lam}")
    for f in sorted((RESULTS / "models").glob("*.npz")):
        m = PAT.match(f.stem)
        if not m or m["level"] != level or m["pattern"] != pattern or int(m["c"]) != n_comp or float(m["p"]) != penalty:
            continue
        feat = FEAT[m["arm"]]
        e = CourierEnv(w, feat, n_comp=n_comp, seed=999)
        ag = D3QN(e.obs_dim, e.n_actions)
        ag.load(f)
        rec(m["arm"], int(m["seed"]), e, Greedy(ag))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="excess", choices=list(LEVELS))
    ap.add_argument("--pattern", default="hotspot_random", choices=PATTERNS)
    ap.add_argument("--n-comp", type=int, default=N_COMPETITORS)
    ap.add_argument("--reject-penalty", type=float, default=0.0)
    a = ap.parse_args()
    df = evaluate(a.level, a.pattern, a.n_comp, a.reject_penalty)
    out = RESULTS / f"eval_{a.level}_{a.pattern}_c{a.n_comp}_p{a.reject_penalty:g}.csv"
    df.to_csv(out, index=False)
    print(df.groupby("policy").net.mean().round(2).to_string())
