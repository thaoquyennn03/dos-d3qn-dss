"""Train one D3QN arm on one scenario.

Example:
    python src/train.py --level excess --pattern hotspot_random --feature dos --seed 0
"""
import argparse, time
import numpy as np
import pandas as pd
from config import *
from env import World, CourierEnv
from agent import D3QN

ARM = {"base": "A0_d3qn", "dhat": "A1_d3qn_dhat", "dos": "A2_d3qn_dos", "oracle": "A3_d3qn_oracle", "noise": "A4_d3qn_noise"}


def tag(level, pattern, feature, seed, n_comp, penalty):
    return f"{level}_{pattern}_{ARM[feature]}_c{n_comp}_p{penalty:g}_s{seed}"


def train(level, pattern, feature, seed, episodes, n_comp=N_COMPETITORS, penalty=0.0, world=None, verbose=True):
    w = world or World()
    env = CourierEnv(w, feature, n_comp=n_comp, reject_penalty=penalty, seed=seed)
    agent = D3QN(env.obs_dim, env.n_actions, seed=seed)
    days = w.day_ids(level, pattern, "train")
    rng = np.random.default_rng(seed)
    curve, t0 = [], time.time()
    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / (0.6 * episodes))
        s, m = env.reset(days[rng.integers(len(days))])
        done = False
        while not done:
            a = agent.act(s, m, eps)
            s2, r, done, m2, info = env.step(a)
            agent.remember(s, a, r, s2, m2, float(done))
            agent.learn()
            s, m = s2, m2
        st = env.stats
        curve.append(dict(episode=ep, eps=round(eps, 3), net=round(st["net"], 2), n=st["n"],
                          reject_rate=round(st["rejects"] / max(1, st["offers"]), 3)))
        if verbose and (ep + 1) % max(1, episodes // 10) == 0:
            last = pd.DataFrame(curve[-max(1, episodes // 10):])
            print(f"  ep {ep+1}/{episodes} eps={eps:.2f} net={last.net.mean():.1f} rej={last.reject_rate.mean():.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    name = tag(level, pattern, feature, seed, n_comp, penalty)
    (RESULTS / "models").mkdir(parents=True, exist_ok=True)
    (RESULTS / "curves").mkdir(parents=True, exist_ok=True)
    agent.save(RESULTS / "models" / f"{name}.npz")
    pd.DataFrame(curve).to_csv(RESULTS / "curves" / f"{name}.csv", index=False)
    return name


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="excess", choices=list(LEVELS))
    ap.add_argument("--pattern", default="hotspot_random", choices=PATTERNS)
    ap.add_argument("--feature", default="dos", choices=list(ARM))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=800)
    ap.add_argument("--n-comp", type=int, default=N_COMPETITORS)
    ap.add_argument("--reject-penalty", type=float, default=0.0)
    a = ap.parse_args()
    print(train(a.level, a.pattern, a.feature, a.seed, a.episodes, a.n_comp, a.reject_penalty))
