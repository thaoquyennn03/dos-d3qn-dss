"""Run a whole plan: train every (scenario, arm, seed) not trained yet, then evaluate and analyse.

Plans:
  smoke : 1 scenario, arms A0 + A2, 1 seed, 60 episodes   (~1 min, checks the install)
  core  : 9 scenarios x {A0, A2} x 3 seeds + {A1, A3, A4} on the two 'excess' scenarios
          (what was run to produce results/ in this package)
  full  : 9 scenarios x all 5 arms x 5 seeds (the complete design; several hours on 4-8 cores)

Example:
  python src/run_experiments.py --plan core --episodes 600 --jobs 4
"""
import argparse, itertools, os, time
from multiprocessing import Pool
import pandas as pd
from config import *
from train import train, tag
from evaluate import evaluate
import analyze

ALL = ["base", "dhat", "dos", "oracle", "noise"]


def plan_jobs(plan, n_comp, penalty):
    jobs = []
    if plan == "smoke":
        return [("excess", "hotspot_random", f, 0) for f in ["base", "dos"]]
    for lv, pt in itertools.product(LEVELS, PATTERNS):
        if plan == "full":
            feats, seeds = ALL, range(5)
        else:
            feats = ALL if (lv == "excess" and pt in ("stationary", "hotspot_random")) else ["base", "dos"]
            seeds = range(3)
        jobs += [(lv, pt, f, s) for f in feats for s in seeds]
    return jobs


def _train(args):
    lv, pt, f, s, eps, n_comp, penalty = args
    name = tag(lv, pt, f, s, n_comp, penalty)
    if (RESULTS / "models" / f"{name}.npz").exists():
        return name + " (skip)"
    t = time.time()
    train(lv, pt, f, s, eps, n_comp, penalty, verbose=False)
    return f"{name} ({time.time()-t:.0f}s)"


def _eval(args):
    lv, pt, n_comp, penalty = args
    df = evaluate(lv, pt, n_comp, penalty)
    df.to_csv(RESULTS / f"eval_{lv}_{pt}_c{n_comp}_p{penalty:g}.csv", index=False)
    return f"eval {lv} {pt}: {len(df)} rows"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="smoke", choices=["smoke", "core", "full"])
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--n-comp", type=int, default=N_COMPETITORS)
    ap.add_argument("--reject-penalty", type=float, default=0.0)
    a = ap.parse_args()
    eps = 60 if a.plan == "smoke" else a.episodes
    jobs = [j + (eps, a.n_comp, a.reject_penalty) for j in plan_jobs(a.plan, a.n_comp, a.reject_penalty)]
    print(f"{len(jobs)} training runs, {eps} episodes each, {a.jobs} parallel jobs", flush=True)
    t0 = time.time()
    with Pool(a.jobs) as pool:
        for i, msg in enumerate(pool.imap_unordered(_train, jobs), 1):
            print(f"[{i}/{len(jobs)}] {msg}  elapsed {time.time()-t0:.0f}s", flush=True)
        scen = sorted({(j[0], j[1], a.n_comp, a.reject_penalty) for j in jobs})
        for msg in pool.imap_unordered(_eval, scen):
            print(msg, flush=True)
    analyze.main()
