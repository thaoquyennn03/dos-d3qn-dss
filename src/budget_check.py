"""Training-budget check: are the extra-input arms only behind because 600 episodes is too few?
Trains A0 / A2 / A4 on one scenario with more episodes into a separate results folder.

    DOS_RESULTS=results_budget1500 python src/budget_check.py --episodes 1500
(Windows PowerShell:  $env:DOS_RESULTS="results_budget1500"; python src/budget_check.py --episodes 1500)
"""
import argparse, os
from multiprocessing import Pool
from config import *
from run_experiments import _train, _eval
import analyze

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="excess")
    ap.add_argument("--pattern", default="hotspot_random")
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    jobs = [(a.level, a.pattern, f, s, a.episodes, N_COMPETITORS, 0.0) for f in ["base", "dos", "noise"] for s in range(a.seeds)]
    with Pool(a.jobs) as pool:
        for m in pool.imap_unordered(_train, jobs):
            print(m, flush=True)
    print(_eval((a.level, a.pattern, N_COMPETITORS, 0.0)))
    analyze.main()
