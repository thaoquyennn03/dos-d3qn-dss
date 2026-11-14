import sys, json, itertools, math
from multiprocessing import Pool
from statistics import mean, stdev
import sim

TRAIN = list(range(1000, 1080))
TEST = list(range(0, 400))
THETAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
LAMS = [0.01, 0.03, 0.1, 0.3]

def avg(args):
    lv, pat, pol, seeds = args
    return mean(sim.run(lv, pat, s, pol)["net"] for s in seeds)

def tune(pool, lv, pat, kind):
    if kind == "thr":
        grid = [dict(kind=kind, theta=th) for th in THETAS]
    else:
        grid = [dict(kind=kind, theta=th, lam=la) for th in THETAS for la in LAMS]
    scores = pool.map(avg, [(lv, pat, g, TRAIN) for g in grid])
    return grid[max(range(len(grid)), key=lambda i: scores[i])]

def evalp(args):
    lv, pat, pol, s = args
    return sim.run(lv, pat, s, pol)

def ci(xs):
    m = mean(xs); h = 1.96 * stdev(xs) / math.sqrt(len(xs)); return m, h

if __name__ == "__main__":
    out = []
    with Pool(2) as pool:
        for lv in sim.LEVELS:
            for pat in ["stationary", "stochastic"]:
                pols = {"centroid": dict(kind="centroid_all")}
                pols["thr"] = tune(pool, lv, pat, "thr")
                pols["thr_dos"] = tune(pool, lv, pat, "thr_dos")
                pols["thr_dos_reloc"] = tune(pool, lv, pat, "thr_dos_reloc")
                res = {k: pool.map(evalp, [(lv, pat, p, s) for s in TEST]) for k, p in pols.items()}
                row = dict(level=lv, pattern=pat, params={k: v for k, v in pols.items()})
                for k, r in res.items():
                    m, h = ci([x["net"] for x in r])
                    row[k] = dict(net=round(m, 2), ci=round(h, 2), idle=round(mean(x["idle"] for x in r), 1),
                                  rej=round(100 * mean(x["rej"] for x in r), 1))
                for k in ["thr_dos", "thr_dos_reloc"]:
                    d = [a["net"] - b["net"] for a, b in zip(res[k], res["thr"])]
                    m, h = ci(d)
                    row["diff_" + k] = dict(mean=round(m, 2), ci=round(h, 2), pct=round(100 * m / row["thr"]["net"], 1))
                print(json.dumps(row), flush=True)
                out.append(row)
    json.dump(out, open("results.json", "w"), indent=1)
