"""Summarise results/eval_*.csv: mean ± CI95 per policy, paired tests vs the original D3QN (A0),
Holm correction, convergence speed from training curves, and two charts."""
import glob, re
import numpy as np
import pandas as pd
from scipy import stats
from config import *


def ci95(x):
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); out = np.empty_like(p); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - k) * p[i])); out[i] = run
    return out


def main():
    files = sorted(glob.glob(str(RESULTS / "eval_*.csv")))
    if not files:
        print("No eval_*.csv in results/. Run src/evaluate.py first."); return
    df = pd.concat(pd.read_csv(f) for f in files)
    keys = ["level", "pattern", "n_comp", "penalty"]
    # average over training seeds per test day -> one value per (policy, day)
    per_day = df.groupby(keys + ["policy", "day_id"]).agg(net=("net", "mean"), reject_rate=("reject_rate", "mean"),
                                                          idle_min=("idle_min", "mean"), seeds=("seed", "nunique")).reset_index()
    summ = per_day.groupby(keys + ["policy"]).agg(net=("net", "mean"), net_ci=("net", ci95), reject_pct=("reject_rate", "mean"),
                                                 idle_min=("idle_min", "mean"), seeds=("seeds", "max"), days=("day_id", "nunique")).reset_index()
    summ["reject_pct"] *= 100
    summ.round(2).to_csv(RESULTS / "summary.csv", index=False)

    tests = []
    for k, g in per_day.groupby(keys):
        piv = g.pivot(index="day_id", columns="policy", values="net")
        pairs = [(a, "A0_d3qn") for a in ["A1_d3qn_dhat", "A2_d3qn_dos", "A3_d3qn_oracle", "A4_d3qn_noise"]]
        pairs += [("A1_d3qn_dhat", "A4_d3qn_noise"), ("A2_d3qn_dos", "A4_d3qn_noise"), ("A3_d3qn_oracle", "A4_d3qn_noise"),
                  ("A2_d3qn_dos", "A1_d3qn_dhat"),("A0_d3qn", "B1_threshold"), ("A2_d3qn_dos", "B1_threshold"),
                  ("B2_threshold_dos", "B1_threshold")]
        for a, b in pairs:
            if a in piv and b in piv:
                d = (piv[a] - piv[b]).dropna()
                t, p = stats.ttest_rel(piv[a][d.index], piv[b][d.index])
                base = abs(piv[b][d.index].mean())
                tests.append(dict(zip(keys, k), compare=f"{a} - {b}", diff=d.mean(), diff_ci=ci95(d),
                                  pct=100 * d.mean() / base, pct_ci=100 * ci95(d) / base, p=p, n_days=len(d)))
    tt = pd.DataFrame(tests)
    if len(tt):
        tt["p_holm"] = holm(tt.p.values)
        tt["significant"] = tt.p_holm < 0.05
        tt.round(4).to_csv(RESULTS / "paired_tests.csv", index=False)

    # convergence: first episode where the 50-episode rolling mean reaches 90% of its final value
    conv = []
    for f in glob.glob(str(RESULTS / "curves" / "*.csv")):
        c = pd.read_csv(f); r = c.net.rolling(50, min_periods=50).mean()
        final = r.iloc[-1]
        hit = r[r >= 0.9 * final].index.min() if final > 0 else np.nan
        conv.append(dict(run=re.sub(r"\.csv$", "", f.split("/")[-1].split("\\")[-1]), final_rolling_net=final, episodes_to_90pct=hit))
    pd.DataFrame(conv).to_csv(RESULTS / "convergence.csv", index=False)

    pd.set_option("display.width", 200)
    print(summ.round(2).to_string(index=False))
    if len(tt):
        print(tt[keys + ["compare", "diff", "diff_ci", "pct", "pct_ci", "p_holm", "significant"]].round(3).to_string(index=False))
    try:
        plot(summ, tt)
    except Exception as e:  # matplotlib optional
        print("plot skipped:", e)


def plot(summ, tt):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sel = tt[tt["compare"] == "A2_d3qn_dos - A0_d3qn"].copy()
    if sel.empty:
        return
    sel["label"] = sel.level + "\n" + sel.pattern.str.replace("_", " ")
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#1f6feb" if p == "hotspot_random" else "#9aa0a6" for p in sel.pattern]
    ax.bar(range(len(sel)), sel.pct, color=colors)
    ax.errorbar(range(len(sel)), sel.pct, yerr=sel.pct_ci, fmt="none", ecolor="black", capsize=3)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(sel))); ax.set_xticklabels(sel.label, fontsize=8)
    ax.set_ylabel("D3QN+DOS vs D3QN goc (%)")
    ax.set_title("Chenh lech thu nhap rong/ca cua D3QN+DOS so voi D3QN goc (± CI95)")
    fig.tight_layout(); fig.savefig(RESULTS / "dos_vs_d3qn.png", dpi=150)


if __name__ == "__main__":
    main()
