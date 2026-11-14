"""Build dataset v2 for the DOS study from the user's source restaurant profiles.

Kept from the source dataset: 30 restaurants (positions, popularity, prep time), fee/cost/tip
formulas, hourly demand weights, km/CNY units.
Added for the DOS study: many days, 3 demand levels x 3 demand patterns, train/test split,
per-restaurant tip means (paper Fig. 6 shows tips differ by restaurant), and the true demand
profile of every day (needed only for the DOS-oracle arm).
"""
import argparse
import numpy as np
import pandas as pd
from config import *


def restaurants(seed=20261108):
    src = pd.read_csv(DATA / "source_restaurants.csv")
    rng = np.random.default_rng(seed)
    src["tip_mean_cny"] = np.round(rng.uniform(2.5, 6.5, len(src)), 2)
    return src


def riders(n=60, seed=20261108):
    rng = np.random.default_rng(seed + 1)
    return pd.DataFrame({
        "rider_id": [f"R{i:03d}" for i in range(n)],
        "start_x_km": np.round(rng.uniform(0, GRID_KM, n), 3),
        "start_y_km": np.round(rng.uniform(0, GRID_KM, n), 3),
        "role": ["focal"] + ["competitor"] * (n - 1),
    })


def peak_sets(n_rest, seed=7):
    """Fixed hot restaurants per 2-h block, identical every day (learnable from x, y, t)."""
    rng = np.random.default_rng(seed)
    return [rng.choice(n_rest, HOT_PER_BLOCK, replace=False) for _ in range(4)]


def day_profile(pattern, n_rest, rng, fixed):
    mult = np.ones((4, n_rest))
    for b in range(4):
        if pattern == "peak_fixed":
            mult[b, fixed[b]] = HOT_MULT
        elif pattern == "hotspot_random":
            mult[b, rng.choice(n_rest, HOT_PER_BLOCK, replace=False)] = HOT_MULT
    return mult


def gen_day(rest, level, pattern, day_seed, fixed):
    rng = np.random.default_rng(day_seed)
    n_rest = len(rest)
    mult = day_profile(pattern, n_rest, rng, fixed)
    total = LEVELS[level]
    hw = np.array(HOURLY_WEIGHTS) / sum(HOURLY_WEIGHTS)
    n_per_hour = rng.poisson(total * hw)
    pop = rest.popularity.values
    rows = []
    for h, n in enumerate(n_per_hour):
        w = pop * mult[h // 2]
        w = w / w.sum()
        ts = np.sort(rng.uniform(h * 60, (h + 1) * 60, n))
        rid = rng.choice(n_rest, n, p=w)
        for t, r in zip(ts, rid):
            R = rest.iloc[r]
            cx = float(np.clip(R.x_km + rng.uniform(-3, 3), 0, GRID_KM))
            cy = float(np.clip(R.y_km + rng.uniform(-3, 3), 0, GRID_KM))
            d = abs(cx - R.x_km) + abs(cy - R.y_km)
            prep = float(np.clip(rng.normal(R.prep_mean_min, R.prep_sd_min), 5, 40))
            rows.append((t, R.restaurant_id, r, R.x_km, R.y_km, cx, cy, d, prep, t + prep,
                         FEE_BASE + FEE_PER_KM * d, max(0.0, rng.normal(R.tip_mean_cny, TIP_SD))))
    df = pd.DataFrame(rows, columns=["order_created_min", "restaurant_id", "restaurant_idx",
                                     "restaurant_x_km", "restaurant_y_km", "customer_x_km",
                                     "customer_y_km", "restaurant_to_customer_km", "prep_duration_min",
                                     "ready_min", "delivery_fee_cny", "tip_cny"])
    return df.round(3), mult


def main(train_days, test_days, seed):
    rest = restaurants(seed)
    rest.to_csv(DATA / "restaurants.csv", index=False)
    riders(seed=seed).to_csv(DATA / "riders.csv", index=False)
    fixed = peak_sets(len(rest))
    orders, prof, index = [], [], []
    k = 0
    for level in LEVELS:
        for pattern in PATTERNS:
            for split, nd in (("train", train_days), ("test", test_days)):
                for d in range(nd):
                    day_id = f"{level[:3]}-{pattern[:4]}-{split}-{d:03d}"
                    day_seed = seed * 1000 + k
                    k += 1
                    df, mult = gen_day(rest, level, pattern, day_seed, fixed)
                    df.insert(0, "day_id", day_id)
                    df.insert(1, "order_id", [f"{day_id}-{i:05d}" for i in range(len(df))])
                    orders.append(df)
                    index.append((day_id, level, pattern, split, day_seed, len(df)))
                    for b in range(4):
                        for r in range(len(rest)):
                            if mult[b, r] != 1.0:
                                prof.append((day_id, b, rest.restaurant_id.iloc[r], mult[b, r]))
    pd.concat(orders).to_csv(DATA / "orders.csv", index=False)
    pd.DataFrame(index, columns=["day_id", "level", "pattern", "split", "seed", "n_orders"]).to_csv(
        DATA / "days.csv", index=False)
    pd.DataFrame(prof, columns=["day_id", "block_2h", "restaurant_id", "demand_multiplier"]).to_csv(
        DATA / "demand_profile.csv", index=False)
    print(f"days={len(index)} orders={sum(len(o) for o in orders)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-days", type=int, default=40)
    ap.add_argument("--test-days", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261108)
    a = ap.parse_args()
    main(a.train_days, a.test_days, a.seed)
