"""Pilot simulator: does a DOS signal add value to a courier's accept/relocate rule?

Simplified replica of Zhou et al. (2025) environment:
10x10-mile Manhattan grid, 10 restaurants, 8-h shift, platform assigns pending
orders every 2 min to the nearest idle courier. One focal courier ("ours") plus
N competitor couriers that follow the paper's static Centroid policy with 80% acceptance.
"""
import math, random, sys, json
from statistics import mean, stdev

V = 0.3            # miles per minute (~18 mph)
SHIFT = 480        # minutes
ASSIGN_EVERY = 2   # minutes
EXPIRE = 40        # pending order lost after 40 min
COST = 0.5         # $ per mile driving cost
N_COMP = 8         # competitor couriers
WIN = 30           # DOS demand nowcast window (min)
RAD = 2.0          # DOS radius (miles, Manhattan)

LEVELS = {"few": 0.15, "sufficient": 0.30, "excess": 0.70}  # orders / minute

def layout():
    r = random.Random(7)
    rest = [(r.uniform(1, 9), r.uniform(1, 9)) for _ in range(10)]
    tipmu = [r.uniform(2, 8) for _ in range(10)]
    return rest, tipmu

REST, TIPMU = layout()
CENTROID = (mean(p[0] for p in REST), mean(p[1] for p in REST))

def md(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def weights_for(pattern, rng):
    """Restaurant weights per 2-hour block (4 blocks)."""
    if pattern == "stationary":
        return [[1.0] * 10 for _ in range(4)]
    # stochastic hotspots: each episode, 3 random restaurants are 5x hot per block
    out = []
    for _ in range(4):
        hot = set(rng.sample(range(10), 3))
        out.append([5.0 if i in hot else 1.0 for i in range(10)])
    return out

def gen_orders(level, pattern, seed):
    rng = random.Random(seed)
    W = weights_for(pattern, rng)
    lam = LEVELS[level]
    orders, t = [], 0.0
    while True:
        t += rng.expovariate(lam)
        if t >= SHIFT - 30:
            break
        w = W[min(3, int(t // 120))]
        ri = rng.choices(range(10), weights=w)[0]
        rx, ry = REST[ri]
        cx = min(10, max(0, rx + rng.uniform(-3, 3)))
        cy = min(10, max(0, ry + rng.uniform(-3, 3)))
        pay = 3.0 + 1.5 * md((rx, ry), (cx, cy))
        tip = max(0.0, rng.gauss(TIPMU[ri], 1.5))
        ready = t + max(3.0, rng.gauss(12, 3))
        orders.append(dict(id=len(orders), t=t, r=ri, rest=(rx, ry), cust=(cx, cy),
                           pay=pay, tip=tip, ready=ready))
    return orders

class Courier:
    def __init__(self, pos, ours=False):
        self.pos = pos; self.free_at = 0.0; self.target = None
        self.ours = ours; self.excl = {}  # order id -> until time
        self.earn = 0.0; self.miles = 0.0; self.idle_min = 0.0; self.n = 0; self.rej = 0; self.offers = 0

    def idle(self, t):
        return t >= self.free_at

    def step_move(self, dt=1.0):
        if self.target is None:
            return
        d = md(self.pos, self.target)
        step = V * dt
        if d <= step:
            self.miles += d; self.pos = self.target; self.target = None; return
        # move along x first then y (Manhattan)
        x, y = self.pos; tx, ty = self.target
        if abs(tx - x) > 1e-9:
            mv = min(step, abs(tx - x)); x += math.copysign(mv, tx - x); step -= mv; self.miles += mv
        if step > 0 and abs(ty - y) > 1e-9:
            mv = min(step, abs(ty - y)); y += math.copysign(mv, ty - y); self.miles += mv
        self.pos = (x, y)

def dos(point, t, orders_seen, couriers, ours):
    """DOS(z,t) = D_hat / (C_hat + 1). D_hat: orders placed in the last WIN min at
    restaurants within RAD of z (what a platform heatmap shows). C_hat: idle competitor couriers within RAD."""
    D = sum(1 for o in orders_seen if t - WIN <= o["t"] <= t and md(o["rest"], point) <= RAD)
    C = sum(1 for c in couriers if c is not ours and c.idle(t) and md(c.pos, point) <= RAD)
    return D / (C + 1)

def run(level, pattern, seed, policy):
    """policy: dict(kind, theta, lam).  kind in {centroid_all, thr, thr_dos, thr_dos_reloc}"""
    orders = gen_orders(level, pattern, seed)
    rng = random.Random(seed * 7919 + 1)
    comps = [Courier((rng.uniform(0, 10), rng.uniform(0, 10))) for _ in range(N_COMP)]
    ours = Courier((rng.uniform(0, 10), rng.uniform(0, 10)), ours=True)
    allc = comps + [ours]
    pending, oi, seen = [], 0, []
    kind = policy["kind"]
    for minute in range(SHIFT):
        t = float(minute)
        while oi < len(orders) and orders[oi]["t"] <= t:
            pending.append(orders[oi]); seen.append(orders[oi]); oi += 1
        pending = [o for o in pending if t - o["t"] <= EXPIRE]
        # movement & idle relocation
        for c in allc:
            if not c.idle(t):
                continue
            if c.ours:
                c.idle_min += 1
                if kind == "thr_dos_reloc" and minute % 5 == 0:
                    cands = [c.pos] + [(min(10, max(0, c.pos[0] + dx)), min(10, max(0, c.pos[1] + dy)))
                                       for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
                    vals = [dos(p, t, seen, allc, ours) for p in cands]
                    best = max(range(5), key=lambda i: vals[i])
                    if best != 0 and vals[best] > vals[0] + 0.25:
                        c.target = cands[best]
                elif kind != "thr_dos_reloc" and c.target is None and md(c.pos, CENTROID) > 0.1:
                    c.target = CENTROID
            else:
                if c.target is None and md(c.pos, CENTROID) > 0.1:
                    c.target = CENTROID
            c.step_move()
        # assignment
        if minute % ASSIGN_EVERY == 0 and pending:
            pending.sort(key=lambda o: o["t"])
            remaining = []
            for o in pending:
                cands = [c for c in allc if c.idle(t) and c.excl.get(o["id"], -1) < t and md(c.pos, o["rest"]) <= 6]
                if not cands:
                    remaining.append(o); continue
                c = min(cands, key=lambda c: (round(md(c.pos, o["rest"]), 1), c.free_at))
                d1 = md(c.pos, o["rest"]); d2 = md(o["rest"], o["cust"])
                fin = max(t + d1 / V, o["ready"]) + d2 / V
                if c.ours:
                    c.offers += 1
                    profit = o["pay"] + o["tip"] - COST * (d1 + d2)
                    rate = profit / (fin - t)
                    if kind == "centroid_all":
                        acc = True
                    elif kind == "thr":
                        acc = rate >= policy["theta"]
                    else:
                        gain = dos(o["cust"], t, seen, allc, ours) - dos(c.pos, t, seen, allc, ours)
                        acc = rate + policy["lam"] * gain >= policy["theta"]
                else:
                    acc = rng.random() < 0.8
                if acc:
                    c.free_at = fin; c.target = None
                    c.miles += d1 + d2; c.pos = o["cust"]; c.n += 1
                    c.earn += o["pay"] + o["tip"]
                else:
                    c.excl[o["id"]] = t + 10
                    if c.ours: c.rej += 1
                    remaining.append(o)
            pending = remaining
    net = ours.earn - COST * ours.miles
    return dict(net=net, n=ours.n, idle=ours.idle_min, rej=ours.rej / max(1, ours.offers))
