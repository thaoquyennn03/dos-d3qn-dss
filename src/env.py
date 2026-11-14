"""Courier environment following Zhou et al. (2025), Sec. 3.

One focal courier (the learning agent) plus N competitor couriers that follow the paper's
static policies (Last / Centroid / Closest / Random restaurant, in rotation) and accept 80% of offers. Every ASSIGN_EVERY minutes the platform
offers each pending order (oldest first) to the nearest idle courier, ties broken by the
longest idle time. A rejected order returns to the pending pool and is not offered to the
same courier for REJECT_LOCK minutes.

Focal decision points (semi-MDP):
  * an order offer        -> actions accept / reject
  * idle, every TAU min   -> actions stay / up / down / left / right (move for TAU min)
Reward (paper): accept = fee + tip - cost*km; move = -cost*km actually driven; reject/stay = 0.

Feature sets (the arms of the study):
  base   : paper state (courier xy, customer xy, fee, tip, time, restaurant one-hot)
  dhat   : base + recent demand D_hat at 6 points
  dos    : base + DOS = D_hat / (C_hat + 1) at 6 points
  oracle : base + true expected demand / (C_hat + 1) at 6 points
  noise  : base + 6 uniform random numbers (control for wider input)
6 points = current position, destination (customer) and the 4 neighbour cells.
"""
import numpy as np
import pandas as pd
from config import *

FEATURES = ["base", "dhat", "dos", "oracle", "noise"]


class World:
    """Static data shared by all episodes."""

    def __init__(self):
        self.rest = pd.read_csv(DATA / "restaurants.csv")
        self.riders = pd.read_csv(DATA / "riders.csv")
        self.days = pd.read_csv(DATA / "days.csv")
        orders = pd.read_csv(DATA / "orders.csv")
        self.orders = {d: g.sort_values("order_created_min").reset_index(drop=True)
                       for d, g in orders.groupby("day_id")}
        prof = pd.read_csv(DATA / "demand_profile.csv")
        rid = {r: i for i, r in enumerate(self.rest.restaurant_id)}
        self.mult = {}
        for d in self.days.day_id:
            self.mult[d] = np.ones((4, len(self.rest)))
        for r in prof.itertuples():
            self.mult[r.day_id][r.block_2h, rid[r.restaurant_id]] = r.demand_multiplier
        self.rxy = self.rest[["x_km", "y_km"]].values
        self.centroid = tuple(self.rxy.mean(0))
        self.n_rest = len(self.rest)

    def day_ids(self, level, pattern, split):
        q = self.days[(self.days.level == level) & (self.days.pattern == pattern) & (self.days.split == split)]
        return list(q.day_id)


def md(ax, ay, bx, by):
    return abs(ax - bx) + abs(ay - by)


class Courier:
    __slots__ = ("x", "y", "free_at", "tx", "ty", "lock", "focal", "policy", "home")

    def __init__(self, x, y, focal=False, policy="centroid"):
        self.x, self.y, self.free_at = x, y, 0.0
        self.tx = self.ty = None
        self.lock = {}
        self.focal = focal
        self.policy = policy      # competitor static policy (paper Table 3)
        self.home = None          # target chosen after the last delivery


class CourierEnv:
    def __init__(self, world, feature="dos", n_comp=N_COMPETITORS, reject_penalty=0.0, seed=0):
        assert feature in FEATURES
        self.w, self.feature, self.n_comp = world, feature, n_comp
        self.reject_penalty = reject_penalty
        self.rng = np.random.default_rng(seed)
        self.obs_dim = 7 + world.n_rest + (0 if feature == "base" else 6)
        self.n_actions = len(ACTIONS)

    # ------------------------------------------------------------------ features
    def _idle_comp_xy(self, t):
        return np.array([(c.x, c.y) for c in self.comps if c.free_at <= t]).reshape(-1, 2)

    def _demand_recent(self, px, py, t):
        i0 = np.searchsorted(self.ot, t - DOS_WINDOW, "left")
        i1 = np.searchsorted(self.ot, t, "right")
        if i1 <= i0:
            return 0.0
        d = np.abs(self.orx[i0:i1] - px) + np.abs(self.ory[i0:i1] - py)
        return float((d <= DOS_RADIUS).sum())

    def _demand_true(self, px, py, t):
        hw = np.array(HOURLY_WEIGHTS) / sum(HOURLY_WEIGHTS)
        h = min(7, int(t // 60))
        lam_min = self.n_expected * hw[h] / 60.0
        w = self.w.rest.popularity.values * self.mult[min(3, h // 2)]
        w = w / w.sum()
        near = (np.abs(self.w.rxy[:, 0] - px) + np.abs(self.w.rxy[:, 1] - py)) <= DOS_RADIUS
        return float(lam_min * w[near].sum() * DOS_WINDOW)

    def _points(self, cust):
        x, y = self.me.x, self.me.y
        pts = [(x, y), cust if cust is not None else (x, y)]
        for dx, dy in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
            pts.append((min(GRID_KM, max(0, x + dx * NEIGHBOR_STEP)), min(GRID_KM, max(0, y + dy * NEIGHBOR_STEP))))
        return pts

    def dos_at(self, px, py, t, idle_xy=None):
        if idle_xy is None:
            idle_xy = self._idle_comp_xy(t)
        c = float(((np.abs(idle_xy[:, 0] - px) + np.abs(idle_xy[:, 1] - py)) <= DOS_RADIUS).sum()) if len(idle_xy) else 0.0
        return self._demand_recent(px, py, t) / (c + 1.0)

    def _extra(self, t, cust):
        f = self.feature
        if f == "base":
            return []
        if f == "noise":
            return list(self.rng.uniform(0, 1, 6))
        idle = self._idle_comp_xy(t)
        out = []
        for px, py in self._points(cust):
            if f == "dhat":
                v = self._demand_recent(px, py, t)
            else:
                c = float(((np.abs(idle[:, 0] - px) + np.abs(idle[:, 1] - py)) <= DOS_RADIUS).sum()) if len(idle) else 0.0
                d = self._demand_true(px, py, t) if f == "oracle" else self._demand_recent(px, py, t)
                v = d / (c + 1.0)
            out.append(np.log1p(v) / np.log1p(30.0))
        return out

    def _obs(self, t, order=None):
        oh = np.zeros(self.w.n_rest)
        if order is not None:
            oh[order["ri"]] = 1.0
            cust, fee, tip = (order["cx"], order["cy"]), order["fee"], order["tip"]
        else:
            cust, fee, tip = None, 0.0, 0.0
        cx, cy = cust if cust else (0.0, 0.0)
        base = [self.me.x / GRID_KM, self.me.y / GRID_KM, cx / GRID_KM, cy / GRID_KM, fee / 10, tip / 10, t / SHIFT_MIN]
        return np.array(base + list(oh) + self._extra(t, cust), dtype=np.float32)

    # ------------------------------------------------------------------ episode
    def reset(self, day_id):
        self.day = day_id
        od = self.w.orders[day_id]
        self.ot = od.order_created_min.values
        self.orx, self.ory = od.restaurant_x_km.values, od.restaurant_y_km.values
        self.mult = self.w.mult[day_id]
        self.n_expected = len(od)
        self.orders = [dict(id=i, t=r.order_created_min, ri=int(r.restaurant_idx), rx=r.restaurant_x_km,
                            ry=r.restaurant_y_km, cx=r.customer_x_km, cy=r.customer_y_km, ready=r.ready_min,
                            fee=r.delivery_fee_cny, tip=r.tip_cny) for i, r in enumerate(od.itertuples())]
        rd = self.w.riders
        f = rd[rd.role == "focal"].iloc[0]
        comps = rd[rd.role == "competitor"].iloc[: self.n_comp]
        self.me = Courier(f.start_x_km, f.start_y_km, focal=True)
        pols = ["last", "centroid", "closest", "random_restaurant"]
        self.comps = [Courier(r.start_x_km, r.start_y_km, policy=pols[i % 4]) for i, r in enumerate(comps.itertuples())]
        self.stats = dict(net=0.0, income=0.0, km=0.0, n=0, offers=0, rejects=0, idle_min=0, delivered_all=0)
        self.gen = self._run()
        self._acc = 0.0
        self.pending_decision = next(self.gen)
        return self.pending_decision[0], self.pending_decision[1]

    def step(self, action):
        self._acc = 0.0
        try:
            self.pending_decision = self.gen.send(action)
            done = False
        except StopIteration:
            self.pending_decision = (np.zeros(self.obs_dim, np.float32), np.ones(self.n_actions, bool), None)
            done = True
        obs, mask, info = self.pending_decision
        return obs, self._acc, done, mask, info

    def _drive(self, c, km):
        if c.focal:
            self._acc -= COST_PER_KM * km
            self.stats["km"] += km
            self.stats["net"] -= COST_PER_KM * km

    def _move(self, c):
        if c.tx is None:
            return
        step = SPEED_KM_MIN
        dx, dy = c.tx - c.x, c.ty - c.y
        mvx = min(step, abs(dx))
        c.x += np.sign(dx) * mvx
        step -= mvx
        mvy = min(step, abs(dy))
        c.y += np.sign(dy) * mvy
        self._drive(c, mvx + mvy)
        if abs(c.tx - c.x) < 1e-9 and abs(c.ty - c.y) < 1e-9:
            c.tx = c.ty = None

    def _comp_target(self, c):
        rxy = self.w.rxy
        if c.policy == "last":
            return (c.x, c.y)
        if c.policy == "centroid":
            return self.w.centroid
        if c.policy == "closest":
            i = int(np.argmin(np.abs(rxy[:, 0] - c.x) + np.abs(rxy[:, 1] - c.y)))
        else:
            i = int(self.rng.integers(len(rxy)))
        return (float(rxy[i, 0]), float(rxy[i, 1]))

    def _run(self):
        """Generator: yields (obs, mask, info) whenever the focal courier must decide."""
        orders, oi, pending = self.orders, 0, []
        cen = self.w.centroid
        offer_mask = np.zeros(self.n_actions, bool); offer_mask[OFFER_ACTIONS] = True
        move_mask = np.zeros(self.n_actions, bool); move_mask[MOVE_ACTIONS] = True
        next_reloc = 0
        for minute in range(SHIFT_MIN):
            t = float(minute)
            while oi < len(orders) and orders[oi]["t"] <= t:
                pending.append(orders[oi]); oi += 1
            pending = [o for o in pending if t - o["t"] <= ORDER_EXPIRE]
            # competitors: the paper's static policies (Last, Centroid, Closest, Random restaurant)
            for c in self.comps:
                if c.free_at <= t:
                    if c.home is None:
                        c.home = self._comp_target(c)
                    if c.tx is None and md(c.x, c.y, *c.home) > 0.05:
                        c.tx, c.ty = c.home
                    self._move(c)
            # focal: idle repositioning decision every TAU minutes
            me = self.me
            if me.free_at <= t:
                self.stats["idle_min"] += 1
                if minute >= next_reloc:
                    a = yield (self._obs(t), move_mask, {"type": "move", "t": t})
                    dx, dy = DIRS[a]
                    if a == 2:
                        me.tx = me.ty = None
                    else:
                        L = SPEED_KM_MIN * TAU
                        me.tx = min(GRID_KM, max(0.0, me.x + dx * L))
                        me.ty = min(GRID_KM, max(0.0, me.y + dy * L))
                    next_reloc = minute + TAU
                self._move(me)
            # platform assignment
            if minute % ASSIGN_EVERY == 0 and pending:
                pending.sort(key=lambda o: o["t"])
                keep = []
                for o in pending:
                    cands = [c for c in self.comps + [me] if c.free_at <= t and c.lock.get(o["id"], -1) < t
                             and md(c.x, c.y, o["rx"], o["ry"]) <= MAX_PICKUP_KM]
                    if not cands:
                        keep.append(o); continue
                    c = min(cands, key=lambda c: (round(md(c.x, c.y, o["rx"], o["ry"]), 1), c.free_at))
                    d1 = md(c.x, c.y, o["rx"], o["ry"]); d2 = md(o["rx"], o["ry"], o["cx"], o["cy"])
                    fin = max(t + d1 / SPEED_KM_MIN, o["ready"]) + d2 / SPEED_KM_MIN
                    if c.focal:
                        self.stats["offers"] += 1
                        a = yield (self._obs(t, o), offer_mask,
                                   {"type": "offer", "t": t, "order": o, "d1": d1, "d2": d2, "fin": fin})
                        acc = (a == 0)
                    else:
                        acc = self.rng.random() < COMP_ACCEPT
                    if acc:
                        c.free_at, c.tx, c.ty, c.home = fin, None, None, None
                        c.x, c.y = o["cx"], o["cy"]
                        self.stats["delivered_all"] += 1
                        if c.focal:
                            inc = o["fee"] + o["tip"]
                            self._acc += inc
                            self.stats["income"] += inc; self.stats["net"] += inc; self.stats["n"] += 1
                            self._drive(c, d1 + d2)
                            next_reloc = int(np.ceil(fin))
                    else:
                        c.lock[o["id"]] = t + REJECT_LOCK
                        if c.focal:
                            self.stats["rejects"] += 1
                            self._acc -= self.reject_penalty
                        keep.append(o)
                pending = keep
