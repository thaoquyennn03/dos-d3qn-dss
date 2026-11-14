"""Rule-based policies: paper's best static policy (Centroid) and the profit-rate threshold rules."""
from config import *
from env import md


class Centroid:
    """Accept every offer; when idle move toward the restaurant centroid (paper policy 3)."""
    name = "B0_centroid"

    def act(self, obs, mask, info, env):
        if info["type"] == "offer":
            return 0
        cx, cy = env.w.centroid
        x, y = env.me.x, env.me.y
        if md(x, y, cx, cy) < SPEED_KM_MIN * TAU / 2:
            return 2
        if abs(cx - x) >= abs(cy - y):
            return 6 if cx > x else 5
        return 3 if cy > y else 4


class Threshold(Centroid):
    """Accept if (fee + tip - cost) / minutes-to-finish >= theta. Optional DOS term:
    rate + lam * (DOS(destination) - DOS(current)) >= theta. Idle moves = Centroid."""

    def __init__(self, theta, lam=0.0):
        self.theta, self.lam = theta, lam
        self.name = "B2_threshold_dos" if lam else "B1_threshold"

    def act(self, obs, mask, info, env):
        if info["type"] != "offer":
            return super().act(obs, mask, info, env)
        o, t = info["order"], info["t"]
        profit = o["fee"] + o["tip"] - COST_PER_KM * (info["d1"] + info["d2"])
        rate = profit / max(1.0, info["fin"] - t)
        if self.lam:
            idle = env._idle_comp_xy(t)
            rate += self.lam * (env.dos_at(o["cx"], o["cy"], t, idle) - env.dos_at(env.me.x, env.me.y, t, idle))
        return 0 if rate >= self.theta else 1


def run_policy(env, policy, day_id):
    obs, mask = env.reset(day_id)
    info = env.pending_decision[2]
    done = False
    while not done:
        a = policy.act(obs, mask, info, env)
        obs, r, done, mask, info = env.step(a)
    return dict(env.stats)
