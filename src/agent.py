"""Dueling Double DQN (D3QN) in pure NumPy, hyper-parameters from the paper's Table B-1:
lr 1e-3 (Adam), gamma 0.99, replay memory 5120, batch 32, ReLU, 70-unit hidden layers.
Network: shared trunk in -> 70 -> 70, then value head 70 -> 70 -> 1 and advantage head
70 -> 70 -> |A|; Q = V + A - mean(A). Invalid actions are masked when choosing the argmax.
"""
import numpy as np


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps, self.t = params, lr, b1, b2, eps, 0
        self.m = [np.zeros_like(x) for x in params]
        self.v = [np.zeros_like(x) for x in params]

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.p, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


def _init(rng, i, o):
    return rng.normal(0, np.sqrt(2.0 / i), (i, o)).astype(np.float64), np.zeros(o)


class DuelingNet:
    def __init__(self, n_in, n_act, h=70, seed=0):
        rng = np.random.default_rng(seed)
        shapes = [(n_in, h), (h, h), (h, h), (h, 1), (h, h), (h, n_act)]
        self.params = []
        for i, o in shapes:
            W, b = _init(rng, i, o)
            self.params += [W, b]

    def copy_from(self, other):
        for p, q in zip(self.params, other.params):
            p[...] = q

    def forward(self, x, mask):
        W1, b1, W2, b2, Wv1, bv1, Wv2, bv2, Wa1, ba1, Wa2, ba2 = self.params
        z1 = x @ W1 + b1; h1 = np.maximum(z1, 0)
        z2 = h1 @ W2 + b2; h2 = np.maximum(z2, 0)
        zv = h2 @ Wv1 + bv1; hv = np.maximum(zv, 0); v = hv @ Wv2 + bv2
        za = h2 @ Wa1 + ba1; ha = np.maximum(za, 0); a = ha @ Wa2 + ba2
        q = v + a - a.mean(1, keepdims=True)   # standard dueling; masks are applied at argmax
        cache = (x, z1, h1, z2, h2, zv, hv, za, ha)
        return q, cache

    def backward(self, dq, cache):
        W1, b1, W2, b2, Wv1, bv1, Wv2, bv2, Wa1, ba1, Wa2, ba2 = self.params
        x, z1, h1, z2, h2, zv, hv, za, ha = cache
        dv = dq.sum(1, keepdims=True)
        da = dq - dq.sum(1, keepdims=True) / dq.shape[1]
        gWv2 = hv.T @ dv; gbv2 = dv.sum(0)
        dhv = dv @ Wv2.T * (zv > 0)
        gWv1 = h2.T @ dhv; gbv1 = dhv.sum(0)
        gWa2 = ha.T @ da; gba2 = da.sum(0)
        dha = da @ Wa2.T * (za > 0)
        gWa1 = h2.T @ dha; gba1 = dha.sum(0)
        dh2 = (dhv @ Wv1.T + dha @ Wa1.T) * (z2 > 0)
        gW2 = h1.T @ dh2; gb2 = dh2.sum(0)
        dh1 = dh2 @ W2.T * (z1 > 0)
        gW1 = x.T @ dh1; gb1 = dh1.sum(0)
        return [gW1, gb1, gW2, gb2, gWv1, gbv1, gWv2, gbv2, gWa1, gba1, gWa2, gba2]


class D3QN:
    def __init__(self, n_in, n_act, seed=0, lr=1e-3, gamma=0.99, memory=5120, batch=32,
                 target_every=500, reward_scale=0.1):
        self.q = DuelingNet(n_in, n_act, seed=seed)
        self.tq = DuelingNet(n_in, n_act, seed=seed)
        self.tq.copy_from(self.q)
        self.opt = Adam(self.q.params, lr)
        self.gamma, self.batch, self.target_every, self.rs = gamma, batch, target_every, reward_scale
        self.rng = np.random.default_rng(seed)
        self.n_act, self.cap, self.ptr, self.size, self.updates = n_act, memory, 0, 0, 0
        self.S = np.zeros((memory, n_in)); self.A = np.zeros(memory, int); self.R = np.zeros(memory)
        self.S2 = np.zeros((memory, n_in)); self.M2 = np.zeros((memory, n_act), bool); self.D = np.zeros(memory)

    def act(self, s, mask, eps=0.0):
        valid = np.flatnonzero(mask)
        if self.rng.random() < eps:
            return int(self.rng.choice(valid))
        q, _ = self.q.forward(s[None].astype(np.float64), mask[None])
        q = np.where(mask[None], q, -np.inf)
        return int(q.argmax())

    def remember(self, s, a, r, s2, m2, done):
        i = self.ptr
        self.S[i], self.A[i], self.R[i], self.S2[i], self.M2[i], self.D[i] = s, a, r * self.rs, s2, m2, done
        self.ptr = (i + 1) % self.cap; self.size = min(self.size + 1, self.cap)

    def learn(self):
        if self.size < max(self.batch, 500):
            return
        idx = self.rng.integers(0, self.size, self.batch)
        S, A, R, S2, M2, D = self.S[idx], self.A[idx], self.R[idx], self.S2[idx], self.M2[idx], self.D[idx]
        q2_online, _ = self.q.forward(S2, M2)
        a2 = np.where(M2, q2_online, -np.inf).argmax(1)          # Double DQN: select with online net
        q2_target, _ = self.tq.forward(S2, M2)                    # ... evaluate with target net
        y = R + self.gamma * (1 - D) * q2_target[np.arange(self.batch), a2]
        full = np.ones((self.batch, self.n_act), bool)
        q, cache = self.q.forward(S, full)
        err = q[np.arange(self.batch), A] - y
        g = np.clip(err, -1, 1) / self.batch                       # Huber loss gradient
        dq = np.zeros_like(q); dq[np.arange(self.batch), A] = g
        self.opt.step(self.q.backward(dq, cache))
        self.updates += 1
        if self.updates % self.target_every == 0:
            self.tq.copy_from(self.q)

    def save(self, path):
        np.savez(path, *self.q.params)

    def load(self, path):
        z = np.load(path)
        for i, p in enumerate(self.q.params):
            p[...] = z[f"arr_{i}"]
        self.tq.copy_from(self.q)
