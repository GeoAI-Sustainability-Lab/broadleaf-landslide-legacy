"""Conditional logistic regression (exact stratified likelihood) used by the method comparisons."""
import numpy as np

def _stratum(eta, y):
    N = len(eta); n1 = int(y.sum())
    if n1 == 0 or n1 == N: return 0.0, y.astype(float)
    NEG = -np.inf
    F = np.full((N + 1, n1 + 1), NEG); F[0, 0] = 0.0
    for i in range(1, N + 1):
        prev = F[i - 1]; cur = F[i]
        cur[0] = 0.0
        cur[1:] = np.logaddexp(prev[1:], eta[i - 1] + prev[:-1])
    logE = F[N, n1]
    B = np.full(n1 + 1, NEG); B[0] = 0.0
    pi = np.empty(N)
    for i in range(N, 0, -1):
        f = F[i - 1, :n1]
        b = B[n1 - 1::-1] if n1 - 1 >= 0 else B[:0]
        s = f + b
        m = np.max(s)
        lse = m + np.log(np.sum(np.exp(s - m))) if np.isfinite(m) else NEG
        pi[i - 1] = np.exp(eta[i - 1] + lse - logE)
        Bn = B.copy(); Bn[1:] = np.logaddexp(B[1:], eta[i - 1] + B[:-1]); B = Bn
    return logE, pi

class CLR:
    def __init__(self, X, y, strata, cluster=None, names=None):
        self.X = np.asarray(X, float); self.y = np.asarray(y, int); self.names = names or [f'x{j}' for j in range(self.X.shape[1])]
        st = np.asarray(strata); self.groups = [np.nonzero(st == s)[0] for s in np.unique(st)]
        self.groups = [g for g in self.groups if 0 < self.y[g].sum() < len(g)]
        self.cluster = None if cluster is None else np.asarray(cluster)
        self.n_used = int(sum(len(g) for g in self.groups))
    def ll_score(self, beta, per_row=False):
        eta = self.X @ beta; ll = 0.0; U = np.zeros_like(beta); R = np.zeros((len(self.y), len(beta))) if per_row else None
        for g in self.groups:
            e = eta[g]; yy = self.y[g]; logE, pi = _stratum(e, yy)
            ll += e[yy == 1].sum() - logE
            r = (yy - pi)[:, None] * self.X[g]
            U += r.sum(0)
            if per_row: R[g] = r
        return ll, U, R
    def hessian(self, beta, h=1e-5):
        p = len(beta); Hm = np.zeros((p, p))
        for j in range(p):
            d = np.zeros(p); d[j] = h
            Hm[:, j] = (self.ll_score(beta + d)[1] - self.ll_score(beta - d)[1]) / (2 * h)
        return 0.5 * (Hm + Hm.T)
    def fit(self, start, tol=1e-8, maxit=25):
        b = np.asarray(start, float).copy(); hist = []
        for it in range(maxit):
            ll, U, _ = self.ll_score(b); Hm = self.hessian(b)
            step = np.linalg.solve(-Hm, U); b = b + step; hist.append((it, ll, float(np.max(np.abs(step)))))
            if np.max(np.abs(step)) < tol: break
        ll, U, R = self.ll_score(b, per_row=True); Hm = self.hessian(b); Hinv = np.linalg.inv(-Hm)
        out = {'beta': b, 'll': ll, 'cov_model': Hinv, 'iter': hist, 'score_max': float(np.max(np.abs(U)))}
        if self.cluster is not None:
            used = np.concatenate(self.groups); cl = self.cluster[used]; Rr = R[used]
            _, inv = np.unique(cl, return_inverse=True); G = inv.max() + 1
            Uc = np.zeros((G, len(b))); np.add.at(Uc, inv, Rr)
            meat = Uc.T @ Uc; n, k = len(used), len(b)
            fac = G / (G - 1) * (n - 1) / (n - k)
            out['cov_cluster'] = fac * Hinv @ meat @ Hinv; out['n_clusters'] = int(G)
        return out
