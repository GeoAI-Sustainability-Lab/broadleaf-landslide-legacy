"""Shared functions for the attributable-fraction comparisons (fits, Bruzzi fractions, event bootstrap)."""
import math, zlib, warnings, numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
from scipy.special import expit
warnings.filterwarnings('ignore')
H = '@WORK@'
DATA = H + '/v11/data'
BRO = '闊葉樹林型'; TV = ['slope', 'slope2', 'north', 'east', 'planc', 'profc', 'elev']; TERR = '+'.join(TV)
SEED = 20261029; NDRAW = 4000; NBOOT = 2000; CAT0 = pd.Timestamp('2004-07-10')
FALLBACK = []

def seed_for(label):
    return (SEED + zlib.crc32(label.encode('utf-8'))) % (2 ** 32)

def drop_uninf(d, col='event'):
    g = d.groupby(col).case.agg(['sum', 'size']); keep = g[(g['sum'] > 0) & (g['sum'] < g['size'])].index
    return d[d[col].isin(keep)].copy(), int(len(g) - len(keep))

def fitfe(d, form, extra_fe=''):
    d, nd = drop_uninf(d); f2 = form + extra_fe + '+C(event)'
    m = smf.logit(f2, data=d).fit(disp=0, maxiter=100, method='newton')
    if not m.mle_retvals.get('converged', True):
        m0 = m
        m = smf.logit(f2, data=d).fit(disp=0, maxiter=5000, method='bfgs', gtol=1e-8)
        mr = smf.logit(f2, data=d).fit(disp=0, maxiter=5000, method='bfgs', gtol=1e-8, cov_type='cluster', cov_kwds={'groups': pd.factorize(d.uid)[0]})
        assert m.mle_retvals.get('converged', True) or abs(m.llf - m0.llf) < 1e-5, f2
        FALLBACK.append({'formula': f2, 'rows': int(len(d)), 'llf_newton': float(m0.llf), 'llf_bfgs': float(m.llf)})
        print('FALLBACK_BFGS', f2, len(d), m0.llf, m.llf, flush=True)
        return m, mr, d, nd
    mr = smf.logit(f2, data=d).fit(disp=0, maxiter=100, method='newton', cov_type='cluster', cov_kwds={'groups': pd.factorize(d.uid)[0]})
    return m, mr, d, nd

def term(m, mr, t):
    b = float(m.params[t]); sr = float(mr.bse[t]); s = float(m.bse[t])
    return {'OR': math.exp(b), 'ci': [math.exp(b - 1.96 * sr), math.exp(b + 1.96 * sr)],
            'ci_model': [math.exp(b - 1.96 * s), math.exp(b + 1.96 * s)], 'p': float(2 * (1 - norm.cdf(abs(b / sr)))), 'beta': b, 'se': sr}

def _ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

def af_from(b, V, dd, terms, label, sums=None):
    rng = np.random.default_rng(seed_for(label))
    cases = dd[dd.case == 1]; n = len(cases)
    rho = np.array([float(cases[t].mean()) for t in terms]); full = np.append(rho, max(0.0, 1 - rho.sum()))
    draws = rng.multivariate_normal(b, V, size=NDRAW)
    comp = lambda bv, r: r * (1 - np.exp(-bv))
    pt = comp(np.asarray(b), rho)
    sims = np.array([comp(bv, rng.multinomial(n, full)[:-1] / n) for bv in draws])
    out = {'label': label, 'seed': seed_for(label), 'n_rows': int(len(dd)), 'n_cases': n, 'n_events': int(dd.event.nunique()),
           'rho_cases': {t: float(r) for t, r in zip(terms, rho)}}
    for i, t in enumerate(terms):
        out[f'AF_{t}'] = {'point': float(pt[i]), 'ci_mc': _ci(sims[:, i])}
    out['AF_total'] = {'point': float(pt.sum()), 'ci_mc': _ci(sims.sum(1))}
    for lab, ts in (sums or {}).items():
        idx = [terms.index(t) for t in ts if t in terms]
        out[f'AF_{lab}'] = {'point': float(pt[idx].sum()), 'ci_mc': _ci(sims[:, idx].sum(1))}
    return out

def af_model(d, terms, label, sums=None, extra_fe=''):
    m, mr, dd, nd = fitfe(d, f"case~{TERR}+" + "+".join(terms), extra_fe)
    b = m.params[terms].values; V = mr.cov_params().loc[terms, terms].values
    out = af_from(b, V, dd, terms, label, sums)
    out['events_dropped'] = nd; out['OR'] = {t: term(m, mr, t) for t in terms}
    out['controls_flagged_share_fitted'] = float((dd.loc[dd.case == 0, terms].sum(1) > 0).mean())
    return out

def ratio(a, b):
    d = a['beta'] - b['beta']; se = math.sqrt(a['se'] ** 2 + b['se'] ** 2)
    return {'ratio': math.exp(d), 'ci': [math.exp(d - 1.96 * se), math.exp(d + 1.96 * se)]}

class FastFE:
    def __init__(self, d, terms):
        d, nd = drop_uninf(d)
        self.terms = list(terms); self.kt = len(terms); self.dropped = nd
        Xt = d[TV].astype(float).values; Xt = (Xt - Xt.mean(0)) / Xt.std(0)
        self.X = np.column_stack([Xt, d[self.terms].astype(float).values]); self.y = d.case.astype(float).values
        codes, uni = pd.factorize(d.event); self.K = len(uni)
        self.rows = [np.where(codes == k)[0] for k in range(self.K)]
        self.n_cases = int(self.y.sum())

    @staticmethod
    def fit_rows(X, y, ev, K, maxit=100, tol=1e-10):
        p = X.shape[1]; beta = np.zeros(p)
        ny = np.bincount(ev, weights=y, minlength=K); nn = np.bincount(ev, minlength=K).astype(float)
        alpha = np.log((ny + 0.5) / (nn - ny + 0.5))
        def ll(bb, aa):
            eta = X @ bb + aa[ev]; return float(np.sum(y * eta - np.logaddexp(0, eta)))
        cur = ll(beta, alpha)
        for it in range(maxit):
            eta = X @ beta + alpha[ev]; mu = expit(eta); w = mu * (1 - mu); r = y - mu
            gb = X.T @ r; ga = np.bincount(ev, weights=r, minlength=K); Waa = np.bincount(ev, weights=w, minlength=K)
            WX = X * w[:, None]; Hbb = X.T @ WX
            Hba = np.vstack([np.bincount(ev, weights=WX[:, j], minlength=K) for j in range(p)])
            Am = Hba / Waa
            db = np.linalg.solve(Hbb - Am @ Hba.T, gb - Am @ ga); da = (ga - Hba.T @ db) / Waa
            step = 1.0
            while True:
                nb, na = beta + step * db, alpha + step * da; new = ll(nb, na)
                if new >= cur - 1e-12 or step < 1e-6: break
                step /= 2
            beta, alpha = nb, na
            if max(np.abs(step * db).max(), np.abs(step * da).max()) < tol: return beta, it + 1, True
            cur = new
        return beta, maxit, False

    def fit_all(self):
        ev = np.empty(len(self.y), dtype=int)
        for k, r in enumerate(self.rows): ev[r] = k
        beta, it, ok = self.fit_rows(self.X, self.y, ev, self.K)
        return beta[-self.kt:], ok

    def af_point(self, X, y, bl):
        rho = X[y == 1][:, -self.kt:].mean(0); return rho * (1 - np.exp(-bl))

    def boot(self, label, B=NBOOT):
        rng = np.random.default_rng(seed_for('boot:' + label)); out = []; fails = 0
        for _ in range(B):
            pick = rng.integers(0, self.K, self.K)
            rows = [self.rows[k] for k in pick]; idx = np.concatenate(rows)
            ev = np.concatenate([np.full(len(r), j) for j, r in enumerate(rows)])
            X = self.X[idx]; y = self.y[idx]
            T = X[:, -self.kt:]
            if any(T[y == 1, j].sum() == 0 or T[y == 0, j].sum() == 0 for j in range(self.kt)): fails += 1; continue
            beta, it, ok = self.fit_rows(X, y, ev, self.K)
            if not ok: fails += 1; continue
            out.append(self.af_point(X, y, beta[-self.kt:]))
        return np.array(out), fails

def load_shared():
    E = pd.read_csv(f'{DATA}/tags_event_v4.csv', dtype={'uid': str}, low_memory=False, parse_dates=['cutoff'])
    M = E[E.ft == BRO].copy(); M['case'] = M.case.astype(int); M['slope2'] = M.slope ** 2
    M = M.dropna(subset=['slope', 'north', 'east', 'planc', 'profc', 'elev'])
    M['R1'] = (M.class1_50m == 1).astype(int); M['scar_age'] = M.scar_50m_min_yrs
    B = pd.read_csv(f'{DATA}/rep_rows_tagged_v1.csv', dtype={'uid': str}, low_memory=False, parse_dates=['cutoff'])
    assert set(B.TypeName.unique()) == {BRO}
    for d in (M, B):
        d['R3'] = ((d.lt_detected == 1) & (d.R1 == 0)).astype(int)
        d['lookback_event'] = d.groupby('event').cutoff.transform('min').sub(CAT0).dt.days / 365.25
    B['y2009'] = B.event.str.startswith('2009_')
    return M, B

def load_main_classes():
    E = pd.read_csv(f'{DATA}/tags_event_v4.csv', dtype={'uid': str}, low_memory=False)
    EV = E[E.ft == BRO].copy(); EV['case'] = EV.case.astype(int); EV['slope2'] = EV.slope ** 2
    EV = EV.dropna(subset=['slope', 'north', 'east', 'planc', 'profc', 'elev'])
    for c in (1, 2, 3): EV[f'sc{c}'] = (EV.site_class4 == c).astype(int)
    return EV

RNG = {'all': lambda d: d, 'le39.5': lambda d: d[d.slope <= 39.5]}
