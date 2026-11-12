"""Shared functions of stage s9: data preparation, event fixed-effects logistic models, event-level bootstrap and attributable fractions."""
import math, json, os, sys, zlib, warnings, numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
from scipy.special import expit
warnings.filterwarnings('ignore')
H = '@WORK@'
sys.path.insert(0, H + '/v11/code')
import afcore_v1 as AC
BRO = '闊葉樹林型'; BAM = '竹林'; TV = ['slope', 'slope2', 'north', 'east', 'planc', 'profc', 'elev']; TERR = '+'.join(TV)
RAINV = ['lmax24', 'ltotal']; RAINF = '+'.join(RAINV)
NBOOT = 2000; NDRAW = 4000
TAGS = {'v4': H + '/v11/data/tags_event_v4.csv', 'v5': H + '/v27/data/tags_event_v5.csv'}
REPS = {'v1': H + '/v11/data/rep_rows_tagged_v1.csv', 'v5': H + '/v27/rep/rep_rows_tagged_v5.csv'}
BRIGHT = H + '/v11/unround/g3/bright_rows_v1.csv'; AROWS = H + '/g3/analysis_rows_v1.csv'
FALLBACK = []
seed_for = AC.seed_for

def prep(df):
    """Derived columns of final_models_v2.py / stand_origin_v1.py."""
    d = df.copy(); d['case'] = d.case.astype(int); d['slope2'] = d.slope ** 2; d['s20'] = d.slope - 20.0
    d['legacy_any4'] = (d.site_class4 > 0).astype(int); d['legacy'] = d.legacy_any4
    d['cluster'] = np.where(d.kind == 'case', d.event.astype(str), 'controls')
    for c in (1, 2, 3): d[f'sc{c}'] = (d.site_class4 == c).astype(int)
    d['tsd3_bin'] = pd.Categorical(d.tsd3_bin, categories=['none', '0-5', '5-10', '10-20', '20+'])
    d['tsd3_bin2'] = pd.Categorical(np.select([d.class3 == 0, d.TSD < 10, d.TSD < 20], ['none', '0-10', '10-20'], '20+'), categories=['none', '0-10', '10-20', '20+'])
    d['tss1_bin'] = pd.Categorical(np.where(d.site_class4 == 1, d.tss_bin.replace({'20+': '10+', '10-20': '10+'}), 'none'), categories=['none', '0-5', '5-10', '10+'])
    yrs = d.scar_50m_min_yrs; age = np.select([yrs < 1, yrs < 10], ['lt1', '1-10'], '10+')
    d['cell'] = np.select([d.site_class4 == 0, d.site_class4 == 3, (d.site_class4 == 1) & (age == 'lt1'), (d.site_class4 == 1) & (age == '1-10'), (d.site_class4 == 1),
                           (d.site_class4 == 2) & yrs.isna(), (d.site_class4 == 2) & (age == 'lt1'), (d.site_class4 == 2) & (age == '1-10')],
                          ['c0', 'c3', 's_lt1', 's_1_10', 's_10p', 'b_noscar', 'b_lt1', 'b_1_10'], 'b_10p')
    d['cell'] = pd.Categorical(d.cell, categories=['c0', 's_lt1', 's_1_10', 's_10p', 'b_lt1', 'b_1_10', 'b_10p', 'b_noscar', 'c3'])
    return d.dropna(subset=['slope', 'north', 'east', 'planc', 'profc', 'elev'])

def load_event(tags='v5'):
    E = pd.read_csv(TAGS[tags], dtype={'uid': str}, low_memory=False)
    return prep(E)

def with_bright(E):
    BR = pd.read_csv(BRIGHT, dtype={'uid': str})
    X = E.merge(BR[['uid', 'event', 'max24', 'total', 'score', 'score_terrain', 'score_insample', 'Q', 'D', 'Qt', 'Qin']], on=['uid', 'event'], how='left')
    for k in ('total', 'max24'): X['l' + k] = np.log1p(X[k])
    return X

def with_arows(E):
    A = pd.read_csv(AROWS, dtype={'uid': str}, low_memory=False)
    keep = [c for c in ('uid', 'chm', 'chm_valid', 'chm_date', 'stand_function', 'area_ha') if c in A.columns]
    return E.merge(A[keep].drop_duplicates('uid'), on='uid', how='left')

def drop_uninf(d):
    g = d.groupby('event').case.agg(['sum', 'size']); keep = g[(g['sum'] > 0) & (g['sum'] < g['size'])].index
    return d[d.event.isin(keep)].copy(), int(len(g) - len(keep))

def fitfe(d, form, cluster='uid'):
    d, nd = drop_uninf(d); f2 = form + '+C(event)'
    m = smf.logit(f2, data=d).fit(disp=0, maxiter=100, method='newton')
    grp = pd.factorize(d[cluster])[0]
    if not m.mle_retvals.get('converged', True):
        m0 = m
        m = smf.logit(f2, data=d).fit(disp=0, maxiter=5000, method='bfgs', gtol=1e-8)
        mr = smf.logit(f2, data=d).fit(disp=0, maxiter=5000, method='bfgs', gtol=1e-8, cov_type='cluster', cov_kwds={'groups': grp})
        assert m.mle_retvals.get('converged', True) or abs(m.llf - m0.llf) < 1e-5, f2
        FALLBACK.append({'formula': f2, 'rows': int(len(d))})
        return m, mr, d, nd
    mr = smf.logit(f2, data=d).fit(disp=0, maxiter=100, method='newton', cov_type='cluster', cov_kwds={'groups': grp})
    return m, mr, d, nd

def lin(m, mr, coefs):
    b = sum(w * m.params[t] for t, w in coefs.items())
    def se_of(mod):
        c = mod.cov_params(); return math.sqrt(sum(wi * wj * c.loc[ti, tj] for ti, wi in coefs.items() for tj, wj in coefs.items()))
    sr = se_of(mr)
    return {'OR': math.exp(b), 'ci': [math.exp(b - 1.96 * sr), math.exp(b + 1.96 * sr)], 'p': float(2 * (1 - norm.cdf(abs(b / sr)))), 'beta': float(b), 'se': float(sr)}

def term(m, mr, t): return lin(m, mr, {t: 1})

def pct(x):
    x = np.asarray(x, float); return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

class Boot:
    """Event-level bootstrap of an event fixed-effect logit. covs are standardised, terms enter as they are."""
    def __init__(self, d, covs, terms, indicators=None):
        d, nd = drop_uninf(d)
        self.d = d; self.dropped = nd; self.covs = list(covs); self.terms = list(terms); self.kt = len(terms)
        Xc = d[self.covs].astype(float).values if self.covs else np.zeros((len(d), 0))
        if self.covs: Xc = (Xc - Xc.mean(0)) / Xc.std(0)
        self.X = np.column_stack([Xc, d[self.terms].astype(float).values]); self.y = d.case.astype(float).values
        codes, uni = pd.factorize(d.event); self.K = len(uni); self.rows = [np.where(codes == k)[0] for k in range(self.K)]
        if indicators is None:
            indicators = [t for t in self.terms if set(np.unique(d[t].dropna().values)) <= {0, 1}]
        self.ind = [self.terms.index(t) for t in indicators]
    def fit_all(self):
        ev = np.empty(len(self.y), dtype=int)
        for k, r in enumerate(self.rows): ev[r] = k
        beta, it, ok = AC.FastFE.fit_rows(self.X, self.y, ev, self.K); assert ok
        return beta[-self.kt:]
    def run(self, label, B=NBOOT, rho=False):
        rng = np.random.default_rng(seed_for('boot:' + label)); out = []; rh = []; fails = 0
        for _ in range(B):
            pick = rng.integers(0, self.K, self.K); rows = [self.rows[k] for k in pick]; idx = np.concatenate(rows)
            ev = np.concatenate([np.full(len(r_), j) for j, r_ in enumerate(rows)]); X = self.X[idx]; y = self.y[idx]
            T = X[:, -self.kt:]
            if any(T[y == 1, j].sum() == 0 or T[y == 0, j].sum() == 0 for j in self.ind): fails += 1; continue
            try: beta, it, ok = AC.FastFE.fit_rows(X, y, ev, self.K)
            except np.linalg.LinAlgError: fails += 1; continue
            if not ok: fails += 1; continue
            out.append(beta[-self.kt:])
            if rho: rh.append(T[y == 1].mean(0))
        res = {'beta': np.array(out), 'n_ok': len(out), 'fails': fails, 'events': self.K, 'label': label}
        if rho: res['rho'] = np.array(rh)
        return res

def af_point(rho, beta): return rho * (1 - np.exp(-beta))

def af_mc(b, V, dd, terms, label):
    """Bruzzi attributable fraction with the Monte Carlo interval of afcore_v1.af_from."""
    return AC.af_from(np.asarray(b), np.asarray(V), dd, terms, label)

def main_model(d, terms, label, covs=TV, extra='', boot=True, B=NBOOT):
    """Event fixed-effect logit of case on covariates + terms: odds ratios with cluster-robust and event-bootstrap intervals,
    attributable fraction of the terms with Monte Carlo and event-bootstrap intervals."""
    form = 'case~' + '+'.join(list(covs) + list(terms)) + extra
    m, mr, dd, nd = fitfe(d, form)
    b = m.params[terms].values; V = mr.cov_params().loc[terms, terms].values
    o = {'n_rows': int(len(dd)), 'n_cases': int(dd.case.sum()), 'n_control_rows': int((dd.case == 0).sum()), 'n_control_points': int(dd[dd.case == 0].uid.nunique()),
         'n_events': int(dd.event.nunique()), 'events_dropped': nd, 'OR': {t: term(m, mr, t) for t in terms},
         'rho_cases': {t: float(dd.loc[dd.case == 1, t].mean()) for t in terms}, 'rho_controls': {t: float(dd.loc[dd.case == 0, t].mean()) for t in terms}}
    indic = all(set(np.unique(dd[t].values)) <= {0, 1} for t in terms)
    if indic and dd.loc[dd.case == 1, terms].sum(1).max() <= 1:
        mc = af_mc(b, V, dd, terms, 'v27:mc:' + label)
        for t in terms: o[f'AF_{t}'] = {'point': mc[f'AF_{t}']['point'], 'ci_mc': mc[f'AF_{t}']['ci_mc']}
        o['AF_total'] = {'point': mc['AF_total']['point'], 'ci_mc': mc['AF_total']['ci_mc']}
    else: indic = False
    if boot and not extra:
        bt = Boot(d, covs, terms); bl = bt.fit_all()
        assert np.allclose(bl, b, atol=1e-6), (label, bl, b)
        r = bt.run('v27:' + label, B, rho=True); be = r['beta']
        for j, t in enumerate(terms): o['OR'][t]['ci_boot'] = [math.exp(v) for v in pct(be[:, j])]
        if indic:
            af = af_point(r['rho'], be)
            for j, t in enumerate(terms): o[f'AF_{t}']['ci_boot'] = pct(af[:, j])
            o['AF_total']['ci_boot'] = pct(af.sum(1))
        o['boot'] = {'n_ok': r['n_ok'], 'fails': r['fails'], 'events': r['events']}
        o['_boot'] = r
    o['_m'] = (m, mr, dd)
    return o

def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items() if not str(k).startswith('_')}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    return o

RNG = {'all': lambda d: d, 'le39.5': lambda d: d[d.slope <= 39.5]}
