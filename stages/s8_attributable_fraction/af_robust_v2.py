"""Attributable fractions under other radii, estimators and the event bootstrap."""
import json, sys, time, numpy as np, pandas as pd
sys.path.insert(0, '@WORK@/v11/code')
from afcore_v1 import *
from clr_exact import CLR
OUT = H + '/v11/an/af_robust_register_v2.json'
REG = json.load(open(H + '/reference/reference_values.json', encoding='utf-8'))
RV = lambda k: REG[k]['value']
SC = ['sc1', 'sc2', 'sc3']
R = {'status': {'B_radius': 'pre-specified sensitivity (data contract v1: r = 50 m primary, 0 and 100 m sensitivity)',
                'G_rings': 'exploratory, after all results (2026-10-29)',
                'C_estimators': 'exploratory, comparison methods (chosen after the 2026-10-13 literature review)',
                'S_bootstrap': 'exploratory, after all results (2026-10-29)'},
     'seed_base': SEED, 'ndraw': NDRAW, 'nboot': NBOOT}
EV = load_main_classes()
def with_classes(d, r):
    d = d.copy(); c1 = d[f'class1_{r}m'].astype(int); c2 = d[f'bare_at_cutoff_{0 if r == 0 else 50}m'].astype(int)
    sc = np.where(c2 == 1, 2, np.where(c1 == 1, 1, 0)); sc = np.where((sc == 0) & (d.lt_detected == 1), 3, sc)
    d['sc4'] = sc
    for c in (1, 2, 3): d[f'sc{c}'] = (sc == c).astype(int)
    return d

B = {}
for r in (0, 50, 100):
    x = with_classes(EV, r)
    if r == 50: assert (x.sc4 == x.site_class4).all()
    for rg, sel in RNG.items():
        o = af_model(sel(x), SC, f'radius:{r}:{rg}'); o['controls_flagged_share'] = o['controls_flagged_share_fitted']
        B[f'r{r}_{rg}'] = o; print('B', r, rg, round(o['AF_total']['point'], 4), {t: round(o['OR'][t]['OR'], 3) for t in SC}, flush=True)
for rg, k in (('le39.5', 'AF4.event.le39_5'), ('all', 'AF4.event.all')):
    assert abs(B[f'r50_{rg}']['AF_total']['point'] - RV(k)['total']['point']) < 2e-3, rg
R['B_radius'] = B

G = {}
x = EV.copy()
lev = [('bare_0', x.bare_at_cutoff_0m == 1), ('scar_0', x.class1_0m == 1), ('bare_1_50', x.bare_at_cutoff_50m == 1),
       ('scar_1_50', x.class1_50m == 1), ('scar_50_100', x.class1_100m == 1), ('lt', x.lt_detected == 1)]
taken = pd.Series(False, index=x.index)
for nm, cond in lev:
    x[nm] = (cond.fillna(False) & ~taken).astype(int); taken = taken | (x[nm] == 1)
RINGS = [nm for nm, _ in lev]
for rg, sel in RNG.items():
    s = sel(x)
    o = af_model(s, RINGS, f'rings:{rg}', sums={'at_point': ['bare_0', 'scar_0'], 'within_50': ['bare_0', 'scar_0', 'bare_1_50', 'scar_1_50'],
                                                 'within_100': ['bare_0', 'scar_0', 'bare_1_50', 'scar_1_50', 'scar_50_100'],
                                                 'scar_only': ['scar_0', 'scar_1_50', 'scar_50_100'], 'bare_only': ['bare_0', 'bare_1_50']})
    o['cases_by_ring'] = {t: int(s.loc[s.case == 1, t].sum()) for t in RINGS}; o['controls_by_ring'] = {t: int(s.loc[s.case == 0, t].sum()) for t in RINGS}
    G[rg] = o; print('G', rg, {t: round(o['OR'][t]['OR'], 2) for t in RINGS}, round(o['AF_total']['point'], 4), flush=True)
    assert abs(o['n_cases'] - B[f'r100_{rg}']['n_cases']) == 0
R['G_rings'] = G

x50 = with_classes(EV, 50); CE = {}
for rg, sel in RNG.items():
    m, mr, dd, nd = fitfe(sel(x50), f"case~{TERR}+sc1+sc2+sc3")
    X = dd[TV + SC].astype(float).copy(); mu = X[TV].mean(); sd = X[TV].std(); X[TV] = (X[TV] - mu) / sd
    start = np.concatenate([m.params[TV].values * sd.values, m.params[SC].values])
    f = CLR(X.values, dd.case.values, dd.event.values, cluster=dd.uid.values, names=TV + SC).fit(start)
    k0 = len(TV); b = np.array(f['beta'][k0:k0 + 3]); V = np.array(f['cov_cluster'])[k0:k0 + 3, k0:k0 + 3]
    o = af_from(b, V, dd, SC, f'clr:{rg}')
    o['OR'] = {t: {'OR': float(np.exp(b[i])), 'ci': [float(np.exp(b[i] - 1.96 * np.sqrt(V[i, i]))), float(np.exp(b[i] + 1.96 * np.sqrt(V[i, i])))], 'beta': float(b[i]), 'se': float(np.sqrt(V[i, i]))} for i, t in enumerate(SC)}
    reg = RV(f"MC1.{'all' if rg == 'all' else 'le39_5'}.classes")['clr']
    assert all(abs(o['OR'][t]['OR'] - reg[t]['OR']) < 2e-3 for t in SC), (rg, o['OR'], reg)
    CE[f'clr_{rg}'] = o; print('C clr', rg, round(o['AF_total']['point'], 4), flush=True)
for km in (20, 10):
    for rg, sel in RNG.items():
        x = sel(x50).copy(); x['blk'] = (np.floor(x.x / (km * 1000)).astype(int).astype(str) + '_' + np.floor(x.y / (km * 1000)).astype(int).astype(str))
        g = x.groupby('blk').case.agg(['sum', 'size']); keep = g[(g['sum'] > 0) & (g['sum'] < g['size'])].index; x = x[x.blk.isin(keep)].copy()
        o = af_model(x, SC, f'block{km}:{rg}', extra_fe='+C(blk)'); o['blocks_used'] = int(len(keep))
        reg = RV(f"MC2.{km}km.{'all' if rg == 'all' else 'le39_5'}")
        assert all(abs(o['OR'][t]['OR'] - reg[t]['OR']) < 2e-3 for t in SC), (km, rg)
        CE[f'block{km}_{rg}'] = o; print('C block', km, rg, round(o['AF_total']['point'], 4), flush=True)
R['C_estimators'] = CE

S = {}; T0 = time.time()
for rg, sel in RNG.items():
    f = FastFE(sel(EV), SC); bl, ok = f.fit_all(); assert ok
    pt = f.af_point(f.X, f.y, bl); assert abs(pt.sum() - B[f'r50_{rg}']['AF_total']['point']) < 1e-6
    a, fails = f.boot(f'AF4:{rg}')
    S[rg] = {'point': float(pt.sum()), 'ci_boot': [float(np.percentile(a.sum(1), 2.5)), float(np.percentile(a.sum(1), 97.5))],
             'components': {t: {'point': float(pt[i]), 'ci_boot': [float(np.percentile(a[:, i], 2.5)), float(np.percentile(a[:, i], 97.5))]} for i, t in enumerate(SC)},
             'n_ok': int(len(a)), 'fails': fails, 'events': f.K}
    print('S', rg, round(pt.sum(), 4), [round(v, 4) for v in S[rg]['ci_boot']], fails, round(time.time() - T0), 's', flush=True)
R['S_bootstrap'] = S
R['fallback_bfgs'] = FALLBACK
json.dump(R, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
print('written', OUT)
