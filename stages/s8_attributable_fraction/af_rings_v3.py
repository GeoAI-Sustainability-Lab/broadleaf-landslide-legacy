"""Attributable fractions by distance ring (at the point, within 50 m, within 100 m)."""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, '@WORK@/v11/code')
from afcore_v1 import *
OUT = H + '/v11/an/af_rings_register_v3.json'
R = {'status': 'exploratory, after all results (2026-10-29)', 'seed_base': SEED, 'ndraw': NDRAW,
     'order_bare_first': ['bare_0', 'bare_1_50', 'scar_0', 'scar_1_50', 'scar_50_100', 'lt'],
     'order_scar_first_v2': ['bare_0', 'scar_0', 'bare_1_50', 'scar_1_50', 'scar_50_100', 'lt'],
     'reference': 'no catalogue scar within 100 m, no bare ground within 50 m, no LandTrendr disturbance'}
EV = load_main_classes()
COND = lambda x: {'bare_0': x.bare_at_cutoff_0m == 1, 'bare_1_50': x.bare_at_cutoff_50m == 1, 'scar_0': x.class1_0m == 1,
                  'scar_1_50': x.class1_50m == 1, 'scar_50_100': x.class1_100m == 1, 'lt': x.lt_detected == 1}
def rings(order):
    x = EV.copy(); c = COND(x); taken = pd.Series(False, index=x.index)
    for nm in order:
        x[nm] = (c[nm].fillna(False) & ~taken).astype(int); taken = taken | (x[nm] == 1)
    return x
SUMS = {'at_point': ['bare_0', 'scar_0'], 'within_50': ['bare_0', 'bare_1_50', 'scar_0', 'scar_1_50'],
        'within_100': ['bare_0', 'bare_1_50', 'scar_0', 'scar_1_50', 'scar_50_100'],
        'scar_only': ['scar_0', 'scar_1_50', 'scar_50_100'], 'bare_only': ['bare_0', 'bare_1_50']}
for key, order, lab in (('bare_first', R['order_bare_first'], 'rings_barefirst'), ('scar_first_v2', R['order_scar_first_v2'], 'rings')):
    x = rings(order); R[key] = {}
    for rg, sel in RNG.items():
        s = sel(x)
        o = af_model(s, order, f'{lab}:{rg}', sums=SUMS)
        o['cases_by_ring'] = {t: int(s.loc[s.case == 1, t].sum()) for t in order}
        o['controls_by_ring'] = {t: int(s.loc[s.case == 0, t].sum()) for t in order}
        R[key][rg] = o
        print(key, rg, {t: round(o['OR'][t]['OR'], 2) for t in order}, {k: round(o[f'AF_{k}']['point'], 4) for k in SUMS}, round(o['AF_total']['point'], 4), flush=True)
V2 = json.load(open(H + '/v11/an/af_robust_register_v2.json', encoding='utf-8'))['G_rings']
for rg in RNG:
    for t in R['order_scar_first_v2']:
        assert abs(R['scar_first_v2'][rg]['OR'][t]['OR'] - V2[rg]['OR'][t]['OR']) < 1e-9, (rg, t)
    assert abs(R['scar_first_v2'][rg]['AF_total']['point'] - V2[rg]['AF_total']['point']) < 1e-12
for rg in RNG:
    a, b = R['bare_first'][rg], R['scar_first_v2'][rg]
    for t in ('bare_0', 'scar_1_50', 'scar_50_100', 'lt'):
        assert a['cases_by_ring'][t] == b['cases_by_ring'][t] and a['controls_by_ring'][t] == b['controls_by_ring'][t], (rg, t)
    assert sum(a['cases_by_ring'][t] for t in SUMS['within_50']) == sum(b['cases_by_ring'][t] for t in SUMS['within_50'])
    o = a['OR']; assert o['scar_0']['OR'] > o['scar_1_50']['OR'] > o['scar_50_100']['OR'] > 1 and o['bare_0']['OR'] > o['bare_1_50']['OR'] > 1, rg
R['fallback_bfgs'] = FALLBACK
json.dump(R, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
print('saved', OUT)
