"""Attributable fractions with matched catalogue look-back windows in both periods, by scar age and per event, event bootstrap."""
import json, sys, time, numpy as np, pandas as pd
sys.path.insert(0, '@WORK@/v11/code')
from afcore_v1 import *
OUT = H + '/v11/an/af_window_register_v2.json'
REG = json.load(open(H + '/reference/reference_values.json', encoding='utf-8'))
R = {'status': "exploratory, written 2026-10-29 after all charter results at the author's request; not in any charter",
     'seed_base': SEED, 'ndraw': NDRAW, 'nboot': NBOOT, 'catalogue_start': str(CAT0.date())}
M, B = load_shared()
PER = {'main': M, 'rep': B, 'rep_excl2009': B[~B.y2009].copy(), 'rep_2009': B[B.y2009].copy()}
R['lookback_years'] = {k: {'min': float(d.lookback_event.min()), 'max': float(d.lookback_event.max())} for k, d in PER.items()}
R['max_scar_age_years'] = {k: float(d.loc[d.R1 == 1, 'scar_age'].max()) for k, d in PER.items() if (d.R1 == 1).any()}
R['landsat_years_before_cutoff'] = {k: {'min': int((d.cutoff.dt.year - 1988).min()), 'max': int((d.cutoff.dt.year - 1988).max())} for k, d in (('main', M), ('rep', B))}
R['rep_2009_share'] = {rg: {'events': int(sel(B)[sel(B).y2009].event.nunique()), 'cases_2009': int(sel(B)[sel(B).y2009].case.sum()),
                            'cases_rep': int(sel(B).case.sum()), 'share_cases': float(sel(B)[sel(B).y2009].case.sum() / sel(B).case.sum())} for rg, sel in RNG.items()}
R['rep_2009_events'] = sorted(B[B.y2009].event.unique().tolist())
print('2009 share', R['rep_2009_share'], flush=True)

def window(d, W, cap=None):
    x = d.copy()
    if W is None: x['R1w'] = x.R1
    else: x['R1w'] = ((x.R1 == 1) & (x.scar_age < W)).astype(int)
    lt = (x.lt_detected == 1) if cap is None else ((x.lt_detected == 1) & (x.TSD < cap))
    x['R3w'] = (lt & (x.R1w == 0)).astype(int)
    return x

for k in ('main', 'rep'):
    m, mr, dd, nd = fitfe(PER[k][PER[k].slope <= 39.5], f"case~{TERR}+R1+R3")
    reg = REG['R5.main' if k == 'main' else 'R5.rep']['value']['MR2_le39.5']
    assert abs(term(m, mr, 'R1')['OR'] - reg['R1']['OR']) < 2e-3 and abs(term(m, mr, 'R3')['OR'] - reg['R3']['OR']) < 2e-3, k

WIN = {}
for lab, W, cap in (('full', None, None), ('W4', 4.0, None), ('W10', 10.0, None), ('W4_lt20', 4.0, 20.0)):
    WIN[lab] = {}
    for k, d in PER.items():
        if lab == 'W10' and k in ('rep_excl2009', 'rep_2009'): continue
        x = window(d, W, cap)
        if lab == 'W10': x = x[x.lookback_event >= W]
        WIN[lab][k] = {}
        for rg, sel in RNG.items():
            s = sel(x); o = af_model(s, ['R1w', 'R3w'], f'window:{lab}:{k}:{rg}'); o['events_kept'] = int(s.event.nunique())
            WIN[lab][k][rg] = o
            print(lab, k, rg, round(o['AF_total']['point'], 4), [round(v, 4) for v in o['AF_total']['ci_mc']], flush=True)
WIN['W4']['note'] = 'all events in both periods (every event sees at least 4.35 years of catalogue)'
WIN['W10']['note'] = '2009-2017 events kept only if the earliest cutoff is at least 10 years after the catalogue start (not reported in v11)'
WIN['W4_lt20']['note'] = '4-year catalogue window and R3 only for LandTrendr disturbances less than 20 years before the cutoff'
R['windows'] = WIN
for k in ('main', 'rep'):
    assert abs(WIN['full'][k]['le39.5']['AF_total']['point'] - REG['R5.main' if k == 'main' else 'R5.rep']['value']['AF_le39.5']['AF_total']['point']) < 2e-3

AGE = {}
for k, d in PER.items():
    x = d.copy()
    x['s05'] = ((x.R1 == 1) & (x.scar_age < 5)).astype(int)
    x['s510'] = ((x.R1 == 1) & (x.scar_age >= 5) & (x.scar_age < 10)).astype(int)
    x['s10p'] = ((x.R1 == 1) & (x.scar_age >= 10)).astype(int)
    AGE[k] = {}
    for rg, sel in RNG.items():
        s = sel(x); terms = [t for t in ('s05', 's510', 's10p') if s.loc[s.case == 1, t].sum() > 0] + ['R3']
        o = af_model(s, terms, f'age:{k}:{rg}', sums={'s0_10': ['s05', 's510'], 'scar_all': [t for t in terms if t != 'R3']})
        o['cases_by_bin'] = {t: int(s.loc[s.case == 1, t].sum()) for t in ('s05', 's510', 's10p', 'R3')}
        o['terms'] = terms; AGE[k][rg] = o
        print('age', k, rg, {t: round(o[f'AF_{t}']['point'], 4) for t in terms}, round(o['AF_total']['point'], 4), flush=True)
AGE['note'] = ('2009-2017: the two 2009 Morakot entries (cutoffs from 2008-11) see 4.35 and 4.52 years of catalogue, so their '
               '0-5 yr bin is truncated at the look-back and their 5-10 and >= 10 yr bins are empty')
AGE['events_lookback_lt5_rep'] = {'events': int(B.loc[B.lookback_event < 5, 'event'].nunique()), 'cases': int(B.loc[(B.lookback_event < 5) & (B.case == 1)].shape[0])}
R['by_scar_age'] = AGE

EVT = []
for k, d in (('main', M), ('rep', B)):
    nc = d.groupby('event').case.sum()
    for e in nc[nc >= 100].index:
        s = d[d.event == e]
        m = smf.logit(f"case~{TERR}+R1+R3", data=s).fit(disp=0, maxiter=100, method='newton')
        mr = smf.logit(f"case~{TERR}+R1+R3", data=s).fit(disp=0, maxiter=100, method='newton', cov_type='cluster', cov_kwds={'groups': pd.factorize(s.uid)[0]})
        assert m.mle_retvals.get('converged', True), e
        b = m.params[['R1', 'R3']].values; V = mr.cov_params().loc[['R1', 'R3'], ['R1', 'R3']].values
        o = af_from(b, V, s, ['R1', 'R3'], f'event:{k}:{e}')
        o.update({'period': k, 'event': e, 'lookback': float(s.lookback_event.iloc[0]), 'y2009': bool(str(e).startswith('2009_')),
                  'OR': {t: term(m, mr, t) for t in ('R1', 'R3')}})
        EVT.append(o); print('event', k, e, o['n_cases'], round(o['AF_total']['point'], 4), flush=True)
R['per_event'] = {'note': 'events with at least 100 broadleaf cases, all slopes, one logit per event (terrain + R1 + R3)', 'events': EVT,
                  'range': {k: [min(o['AF_total']['point'] for o in EVT if o['period'] == k), max(o['AF_total']['point'] for o in EVT if o['period'] == k)] for k in ('main', 'rep')},
                  'range_rep_excl2009': [min(o['AF_total']['point'] for o in EVT if o['period'] == 'rep' and not o['y2009']), max(o['AF_total']['point'] for o in EVT if o['period'] == 'rep' and not o['y2009'])]}

BOOT = {}; T0 = time.time()
for lab, W in (('full', None), ('W4', 4.0)):
    BOOT[lab] = {}; arrs = {}
    for k in ('main', 'rep', 'rep_excl2009'):
        BOOT[lab][k] = {}
        for rg, sel in RNG.items():
            x = sel(window(PER[k], W)); f = FastFE(x, ['R1w', 'R3w']); bl, ok = f.fit_all(); assert ok
            pt = float(f.af_point(f.X, f.y, bl).sum())
            assert abs(pt - WIN[lab][k][rg]['AF_total']['point']) < 1e-6, (lab, k, rg, pt)
            a, fails = f.boot(f'{lab}:{k}:{rg}'); tot = a.sum(1); arrs[(k, rg)] = tot
            BOOT[lab][k][rg] = {'point': pt, 'ci_boot': [float(np.percentile(tot, 2.5)), float(np.percentile(tot, 97.5))], 'n_ok': int(len(tot)), 'fails': fails, 'events': f.K}
            print('boot', lab, k, rg, round(pt, 4), [round(v, 4) for v in BOOT[lab][k][rg]['ci_boot']], fails, round(time.time() - T0), 's', flush=True)
    for rg in RNG:
        for other in ('rep', 'rep_excl2009'):
            a, b = arrs[('main', rg)], arrs[(other, rg)]; n = min(len(a), len(b)); dlt = a[:n] - b[:n]
            BOOT[lab][f'diff_main_minus_{other}_{rg}'] = {'point': BOOT[lab]['main'][rg]['point'] - BOOT[lab][other][rg]['point'],
                                                          'ci_boot': [float(np.percentile(dlt, 2.5)), float(np.percentile(dlt, 97.5))], 'n': int(n)}
R['bootstrap'] = BOOT
R['fallback_bfgs'] = FALLBACK
json.dump(R, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
print('written', OUT, round(time.time() - T0), 's')
