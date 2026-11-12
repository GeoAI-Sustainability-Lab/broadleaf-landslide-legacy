"""Both periods with the shared definitions: R1 odds ratios, gradients and attributable fractions, R2 inventory look-back windows, R3 without the 2009 inventory, R4 geologically sensitive areas and temporal transfer. Writes rep27_<TAGS>.json."""
import os, json, math, time
import numpy as np, pandas as pd
from lib27 import *
TAG = os.environ.get('TAGS', 'v5'); DOBOOT = os.environ.get('BOOT', '1') == '1'; REPV = 'v1' if TAG == 'v4' else 'v5'
OUTD = f'{H}/v27/an/rep_{TAG}'; os.makedirs(OUTD, exist_ok=True)
F = json.load(open(H + '/reference/reference_values_s9.json', encoding='utf-8')); FV = lambda k: F[k]
T0 = time.time()
def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)
def cached(name, fn):
    p = f'{OUTD}/{name}.json'
    if os.path.exists(p): return json.load(open(p, encoding='utf-8'))
    r = clean(fn()); json.dump(r, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1); log('saved', name); return r
def close(a, b, tol=1e-6): return abs(a - b) < tol
def MM(d, terms, label, covs=TV, boot=None): return main_model(d, terms, label, covs=covs, boot=(DOBOOT if boot is None else boot))
def strip(o): return {k: v for k, v in o.items() if not k.startswith('_')}
def comb(o, coefs):
    m, mr, dd = o['_m']; r = lin(m, mr, coefs)
    if '_boot' in o:
        terms = list(o['OR'].keys()); be = o['_boot']['beta']; v = sum(w * be[:, terms.index(t)] for t, w in coefs.items())
        r['ci_boot'] = [math.exp(x) for x in pct(v)]; r['_draws'] = v
    return r
CAT0 = pd.Timestamp('2004-07-10')
E = load_event(TAG); BR = pd.read_csv(BRIGHT, dtype={'uid': str})
M = E[E.ft == BRO].merge(BR[['uid', 'event', 'Qt', 'score_terrain']], on=['uid', 'event'], how='left')
M['R1'] = (M.class1_50m == 1).astype(int); M['R3'] = ((M.lt_detected == 1) & (M.R1 == 0)).astype(int); M['legacy'] = ((M.R1 == 1) | (M.R3 == 1)).astype(int); M['R0'] = 1 - M.legacy
M['scar_age'] = M.scar_50m_min_yrs
M['scar_bin'] = np.select([M.R1 == 0, M.scar_age < 5, M.scar_age < 10], ['none', '0-5', '5-10'], '10+')
M['tsd_bin'] = np.select([M.R3 == 0, M.TSD < 5, M.TSD < 10, M.TSD < 20], ['none', '0-5', '5-10', '10-20'], '20+')
M['cutoff_dt'] = pd.to_datetime(M.cutoff)
Rp = pd.read_csv(REPS[REPV], dtype={'uid': str}, low_memory=False, parse_dates=['cutoff'])
Rp['case'] = Rp.case.astype(int); Rp['slope2'] = Rp.slope ** 2; Rp['s20'] = Rp.slope - 20; Rp = Rp.dropna(subset=['slope', 'north', 'east', 'planc', 'profc', 'elev'])
Rp['cutoff_dt'] = Rp.cutoff; Rp['scar_bin'] = Rp.scar_bin.astype(str); Rp['tsd_bin'] = Rp.tsd_bin.astype(str)
for d in (M, Rp): d['lookback_event'] = (d.groupby('event').cutoff_dt.transform('min') - CAT0).dt.days / 365.25
Rp['y2009'] = Rp.event.str.startswith('2009_')
PER = {'main': M, 'rep': Rp}
def dummies(d, col, cats, prefix):
    d = d.copy(); names = []
    for c in cats:
        n = f'{prefix}_{c}'.replace('-', '_').replace('+', 'p'); d[n] = (d[col].astype(str) == c).astype(int); names.append(n)
    return d, names
def period_block(Bk, lab):
    o = {'n_rows': int(len(Bk)), 'n_cases': int(Bk.case.sum()), 'n_control_points': int(Bk[Bk.case == 0].uid.nunique()), 'n_events': int(Bk.event.nunique()),
         'n_cases_le39.5': int(((Bk.case == 1) & (Bk.slope <= 39.5)).sum()), 'n_control_rows_le39.5': int(((Bk.case == 0) & (Bk.slope <= 39.5)).sum())}
    for rg, sel in RNG.items():
        x = sel(Bk); o[f'composition_{rg}'] = {'cases': {c: float(x[x.case == 1][c].mean()) for c in ('R1', 'R3', 'R0')}, 'controls': {c: float(x[x.case == 0][c].mean()) for c in ('R1', 'R3', 'R0')}}
        a = MM(x, ['legacy'], f'{lab}:MR1:{rg}'); o[f'MR1_{rg}'] = strip(a); o[f'_MR1_{rg}'] = a
        b = MM(x, ['R1', 'R3'], f'{lab}:MR2:{rg}'); o[f'MR2_{rg}'] = strip(b); o[f'_MR2_{rg}'] = b
    x, ts = dummies(Bk, 'scar_bin', ['0-5', '5-10', '10+'], 'sb'); ts = [t for t in ts if x.loc[x.case == 1, t].sum() > 0 and x.loc[x.case == 0, t].sum() > 0]
    a = MM(x, ts + ['R3'], f'{lab}:MR3a'); o['MR3a'] = strip(a); o['_MR3a'] = a
    o['MR3a_counts'] = {t: {'cases': int(x.loc[x.case == 1, t].sum()), 'controls': int(x.loc[x.case == 0, t].sum())} for t in ts}
    if 'sb_10p' in ts: o['MR3a_05_vs_10p'] = {k: v for k, v in comb(a, {'sb_0_5': 1, 'sb_10p': -1}).items() if k != '_draws'}
    x, tt = dummies(Bk, 'tsd_bin', ['0-5', '5-10', '10-20', '20+'], 'tb')
    a = MM(x, ['R1'] + tt, f'{lab}:MR3b'); o['MR3b'] = strip(a); o['_MR3b'] = a
    o['MR3b_counts'] = {t: {'cases': int(x.loc[x.case == 1, t].sum()), 'controls': int(x.loc[x.case == 0, t].sum())} for t in tt}
    x = Bk.copy(); x['legacy_s20'] = x.legacy * x.s20
    a = MM(x, ['legacy', 'legacy_s20'], f'{lab}:MR4'); o['MR4'] = {k: v for k, v in comb(a, {'legacy_s20': 1}).items() if k != '_draws'}; o['_MR4'] = a
    X = Bk.copy(); qs = sorted(X.Qt.astype(int).unique()); top = qs[-1]
    o['Q_composition'] = {str(q): {'R0_cases': float(X[(X.Qt == q) & (X.case == 1)].R0.mean()), 'R0_controls': float(X[(X.Qt == q) & (X.case == 0)].R0.mean()),
                                   'n_cases': int(((X.Qt == q) & (X.case == 1)).sum())} for q in qs}
    for q in qs[1:]: X[f'q{q}'] = (X.Qt.astype(int) == q).astype(int)
    X['legacy_qrank'] = X.legacy * (X.Qt.astype(int) - 1)
    a = MM(X, ['legacy', 'legacy_qrank'], f'{lab}:B2trend', covs=TV + [f'q{q}' for q in qs[1:]]); o['B2_trend'] = {k: v for k, v in comb(a, {'legacy_qrank': 1}).items() if k != '_draws'}; o['_B2'] = a
    cats = [f'{q}_{s}' for q in qs for s in ('intact', 'legacy')]
    X['cellq'] = X.Qt.astype(int).astype(str) + np.where(X.legacy == 1, '_legacy', '_intact')
    for c in cats[1:]: X['cq_' + c] = (X.cellq == c).astype(int)
    a = MM(X, ['cq_' + c for c in cats[1:]], f'{lab}:B3', covs=[])
    o['B3'] = {k: {kk: vv for kk, vv in comb(a, cf).items() if kk != '_draws'} for k, cf in (('5_intact_vs_1_legacy', {'cq_5_intact': 1, 'cq_1_legacy': -1}),
                                                                                       ('5_intact_vs_3_legacy', {'cq_5_intact': 1, 'cq_3_legacy': -1}),
                                                                                       ('5_legacy_vs_5_intact', {'cq_5_legacy': 1, 'cq_5_intact': -1}))}
    return o
def blocks():
    return {k: period_block(d, k) for k, d in PER.items()}
PB = None
p = f'{OUTD}/R1_periods.json'
if os.path.exists(p): RES = json.load(open(p, encoding='utf-8'))
else:
    PB = blocks(); RES = {'periods': {k: {kk: vv for kk, vv in v.items() if not kk.startswith('_')} for k, v in PB.items()}}
    def ratio(a, b, key):
        ra, rb = a['OR'][key], b['OR'][key]; dlt = ra['beta'] - rb['beta']; se = math.sqrt(ra['se'] ** 2 + rb['se'] ** 2)
        out = {'ratio': math.exp(dlt), 'ci': [math.exp(dlt - 1.96 * se), math.exp(dlt + 1.96 * se)]}
        if '_boot' in a and '_boot' in b:
            ia = list(a['OR']).index(key); ib = list(b['OR']).index(key); n = min(a['_boot']['n_ok'], b['_boot']['n_ok'])
            out['ci_boot'] = [math.exp(v) for v in pct(a['_boot']['beta'][:n, ia] - b['_boot']['beta'][:n, ib])]
        return out
    Pm, Pr = PB['main'], PB['rep']; RT = {}
    for rg in RNG:
        RT[f'MR1_{rg}'] = ratio(Pr[f'_MR1_{rg}'], Pm[f'_MR1_{rg}'], 'legacy')
        for c in ('R1', 'R3'): RT[f'MR2_{rg}_{c}'] = ratio(Pr[f'_MR2_{rg}'], Pm[f'_MR2_{rg}'], c)
    for t in ('sb_0_5', 'sb_5_10', 'sb_10p'):
        if t in Pr['_MR3a']['OR'] and t in Pm['_MR3a']['OR']: RT[t] = ratio(Pr['_MR3a'], Pm['_MR3a'], t)
    for t in ('tb_0_5', 'tb_5_10', 'tb_10_20', 'tb_20p'): RT[t] = ratio(Pr['_MR3b'], Pm['_MR3b'], t)
    RT['MR4_per_degree'] = ratio(Pr['_MR4'], Pm['_MR4'], 'legacy_s20'); RT['B2_trend'] = ratio(Pr['_B2'], Pm['_B2'], 'legacy_qrank')
    RES['ratios_rep_over_main'] = RT
    RES = clean(RES); json.dump(RES, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1); log('saved R1_periods')
if TAG == 'v4':
    R5m = FV('R5.main'); R5r = FV('R5.rep')
    for per, reg in (('main', R5m), ('rep', R5r)):
        P_ = RES['periods'][per]
        assert close(P_['MR1_all']['OR']['legacy']['OR'], reg['MR1_all']['legacy']['OR'], 1e-3), per
        assert close(P_['MR2_le39.5']['OR']['R1']['OR'], reg['MR2_le39.5']['R1']['OR'], 1e-3), per
        assert close(P_['MR4']['OR'], reg['MR4_per_degree']['OR'], 1e-3), per
        assert close(P_['B2_trend']['OR'], reg['bright_terrain_only']['B2_trend_per_stratum_step']['OR'], 1e-3), per
        assert close(P_['MR2_le39.5']['AF_total']['point'], reg['AF_le39.5']['AF_total']['point'], 2e-3), per
    log('R1 validated against R5.main and R5.rep')
log('R1', {k: (round(v['MR1_all']['OR']['legacy']['OR'], 3), round(v['MR2_le39.5']['AF_total']['point'], 4), v['MR2_le39.5']['AF_total'].get('ci_boot')) for k, v in RES['periods'].items()})

def window(d, W):
    x = d.copy(); x['R1w'] = x.R1 if W is None else ((x.R1 == 1) & (x.scar_age < W)).astype(int)
    x['R3w'] = ((x.lt_detected == 1) & (x.R1w == 0)).astype(int); return x
def w2():
    SUB = {'main': M, 'rep': Rp, 'rep_excl2009': Rp[~Rp.y2009], 'rep_2009': Rp[Rp.y2009]}
    W = {}; draws = {}
    for lab, Wy in (('full', None), ('W4', 4.0)):
        W[lab] = {}
        for k, d in SUB.items():
            W[lab][k] = {}
            for rg, sel in RNG.items():
                o = MM(sel(window(d, Wy)), ['R1w', 'R3w'], f'win:{lab}:{k}:{rg}'); W[lab][k][rg] = strip(o)
                if '_boot' in o: draws[(lab, k, rg)] = af_point(o['_boot']['rho'], o['_boot']['beta']).sum(1)
        for rg in RNG:
            for other in ('rep', 'rep_excl2009'):
                if (lab, 'main', rg) in draws and (lab, other, rg) in draws:
                    a, b = draws[(lab, 'main', rg)], draws[(lab, other, rg)]; n = min(len(a), len(b))
                    W[lab][f'diff_main_minus_{other}_{rg}'] = {'point': W[lab]['main'][rg]['AF_total']['point'] - W[lab][other][rg]['AF_total']['point'], 'ci_boot': pct(a[:n] - b[:n])}
    AGE = {}
    for k, d in SUB.items():
        x = d.copy(); x['s05'] = ((x.R1 == 1) & (x.scar_age < 5)).astype(int); x['s510'] = ((x.R1 == 1) & (x.scar_age >= 5) & (x.scar_age < 10)).astype(int); x['s10p'] = ((x.R1 == 1) & (x.scar_age >= 10)).astype(int)
        AGE[k] = {}
        for rg, sel in RNG.items():
            s = sel(x); terms = [t for t in ('s05', 's510', 's10p') if s.loc[s.case == 1, t].sum() > 0 and s.loc[s.case == 0, t].sum() > 0] + ['R3']
            o = MM(s, terms, f'age:{k}:{rg}', boot=False); AGE[k][rg] = {'terms': terms, **{f'AF_{t}': o[f'AF_{t}'] for t in terms}, 'AF_total': o['AF_total'],
                                                                          'cases_by_bin': {t: int(s.loc[s.case == 1, t].sum()) for t in ('s05', 's510', 's10p', 'R3')}}
    share = {}
    for rg, sel in RNG.items():
        s = sel(Rp); share[rg] = {'cases_2009': int(s[s.y2009].case.sum()), 'cases_rep': int(s.case.sum()), 'share': float(s[s.y2009].case.sum() / s.case.sum()),
                                  'events_2009': int(s[s.y2009].event.nunique())}
        nc = s[s.case == 1].groupby('event').size(); share[rg]['largest_event'] = nc.idxmax(); share[rg]['largest_share'] = float(nc.max() / nc.sum())
        nm = sel(M)[sel(M).case == 1].groupby('event').size(); share[rg]['main_largest_event'] = nm.idxmax(); share[rg]['main_largest_share'] = float(nm.max() / nm.sum())
    lb = {k: {'min': float(d.lookback_event.min()), 'max': float(d.lookback_event.max())} for k, d in (('main', M), ('rep', Rp))}
    return {'windows': W, 'by_scar_age': AGE, 'rep_2009': share, 'lookback_years': lb}
W2 = cached('R2_windows', w2)
if TAG == 'v4':
    W1W4 = FV('W1.window_W4'); W1F = FV('W1.window_full')
    assert close(W2['windows']['W4']['main']['le39.5']['AF_total']['point'], W1W4['main']['le39.5']['AF_total']['point'], 1e-9)
    assert close(W2['windows']['full']['rep_2009']['le39.5']['AF_total']['point'], W1F['rep_2009']['le39.5']['AF_total']['point'], 1e-9)
    log('R2 validated against W1 windows')
log('R2', {lab: {k: round(W2['windows'][lab][k]['le39.5']['AF_total']['point'], 4) for k in ('main', 'rep', 'rep_excl2009', 'rep_2009')} for lab in ('full', 'W4')})

def r3():
    X = Rp[~Rp.y2009].copy(); X['legacy_s20'] = X.legacy * X.s20
    a = MM(X, ['legacy', 'legacy_s20'], 'excl2009:MR4'); o = {'MR4': {k: v for k, v in comb(a, {'legacy_s20': 1}).items() if k != '_draws'}}
    qs = sorted(X.Qt.astype(int).unique())
    for q in qs[1:]: X[f'q{q}'] = (X.Qt.astype(int) == q).astype(int)
    X['legacy_qrank'] = X.legacy * (X.Qt.astype(int) - 1)
    a = MM(X, ['legacy', 'legacy_qrank'], 'excl2009:B2', covs=TV + [f'q{q}' for q in qs[1:]]); o['B2_trend'] = {k: v for k, v in comb(a, {'legacy_qrank': 1}).items() if k != '_draws'}
    return o
R3 = cached('R3_excl2009', r3)
if TAG == 'v4':
    assert close(R3['MR4']['OR'], FV('R5.explore')['E1_rep_excl_2009']['slope']['per_degree']['OR'], 1e-6); log('R3 validated')
log('R3', round(R3['MR4']['OR'], 4), R3['MR4'].get('ci_boot'), round(R3['B2_trend']['OR'], 4))

ZF = H + '/g5/zoning_out_v1.csv'
def r4():
    Z = pd.read_csv(ZF, dtype={'uid': str}); o = {}
    X = Rp.merge(Z, on='uid', how='left'); X['zin'] = X.gsmma_inside.fillna(0).astype(int); L = X[X.slope <= 39.5].copy(); L['leg_zin'] = L.legacy * L.zin
    a = MM(L, ['legacy', 'zin', 'leg_zin'], 'rep_gsa'); o['rep_legacy_OR_outside'] = {k: v for k, v in comb(a, {'legacy': 1}).items() if k != '_draws'}
    def strat_auc(d, s):
        num = 0.0; den = 0.0
        for ev, g in d.assign(s=s).groupby('event'):
            y = g.case.values; n1 = y.sum(); n0 = len(y) - n1
            if n1 == 0 or n0 == 0: continue
            r = pd.Series(g.s.values).rank().values; a_ = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0); num += a_ * n1 * n0; den += n1 * n0
        return num / den
    C7 = {}
    for nm, sel in RNG.items():
        A_ = sel(M); B_ = sel(Rp); oo = {}
        for tr_lab, tr, te_lab, te in (('2018_2025', A_, '2009_2017', B_), ('2009_2017', B_, '2018_2025', A_)):
            for mod, f in (('terrain', ''), ('terrain+legacy', '+R1+R3')):
                dtr, _ = drop_uninf(tr); mm = smf.logit(f"case~{TERR}{f}+C(event)", data=dtr).fit(disp=0, maxiter=100, method='newton')
                cols = TV + (['R1', 'R3'] if f else []); s_te = te[cols].values @ mm.params[cols].values
                oo[f'{tr_lab}_to_{te_lab}_{mod}'] = strat_auc(te, s_te)
        for a_ in ('2018_2025_to_2009_2017', '2009_2017_to_2018_2025'): oo[f'{a_}_gain'] = oo[f'{a_}_terrain+legacy'] - oo[f'{a_}_terrain']
        C7[nm] = oo
    o['transfer'] = C7
    return o
R4 = cached('R4_gsa_transfer', r4)
if TAG == 'v4':
    assert close(R4['rep_legacy_OR_outside']['OR'], FV('R5.Z')['rep']['legacy_OR_outside']['OR'], 1e-3)
    MC = json.load(open(H + '/v11/unround/methods/methods_register_v1.json', encoding='utf-8'))['C7_transfer']
    assert close(R4['transfer']['all']['2018_2025_to_2009_2017_gain'], MC['all']['2018_2025_to_2009_2017_gain'], 1e-3); log('R4 validated (GSA outside, transfer)')
log('R4', round(R4['rep_legacy_OR_outside']['OR'], 3), {k: round(v['2018_2025_to_2009_2017_gain'], 4) for k, v in R4['transfer'].items()})
json.dump({'tags': TAG, 'boot': DOBOOT, 'R1': RES, 'R2': W2, 'R3': R3, 'R4': R4, 'fallback': FALLBACK}, open(f'{H}/v27/an/rep27_{TAG}.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
log('done', TAG)
