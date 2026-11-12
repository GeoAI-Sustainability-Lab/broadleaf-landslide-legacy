"""2018-2025 events, sections S0 to S12 (sample, main model, sensitivities, distance rings, time structure, canopy height, slope, susceptibility, bamboo contrast, stand origin, geologically sensitive areas, random effects, Sentinel-2). TAGS=v5 uses one cutoff date per event, TAGS=v4 the earlier cutoff rule; BOOT=0 skips the event-level bootstrap. Writes main27_<TAGS>.json."""
import os, json, math, time, sys
import numpy as np, pandas as pd
from scipy.stats import chi2, t as tdist
from scipy.spatial import cKDTree
from lib27 import *
TAG = os.environ.get('TAGS', 'v5'); DOBOOT = os.environ.get('BOOT', '1') == '1'
OUTD = f'{H}/v27/an/main_{TAG}'; os.makedirs(OUTD, exist_ok=True)
F = json.load(open(H + '/reference/reference_values_s9.json', encoding='utf-8')); FV = lambda k: F[k]
T0 = time.time()
def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)
E = load_event(TAG); B = E[E.ft == BRO].copy()
EQ = pd.Timestamp('2024-04-03')
def cached(name, fn):
    p = f'{OUTD}/{name}.json'
    if os.path.exists(p): return json.load(open(p, encoding='utf-8'))
    r = clean(fn()); json.dump(r, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1); log('saved', name); return r
def close(a, b, tol=1e-6): return abs(a - b) < tol
def MM(d, terms, label, covs=TV, boot=None):
    return main_model(d, terms, label, covs=covs, boot=(DOBOOT if boot is None else boot))
def comb(o, coefs, key='lin'):
    """linear combination of terms: cluster-robust (lin) and bootstrap percentile."""
    m, mr, dd = o['_m']; r = lin(m, mr, coefs)
    if '_boot' in o:
        terms = list(o['OR'].keys()); be = o['_boot']['beta']
        v = sum(w * be[:, terms.index(t)] for t, w in coefs.items()); r['ci_boot'] = [math.exp(x) for x in pct(v)]
    return r
def strip(o): return {k: v for k, v in o.items() if not k.startswith('_')}
def classes_from(c1, c2, lt):
    sc = np.where(c2 == 1, 2, np.where(c1 == 1, 1, 0)); return np.where((sc == 0) & (lt == 1), 3, sc)
def set_classes(d, sc):
    d = d.copy(); d['site_class4'] = np.asarray(sc)
    for c in (1, 2, 3): d[f'sc{c}'] = (d.site_class4 == c).astype(int)
    d['legacy'] = (d.site_class4 > 0).astype(int); return d

def s0():
    R = {}
    for rg, sel in RNG.items():
        d, nd = drop_uninf(sel(B)); ca = d[d.case == 1]; co = d[d.case == 0]
        R[rg] = {'cases': int(len(ca)), 'control_rows': int(len(co)), 'control_points': int(co.uid.nunique()), 'events': int(d.event.nunique()), 'events_dropped': nd,
                 'share_cases': {str(c): float((ca.site_class4 == c).mean()) for c in range(4)}, 'share_controls': {str(c): float((co.site_class4 == c).mean()) for c in range(4)},
                 'legacy_share_cases': float((ca.site_class4 > 0).mean()), 'legacy_share_controls': float((co.site_class4 > 0).mean()),
                 'all_rows': {'cases': int((sel(B).case == 1).sum()), 'control_rows': int((sel(B).case == 0).sum())}}
        nc = ca.groupby('event').size(); yr = ca.groupby('year').size(); pe = d.groupby('event').case.agg(['sum', 'size']); rr = (pe['size'] - pe['sum']) / pe['sum']
        R[rg].update({'effective_events': float(nc.sum() ** 2 / (nc ** 2).sum()), 'events_ge50': int((nc >= 50).sum()), 'share_cases_events_ge50': float(nc[nc >= 50].sum() / nc.sum()),
                      'largest_event_share': float(nc.max() / nc.sum()), 'share_cases_2024_2025': float(yr[yr.index >= 2024].sum() / yr.sum()),
                      'controls_per_case': float(len(co) / len(ca)), 'controls_per_case_range_ge50': [float(rr[pe['sum'] >= 50].min()), float(rr[pe['sum'] >= 50].max())]})
    if TAG == 'v5':
        cs = B[B.case == 1].copy(); cs['cutoff_own'] = pd.to_datetime(cs.cutoff_own); cs['cutoff_event'] = pd.to_datetime(cs.cutoff_event)
        lag = (cs.cutoff_own - cs.cutoff_event).dt.days
        R['cutoff_lag'] = {'n_cases': int(len(cs)), 'share_zero': float((lag == 0).mean()), 'median_days': float(lag.median()), 'q25': float(lag.quantile(.25)), 'q75': float(lag.quantile(.75)), 'max_days': int(lag.max())}
        c24 = cs[cs.year == 2024]
        R['eq2024'] = {'cases_2024': int(len(c24)), 'image_not_after_quake': int((c24.cutoff_own <= EQ).sum()), 'image_after_quake': int((c24.cutoff_own > EQ).sum()),
                       'events_2024': int(B[B.year == 2024].event.nunique()), 'events_2024_cutoff_before_quake': int((B[B.year == 2024].groupby('event').cutoff_event.first().pipe(pd.to_datetime) <= EQ).sum())}
    X = with_arows(B[B.case == 1])
    for rg, sel in RNG.items():
        x = sel(X); a = {}
        for c in range(4):
            v = x[x.site_class4 == c].area_ha.dropna(); a[str(c)] = {'n': int(len(v)), 'median_ha': float(v.median()), 'q25': float(v.quantile(.25)), 'q75': float(v.quantile(.75))}
        v = x.area_ha.dropna(); a['all'] = {'n': int(len(v)), 'median_ha': float(v.median()), 'q25': float(v.quantile(.25)), 'q75': float(v.quantile(.75)), 'min_ha': float(v.min()), 'share_lt_0.1ha': float((v < 0.1).mean())}
        R[rg]['case_area'] = a
    return R
S0 = cached('S0_sample', s0); log('S0', {k: (v['cases'], v['control_rows'], round(v['legacy_share_cases'], 4), round(v['legacy_share_controls'], 4)) for k, v in S0.items() if k in RNG})

def s1():
    R = {}
    for rg, sel in RNG.items():
        o = MM(sel(B), ['sc1', 'sc2', 'sc3'], f'main:{rg}')
        o['bare_vs_scar'] = comb(o, {'sc2': 1, 'sc1': -1})
        a = MM(sel(B), ['legacy'], f'any:{rg}')
        R[rg] = {'classes': strip(o), 'any_legacy': strip(a)}
    return R
S1 = cached('S1_main', s1)
if TAG == 'v4':
    for rg, key in (('all', 'all'), ('le39.5', 'le39_5')):
        for c in (1, 2, 3):
            reg = FV(f'M2c.event.{key}.sc{c}'); new = S1[rg]['classes']['OR'][f'sc{c}']
            assert close(new['OR'], reg['OR']) and all(close(a, b) for a, b in zip(new['ci'], reg['ci'])), (rg, c)
        assert close(S1[rg]['any_legacy']['OR']['legacy']['OR'], FV(f'MC1.{key}.any_legacy')['fe']['legacy']['OR'])
    log('S1 validated against M2c and MC1')
log('S1', {rg: ({t: round(v['OR'], 3) for t, v in S1[rg]['classes']['OR'].items()}, round(S1[rg]['classes']['AF_total']['point'], 4), S1[rg]['classes']['AF_total'].get('ci_boot')) for rg in RNG})

def s2():
    R = {}
    def row(d_all, d_le, lab, covs=TV, boot=None):
        o = {}
        for rg, d in (('all', d_all), ('le39.5', d_le)):
            if d is None: continue
            x = MM(d, ['sc1', 'sc2', 'sc3'], f'sens:{lab}:{rg}', covs=covs, boot=boot); o[rg] = strip(x)
        return o
    R['radius0'] = row(*[set_classes(sel(B), classes_from(sel(B).class1_0m.values, sel(B).bare_at_cutoff_0m.values, sel(B).lt_detected.values)) for sel in RNG.values()], 'r0')
    R['radius100'] = row(*[set_classes(sel(B), classes_from(sel(B).class1_100m.values, sel(B).bare_at_cutoff_50m.values, sel(B).lt_detected.values)) for sel in RNG.values()], 'r100')
    R['upper31.4'] = row(None, B[B.slope <= 31.4], 'u314'); R['upper35.8'] = row(None, B[B.slope <= 35.8], 'u358')
    xd = B.copy(); xd['lroad'] = np.log1p(xd.d_road); xd['lriver'] = np.log1p(xd.d_river)
    R['road_river'] = row(xd, xd[xd.slope <= 39.5], 'roadriver', covs=TV + ['lroad', 'lriver'])
    if TAG == 'v5':
        keep = ~((B.case == 1) & (B.year == 2024) & (pd.to_datetime(B.cutoff_own) <= EQ))
        R['eq_post_image'] = row(B[keep], B[keep & (B.slope <= 39.5)], 'eqpost')
    R['pre2024'] = row(B[B.year <= 2023], B[(B.year <= 2023) & (B.slope <= 39.5)], 'pre2024')
    L = {}
    for rg, sel in RNG.items():
        d, _ = drop_uninf(sel(B)); res = []
        for ev in sorted(d.event.unique()):
            x = d[d.event != ev]; bt = Boot(x, TV, ['sc1', 'sc2', 'sc3']); bl = bt.fit_all()
            rho = bt.X[bt.y == 1][:, -3:].mean(0); res.append((ev, float(af_point(rho, bl).sum()), [math.exp(v) for v in bl]))
        afs = [r_[1] for r_ in res]
        L[rg] = {'refits': len(res), 'AF_total_min': min(afs), 'AF_total_max': max(afs), 'event_min': res[int(np.argmin(afs))][0], 'event_max': res[int(np.argmax(afs))][0],
                 'OR_range': {f'sc{c}': [min(r_[2][c - 1] for r_ in res), max(r_[2][c - 1] for r_ in res)] for c in (1, 2, 3)}}
    R['loeo'] = L
    sys.path.insert(0, H + '/v10/code'); from clr_exact import CLR
    C = {}
    for rg, sel in RNG.items():
        o = S1[rg]['classes']; terms = ['sc1', 'sc2', 'sc3']
        m, mr, dd, nd = fitfe(sel(B), f"case~{TERR}+sc1+sc2+sc3")
        X = dd[TV + terms].astype(float).copy(); mu = X[TV].mean(); sd = X[TV].std(); X[TV] = (X[TV] - mu) / sd
        start = np.concatenate([m.params[TV].values * sd.values, m.params[terms].values])
        f = CLR(X.values, dd.case.values, dd.event.values, cluster=dd.uid.values, names=TV + terms).fit(start)
        b = np.array(f['beta'][7:10]); rho = dd[dd.case == 1][terms].mean().values
        C[rg] = {'OR': {t: math.exp(b[i]) for i, t in enumerate(terms)}, 'ci': {t: [math.exp(b[i] - 1.96 * math.sqrt(f['cov_cluster'][7 + i, 7 + i])), math.exp(b[i] + 1.96 * math.sqrt(f['cov_cluster'][7 + i, 7 + i]))] for i, t in enumerate(terms)},
                 'AF_total': float(af_point(rho, b).sum())}
    R['clr'] = C
    BL = {}
    for km in (20, 10):
        for rg, sel in RNG.items():
            x = sel(B).copy(); x['blk'] = (np.floor(x.x / (km * 1000)).astype(int).astype(str) + '_' + np.floor(x.y / (km * 1000)).astype(int).astype(str))
            g = x.groupby('blk').case.agg(['sum', 'size']); keep = g[(g['sum'] > 0) & (g['sum'] < g['size'])].index; x = x[x.blk.isin(keep)].copy()
            m, mr, dd, nd = fitfe(x, f"case~{TERR}+sc1+sc2+sc3+C(blk)"); b = m.params[['sc1', 'sc2', 'sc3']].values; rho = dd[dd.case == 1][['sc1', 'sc2', 'sc3']].mean().values
            BL[f'{km}km_{rg}'] = {'OR': {f'sc{c}': term(m, mr, f'sc{c}') for c in (1, 2, 3)}, 'AF_total': float(af_point(rho, b).sum()), 'blocks_used': int(len(keep))}
    R['blocks'] = BL
    return R
S2 = cached('S2_sens', s2)
if TAG == 'v4':
    RV = F['RV']
    assert close(S2['road_river']['le39.5']['AF_total']['point'], RV['D_road_river']['le39.5']['AF_total']['point'], 1e-9)
    assert close(S2['pre2024']['le39.5']['AF_total']['point'], RV['E_earthquake_2024']['pre2024_le39.5']['AF_total']['point'], 1e-9)
    assert close(S2['loeo']['le39.5']['AF_total_min'], RV['L_loeo_hk']['le39.5']['AF_total_min'], 1e-6)
    assert close(S2['clr']['le39.5']['OR']['sc1'], FV('MC1.le39_5.classes')['clr']['sc1']['OR'], 1e-3)
    assert close(S2['blocks']['20km_le39.5']['OR']['sc1']['OR'], FV('MC2.20km.le39_5')['sc1']['OR'], 1e-6)
    W2 = FV('W2.radius'); assert close(S2['radius0']['le39.5']['AF_total']['point'], W2['r0']['le39.5']['AF_total']['point'], 1e-9)
    log('S2 validated (road/river, 2018-2023, LOEO, CLR, blocks, radius 0)')
log('S2', {k: (round(v['le39.5']['AF_total']['point'], 4) if isinstance(v, dict) and 'le39.5' in v and isinstance(v['le39.5'], dict) and 'AF_total' in v['le39.5'] and isinstance(v['le39.5']['AF_total'], dict) else '') for k, v in S2.items()})

ORDER = ['bare_0', 'bare_1_50', 'scar_0', 'scar_1_50', 'scar_50_100', 'lt']
SUMS = {'at_point': ['bare_0', 'scar_0'], 'within_50': ['bare_0', 'bare_1_50', 'scar_0', 'scar_1_50'], 'within_100': ['bare_0', 'bare_1_50', 'scar_0', 'scar_1_50', 'scar_50_100'],
        'scar_only': ['scar_0', 'scar_1_50', 'scar_50_100'], 'bare_only': ['bare_0', 'bare_1_50']}
def rings(x):
    x = x.copy(); c = {'bare_0': x.bare_at_cutoff_0m == 1, 'bare_1_50': x.bare_at_cutoff_50m == 1, 'scar_0': x.class1_0m == 1, 'scar_1_50': x.class1_50m == 1,
                       'scar_50_100': x.class1_100m == 1, 'lt': x.lt_detected == 1}
    taken = pd.Series(False, index=x.index)
    for nm in ORDER: x[nm] = (c[nm].fillna(False) & ~taken).astype(int); taken = taken | (x[nm] == 1)
    return x
def s3():
    R = {}
    for rg, sel in RNG.items():
        x = rings(sel(B)); o = MM(x, ORDER, f'rings:{rg}')
        for k, ts in SUMS.items():
            ix = [ORDER.index(t) for t in ts]; m, mr, dd = o['_m']
            pt = sum(o[f'AF_{t}']['point'] for t in ts); o[f'AFsum_{k}'] = {'point': pt}
            if '_boot' in o: o[f'AFsum_{k}']['ci_boot'] = pct(af_point(o['_boot']['rho'], o['_boot']['beta'])[:, ix].sum(1))
        o['cases_by_ring'] = {t: int(x.loc[x.case == 1, t].sum()) for t in ORDER}; o['controls_by_ring'] = {t: int(x.loc[x.case == 0, t].sum()) for t in ORDER}
        R[rg] = strip(o)
    return R
S3 = cached('S3_rings', s3)
if TAG == 'v4':
    W = FV('W2.rings')
    for rg in RNG:
        for t in ORDER: assert close(S3[rg]['OR'][t]['OR'], W[rg]['OR'][t]['OR']), (rg, t)
    log('S3 validated against W2.rings')
log('S3', {rg: {t: round(v['OR'], 2) for t, v in S3[rg]['OR'].items()} for rg in RNG}, {rg: round(S3[rg]['AFsum_within_50']['point'], 4) for rg in RNG})

def dummies(d, col, cats, prefix):
    d = d.copy(); names = []
    for c in cats:
        n = f'{prefix}_{c}'.replace('-', '_').replace('+', 'p').replace('.', '_'); d[n] = (d[col].astype(str) == c).astype(int); names.append(n)
    return d, names
def s4():
    R = {}
    d, t1 = dummies(B, 'tss1_bin', ['0-5', '5-10', '10+'], 'age')
    o = MM(d, t1 + ['sc2', 'sc3'], 'M1b')
    o['r05_10p'] = comb(o, {t1[0]: 1, t1[2]: -1}); o['r05_510'] = comb(o, {t1[0]: 1, t1[1]: -1})
    o['cases_by_bin'] = {t: int(d.loc[d.case == 1, t].sum()) for t in t1}; o['controls_by_bin'] = {t: int(d.loc[d.case == 0, t].sum()) for t in t1}
    R['M1b'] = strip(o)
    cells = ['s_lt1', 's_1_10', 's_10p', 'b_lt1', 'b_1_10', 'b_10p', 'b_noscar', 'c3']
    d, tc = dummies(B, 'cell', cells, 'cell')
    o = MM(d, tc, 'M2d')
    nm = dict(zip(cells, tc))
    for a in ('lt1', '1_10', '10p'): o[f'bare_vs_scar_{a}'] = comb(o, {nm[f'b_{a}']: 1, nm[f's_{a}']: -1})
    o['reveg_1_10_vs_10p'] = comb(o, {nm['s_1_10']: 1, nm['s_10p']: -1}); o['reveg_lt1_vs_1_10'] = comb(o, {nm['s_lt1']: 1, nm['s_1_10']: -1})
    o['counts'] = {c: {'cases': int(d.loc[d.case == 1, nm[c]].sum()), 'controls': int(d.loc[d.case == 0, nm[c]].sum())} for c in cells}
    o['counts']['c0'] = {'cases': int(((d.cell == 'c0') & (d.case == 1)).sum()), 'controls': int(((d.cell == 'c0') & (d.case == 0)).sum())}
    R['M2d'] = strip(o)
    d, t3 = dummies(B, 'tsd3_bin2', ['0-10', '10-20', '20+'], 'tsd')
    o = MM(d, ['sc1', 'sc2'] + t3, 'M3b'); o['r010_1020'] = comb(o, {t3[0]: 1, t3[1]: -1}); o['r010_20p'] = comb(o, {t3[0]: 1, t3[2]: -1})
    o['counts'] = {t: {'cases': int(d.loc[d.case == 1, t].sum()), 'controls': int(d.loc[d.case == 0, t].sum())} for t in t3}
    R['M3b'] = strip(o)
    d, t4 = dummies(B, 'tsd3_bin', ['0-5', '5-10', '10-20', '20+'], 'tsdf')
    o = MM(d, ['sc1', 'sc2'] + t4, 'M3'); o['counts'] = {t: {'cases': int(d.loc[d.case == 1, t].sum()), 'controls': int(d.loc[d.case == 0, t].sum())} for t in t4}
    R['M3'] = strip(o)
    for rg, sel in RNG.items():
        x = sel(B).copy(); c3 = x.site_class4 == 3
        x['c3_pre'] = (c3 & (x.dist_year < 2004)).astype(int); x['c3_post'] = (c3 & (x.dist_year >= 2004)).astype(int)
        o = MM(x, ['sc1', 'sc2', 'c3_pre', 'c3_post'], f'c3year:{rg}'); o['pre_vs_post'] = comb(o, {'c3_pre': 1, 'c3_post': -1})
        ca = x[x.case == 1]; co = x[x.case == 0]
        o['share_pre2004'] = {'cases': float(ca.c3_pre.sum() / (ca.c3_pre.sum() + ca.c3_post.sum())), 'controls': float(co.c3_pre.sum() / (co.c3_pre.sum() + co.c3_post.sum()))}
        R[f'c3_loss_year_{rg}'] = strip(o)
    return R
S4 = cached('S4_time', s4)
if TAG == 'v4':
    assert close(S4['M1b']['OR']['age_0_5']['OR'], FV('M1b.event.0_5')['OR']) and close(S4['M1b']['r05_10p']['OR'], FV('M1b.event.0_5_vs_10p')['ratio'], 1e-6)
    assert close(S4['M2d']['bare_vs_scar_10p']['OR'], FV('M2d.event.bare_vs_scar.10p')['ratio'], 1e-6)
    assert close(S4['M3b']['OR']['tsd_0_10']['OR'], FV('M3b.event.0_10')['OR']) and close(S4['M3b']['r010_1020']['OR'], FV('M3b.event.0_10_vs_10_20')['ratio'], 1e-6)
    RV = F['RV']
    assert close(S4['c3_loss_year_all']['OR']['c3_pre']['OR'], RV['F_class3_loss_year']['all']['OR_pre2004']['OR'])
    log('S4 validated (M1b, M2d, M3b, class-3 loss year)')
log('S4', {t: round(v['OR'], 2) for t, v in S4['M1b']['OR'].items()}, round(S4['M1b']['r05_10p']['OR'], 3), S4['M1b']['r05_10p'].get('ci_boot'))

def s5():
    X = with_arows(B); X = X[(X.year >= 2020) & X.chm.notna()].copy(); X['chm5'] = X.chm / 5.0; X['chm12'] = (X.chm >= 12).astype(int)
    R = {}
    for rg, sel in RNG.items():
        x1 = sel(X[X.site_class4 == 1]).copy(); x1, ta = dummies(x1, 'tss1_bin', ['5-10', '10+'], 'age')
        o = MM(x1, ta + ['chm5'], f'chm1:{rg}'); r = {'class1': strip(o)}
        r['class1_ge12'] = strip(MM(x1, ta + ['chm12'], f'chm1_12:{rg}'))
        r['class1_ge2m'] = strip(MM(x1[x1.chm >= 2], ta + ['chm5'], f'chm1_ge2:{rg}'))
        x0 = sel(X[X.site_class4 == 0]); r['class0'] = strip(MM(x0, ['chm5'], f'chm0:{rg}'))
        r['class0_ge2m'] = strip(MM(x0[x0.chm >= 2], ['chm5'], f'chm0_ge2:{rg}'))
        x3 = sel(X[X.site_class4 == 3]); r['class3'] = strip(MM(x3, ['chm5'], f'chm3:{rg}'))
        x01 = sel(X[X.site_class4.isin([0, 1])]).copy(); x01['c1'] = (x01.site_class4 == 1).astype(int); x01['chm5_c1'] = x01.chm5 * x01.c1
        x01['agec'] = np.where(x01.site_class4 == 1, x01.tss1_bin.astype(str), 'none'); x01, tc = dummies(x01, 'agec', ['0-5', '5-10', '10+'], 'agec')
        r['interaction'] = strip(MM(x01, tc + ['chm5', 'chm5_c1'], f'chm01:{rg}'))
        r['share_lt2m'] = {str(c): {'cases': float((sel(X[(X.site_class4 == c) & (X.case == 1)]).chm < 2).mean()), 'controls': float((sel(X[(X.site_class4 == c) & (X.case == 0)]).chm < 2).mean())} for c in range(4)}
        r['median_chm'] = {str(c): {'cases': float(sel(X[(X.site_class4 == c) & (X.case == 1)]).chm.median()), 'controls': float(sel(X[(X.site_class4 == c) & (X.case == 0)]).chm.median())} for c in range(4)}
        R[rg] = r
    return R
S5 = cached('S5_canopy', s5)
if TAG == 'v4':
    assert close(S5['all']['class1']['OR']['chm5']['OR'], FV('G4.M10_class1')['all']['chm_per_5m']['OR'])
    RV = F['RV']
    assert close(S5['all']['class0']['OR']['chm5']['OR'], RV['J_canopy']['all']['class0']['OR']) and close(S5['all']['interaction']['OR']['chm5_c1']['OR'], RV['J_canopy']['all']['class1_vs_class0_interaction']['OR'])
    log('S5 validated (class 1, class 0, interaction)')
log('S5', {rg: (round(S5[rg]['class1']['OR']['chm5']['OR'], 3), round(S5[rg]['class0']['OR']['chm5']['OR'], 3), round(S5[rg]['interaction']['OR']['chm5_c1']['OR'], 3)) for rg in RNG})

GRID = list(range(0, 61))
def s6():
    R = {}
    d = B.copy(); d['legacy_s20'] = d.legacy * d.s20
    o = MM(d, ['legacy', 'legacy_s20'], 'M4'); o['per_degree'] = comb(o, {'legacy_s20': 1})
    o['OR_at'] = {str(g): comb(o, {'legacy': 1, 'legacy_s20': g - 20}) for g in (10, 20, 30, 40, 50)}
    R['M4'] = strip(o)
    for c in (1, 2, 3): d[f'sc{c}_s20'] = d[f'sc{c}'] * d.s20
    tt = ['sc1', 'sc2', 'sc3', 'sc1_s20', 'sc2_s20', 'sc3_s20']
    o = MM(d, tt, 'M4b'); o['per_degree'] = {f'sc{c}': comb(o, {f'sc{c}_s20': 1}) for c in (1, 2, 3)}
    o['curve'] = {f'sc{c}': {str(g): comb(o, {f'sc{c}': 1, f'sc{c}_s20': g - 20}) for g in GRID} for c in (1, 2, 3)}
    R['M4b'] = strip(o)
    BANDS = {'all': (0, 90), 'le39.5': (0, 39.5), 'gt39.5': (39.5, 90), 'b0_20': (0, 20), 'b20_30': (20, 30), 'b30_39.5': (30, 39.5), 'b39.5_50': (39.5, 50), 'b50_90': (50, 90)}
    SB = {}
    for k, (lo, hi) in BANDS.items():
        x = B[(B.slope > lo) & (B.slope <= hi)] if lo > 0 else B[B.slope <= hi]
        a = MM(x, ['legacy'], f'band_any:{k}'); c = MM(x, ['sc1', 'sc2', 'sc3'], f'band_cls:{k}')
        dd = a['_m'][2]
        SB[k] = {'events': int(dd.event.nunique()), 'cases': int(dd.case.sum()), 'control_rows': int((dd.case == 0).sum()), 'any': strip(a), 'classes': strip(c),
                 '_ab': a.get('_boot', {}).get('beta')}
    adj = {}
    for a_, b_ in (('b20_30', 'b0_20'), ('b30_39.5', 'b20_30'), ('b39.5_50', 'b30_39.5'), ('b50_90', 'b39.5_50')):
        ra = SB[a_]['any']['OR']['legacy']; rb = SB[b_]['any']['OR']['legacy']; dlt = ra['beta'] - rb['beta']; se = math.sqrt(ra['se'] ** 2 + rb['se'] ** 2)
        adj[f'{a_}_vs_{b_}'] = {'ratio': math.exp(dlt), 'ci': [math.exp(dlt - 1.96 * se), math.exp(dlt + 1.96 * se)]}
        if SB[a_]['_ab'] is not None and SB[b_]['_ab'] is not None:
            n = min(len(SB[a_]['_ab']), len(SB[b_]['_ab'])); adj[f'{a_}_vs_{b_}']['ci_boot'] = [math.exp(v) for v in pct(SB[a_]['_ab'][:n, 0] - SB[b_]['_ab'][:n, 0])]
    for k in SB: SB[k].pop('_ab')
    R['bands'] = SB; R['bands_adjacent'] = adj
    return R
S6 = cached('S6_slope', s6)
if TAG == 'v4':
    assert close(S6['M4']['per_degree']['OR'], FV('M4.event.per_degree')['OR'], 1e-9) and close(S6['M4b']['per_degree']['sc2']['OR'], FV('M4b.event.sc2_per_degree')['OR'], 1e-9)
    E6 = FV('E6.slope_bands_adjacent'); assert close(S6['bands_adjacent']['b20_30_vs_b0_20']['ratio'], E6['20-30_vs_0-20']['ratio'], 1e-6)
    log('S6 validated (M4, M4b, slope bands)')
log('S6', round(S6['M4']['per_degree']['OR'], 4), S6['M4']['per_degree'].get('ci_boot'))

def s7():
    X = with_bright(B); assert X.Q.notna().all()
    R = {}
    def block(Bx, qcol, label, full=True):
        out = {}; Bx = Bx.copy(); qs = sorted(Bx[qcol].astype(int).unique()); top = qs[-1]; mid = qs[len(qs) // 2]
        comp = {}
        for q in qs:
            s = Bx[Bx[qcol] == q]
            comp[str(q)] = {'n_cases': int(s.case.sum()), 'n_control_rows': int((s.case == 0).sum()),
                            'share_cases': {str(c): float((s[s.case == 1].site_class4 == c).mean()) for c in range(4)}, 'share_controls': {str(c): float((s[s.case == 0].site_class4 == c).mean()) for c in range(4)}}
        out['composition'] = comp
        for q in qs[1:]: Bx[f'q{q}'] = (Bx[qcol].astype(int) == q).astype(int)
        qd = [f'q{q}' for q in qs[1:]]
        for q in qs: Bx[f'leg_q{q}'] = Bx.legacy * (Bx[qcol].astype(int) == q)
        o = MM(Bx, [f'leg_q{q}' for q in qs], f'B2:{label}', covs=TV + RAINV + qd); out['legacy_by_Q'] = strip(o)['OR']
        Bx['legacy_qrank'] = Bx.legacy * (Bx[qcol].astype(int) - 1)
        o = MM(Bx, ['legacy', 'legacy_qrank'], f'B2trend:{label}', covs=TV + RAINV + qd); out['trend'] = strip(o)['OR']['legacy_qrank']
        if not full: return out
        S = Bx[Bx[qcol] == top]; o = MM(S, ['sc1', 'sc2', 'sc3'], f'B1b:{label}', covs=TV + RAINV); out['af_top'] = strip(o)
        cats = [f'{q}_{s}' for q in qs for s in ('intact', 'legacy')]
        Bx['cellq'] = Bx[qcol].astype(int).astype(str) + np.where(Bx.legacy == 1, '_legacy', '_intact')
        for c in cats[1:]: Bx['cq_' + c] = (Bx.cellq == c).astype(int)
        tq = ['cq_' + c for c in cats[1:]]
        o = MM(Bx, tq, f'B3:{label}', covs=[])
        cells = {cats[0]: {'OR': 1.0}}; cells.update({c: o['OR']['cq_' + c] for c in cats[1:]})
        dd = o['_m'][2]; cn = dd.groupby('cellq').case.agg(['sum', 'size'])
        for c in cats: cells[c]['n_cases'] = int(cn.loc[c, 'sum']) if c in cn.index else 0; cells[c]['n_control_rows'] = int(cn.loc[c, 'size'] - cn.loc[c, 'sum']) if c in cn.index else 0
        out['cells'] = cells
        out['contrasts'] = {f'{top}_intact_vs_{qs[0]}_legacy': comb(o, {f'cq_{top}_intact': 1, f'cq_{qs[0]}_legacy': -1}),
                            f'{top}_intact_vs_{mid}_legacy': comb(o, {f'cq_{top}_intact': 1, f'cq_{mid}_legacy': -1}),
                            f'{top}_legacy_vs_{top}_intact': comb(o, {f'cq_{top}_legacy': 1, f'cq_{top}_intact': -1})}
        m, mr, _ = o['_m']; V = mr.cov_params(); b = m.params
        diffs = {}
        for q in qs:
            tl = f'cq_{q}_legacy'; ti = f'cq_{q}_intact'
            el = math.exp(b[tl]); ei = 1.0 if q == qs[0] else math.exp(b[ti])
            g = {tl: el};
            if q != qs[0]: g[ti] = -ei
            var = sum(g[i] * g[j] * V.loc[i, j] for i in g for j in g); pt_ = el - ei
            diffs[str(q)] = {'point': pt_, 'ci_delta': [pt_ - 1.96 * math.sqrt(var), pt_ + 1.96 * math.sqrt(var)]}
            if '_boot' in o:
                be = o['_boot']['beta']; jl = tq.index(tl); vl = np.exp(be[:, jl]); vi = np.ones(len(be)) if q == qs[0] else np.exp(be[:, tq.index(ti)])
                diffs[str(q)]['ci_boot'] = pct(vl - vi); diffs[str(q)]['_draws'] = vl - vi
        if all('_draws' in diffs[str(q)] for q in (qs[0], top)):
            dd_ = diffs[str(top)]['_draws'] - diffs[str(qs[0])]['_draws']; diffs['top_minus_bottom'] = {'point': diffs[str(top)]['point'] - diffs[str(qs[0])]['point'], 'ci_boot': pct(dd_)}
        for k in list(diffs): diffs[k].pop('_draws', None)
        out['differences'] = diffs
        return out
    R['main'] = block(X, 'Q', 'Q')
    R['le39.5'] = block(X[X.slope <= 39.5], 'Q', 'Q_le')
    R['insample'] = block(X, 'Qin', 'Qin', full=False); R['terrain_only'] = block(X, 'Qt', 'Qt', full=False); R['deciles'] = block(X, 'D', 'D', full=False)
    return R
S7 = cached('S7_risk', s7)
if TAG == 'v4':
    assert close(S7['main']['trend']['OR'], FV('G3.trend')['OR'], 1e-9) and close(S7['main']['legacy_by_Q']['leg_q1']['OR'], FV('G3.legacy_by_Q')['1']['OR'], 1e-9)
    assert close(S7['main']['contrasts']['5_intact_vs_1_legacy']['OR'], FV('G3.contrasts')['5_intact_vs_1_legacy']['ratio'], 1e-6)
    assert close(S7['main']['differences']['5']['point'], FV('RV.cell_diff')['difference_by_quintile']['5']['point'], 1e-6)
    assert close(S7['main']['af_top']['AF_total']['point'], FV('G3.af_top')['AF_total']['point'], 1e-6)
    log('S7 validated (trend, legacy by quintile, contrasts, differences, top-quintile AF)')
log('S7', round(S7['main']['trend']['OR'], 4), S7['main']['trend'].get('ci_boot'), {k: round(v['point'], 2) for k, v in S7['main']['differences'].items()})

def s8():
    X = E[E.ft.isin([BAM, BRO])].copy(); X['bamboo'] = (X.ft == BAM).astype(int); X['bamboo_slope'] = X.bamboo * X.slope
    POP = {'full': X, 'broadleaf_class0_v2': X[(X.ft == BAM) | ((X.ft == BRO) & (X.site_class == 0))], 'broadleaf_class0_v3': X[(X.ft == BAM) | ((X.ft == BRO) & (X.site_class4 == 0))],
           'both_class0_v3': X[X.site_class4 == 0]}
    R = {}
    for k, d in POP.items():
        o = MM(d, ['bamboo', 'bamboo_slope'], f'H4:{k}', boot=DOBOOT)
        r = {'n_bamboo_case': int(((o['_m'][2].case == 1) & (o['_m'][2].bamboo == 1)).sum()), 'n_broadleaf_case': int(((o['_m'][2].case == 1) & (o['_m'][2].bamboo == 0)).sum()),
             'interaction': comb(o, {'bamboo_slope': 1})}
        for deg in (20, 30, 40): r[f'OR_at_{deg}'] = comb(o, {'bamboo': 1, 'bamboo_slope': deg})
        R[k] = r
    bb = X[X.ft == BAM]
    for rg, sel in RNG.items():
        d = sel(bb); ca = d[d.case == 1]; co = d[d.case == 0]
        R[f'bamboo_classes_{rg}'] = {'cases': int(len(ca)), 'control_rows': int(len(co)), 'cases_by_class': {str(c): int((ca.site_class4 == c).sum()) for c in range(4)},
                                    'controls_by_class': {str(c): int((co.site_class4 == c).sum()) for c in range(4)}, 'legacy_share_cases': float((ca.site_class4 > 0).mean()), 'legacy_share_controls': float((co.site_class4 > 0).mean())}
    return R
S8 = cached('S8_bamboo', s8)
if TAG == 'v4':
    for k in ('full', 'broadleaf_class0_v3', 'both_class0_v3'):
        assert close(S8[k]['interaction']['OR'], FV(f'H4.event.{k}')['interaction']['OR'], 1e-6), k
    log('S8 validated (H4 interactions)')
log('S8', {k: (round(v['OR_at_20']['OR'], 3), v['OR_at_20'].get('ci_boot'), v['n_bamboo_case']) for k, v in S8.items() if k.startswith(('full', 'broad', 'both'))})

SRC = {'原生林': 'primary', '經改造天然林': 'modified', '保護性人工林': 'plantation', '生產性人工林': 'plantation', '半天然林': 'other', '未確認': 'other'}
CMP = ['primary', 'modified', 'plantation']
def s9():
    X = with_arows(B); X['src'] = X.stand_function.map(SRC).fillna('missing'); R = {'M8': {}, 'M9': {}, 'M11': {}}
    for rg, sel in RNG.items():
        o = {}
        for s in CMP:
            x = sel(X[X.src == s]); a = MM(x, ['sc1', 'sc2', 'sc3'], f'M8:{rg}:{s}')
            o[s] = {'n_cases': int(x.case.sum()), 'n_control_rows': int((x.case == 0).sum()), 'share_controls': {str(c): float((x[x.case == 0].site_class4 == c).mean()) for c in range(4)},
                    'legacy_share_controls': float(x[x.case == 0].legacy.mean()), 'AF': strip(a)['AF_total'], 'any_OR': strip(MM(x, ['legacy'], f'M8any:{rg}:{s}'))['OR']['legacy']}
        R['M8'][rg] = o
        x = sel(X[X.src.isin(CMP)]).copy(); x['mod'] = (x.src == 'modified').astype(int); x['pla'] = (x.src == 'plantation').astype(int)
        x['leg_mod'] = x.legacy * x['mod']; x['leg_pla'] = x.legacy * x.pla
        o = MM(x, ['legacy', 'leg_mod', 'leg_pla'], f'M9:{rg}', covs=TV + ['mod', 'pla'])
        R['M9'][rg] = {'interaction_modified': comb(o, {'leg_mod': 1}), 'interaction_plantation': comb(o, {'leg_pla': 1}), 'legacy_primary': comb(o, {'legacy': 1})}
    POPS = {'all': lambda d: d, 'excl_1_2': lambda d: d[~d.site_class4.isin([1, 2])], 'excl_1_3': lambda d: d[d.site_class4 == 0], 'excl_1_3_le39.5': lambda d: d[(d.site_class4 == 0) & (d.slope <= 39.5)]}
    for k, f in POPS.items():
        x = f(X[X.src.isin(CMP)]).copy(); x['modified'] = (x.src == 'modified').astype(int); x['plantation'] = (x.src == 'plantation').astype(int)
        x['modified_s20'] = x.modified * x.s20; x['plantation_s20'] = x.plantation * x.s20
        o = MM(x, ['modified', 'plantation', 'modified_s20', 'plantation_s20'], f'M11:{k}')
        R['M11'][k] = {s: {'OR_at_20': comb(o, {s: 1})} for s in ('modified', 'plantation')}
    return R
S9 = cached('S9_origin', s9)
if TAG == 'v4':
    assert close(S9['M11']['all']['plantation']['OR_at_20']['OR'], FV('G4.M11_event')['all']['plantation']['OR_at_20']['OR'], 1e-3)
    assert close(S9['M9']['le39.5']['interaction_modified']['OR'], FV('G4.M9')['le39.5']['interaction_vs_primary']['modified']['OR'], 1e-3)
    log('S9 validated (M9, M11)')
log('S9', {k: round(v['plantation']['OR_at_20']['OR'], 3) for k, v in S9['M11'].items()})

ZF = H + '/g5/zoning_out_v1.csv'
def s10():
    Z = pd.read_csv(ZF, dtype={'uid': str}); GAP = json.load(open(H + '/reference/gsa_gap_register_v2.json', encoding='utf-8'))
    M = B.copy(); M['R1'] = (M.class1_50m == 1).astype(int); M['R3'] = ((M.lt_detected == 1) & (M.R1 == 0)).astype(int); M['legacy_s'] = ((M.R1 == 1) | (M.R3 == 1)).astype(int)
    M = M.merge(Z, on='uid', how='left'); M['zin'] = M.gsmma_inside.fillna(0).astype(int)
    L = M[M.slope <= 39.5].copy(); c = L[L.case == 1]; k = L[L.case == 0]
    R = {'cases_inside_share': float(c.zin.mean()), 'controls_inside_share': float(k.zin.mean()), 'n_cases': int(len(c)), 'n_control_rows': int(len(k)),
         'legacy_cases': int(c.legacy_s.sum()), 'legacy_cases_outside': int(((c.legacy_s == 1) & (c.zin == 0)).sum())}
    R['legacy_cases_outside_share'] = R['legacy_cases_outside'] / R['legacy_cases']
    L['leg'] = L.legacy_s; L['leg_zin'] = L.leg * L.zin
    o = MM(L, ['leg', 'zin', 'leg_zin'], 'gsa_model')
    R['legacy_OR_outside'] = comb(o, {'leg': 1}); R['legacy_OR_inside'] = comb(o, {'leg': 1, 'leg_zin': 1}); R['interaction'] = comb(o, {'leg_zin': 1}); R['zone_OR_intact'] = comb(o, {'zin': 1})
    a = MM(L[L.zin == 0], ['R1', 'R3'], 'gsa_af_out'); R['AF_outside'] = strip(a)['AF_total']
    P = pd.concat([M[['uid', 'x', 'y']], pd.read_csv(REPS['v1'], usecols=['uid', 'x', 'y'], low_memory=False)]).drop_duplicates('uid').merge(Z, on='uid', how='inner')
    ins = P[P.gsmma_inside == 1].dropna(subset=['gsmma_zone']); tree = cKDTree(ins[['x', 'y']].values); dist, j = tree.query(L[['x', 'y']].values)
    L['zone'] = np.where(L.zin == 1, L.gsmma_zone, ins.gsmma_zone.values[j]); plan = {z: pd.Timestamp(v['plan_date']) for z, v in GAP['zones'].items()}
    L['plan_date'] = L.zone.map(plan); assert L.plan_date.notna().all()
    L['phase'] = np.where(L.plan_date < pd.Timestamp('2018-01-01'), 'plan_2014_2016', 'plan_2021_2025')
    for col in ('scar_50m_first_after', 'scar_0m_first_after'): L[col] = pd.to_datetime(L[col])
    cl = L[(L.case == 1) & (L.legacy_s == 1)].copy()
    def comp(g):
        newer = (g.scar_50m_first_after > g.plan_date) | ((g.class1_0m == 1) & (g.scar_0m_first_after > g.plan_date))
        return np.select([g.R3 == 1, newer, g.class1_0m == 1], ['disturbance_without_catalogue_scar', 'scar_newer_than_zone_plan', 'older_scar_polygon_contains_point'], 'older_scar_within_50m_polygon_not_containing_point')
    out = cl[cl.zin == 0].copy(); out['gap'] = comp(out)
    R['outside_by_component'] = {k_: int(v) for k_, v in out.gap.value_counts().items()}; R['outside_share_by_component'] = {k_: float(v / len(out)) for k_, v in out.gap.value_counts().items()}
    cov = {}
    for ph, g in L.groupby('phase'):
        kk = g[g.case == 1]; ct = g[g.case == 0]; lg = kk[kk.legacy_s == 1]
        cov[ph] = {'cases': int(len(kk)), 'cases_inside_share': float(kk.zin.mean()), 'control_rows': int(len(ct)), 'controls_inside_share': float(ct.zin.mean()), 'legacy_cases_outside_share': float(1 - lg.zin.mean()),
                   'zones': int(g.zone.nunique())}
    R['coverage_by_phase'] = cov
    cm = L[(L.case == 1) & (L.site_class4 > 0)]
    R['main_definition_outside'] = {'n': int(len(cm)), 'outside': int((cm.zin == 0).sum()), 'outside_by_class': {str(k_): int(v) for k_, v in cm[cm.zin == 0].site_class4.value_counts().sort_index().items()}}
    def share(f): return {'cases': float(f[L.case == 1].mean()), 'controls': float(f[L.case == 0].mean())}
    lay = {'class2': L.site_class4 == 2, 'class1': L.site_class4 == 1, 'class3': L.site_class4 == 3, 'classes_1_3': L.site_class4 > 0, 'legacy_shared': L.legacy_s == 1,
           'gsa': L.zin == 1, 'gsa_or_legacy': (L.zin == 1) | (L.legacy_s == 1), 'legacy_outside_gsa': (L.zin == 0) & (L.legacy_s == 1)}
    R['screening'] = {k_: share(v) for k_, v in lay.items()}
    for k_ in R['screening']: R['screening'][k_]['ratio'] = R['screening'][k_]['cases'] / R['screening'][k_]['controls']
    return R
S10 = cached('S10_gsa', s10)
if TAG == 'v4':
    Zr = FV('R5.Z')['main']; assert close(S10['cases_inside_share'], Zr['cases_le39.5_inside_share'], 1e-4) and close(S10['legacy_cases_outside_share'], Zr['legacy_cases_le39.5_outside_share'], 1e-4)
    assert close(S10['legacy_OR_outside']['OR'], Zr['legacy_OR_outside']['OR'], 1e-3)
    log('S10 validated (GSA shares and odds ratio)')
log('S10', round(S10['legacy_cases_outside_share'], 4), round(S10['legacy_OR_outside']['OR'], 3), {k: round(v['cases'], 4) for k, v in S10['screening'].items()})

def s11():
    R = {}
    for rg, sel in RNG.items():
        rows = []
        for ev, g in sel(B).groupby('event'):
            if g.case.sum() < 50: continue
            mm = smf.logit(f"case~{TERR}+legacy", data=g).fit(disp=0, maxiter=200, method='newton')
            rows.append({'event': ev, 'n_cases': int(g.case.sum()), 'beta': float(mm.params['legacy']), 'se': float(mm.bse['legacy'])})
        b = np.array([r['beta'] for r in rows]); se = np.array([r['se'] for r in rows]); w = 1 / se ** 2; k = len(b)
        bf = np.sum(w * b) / np.sum(w); Q = float(np.sum(w * (b - bf) ** 2)); tau2 = max(0.0, (Q - (k - 1)) / (np.sum(w) - np.sum(w ** 2) / np.sum(w)))
        ws = 1 / (se ** 2 + tau2); br = np.sum(ws * b) / np.sum(ws); ser = math.sqrt(1 / np.sum(ws)); tq = tdist.ppf(0.975, k - 2); pse = math.sqrt(tau2 + ser ** 2)
        q = (ws * (b - br) ** 2).sum() / (k - 1); se_hk = math.sqrt(q / ws.sum()); th = tdist.ppf(0.975, k - 1)
        R[rg] = {'k': k, 'pooled': math.exp(br), 'ci': [math.exp(br - 1.96 * ser), math.exp(br + 1.96 * ser)], 'pi': [math.exp(br - tq * pse), math.exp(br + tq * pse)],
                 'hk': [math.exp(br - th * se_hk), math.exp(br + th * se_hk)], 'tau2': tau2, 'events': rows}
    return R
S11 = cached('S11_events', s11)
if TAG == 'v4':
    assert close(S11['le39.5']['pooled'], FV('MC3.le39_5')['pooled_random']['OR'], 1e-3); log('S11 validated (random effects)')
def s12():
    P2 = pd.read_csv(H + '/v11/unround/g1/s2_pairs_v1.csv', dtype={'uid': str})
    X = B[B.cutoff_year >= 2021].merge(P2[['uid', 'cutoff_year', 's2_detected', 'lt_detected_win']], on=['uid', 'cutoff_year'], how='left'); X = X[X.s2_detected.notna()].copy()
    for c in (1, 2): X[f'k{c}'] = (X.site_class == c).astype(int)
    X['c3_lt'] = ((X.site_class == 0) & (X.lt_detected_win == 1)).astype(int); X['c3_s2'] = ((X.site_class == 0) & (X.s2_detected == 1)).astype(int)
    R = {}
    for lab in ('c3_lt', 'c3_s2'):
        o = MM(X, ['k1', 'k2', lab], f'S2:{lab}'); dd = o['_m'][2]
        R[lab] = dict(strip(o)['OR'][lab], n_exposed_cases=int(dd[dd.case == 1][lab].sum()), n_exposed_controls=int(dd[dd.case == 0][lab].sum()), n_cases=int(dd.case.sum()))
    return R
S12 = cached('S12_s2', s12)
if TAG == 'v4':
    assert close(S12['c3_s2']['OR'], FV('S2.class3_or')['s2_010']['OR'], 1e-3); log('S12 validated (Sentinel-2)')
log('S11', {rg: (S11[rg]['k'], round(S11[rg]['pooled'], 3)) for rg in RNG}, 'S12', {k: round(v['OR'], 3) for k, v in S12.items()})

json.dump({'tags': TAG, 'boot': DOBOOT, 'S0': S0, 'S1': S1, 'S2': S2, 'S3': S3, 'S4': S4, 'S5': S5, 'S6': S6, 'S7': S7, 'S8': S8, 'S9': S9, 'S10': S10, 'S11': S11, 'S12': S12, 'fallback': FALLBACK},
          open(f'{H}/v27/an/main27_{TAG}.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
log('done', TAG)
