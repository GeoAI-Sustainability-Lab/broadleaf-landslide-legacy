"""Comparison methods: conditional logistic regression, spatial block fixed effects, event heterogeneity, path dependency, scar-age splines, transfer to the replication."""
import json, math, sys, os, time, itertools, warnings
H = '@WORK@'
src = open(H + '/v10/an/g4/stand_origin_v1.py', encoding='utf-8').read()
exec(compile(src[:src.index('rng=np.random.default_rng(SEED)')], 'stand_origin_v1_defs', 'exec'))
sys.path.insert(0, H + '/v10/code'); from clr_exact import CLR
from scipy.stats import norm, chi2, t as tdist
warnings.filterwarnings('ignore')
OUT = H + '/v10/an/methods'; os.makedirs(OUT, exist_ok=True)
SEED_M = 20261013; rng = np.random.default_rng(SEED_M)
TV = ['slope', 'slope2', 'north', 'east', 'planc', 'profc', 'elev']
R = {'status': 'exploratory comparison methods chosen after the 2026-10-13 literature review; not in any charter', 'seed': SEED_M, 'data': 'v10/data/tags_event_v4.csv (corrected tags)'}
SUB = {'all': EV, 'le39.5': EV[EV.slope <= 39.5]}
def orci(b, se): return {'OR': round(math.exp(b), 4), 'ci': [round(math.exp(b - 1.96 * se), 4), round(math.exp(b + 1.96 * se), 4)], 'beta': float(b), 'se': float(se)}
t0 = time.time(); C1 = {}
for nm, d in SUB.items():
    for lab, terms in (('classes', ['sc1', 'sc2', 'sc3']), ('any_legacy', ['legacy'])):
        m, mr, dd, nd = fitfe(d, f"case~{TERR}+" + '+'.join(terms))
        fe = {t: lin(m, mr, {t: 1}) for t in terms}
        X = dd[TV + terms].astype(float).copy()
        mu = X[TV].mean(); sd = X[TV].std(); X[TV] = (X[TV] - mu) / sd
        start = np.concatenate([m.params[TV].values * sd.values, m.params[terms].values])
        clr = CLR(X.values, dd.case.values, dd.event.values, cluster=dd.uid.values, names=TV + terms)
        f = clr.fit(start)
        k0 = len(TV); out = {}
        for i, t in enumerate(terms):
            b = f['beta'][k0 + i]; se_c = math.sqrt(f['cov_cluster'][k0 + i, k0 + i]); se_m = math.sqrt(f['cov_model'][k0 + i, k0 + i])
            out[t] = {**orci(b, se_c), 'ci_model': [round(math.exp(b - 1.96 * se_m), 4), round(math.exp(b + 1.96 * se_m), 4)],
                      'ratio_to_fe': round(math.exp(b - fe[t]['beta']), 4)}
        if lab == 'classes':
            j1, j2 = k0, k0 + 1; V = f['cov_cluster']; b = f['beta'][j2] - f['beta'][j1]; se = math.sqrt(V[j1, j1] + V[j2, j2] - 2 * V[j1, j2])
            out['bare_vs_scar'] = orci(b, se)
            bf = m.params['sc2'] - m.params['sc1']; cv = mr.cov_params(); sef = math.sqrt(cv.loc['sc1', 'sc1'] + cv.loc['sc2', 'sc2'] - 2 * cv.loc['sc1', 'sc2'])
            fe['bare_vs_scar'] = orci(bf, sef)
        C1[f'{nm}_{lab}'] = {'fe': fe, 'clr': out, 'n_rows': int(len(dd)), 'n_cases': int(dd.case.sum()), 'strata': len(clr.groups),
                             'clr_iterations': len(f['iter']), 'clr_score_max': f['score_max'], 'clr_clusters': f['n_clusters']}
        print('C1', nm, lab, {t: (fe[t]['OR'], out[t]['OR']) for t in terms}, round(time.time() - t0), 's', flush=True)
R['C1_clr_vs_fe'] = C1
C2 = {}
for km in (20, 10):
    for nm, d in SUB.items():
        x = d.copy(); x['blk'] = (np.floor(x.x / (km * 1000)).astype(int).astype(str) + '_' + np.floor(x.y / (km * 1000)).astype(int).astype(str))
        g = x.groupby('blk').case.agg(['sum', 'size']); keep = g[(g['sum'] > 0) & (g['sum'] < g['size'])].index
        x = x[x.blk.isin(keep)].copy()
        m, mr, dd, nd = fitfe(x, f"case~{TERR}+sc1+sc2+sc3+C(blk)")
        o = {t: lin(m, mr, {t: 1}) for t in ('sc1', 'sc2', 'sc3')}; o['bare_vs_scar'] = lin(m, mr, {'sc2': 1, 'sc1': -1})
        m2, mr2, dd2, _ = fitfe(x, f"case~{TERR}+legacy+C(blk)"); o['any_legacy'] = lin(m2, mr2, {'legacy': 1})
        o.update({'blocks_used': int(len(keep)), 'blocks_dropped_no_variation': int(len(g) - len(keep)), 'n_rows': int(len(dd)), 'n_cases': int(dd.case.sum())})
        C2[f'{km}km_{nm}'] = o
        print('C2', km, nm, {t: o[t]['OR'] for t in ('sc1', 'sc2', 'sc3', 'any_legacy')}, flush=True)
R['C2_spatial_block_fe'] = C2
def dl(b, se):
    w = 1 / se ** 2; bf = np.sum(w * b) / np.sum(w); Q = float(np.sum(w * (b - bf) ** 2)); k = len(b)
    tau2 = max(0.0, (Q - (k - 1)) / (np.sum(w) - np.sum(w ** 2) / np.sum(w)))
    ws = 1 / (se ** 2 + tau2); br = np.sum(ws * b) / np.sum(ws); ser = math.sqrt(1 / np.sum(ws))
    I2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
    tq = tdist.ppf(0.975, k - 2); pse = math.sqrt(tau2 + ser ** 2)
    return {'k': k, 'pooled_random': orci(br, ser), 'pooled_fixed': orci(bf, math.sqrt(1 / np.sum(w))), 'tau': round(math.sqrt(tau2), 4), 'I2': round(I2, 4),
            'Q': round(Q, 3), 'Q_p': float(1 - chi2.cdf(Q, k - 1)), 'prediction_interval': [round(math.exp(br - tq * pse), 4), round(math.exp(br + tq * pse), 4)]}
C3 = {}
for nm, d in SUB.items():
    rows = []
    for ev, g in d.groupby('event'):
        if g.case.sum() < 50: continue
        mm = smf.logit(f"case~{TERR}+legacy", data=g).fit(disp=0, maxiter=200, method='newton')
        rows.append({'event': ev, 'n_cases': int(g.case.sum()), 'n_controls': int((g.case == 0).sum()), 'OR': round(math.exp(mm.params['legacy']), 4),
                     'beta': float(mm.params['legacy']), 'se': float(mm.bse['legacy'])})
    b = np.array([r['beta'] for r in rows]); se = np.array([r['se'] for r in rows])
    C3[nm] = {'events': rows, **dl(b, se)}
    print('C3', nm, C3[nm]['k'], C3[nm]['pooled_random']['OR'], C3[nm]['tau'], C3[nm]['prediction_interval'], flush=True)
R['C3_event_heterogeneity'] = C3
cat = pd.read_csv(H + '/g0/catalog_matches_within100m_v1.csv', dtype={'uid': str, 'record_id': str}, parse_dates=['after'])
D0 = pd.read_csv(H + '/g0/point_dates_v1.csv', dtype={'uid': str, 'lid': str})
own_ids = set((D0[D0.kind == 'case'].assign(rid=lambda q: q.lid.str.split('_').str[0] + ':' + q.own_record_index.astype(int).astype(str))).rid)
cat['is_own'] = cat.record_id.isin(own_ids) & cat.uid.str.startswith('case:') & (cat.uid.str[5:] == cat.cat_year.astype(str) + '_' + cat.record_index.astype(str))
EVr = EV.reset_index(drop=True).copy(); EVr['rk'] = np.arange(len(EVr)); EVr['cutoff_dt'] = pd.to_datetime(EVr.cutoff)
mt = cat.merge(EVr[['rk', 'uid', 'cutoff_dt']], on='uid'); mt = mt[(~mt.is_own) & mt.after.notna() & (mt.after <= mt.cutoff_dt)].copy()
mt['age'] = (mt.cutoff_dt - mt.after).dt.days / 365.25; mt['dist'] = np.where(mt.inside == 1, 0.0, mt.distance_m)
def pd_cov(lam, tau):
    v = np.exp(-mt.dist.values / lam - (mt.age.values / tau if np.isfinite(tau) else 0.0))
    s = pd.Series(v).groupby(mt.rk.values).max()
    out = np.zeros(len(EVr)); out[s.index.values] = s.values; return out
LAMS = [15, 30, 45, 60, 90, 150]; TAUS = [3, 5, 8, 12, 17, 25, 40, np.inf]
C4 = {}
for nm in ('all', 'le39.5'):
    base = EVr if nm == 'all' else EVr[EVr.slope <= 39.5]
    grid = []
    for lam, tau in itertools.product(LAMS, TAUS):
        x = base.copy(); x['PD'] = pd_cov(lam, tau)[x.rk.values]
        d2, _ = drop_uninf(x); mm = smf.logit(f"case~{TERR}+PD+C(event)", data=d2).fit(disp=0, maxiter=100, method='newton')
        grid.append({'lambda_m': lam, 'tau_yr': (None if not np.isfinite(tau) else tau), 'll': float(mm.llf), 'beta_PD': float(mm.params['PD'])})
    best = max(grid, key=lambda r: r['ll'])
    lam_b = best['lambda_m']; tau_b = np.inf if best['tau_yr'] is None else best['tau_yr']
    x = base.copy(); x['PD'] = pd_cov(lam_b, tau_b)[x.rk.values]; x['PD60_17'] = pd_cov(60, 17)[x.rk.values]
    x['bare_cat'] = ((x.sc2 == 1) & (x.class1_50m == 1)).astype(int); x['bare_nocat'] = ((x.sc2 == 1) & (x.class1_50m == 0)).astype(int)
    d2, _ = drop_uninf(x)
    forms = {'T': '', 'C_classes': '+sc1+sc2+sc3', 'S_published_60m_17yr': '+PD60_17', 'S_profiled': '+PD', 'S_profiled+bare': '+PD+sc2',
             'S_profiled+bare+uncatalogued': '+PD+sc2+sc3', 'C_classes+S_profiled': '+sc1+sc2+sc3+PD'}
    extra = {'S_profiled': 2, 'S_profiled+bare': 2, 'S_profiled+bare+uncatalogued': 2, 'C_classes+S_profiled': 2}
    fits = {}
    for k, f in forms.items():
        mm = smf.logit(f"case~{TERR}{f}+C(event)", data=d2).fit(disp=0, maxiter=100, method='newton')
        kk = int(mm.df_model + 1 + extra.get(k, 0)); fits[k] = {'ll': round(float(mm.llf), 3), 'k': kk, 'AIC': round(float(-2 * mm.llf + 2 * kk), 2)}
        if 'PD' in f:
            col = 'PD60_17' if 'PD60_17' in f else 'PD'
            mr = smf.logit(f"case~{TERR}{f}+C(event)", data=d2).fit(disp=0, maxiter=100, method='newton', cov_type='cluster', cov_kwds={'groups': pd.factorize(d2.uid)[0]})
            fits[k]['PD_OR_full_vs_none'] = lin(mm, mr, {col: 1})
            for t in ('sc2', 'sc3'):
                if t in mm.params.index: fits[k][t] = lin(mm, mr, {t: 1})
    aicC = fits['C_classes']['AIC']
    for k in fits: fits[k]['dAIC_vs_classes'] = round(fits[k]['AIC'] - aicC, 2)
    def lr(a, b, df):
        s = 2 * (fits[b]['ll'] - fits[a]['ll']); return {'chi2': round(s, 2), 'df': df, 'p': float(1 - chi2.cdf(s, df))}
    C4[nm] = {'grid': grid, 'profiled': {'lambda_m': lam_b, 'tau_yr': best['tau_yr']}, 'fits': fits,
              'lr_bare_added_to_S': lr('S_profiled', 'S_profiled+bare', 1), 'lr_uncatalogued_added_to_S_bare': lr('S_profiled+bare', 'S_profiled+bare+uncatalogued', 1),
              'lr_S_added_to_classes': lr('C_classes', 'C_classes+S_profiled', 1), 'n_rows': int(len(d2)), 'n_cases': int(d2.case.sum()),
              'rows_with_prior_scar_within_100m': int((x.loc[d2.index, 'PD'] > 0).sum()) if True else None}
    print('C4', nm, C4[nm]['profiled'], {k: v['dAIC_vs_classes'] for k, v in fits.items()}, flush=True)
R['C4_path_dependency'] = C4
KN = np.array([0.5, 5.0, 10.0, 18.0])
def ns_basis(a):
    a = np.asarray(a, float); K = len(KN)
    def dk(k): return (np.clip(a - KN[k], 0, None) ** 3 - np.clip(a - KN[-1], 0, None) ** 3) / (KN[-1] - KN[k])
    return np.column_stack([a] + [dk(k) - dk(K - 2) for k in range(K - 2)])
C5 = {}
GRID = np.round(np.arange(0.5, 18.01, 0.5), 2)
for nm, d in SUB.items():
    x = d.copy(); age = x.scar_50m_min_yrs.fillna(0).values; Bm = ns_basis(age)
    x['rev'] = (x.sc1 == 1).astype(int); x['barec'] = ((x.sc2 == 1) & (x.class1_50m == 1)).astype(int); x['barenc'] = ((x.sc2 == 1) & (x.class1_50m == 0)).astype(int)
    terms = []
    for g in ('rev', 'barec'):
        for j in range(Bm.shape[1]): x[f'{g}_b{j}'] = x[g].values * Bm[:, j]; terms.append(f'{g}_b{j}')
    m, mr, dd, nd = fitfe(x, f"case~{TERR}+rev+barec+barenc+sc3+" + '+'.join(terms))
    Bg = ns_basis(GRID); cv = mr.cov_params(); curves = {}
    for g in ('rev', 'barec'):
        names = [g] + [f'{g}_b{j}' for j in range(Bm.shape[1])]
        L = np.column_stack([np.ones(len(GRID)), Bg]); bb = m.params[names].values; V = cv.loc[names, names].values
        est = L @ bb; se = np.sqrt(np.einsum('ij,jk,ik->i', L, V, L))
        curves[g] = [{'age': float(a), 'OR': round(float(np.exp(e)), 4), 'ci': [round(float(np.exp(e - 1.96 * s)), 4), round(float(np.exp(e + 1.96 * s)), 4)]} for a, e, s in zip(GRID, est, se)]
    sup = {g: np.histogram(dd.loc[dd[g] == 1, 'scar_50m_min_yrs'], bins=[0, 1, 5, 10, 15, 20])[0].tolist() for g in ('rev', 'barec')}
    C5[nm] = {'knots_yr': KN.tolist(), 'curves': curves, 'rows_by_age_bin_0_1_5_10_15_20': sup, 'bare_without_catalogue_scar': lin(m, mr, {'barenc': 1}), 'class3': lin(m, mr, {'sc3': 1})}
    print('C5', nm, {g: (curves[g][0]['OR'], curves[g][19]['OR'], curves[g][-1]['OR']) for g in curves}, flush=True)
R['C5_scar_age_splines'] = C5
C6 = {}
for nm, d in SUB.items():
    m, mr, dd, nd = fitfe(d, f"case~{TERR}+sc1+sc2+sc3")
    terms = ['sc1', 'sc2', 'sc3']; cases = dd[dd.case == 1]; n = len(cases)
    rho = np.array([float(cases[t].mean()) for t in terms]); full = np.append(rho, 1 - rho.sum())
    b = m.params[terms].values; V = mr.cov_params().loc[terms, terms].values; draws = rng.multivariate_normal(b, V, size=NDRAW)
    f = lambda bv, r: float(sum(r[i] * (1 - math.exp(-bv[i])) for i in range(3)))
    pt = f(b, rho); sims = np.array([f(bv, rng.multinomial(n, full)[:-1] / n) for bv in draws])
    lz = np.log(1 - sims); s = float(np.std(lz, ddof=1)); c = math.log(1 - pt)
    C6[nm] = {'AF': round(pt, 4), 'ci_mc_percentile': [round(float(np.percentile(sims, 2.5)), 4), round(float(np.percentile(sims, 97.5)), 4)],
              'ci_log1mAF': [round(1 - math.exp(c + 1.96 * s), 4), round(1 - math.exp(c - 1.96 * s), 4)]}
R['C6_af_interval'] = C6
print('C6', C6, flush=True)
BR = pd.read_csv(H + '/v10/an/g3/bright_rows_v1.csv', dtype={'uid': str})
Mn = EV.copy(); Mn['R1'] = (Mn.class1_50m == 1).astype(int); Mn['R3'] = ((Mn.lt_detected == 1) & (Mn.R1 == 0)).astype(int)
Rp = pd.read_csv(H + '/v10/an/g5/rep_rows_tagged_v1.csv', low_memory=False); Rp = Rp[Rp.TypeName == BRO].copy() if 'TypeName' in Rp else Rp
Rp['case'] = Rp.case.astype(int); Rp['slope2'] = Rp.slope ** 2; Rp = Rp.dropna(subset=TV)
def strat_auc(d, s):
    num = 0.0; den = 0.0
    for ev, g in d.assign(s=s).groupby('event'):
        y = g.case.values; n1 = y.sum(); n0 = len(y) - n1
        if n1 == 0 or n0 == 0: continue
        r = pd.Series(g.s.values).rank().values; a = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        num += a * n1 * n0; den += n1 * n0
    return num / den
C7 = {}
for nm in ('all', 'le39.5'):
    A_ = Mn if nm == 'all' else Mn[Mn.slope <= 39.5]; B_ = Rp if nm == 'all' else Rp[Rp.slope <= 39.5]
    o = {}
    for tr_lab, tr, te_lab, te in (('2018_2025', A_, '2009_2017', B_), ('2009_2017', B_, '2018_2025', A_)):
        for mod, f in (('terrain', ''), ('terrain+legacy', '+R1+R3')):
            dtr, _ = drop_uninf(tr); mm = smf.logit(f"case~{TERR}{f}+C(event)", data=dtr).fit(disp=0, maxiter=100, method='newton')
            cols = TV + (['R1', 'R3'] if f else [])
            s_te = te[cols].values @ mm.params[cols].values; s_tr = dtr[cols].values @ mm.params[cols].values
            o[f'{tr_lab}_to_{te_lab}_{mod}'] = {'auc_within_event_test': round(strat_auc(te, s_te), 4), 'auc_within_event_train': round(strat_auc(dtr, s_tr), 4)}
    for a in ('2018_2025_to_2009_2017', '2009_2017_to_2018_2025'):
        o[f'{a}_gain'] = round(o[f'{a}_terrain+legacy']['auc_within_event_test'] - o[f'{a}_terrain']['auc_within_event_test'], 4)
    C7[nm] = o
    print('C7', nm, o, flush=True)
R['C7_transfer'] = C7
json.dump(R, open(OUT + '/methods_register_v1.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print('written', round(time.time() - t0), 's')
