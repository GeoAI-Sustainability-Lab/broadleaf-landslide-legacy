"""Terrain and rainfall expected-risk model (out-of-fold), risk quintiles and the composition of failed and unfailed sites. Needs the public bamboo-landslide-terrain repository for the event rainfall table (environment variable RAIN_DIR)."""
import os, json, math, warnings
import numpy as np, pandas as pd, rasterio
import statsmodels.formula.api as smf
from scipy.stats import norm
from pyproj import Transformer
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); G1 = os.path.join(os.path.dirname(HERE), 'g1')
RAIN = os.environ.get('RAIN_DIR', '@BAMBOO_REPO@/extended_analysis')
SEED = 20260928; NDRAW = 4000
BRO = '闊葉樹林型'; TERR = "slope+slope2+north+east+planc+profc+elev"; RAINF = "lmax24+ltotal"
R = {'charter': 'charter_addendum_3_unfailed_high_risk_sites (2026-09-28, frozen before fitting)', 'seed': SEED}

E = pd.read_csv('@WORK@/v10/data/tags_event_v4.csv', dtype={'uid': str}, low_memory=False)
E['case'] = E.case.astype(int); E['slope2'] = E.slope ** 2
E = E.dropna(subset=['slope', 'north', 'east', 'planc', 'profc', 'elev']).reset_index(drop=True)
E['legacy_any4'] = (E.site_class4 > 0).astype(int)
for c in (1, 2, 3): E[f'sc{c}'] = (E.site_class4 == c).astype(int)

tr = Transformer.from_crs(3826, 4326, always_xy=True)
def sample(fn, xs, ys):
    with rasterio.open(fn) as src:
        arr = src.read(); T = src.transform
    lon, lat = tr.transform(np.asarray(xs, float), np.asarray(ys, float))
    col = np.floor((lon - T.c) / T.a).astype(int); row = np.floor((lat - T.f) / T.e).astype(int)
    ok = (row >= 0) & (row < arr.shape[1]) & (col >= 0) & (col < arr.shape[2])
    out = np.full((arr.shape[0], len(lon)), np.nan); out[:, ok] = arr[:, row[ok], col[ok]]
    return out
vals = np.full((3, len(E)), np.nan); missing_raster = []
for ev, g in E.groupby('event'):
    fn = os.path.join(RAIN, 'gee', 'rain', ev.replace('、', '_').replace('/', '_') + '.tif')
    if not os.path.exists(fn): missing_raster.append(ev); continue
    vals[:, g.index.values] = sample(fn, g.x.values, g.y.values)
E['total'] = vals[0]; E['max24'] = vals[1]; E['max72'] = vals[2]
for k in ('total', 'max24', 'max72'): E['l' + k] = np.log1p(E[k])
n_before = len(E); E = E.dropna(subset=['lmax24', 'ltotal']).reset_index(drop=True)
R['rain'] = {'rows_with_rain': int(len(E)), 'rows_dropped_no_rain': int(n_before - len(E)), 'events_missing_raster': missing_raster,
             'median_case_max24_mm': round(float(E[E.case == 1].max24.median()), 1), 'median_control_max24_mm': round(float(E[E.case == 0].max24.median()), 1),
             'median_case_total_mm': round(float(E[E.case == 1].total.median()), 1), 'median_control_total_mm': round(float(E[E.case == 0].total.median()), 1)}
P = pd.read_csv(os.path.join(RAIN, 'verify', 'rain_exposure_table.csv'))
P['uid'] = 'case:' + P.lid.astype(str)
chk = E[E.case == 1][['uid', 'max24', 'total']].merge(P[['uid', 'max24', 'total']], on='uid', suffixes=('', '_parent'))
R['rain']['parent_check'] = {'cases_matched': int(len(chk)), 'max_abs_diff_max24_mm': round(float((chk.max24 - chk.max24_parent).abs().max()), 6),
                              'max_abs_diff_total_mm': round(float((chk.total - chk.total_parent).abs().max()), 6)}

E['block'] = np.floor(E.x / 20000).astype(int).astype(str) + '_' + np.floor(E.y / 20000).astype(int).astype(str)
ub = np.array(sorted(E.block.unique())); rng = np.random.default_rng(SEED)
fold_of = {b: int(f) for b, f in zip(ub, rng.integers(0, 5, size=len(ub)))}; E['fold'] = E.block.map(fold_of)
def auc(y, s):
    y = np.asarray(y); s = np.asarray(s); r = pd.Series(s).rank().values
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
FORM_TR = f"case~{TERR}+{RAINF}"; FORM_T = f"case~{TERR}"
E['score'] = np.nan; E['score_terrain'] = np.nan; folds = []
for f in range(5):
    tr_ = E[E.fold != f]; te = E.fold == f
    m = smf.logit(FORM_TR, data=tr_).fit(disp=0, maxiter=200); mt = smf.logit(FORM_T, data=tr_).fit(disp=0, maxiter=200)
    E.loc[te, 'score'] = np.asarray(m.predict(E[te], which='linear'))
    E.loc[te, 'score_terrain'] = np.asarray(mt.predict(E[te], which='linear'))
    folds.append({'fold': f, 'n_test': int(te.sum()), 'cases_test': int(E[te].case.sum()), 'auc_terrain_rain': round(auc(E[te].case, E[te].score), 4),
                  'auc_terrain_only': round(auc(E[te].case, E[te].score_terrain), 4)})
mfull = smf.logit(FORM_TR, data=E).fit(disp=0, maxiter=200)
mfull_r = smf.logit(FORM_TR, data=E).fit(disp=0, maxiter=200, cov_type='cluster', cov_kwds={'groups': pd.factorize(E.event)[0]})
E['score_insample'] = np.asarray(mfull.predict(E, which='linear'))
R['expected_risk'] = {'formula': FORM_TR, 'n_rows': int(len(E)), 'n_cases': int(E.case.sum()), 'n_blocks': int(len(ub)), 'folds': folds,
                      'auc_oof_terrain_rain': round(auc(E.case, E.score), 4), 'auc_oof_terrain_only': round(auc(E.case, E.score_terrain), 4),
                      'coef_full': {k: {'beta': round(float(mfull.params[k]), 4), 'OR': round(math.exp(mfull.params[k]), 4),
                                        'ci_cluster_event': [round(math.exp(mfull_r.params[k] - 1.96 * mfull_r.bse[k]), 4), round(math.exp(mfull_r.params[k] + 1.96 * mfull_r.bse[k]), 4)]}
                                    for k in ['slope', 'slope2', 'lmax24', 'ltotal']}}
ctrl = E.case == 0
def strata(col, q=5):
    cuts = np.quantile(E.loc[ctrl, col], np.linspace(0, 1, q + 1)[1:-1])
    return np.searchsorted(cuts, E[col].values, side='right') + 1, [round(float(c), 4) for c in cuts]
E['Q'], cutsQ = strata('score'); E['D'], cutsD = strata('score', 10); E['Qin'], _ = strata('score_insample'); E['Qt'], _ = strata('score_terrain')
R['expected_risk']['quintile_cuts_oof'] = cutsQ
R['expected_risk']['cases_share_by_Q_all_types'] = {int(k): round(float(v), 4) for k, v in (E[E.case == 1].Q.value_counts(normalize=True).sort_index()).items()}

B = E[E.ft == BRO].copy()
def drop_uninf(d):
    g = d.groupby('event').case.agg(['sum', 'size']); keep = g[(g['sum'] > 0) & (g['sum'] < g['size'])].index
    return d[d.event.isin(keep)].copy(), int(len(g) - len(keep))
def fitfe(d, form):
    d, nd = drop_uninf(d); f2 = form + '+C(event)'
    m = smf.logit(f2, data=d).fit(disp=0, maxiter=100, method='newton')
    mr = smf.logit(f2, data=d).fit(disp=0, maxiter=100, method='newton', cov_type='cluster', cov_kwds={'groups': pd.factorize(d.uid)[0]})
    return m, mr, d, nd
def term(m, mr, t):
    b = m.params[t]; s = m.bse[t]; sr = mr.bse[t]
    return {'OR': round(math.exp(b), 4), 'ci': [round(math.exp(b - 1.96 * sr), 4), round(math.exp(b + 1.96 * sr), 4)],
            'ci_model': [round(math.exp(b - 1.96 * s), 4), round(math.exp(b + 1.96 * s), 4)], 'p': float(2 * (1 - norm.cdf(abs(b / sr))))}
def wald(mr, ta, tb):
    b = mr.params[ta] - (mr.params[tb] if tb else 0.0); c = mr.cov_params()
    se = math.sqrt(c.loc[ta, ta] + (c.loc[tb, tb] - 2 * c.loc[ta, tb] if tb else 0.0))
    return {'ratio': round(math.exp(b), 4), 'ci': [round(math.exp(b - 1.96 * se), 4), round(math.exp(b + 1.96 * se), 4)], 'p': float(2 * (1 - norm.cdf(abs(b / se))))}

def run_all(B, qcol, label):
    out = {'label': label}
    B = B.copy(); B['Qc'] = pd.Categorical(B[qcol].astype(int), categories=sorted(B[qcol].unique().astype(int)))
    qs = list(B.Qc.cat.categories); top = qs[-1]
    comp = {}
    for q in qs:
        s = B[B[qcol] == q]
        comp[int(q)] = {'cases': {int(k): int(v) for k, v in s[s.case == 1].site_class4.value_counts().sort_index().items()},
                        'control_rows': {int(k): int(v) for k, v in s[s.case == 0].site_class4.value_counts().sort_index().items()},
                        'n_cases': int(s.case.sum()), 'n_control_rows': int((s.case == 0).sum())}
        comp[int(q)]['intact_share_cases'] = round(comp[int(q)]['cases'].get(0, 0) / max(1, comp[int(q)]['n_cases']), 4)
        comp[int(q)]['intact_share_controls'] = round(comp[int(q)]['control_rows'].get(0, 0) / max(1, comp[int(q)]['n_control_rows']), 4)
    out['B1_composition'] = comp
    S = B[B[qcol] == top]
    m, mr, d, nd = fitfe(S, f"case~{TERR}+{RAINF}+sc1+sc2+sc3")
    beta = np.array([mr.params[f'sc{c}'] for c in (1, 2, 3)]); cov = mr.cov_params().loc[[f'sc{c}' for c in (1, 2, 3)], [f'sc{c}' for c in (1, 2, 3)]].values
    ncase = int(d.case.sum()); cnt = np.array([int(((d.case == 1) & (d.site_class4 == c)).sum()) for c in (0, 1, 2, 3)]); rho = cnt / cnt.sum()
    rg = np.random.default_rng(SEED); draws = []
    for _ in range(NDRAW):
        b = rg.multivariate_normal(beta, cov); r = rg.multinomial(ncase, rho) / ncase
        draws.append([r[c] * (1 - math.exp(-b[c - 1])) for c in (1, 2, 3)])
    draws = np.array(draws); pt = [rho[c] * (1 - math.exp(-beta[c - 1])) for c in (1, 2, 3)]
    out['B1b_AF_top'] = {'stratum': int(top), 'n_cases': ncase, 'events_dropped': nd,
                        'OR': {f'sc{c}': term(m, mr, f'sc{c}') for c in (1, 2, 3)},
                        'AF': {f'AF{c}': {'point': round(pt[c - 1], 4), 'ci_mc': [round(float(np.percentile(draws[:, c - 1], 2.5)), 4), round(float(np.percentile(draws[:, c - 1], 97.5)), 4)]} for c in (1, 2, 3)},
                        'AF_total': {'point': round(sum(pt), 4), 'ci_mc': [round(float(np.percentile(draws.sum(1), 2.5)), 4), round(float(np.percentile(draws.sum(1), 97.5)), 4)]}}
    m, mr, d, nd = fitfe(B, f"case~C(Qc)+C(Qc):legacy_any4+{TERR}+{RAINF}")
    out['B2_legacy_by_stratum'] = {int(q): term(m, mr, f'C(Qc)[{q}]:legacy_any4') for q in qs}
    B['qrank'] = B[qcol].astype(int) - 1
    m, mr, d, nd = fitfe(B, f"case~C(Qc)+legacy_any4+legacy_any4:qrank+{TERR}+{RAINF}")
    out['B2_trend_per_stratum_step'] = term(m, mr, 'legacy_any4:qrank'); out['B2_events_dropped'] = nd
    B['cell'] = B[qcol].astype(int).astype(str) + np.where(B.legacy_any4 == 1, '_legacy', '_intact')
    cats = [f'{int(q)}_{s}' for q in qs for s in ('intact', 'legacy')]
    B['cell'] = pd.Categorical(B.cell, categories=cats)
    m, mr, d, nd = fitfe(B, "case~C(cell)")
    cells = {c: ({'OR': 1.0, 'ci': [1.0, 1.0]} if c == cats[0] else term(m, mr, f'C(cell)[T.{c}]')) for c in cats}
    cnts = d.groupby('cell', observed=True).case.agg(['sum', 'size'])
    for c in cats: cells[c]['n_cases'] = int(cnts.loc[c, 'sum']) if c in cnts.index else 0; cells[c]['n_control_rows'] = int(cnts.loc[c, 'size'] - cnts.loc[c, 'sum']) if c in cnts.index else 0
    out['B3_cells'] = cells; out['B3_events_dropped'] = nd
    mid = qs[len(qs) // 2]
    out['B3_contrasts'] = {f'{int(top)}_intact_vs_{int(qs[0])}_legacy': wald(mr, f'C(cell)[T.{int(top)}_intact]', f'C(cell)[T.{int(qs[0])}_legacy]'),
                           f'{int(top)}_intact_vs_{int(mid)}_legacy': wald(mr, f'C(cell)[T.{int(top)}_intact]', f'C(cell)[T.{int(mid)}_legacy]'),
                           f'{int(top)}_legacy_vs_{int(top)}_intact': wald(mr, f'C(cell)[T.{int(top)}_legacy]', f'C(cell)[T.{int(top)}_intact]')}
    return out

R['main_oof_quintiles'] = run_all(B, 'Q', 'broadleaf, all slopes, out-of-fold expected risk quintiles (main)')
R['sens_insample'] = run_all(B, 'Qin', 'in-sample expected risk quintiles')
R['sens_terrain_only'] = run_all(B, 'Qt', 'terrain-only expected risk quintiles (no rainfall)')
R['sens_le39_5'] = run_all(B[B.slope <= 39.5], 'Q', 'broadleaf slope <= 39.5, main quintiles')
R['sens_deciles'] = run_all(B, 'D', 'out-of-fold expected risk deciles')

A = pd.read_csv(os.path.join(HERE, 'analysis_rows_v1.csv'), low_memory=False)[['uid', 'chm', 'chm_valid', 'chm_date', 'stand_function']]
X = B.merge(A, on='uid', how='left')
fmap = {'原生林': 'primary', '經改造天然林': 'modified_natural', '保護性人工林': 'plantation', '生產性人工林': 'plantation'}
X['func3'] = X.stand_function.map(fmap)
S = X[(X.site_class4 == 0) & (X.Q >= 4) & (X.year >= 2020) & X.chm.notna() & X.chm_valid.astype(bool) & X.func3.notna()].copy()
S['chm5'] = S.chm / 5.0; S['chm12'] = (S.chm >= 12).astype(int)
S['func3'] = pd.Categorical(S.func3, categories=['primary', 'modified_natural', 'plantation'])
b4 = {'n_rows': int(len(S)), 'n_cases': int(S.case.sum()), 'n_control_rows': int((S.case == 0).sum()),
      'chm_date_max': str(pd.to_datetime(S.chm_date, errors='coerce').max().date()),
      'excluded_function_classes': ['半天然林', '未確認'],
      'function_counts': {k: {'cases': int(((S.func3 == k) & (S.case == 1)).sum()), 'control_rows': int(((S.func3 == k) & (S.case == 0)).sum())} for k in ['primary', 'modified_natural', 'plantation']},
      'chm_median': {'cases': float(S[S.case == 1].chm.median()), 'control_rows': float(S[S.case == 0].chm.median())}}
m, mr, d, nd = fitfe(S, f"case~chm5+C(func3)+{TERR}+{RAINF}")
b4['model_chm_per5m'] = {'chm_per_5m': term(m, mr, 'chm5'), 'modified_natural_vs_primary': term(m, mr, 'C(func3)[T.modified_natural]'),
                         'plantation_vs_primary': term(m, mr, 'C(func3)[T.plantation]'), 'events_dropped': nd, 'n_used': int(len(d)), 'cases_used': int(d.case.sum())}
m, mr, d, nd = fitfe(S, f"case~chm12+C(func3)+{TERR}+{RAINF}")
b4['model_chm_ge12'] = {'chm_ge12': term(m, mr, 'chm12'), 'modified_natural_vs_primary': term(m, mr, 'C(func3)[T.modified_natural]'),
                        'plantation_vs_primary': term(m, mr, 'C(func3)[T.plantation]'), 'events_dropped': nd}
S2 = X[(X.site_class4 == 0) & (X.Q <= 3) & (X.year >= 2020) & X.chm.notna() & X.chm_valid.astype(bool) & X.func3.notna()].copy()
S2['chm5'] = S2.chm / 5.0; S2['func3'] = pd.Categorical(S2.func3, categories=['primary', 'modified_natural', 'plantation'])
m, mr, d, nd = fitfe(S2, f"case~chm5+C(func3)+{TERR}+{RAINF}")
b4['contrast_Q1_Q3_chm_per5m'] = {'chm_per_5m': term(m, mr, 'chm5'), 'modified_natural_vs_primary': term(m, mr, 'C(func3)[T.modified_natural]'),
                                   'plantation_vs_primary': term(m, mr, 'C(func3)[T.plantation]'), 'n_used': int(len(d)), 'cases_used': int(d.case.sum())}
b4['posthoc_share_chm_lt2'] = {'cases': round(float((S[S.case == 1].chm < 2).mean()), 4), 'control_rows': round(float((S[S.case == 0].chm < 2).mean()), 4)}
S3 = S[S.chm >= 2].copy()
m, mr, d, nd = fitfe(S3, f"case~chm5+C(func3)+{TERR}+{RAINF}")
b4['posthoc_exclude_chm_lt2'] = {'chm_per_5m': term(m, mr, 'chm5'), 'n_used': int(len(d)), 'cases_used': int(d.case.sum())}
R['B4_forest_structure_intact_Q4_Q5'] = b4

BM = E[(E.ft == '竹林')]
R['bamboo_descriptive_top'] = {'cases_Q5': int(((BM.case == 1) & (BM.Q == 5)).sum()), 'control_rows_Q5': int(((BM.case == 0) & (BM.Q == 5)).sum())}

E[['uid', 'kind', 'ft', 'event', 'year', 'case', 'site_class4', 'slope', 'max24', 'total', 'score', 'score_terrain', 'score_insample', 'Q', 'D', 'Qt', 'Qin', 'fold']].to_csv(os.path.join(HERE, 'bright_rows_v1.csv'), index=False)
json.dump(R, open(os.path.join(HERE, 'bright_register_v1.json'), 'w'), ensure_ascii=False, indent=1)
print(json.dumps({k: R[k] for k in ['rain', 'expected_risk']}, ensure_ascii=False, indent=1)[:3000])
