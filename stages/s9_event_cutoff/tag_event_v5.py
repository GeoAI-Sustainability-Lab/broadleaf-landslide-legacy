"""Event-matched legacy tags with one cutoff date per event (the earliest pre-event image date of the event, for cases and controls). MODE=common writes tags_event_v5.csv and tag_register_v5_common.json; MODE=own rebuilds tags_event_v4.csv column by column."""
import json, os, sys, numpy as np, pandas as pd
H = '@WORK@'
IN, OUT = H + '/v27/in/', H + '/v27/data/'
MODE = os.environ.get('MODE', 'common'); assert MODE in ('own', 'common')
D = pd.read_csv(IN + 'point_dates_v1.csv', dtype={'uid': str, 'lid': str}, parse_dates=['own_before', 'own_before_max', 'own_after', 'cutoff_isl', 'isl_cutoff'])
cat = pd.read_csv(IN + 'catalog_matches_within100m_v1.csv', dtype={'uid': str, 'record_id': str}, parse_dates=['before', 'before_max', 'after', 'after_min'])
st = pd.read_csv(IN + 'annual_state_matches_within50m_v1.csv', dtype={'uid': str}, parse_dates=['img_date'])
own_ids = set((D[D.kind == 'case'].assign(rid=lambda d: d.lid.str.split('_').str[0] + ':' + d.own_record_index.astype(int).astype(str))).rid)
cat['is_own'] = cat.record_id.isin(own_ids) & (cat.uid.str.startswith('case:')) & (cat.uid.str[5:] == cat.cat_year.astype(str) + '_' + cat.record_index.astype(str))
EM = pd.read_csv(IN + 'control_event_footprint_membership_v1.csv', dtype={'uid': str}, parse_dates=['cutoff_event_min', 'cutoff_event_max'])
ctrl = EM.merge(D[['uid', 'kind', 'ft', 'slope']], on='uid').rename(columns={'cutoff_event_min': 'cutoff'}); ctrl['year'] = ctrl.year.astype(str)
cas = D[D.kind == 'case'][['uid', 'kind', 'ft', 'slope', 'year', 'event', 'own_before']].rename(columns={'own_before': 'cutoff_own'}); cas['year'] = cas.year.astype(str)
cas['yk'] = pd.to_numeric(cas.year).astype(int)
em_ev = EM.assign(yk=pd.to_numeric(EM.year).astype(int)).groupby(['yk', 'event']).cutoff_event_min.agg(['min', 'max'])
assert (em_ev['min'] == em_ev['max']).all(), 'controls of one event carry different cutoffs'
ev_min = em_ev['min'].rename('cutoff_event')
assert set(ev_min.index) == set(map(tuple, cas[['yk', 'event']].drop_duplicates().values)), 'events without controls'
cas = cas.merge(ev_min.reset_index(), on=['yk', 'event'], how='left'); assert cas.cutoff_event.notna().all()
assert (cas.cutoff_own >= cas.cutoff_event).all(), 'a case cutoff precedes its event cutoff'
cas['cutoff'] = cas.cutoff_own if MODE == 'own' else cas.cutoff_event
ctrl['cutoff_own'] = pd.NaT; ctrl['cutoff_event'] = ctrl.cutoff
rows = pd.concat([cas[['uid', 'kind', 'ft', 'slope', 'year', 'event', 'cutoff', 'cutoff_own', 'cutoff_event']],
                  ctrl[['uid', 'kind', 'ft', 'slope', 'year', 'event', 'cutoff', 'cutoff_own', 'cutoff_event']]], ignore_index=True)
rows['rk'] = np.arange(len(rows)); rows['Yc'] = rows.cutoff.dt.year
m = cat.merge(rows[['rk', 'uid', 'cutoff', 'Yc']], on='uid'); m = m[(~m.is_own) & m.after.notna() & (m.after <= m.cutoff)].copy()
m['yrs_since'] = (m.cutoff - m.after).dt.days / 365.25
R = rows.copy()
for B, name in [(0, '0m'), (50, '50m'), (100, '100m')]:
    mm = m[(m.inside == 1)] if B == 0 else m[m.distance_m <= B]
    g = mm.groupby('rk').agg(n_rec=('record_id', 'size'), n_years=('cat_year', 'nunique'), last_after=('after', 'max'), first_after=('after', 'min'),
                             min_yrs=('yrs_since', 'min'), max_yrs=('yrs_since', 'max'), max_area_ha=('Area_ha', 'max'))
    R = R.merge(g.add_prefix(f'scar_{name}_'), left_on='rk', right_index=True, how='left'); R[f'class1_{name}'] = R[f'scar_{name}_n_rec'].notna().astype(int)
s = st.merge(rows[['rk', 'uid', 'cutoff', 'Yc']], on='uid'); s = s[s.img_date.notna() & (s.img_date <= s.cutoff) & (s.state_year >= s.Yc - 1)]
for B, name in [(0, '0m'), (50, '50m')]:
    ss = s[s.inside == 1] if B == 0 else s
    g = ss.groupby('rk').agg(bare_years=('state_year', 'nunique'), bare_last_img=('img_date', 'max'), bare_area_m2=('Area', 'max'))
    R = R.merge(g.add_prefix(f'bare_{name}_'), left_on='rk', right_index=True, how='left'); R[f'bare_at_cutoff_{name}'] = R[f'bare_{name}_bare_years'].notna().astype(int)
s2 = st.merge(rows[['rk', 'uid', 'cutoff', 'Yc']], on='uid'); s2 = s2[s2.img_date.notna() & (s2.img_date <= s2.cutoff) & (s2.inside == 1)]
ever = s2.groupby('rk').agg(bare_ever_years=('state_year', 'nunique'), bare_ever_first=('state_year', 'min'), bare_ever_last=('state_year', 'max'))
R = R.merge(ever, left_on='rk', right_index=True, how='left'); R['bare_ever'] = R.bare_ever_years.notna().astype(int)
R['design'] = 'event'
R['legacy_scar'] = R.class1_50m; R['legacy_bare'] = R.bare_at_cutoff_50m
R['legacy_scar_0m'] = R.class1_0m; R['legacy_bare_0m'] = R.bare_at_cutoff_0m; R['legacy_scar_100m'] = R.class1_100m
R['site_class'] = np.where(R.legacy_bare == 1, 2, np.where(R.legacy_scar == 1, 1, 0))
R['bare_subtype'] = np.where(R.site_class == 2, np.where(R.class1_50m == 1, 'bare_with_catalog_scar', 'bare_without_catalog_scar'), 'not_bare')
a = R.scar_50m_min_yrs
R['tss_bin'] = np.select([R.class1_50m == 0, a < 5, a < 10, a < 20], ['none', '0-5', '5-10', '10-20'], '20+')
V3 = pd.read_csv(IN + 'tags_event_v3.csv', dtype={'uid': str}, low_memory=False, parse_dates=['cutoff'])
PT = ['case', 'slope2', 'north', 'east', 'planc', 'profc', 'elev', 'd_road', 'd_river', 'old_scar', 'x', 'y']
assert (V3.groupby('uid')[PT].nunique(dropna=False) <= 1).all().all(), 'point-level columns vary within uid'
R = R.merge(V3.drop_duplicates('uid')[['uid'] + PT], on='uid', how='left')
S = pd.read_csv(IN + 'lt_summary_v1.csv', dtype={'uid': str})
R['cutoff_year'] = R.cutoff.dt.year
R = R.merge(S[['uid', 'cutoff_year', 'fit', 'rmse', 'n_loss_segments', 'dist_year', 'dist_mag', 'dist_dur', 'TSD', 'recovery_ratio', 'end_fit']], on=['uid', 'cutoff_year'], how='left')
R['lt_detected'] = R.dist_year.notna().astype(int)
R['class3'] = ((R.site_class == 0) & (R.lt_detected == 1)).astype(int)
R['site_class4'] = np.where(R.class3 == 1, 3, R.site_class)
R['tsd3_bin'] = np.select([R.class3 == 0, R.TSD < 5, R.TSD < 10, R.TSD < 20], ['none', '0-5', '5-10', '10-20'], '20+')
R['lt_missing'] = R.fit.isna().astype(int)
R['year'] = R.year.astype(float)
REG = {'mode': MODE, 'rows': int(len(R)), 'cases': int((R.kind == 'case').sum()),
       'case_cutoff_year_differs_from_event_year': int((R[R.kind == 'case'].cutoff_own.dt.year != R[R.kind == 'case'].cutoff_event.dt.year).sum())}
assert REG['case_cutoff_year_differs_from_event_year'] == 0
assert R[R.kind == 'case'].fit.notna().sum() == (R[R.kind == 'case'].merge(S[['uid', 'cutoff_year']], on=['uid', 'cutoff_year']).shape[0])
V4 = pd.read_csv(H + '/v11/data/tags_event_v4.csv', dtype={'uid': str}, low_memory=False)
assert len(R) == len(V4) and not R.duplicated(['uid', 'event']).any()
J = V4.merge(R.drop(columns=['rk']), on=['uid', 'event'], suffixes=('_v4', '_new'), how='inner'); assert len(J) == len(V4)
def same(a, b):
    if a.dtype.kind in 'fciub' or b.dtype.kind in 'fciub':
        a = pd.to_numeric(a.astype(object).map(lambda v: float(v) if isinstance(v, (bool, np.bool_)) else v), errors='coerce').astype(float)
        b = pd.to_numeric(b.astype(object).map(lambda v: float(v) if isinstance(v, (bool, np.bool_)) else v), errors='coerce').astype(float)
        return ((a - b).abs() < 1e-9) | (a.isna() & b.isna())
    na_a = a.isna().values; na_b = b.isna().values
    a = a.astype(object).map(lambda v: str(v)[:10]); b = b.astype(object).map(lambda v: str(v)[:10])
    return pd.Series((na_a & na_b) | ((~na_a) & (~na_b) & (a.values == b.values)), index=a.index)
cols = [c for c in V4.columns if c not in ('uid', 'event')]
diff = {c: int((~same(J[c + '_v4'], J[c + '_new'])).sum()) for c in cols}
case = J.kind_v4 == 'case'
if MODE == 'own':
    bad = {c: n for c, n in diff.items() if n}
    assert not bad, bad
    REG['reproduces_tags_event_v4'] = True
else:
    ctl_diff = {c: int((~same(J.loc[~case, c + '_v4'], J.loc[~case, c + '_new'])).sum()) for c in cols if c != 'cutoff'}
    assert not any(ctl_diff.values()), {k: v for k, v in ctl_diff.items() if v}
    eq = case & (J.cutoff_own.astype(str).str[:10] == J.cutoff_v4.astype(str).str[:10]) & (J.cutoff_own.astype(str).str[:10] == J.cutoff_event.astype(str).str[:10])
    same_rows = {c: int((~same(J.loc[eq, c + '_v4'], J.loc[eq, c + '_new'])).sum()) for c in cols}
    assert not any(same_rows.values()), {k: v for k, v in same_rows.items() if v}
    JB = J[case & (J.ft_v4 == '闊葉樹林型')]
    tr = JB.groupby(['site_class4_v4', 'site_class4_new']).size()
    REG['broadleaf_case_transitions'] = {f'{int(a)}->{int(b)}': int(n) for (a, b), n in tr.items() if a != b}
    REG['broadleaf_cases_reclassified'] = int((JB.site_class4_v4 != JB.site_class4_new).sum())
    JM = J[case & (J.ft_v4 == '竹林')]
    REG['bamboo_cases_reclassified'] = int((JM.site_class4_v4 != JM.site_class4_new).sum())
    REG['columns_changed_case_rows'] = {c: n for c, n in diff.items() if n}
    REG['case_rows_own_equals_event_cutoff'] = int(eq.sum())
    RV = json.load(open(H + '/reference/reference_values_s9.json', encoding='utf-8'))['RV']['B_common_cutoff']
    assert REG['broadleaf_cases_reclassified'] == RV['cases_reclassified_all'], (REG['broadleaf_cases_reclassified'], RV['cases_reclassified_all'])
    assert REG['broadleaf_case_transitions'] == RV['case_transitions_all'], (REG['broadleaf_case_transitions'], RV['case_transitions_all'])
out = R.drop(columns=['rk'])
fn = OUT + ('tags_event_v4_rebuilt.csv' if MODE == 'own' else 'tags_event_v5.csv')
out.to_csv(fn, index=False, encoding='utf-8-sig')
json.dump(REG, open(OUT + f'tag_register_v5_{MODE}.json', 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(REG, ensure_ascii=False, indent=1, default=str))
