"""Event-matched legacy tags evaluated per point and event (each row with its own cutoff): prior scars, still-bare ground, uncatalogued disturbance, site classes 0-3. Writes tags_event_v4.csv and tag_fix_register_v1.json."""
import json, os, numpy as np, pandas as pd
H = '@WORK@'
G0, G1, OUT = H + '/g0/', H + '/g1/', H + '/v10/data/'
os.makedirs(OUT, exist_ok=True)
D = pd.read_csv(G0 + 'point_dates_v1.csv', dtype={'uid': str, 'lid': str}, parse_dates=['own_before', 'own_before_max', 'own_after', 'cutoff_isl', 'isl_cutoff'])
cat = pd.read_csv(G0 + 'catalog_matches_within100m_v1.csv', dtype={'uid': str, 'record_id': str}, parse_dates=['before', 'before_max', 'after', 'after_min'])
st = pd.read_csv(G0 + 'annual_state_matches_within50m_v1.csv', dtype={'uid': str}, parse_dates=['img_date'])
own_ids = set((D[D.kind == 'case'].assign(rid=lambda d: d.lid.str.split('_').str[0] + ':' + d.own_record_index.astype(int).astype(str))).rid)
cat['is_own'] = cat.record_id.isin(own_ids) & (cat.uid.str.startswith('case:')) & (cat.uid.str[5:] == cat.cat_year.astype(str) + '_' + cat.record_index.astype(str))
EM = pd.read_csv(G0 + 'control_event_footprint_membership_v1.csv', dtype={'uid': str}, parse_dates=['cutoff_event_min', 'cutoff_event_max'])
ctrl = EM.merge(D[['uid', 'kind', 'ft', 'slope']], on='uid').rename(columns={'cutoff_event_min': 'cutoff'}); ctrl['year'] = ctrl.year.astype(str)
cas = D[D.kind == 'case'][['uid', 'kind', 'ft', 'slope', 'year', 'event', 'own_before']].rename(columns={'own_before': 'cutoff'}); cas['year'] = cas.year.astype(str)
rows = pd.concat([cas, ctrl[['uid', 'kind', 'ft', 'slope', 'year', 'event', 'cutoff']]], ignore_index=True)
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
V3 = pd.read_csv(G1 + 'tags_event_v3.csv', dtype={'uid': str}, low_memory=False, parse_dates=['cutoff'])
PT = ['case', 'slope2', 'north', 'east', 'planc', 'profc', 'elev', 'd_road', 'd_river', 'old_scar', 'x', 'y']
nun = V3.groupby('uid')[PT].nunique(dropna=False)
assert (nun <= 1).all().all(), 'point-level columns vary within uid'
R = R.merge(V3.drop_duplicates('uid')[['uid'] + PT], on='uid', how='left')
S = pd.read_csv(G1 + 'lt_summary_v1.csv', dtype={'uid': str})
R['cutoff_year'] = R.cutoff.dt.year
R = R.merge(S[['uid', 'cutoff_year', 'fit', 'rmse', 'n_loss_segments', 'dist_year', 'dist_mag', 'dist_dur', 'TSD', 'recovery_ratio', 'end_fit']], on=['uid', 'cutoff_year'], how='left')
R['lt_detected'] = R.dist_year.notna().astype(int)
R['class3'] = ((R.site_class == 0) & (R.lt_detected == 1)).astype(int)
R['site_class4'] = np.where(R.class3 == 1, 3, R.site_class)
R['tsd3_bin'] = np.select([R.class3 == 0, R.TSD < 5, R.TSD < 10, R.TSD < 20], ['none', '0-5', '5-10', '10-20'], '20+')
R['lt_missing'] = R.fit.isna().astype(int)
R['year'] = R.year.astype(float)
assert len(R) == len(V3) and not R.duplicated(['uid', 'event']).any()
J = V3.merge(R, on=['uid', 'event'], suffixes=('_v3', '_v4'), how='inner'); assert len(J) == len(V3)
assert (J.cutoff_v3 == J.cutoff_v4).all(), 'cutoff differs'
ncut = R[R.kind == 'control'].groupby('uid').cutoff.nunique()
single = set(ncut[ncut == 1].index) | set(R[R.kind == 'case'].uid)
CMP = ['scar_50m_min_yrs', 'scar_0m_min_yrs', 'scar_100m_min_yrs', 'class1_0m', 'class1_50m', 'class1_100m',
       'bare_at_cutoff_0m', 'bare_at_cutoff_50m', 'bare_ever', 'site_class', 'bare_subtype', 'tss_bin', 'site_class4', 'tsd3_bin', 'lt_detected', 'class3']
CNT = ['scar_0m_n_rec', 'scar_50m_n_rec', 'scar_100m_n_rec']
def same(a, b):
    if a.dtype.kind in 'fc' or b.dtype.kind in 'fc':
        a = pd.to_numeric(a, errors='coerce'); b = pd.to_numeric(b, errors='coerce'); return ((a - b).abs() < 1e-9) | (a.isna() & b.isna())
    return (a.astype(str) == b.astype(str))
chk = {}
for c in CMP + CNT:
    eq = same(J[c + '_v3'], J[c + '_v4']); u = J.uid.isin(single)
    chk[c] = {'unaffected_rows_equal': int(eq[u].sum()), 'unaffected_rows': int(u.sum()), 'affected_rows_changed': int((~eq[~u]).sum()), 'affected_rows': int((~u).sum())}
    if c in CMP: assert eq[u].all(), (c, 'unaffected rows differ')
out = R.drop(columns=['rk'])
out.to_csv(OUT + 'tags_event_v4.csv', index=False, encoding='utf-8-sig')
JB = J[(J['ft_v3'] if 'ft_v3' in J.columns else J['ft']) == '闊葉樹林型']
reg = {'defect': 'gate-0 tag_v1.py grouped catalogue and annual-state summaries by uid; control rows of points entering several events with different cutoffs carried summaries over all events',
       'rows': int(len(R)), 'control_rows': int((R.kind == 'control').sum()), 'control_points': int(R[R.kind == 'control'].uid.nunique()),
       'control_points_multiple_cutoffs': int((ncut > 1).sum()), 'control_rows_of_points_with_multiple_cutoffs': int((~J.uid.isin(single)).sum()),
       'checks': chk,
       'broadleaf_control_rows_site_class4_v3': JB[JB.kind_v3 == 'control'].site_class4_v3.value_counts().sort_index().to_dict() if 'kind_v3' in JB else JB[JB.kind == 'control'].site_class4_v3.value_counts().sort_index().to_dict(),
       'broadleaf_control_rows_site_class4_v4': JB[JB.kind_v3 == 'control'].site_class4_v4.value_counts().sort_index().to_dict() if 'kind_v3' in JB else JB[JB.kind == 'control'].site_class4_v4.value_counts().sort_index().to_dict(),
       'broadleaf_control_rows_class_transition_v3_to_v4': {f'{a}->{b}': int(n) for (a, b), n in JB[(JB.kind_v3 if 'kind_v3' in JB else JB.kind) == 'control'].groupby(['site_class4_v3', 'site_class4_v4']).size().items() if a != b}}
json.dump(reg, open(OUT + 'tag_fix_register_v1.json', 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(reg, ensure_ascii=False, indent=1, default=str))
