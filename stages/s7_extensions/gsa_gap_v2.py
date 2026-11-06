"""Legacy cases outside the landslide geologically sensitive areas and their decomposition (disturbance, scar neighbourhood, scars newer than the area plans)."""
import json, os, numpy as np, pandas as pd
from scipy.spatial import cKDTree
H = '@WORK@'
OUT = H + '/v10/an/necessity'; os.makedirs(OUT, exist_ok=True)
BRO = '闊葉樹林型'
E = pd.read_csv(H + '/v10/data/tags_event_v4.csv', dtype={'uid': str}, low_memory=False, parse_dates=['cutoff', 'scar_50m_first_after', 'scar_50m_last_after', 'scar_0m_first_after', 'scar_0m_last_after'])
M = E[E.ft == BRO].copy(); M['case'] = M.case.astype(int)
M = M.dropna(subset=['slope', 'north', 'east', 'planc', 'profc', 'elev'])
M['R1'] = (M.class1_50m == 1).astype(int); M['R3'] = ((M.lt_detected == 1) & (M.R1 == 0)).astype(int); M['legacy'] = ((M.R1 == 1) | (M.R3 == 1)).astype(int)
Z = pd.read_csv(H + '/g5/zoning_out_v1.csv', dtype={'uid': str})
M = M.merge(Z, on='uid', how='left'); M['zin'] = M.gsmma_inside.fillna(0).astype(int)
P = pd.concat([M[['uid', 'x', 'y']], pd.read_csv(H + '/v10/an/g5/rep_rows_tagged_v1.csv', usecols=['uid', 'x', 'y'], low_memory=False)]).drop_duplicates('uid')
P = P.merge(Z, on='uid', how='inner')
ins = P[P.gsmma_inside == 1].dropna(subset=['gsmma_zone'])
tree = cKDTree(ins[['x', 'y']].values)
d, j = tree.query(M[['x', 'y']].values)
M['zone'] = np.where(M.zin == 1, M.gsmma_zone, ins.gsmma_zone.values[j]); M['zone_dist_m'] = np.where(M.zin == 1, 0.0, d)
A = pd.read_csv(H + '/litrev/dl/L0001.csv', dtype=str)
A = A[A['地質敏感區類型'] == '山崩與地滑地質敏感區']
A['code'] = np.where(A['地質敏感區名稱'].str.contains('-0'), A['地質敏感區編號'] + '_' + A['地質敏感區名稱'].str.extract(r'-(\d+)$')[0].fillna(''), A['地質敏感區編號'])
plan = A.groupby('code')['公告日期'].max().apply(lambda s: pd.Timestamp(s))
names = A.drop_duplicates('code').set_index('code')['地質敏感區名稱'].to_dict()
M['plan_date'] = M.zone.map(plan)
assert M.plan_date.notna().all(), 'zone without plan date'
M['phase'] = np.where(M.plan_date < pd.Timestamp('2018-01-01'), 'plan_2014_2016', 'plan_2021_2025')
L = M[M.slope <= 39.5].copy()
R = {'status': 'exploratory necessity check, written 2026-10-13 after all charter results; not in any charter',
     'zones': {z: {'name': names.get(z), 'plan_date': str(plan[z].date())} for z in sorted(plan.index)},
     'zone_lookup': {'inside_points': int(len(ins)), 'outside_rows_le39.5_median_dist_to_nearest_inside_m': round(float(L[L.zin == 0].zone_dist_m.median()), 1),
                     'outside_rows_le39.5_p90_dist_m': round(float(L[L.zin == 0].zone_dist_m.quantile(0.9)), 1)}}
c = L[(L.case == 1) & (L.legacy == 1)].copy()
out = c[c.zin == 0].copy()
def comp(g):
    newer = (g.scar_50m_first_after > g.plan_date) | ((g.class1_0m == 1) & (g.scar_0m_first_after > g.plan_date))
    return np.select([g.R3 == 1, newer, g.class1_0m == 1],
                     ['disturbance_without_catalogue_scar', 'scar_newer_than_zone_plan', 'older_scar_polygon_contains_point'], 'older_scar_within_50m_polygon_not_containing_point')
out['gap'] = comp(out)
v1cat = np.select([out.R3 == 1, out.scar_50m_first_after > out.plan_date, out.class1_0m == 1], ['d', 'n', 'c'], 'w')
moved = int(((v1cat == 'c') & (out.gap == 'scar_newer_than_zone_plan')).sum())
R['legacy_cases_le39.5'] = {'n': int(len(c)), 'outside': int(len(out)), 'outside_share': round(float(len(out) / len(c)), 4),
                            'outside_by_component': {k: int(v) for k, v in out.gap.value_counts().items()},
                            'outside_share_by_component': {k: round(float(v / len(out)), 4) for k, v in out.gap.value_counts().items()}}
inn = c[c.zin == 1].copy()
inn['gap'] = comp(inn)
R['legacy_cases_le39.5']['inside_by_component'] = {k: int(v) for k, v in inn.gap.value_counts().items()}
R['legacy_cases_le39.5']['v1_to_v2_moved_from_contains_point_to_newer'] = moved
cm = L[(L.case == 1) & (L.site_class4 > 0)].copy()
R['legacy_cases_le39.5_main_definition'] = {'n': int(len(cm)), 'outside': int((cm.zin == 0).sum()),
    'outside_by_class': {f'class{k}': int(v) for k, v in cm[cm.zin == 0].site_class4.value_counts().sort_index().items()},
    'inside_by_class': {f'class{k}': int(v) for k, v in cm[cm.zin == 1].site_class4.value_counts().sort_index().items()},
    'outside_class2_with_catalogue_scar': int(((cm.zin == 0) & (cm.site_class4 == 2) & (cm.class1_50m == 1)).sum()),
    'outside_class2_without_catalogue_scar': int(((cm.zin == 0) & (cm.site_class4 == 2) & (cm.class1_50m == 0)).sum())}
cov = {}
for ph, g in L.groupby('phase'):
    k = g[g.case == 1]; ctl = g[g.case == 0]; lg = k[k.legacy == 1]
    cov[ph] = {'cases': int(len(k)), 'cases_inside_share': round(float(k.zin.mean()), 4), 'control_rows': int(len(ctl)), 'control_rows_inside_share': round(float(ctl.zin.mean()), 4),
               'legacy_cases': int(len(lg)), 'legacy_cases_outside_share': round(float(1 - lg.zin.mean()), 4),
               'zones': sorted(g.zone.unique().tolist())}
R['coverage_by_plan_phase_le39.5'] = cov
chg = sorted(z for z in plan.index if plan[z] >= pd.Timestamp('2018-01-01'))
R['zones_changed_2021_2025'] = {'in_list': chg, 'n_in_list': len(chg), 'with_sample_points_le39.5': cov['plan_2021_2025']['zones'],
                                'without_sample_points_le39.5': sorted(set(chg) - set(cov['plan_2021_2025']['zones']))}
def shares(g):
    k = g[g.case == 1]; ctl = g[g.case == 0]
    o = {}
    for nm, f in (('gsa', g.zin == 1), ('legacy', g.legacy == 1), ('either', (g.zin == 1) | (g.legacy == 1)), ('both', (g.zin == 1) & (g.legacy == 1)),
                  ('legacy_outside_gsa', (g.zin == 0) & (g.legacy == 1))):
        o[nm] = {'cases_flagged_share': round(float(f[k.index].mean()), 4), 'control_rows_flagged_share': round(float(f[ctl.index].mean()), 4)}
    o['n_cases'] = int(len(k)); o['n_control_rows'] = int(len(ctl))
    return o
R['screening_le39.5'] = {'all_zones': shares(L), 'plan_2014_2016_zones': shares(L[L.phase == 'plan_2014_2016'])}
json.dump(R, open(OUT + '/gsa_gap_register_v2.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(R, ensure_ascii=False, indent=1, default=str))
