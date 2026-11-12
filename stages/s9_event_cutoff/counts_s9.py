"""Counts of stage s9 that are not outputs of the model scripts: the class 1 sample of the canopy analysis (all rows with a canopy height and the rows of events with cases and controls) and the number of polygons of the 2026 layer of landslide geologically sensitive areas, a property of the source layer recorded as a constant. Writes counts_s9.json."""
import json, pandas as pd
H = '@WORK@'
E = pd.read_csv(H + '/v27/data/tags_event_v5.csv', dtype={'uid': str}, low_memory=False)
A = pd.read_csv(H + '/g3/analysis_rows_v1.csv', dtype={'uid': str}, low_memory=False)[['uid', 'chm']].drop_duplicates('uid')
X = E[(E.ft == '闊葉樹林型') & (E.site_class4 == 1) & (E.year >= 2020)].merge(A, on='uid', how='left')
X = X[X.chm.notna()].dropna(subset=['slope', 'north', 'east', 'planc', 'profc', 'elev'])
X['case'] = X.case.astype(int)
g = X.groupby('event').case.agg(['sum', 'size']); keep = g[(g['sum'] > 0) & (g['sum'] < g['size'])].index
M = X[X.event.isin(keep)]
out = {'canopy_class1_2020_2025': {'rows': int(len(X)), 'cases': int(X.case.sum()), 'controls': int((X.case == 0).sum()),
                                   'cases_in_events_with_cases_and_controls': int(M.case.sum()), 'controls_in_events_with_cases_and_controls': int((M.case == 0).sum()),
                                   'median_canopy_height_cases_m': float(X[X.case == 1].chm.median()), 'median_canopy_height_controls_m': float(X[X.case == 0].chm.median())},
       'source_layers': {'landslide_geologically_sensitive_areas_2026_polygons': 168714}}
json.dump(out, open(H + '/v27/an/counts_s9.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(out)
