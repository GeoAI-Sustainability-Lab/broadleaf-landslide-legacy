"""Legacy odds ratio by quintile of the terrain-only susceptibility score in both periods. Writes rep27_b2_<TAGS>.json."""
import os, json, math
import numpy as np, pandas as pd
from lib27 import *
TAG = os.environ.get('TAGS', 'v5'); DOBOOT = os.environ.get('BOOT', '1') == '1'; REPV = 'v1' if TAG == 'v4' else 'v5'
F = json.load(open(H + '/reference/reference_values_s9.json', encoding='utf-8')); FV = lambda k: F[k]
E = load_event(TAG); BR = pd.read_csv(BRIGHT, dtype={'uid': str})
M = E[E.ft == BRO].merge(BR[['uid', 'event', 'Qt']], on=['uid', 'event'], how='left')
M['R1'] = (M.class1_50m == 1).astype(int); M['R3'] = ((M.lt_detected == 1) & (M.R1 == 0)).astype(int); M['legacy'] = ((M.R1 == 1) | (M.R3 == 1)).astype(int)
Rp = pd.read_csv(REPS[REPV], dtype={'uid': str}, low_memory=False)
Rp['case'] = Rp.case.astype(int); Rp['slope2'] = Rp.slope ** 2; Rp = Rp.dropna(subset=['slope', 'north', 'east', 'planc', 'profc', 'elev'])
OUT = {}
for per, X in (('main', M), ('rep', Rp)):
    X = X.copy(); qs = sorted(X.Qt.astype(int).unique())
    for q in qs[1:]: X[f'q{q}'] = (X.Qt.astype(int) == q).astype(int)
    for q in qs: X[f'leg_q{q}'] = X.legacy * (X.Qt.astype(int) == q)
    o = main_model(X, [f'leg_q{q}' for q in qs], f'repB2:{per}', covs=TV + [f'q{q}' for q in qs[1:]], boot=DOBOOT)
    OUT[per] = {str(q): o['OR'][f'leg_q{q}'] for q in qs}
if TAG == 'v4':
    for per, k in (('main', 'R5.main'), ('rep', 'R5.rep')):
        reg = FV(k)['bright_terrain_only']['B2_legacy_by_stratum']
        for q in '12345': assert abs(OUT[per][q]['OR'] - reg[q]['OR']) < 1e-3, (per, q, OUT[per][q]['OR'], reg[q]['OR'])
    print('validated against R5 B2_legacy_by_stratum')
json.dump(clean(OUT), open(f'{H}/v27/an/rep27_b2_{TAG}.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print({p: {q: round(v['OR'], 3) for q, v in d.items()} for p, d in OUT.items()})
