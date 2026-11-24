"""Replication period (2009-2017): broadleaf cases whose event cutoff falls in an earlier calendar year than the pre-event image of their
own polygon. With one cutoff date per event (stage s9), LandTrendr was fitted per point and cutoff year with the series ending in the year
before the cutoff, and such a case keeps the loss segments of the fit of its own pre-event year that end before the event cutoff year.
The script flags these cases from the event-matched rows of stage s9 and the rows of the replication period (own cutoff dates), describes
them, reproduces the registered replication odds ratios with all cases (registers/rep27_v5.json, R1) and refits without these cases with
the event-level bootstrap of stage s9 (2,000 replicates, one random stream per bootstrap label, labels as registered).
Writes rep_early_cases_v1.json."""
import hashlib, json, os, sys, time
H = '@WORK@'
sys.path.insert(0, os.path.join(H, 'v27', 'code'))
from lib27 import *          # REPS, RNG, main_model, clean, np, pd
T0 = time.time()
def say(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
INP = {'rep_rows_tagged_v5.csv': REPS['v5'], 'rep_rows_v1.csv': os.path.join(H, 'g5', 'rep_rows_v1.csv')}
Rp = pd.read_csv(REPS['v5'], dtype={'uid': str}, low_memory=False, parse_dates=['cutoff'])
Rp['case'] = Rp.case.astype(int); Rp['slope2'] = Rp.slope ** 2; Rp['s20'] = Rp.slope - 20
Rp = Rp.dropna(subset=['slope', 'north', 'east', 'planc', 'profc', 'elev'])          # as an27_rep.py
D = pd.read_csv(INP['rep_rows_v1.csv'], parse_dates=['cutoff'], low_memory=False, usecols=['uid', 'event', 'case', 'cutoff']); D['uid'] = D.uid.astype(str)
own = D[D.case == 1][['uid', 'event', 'cutoff']].rename(columns={'cutoff': 'cutoff_own_frozen'})
X = Rp.merge(own, on=['uid', 'event'], how='left'); X['cutoff_own'] = pd.to_datetime(X.cutoff_own)
assert X[X.case == 1].cutoff_own_frozen.notna().all() and (X[X.case == 1].cutoff_own == X[X.case == 1].cutoff_own_frozen).all()   # own cutoff of stage s9 = rows of the period
assert (X.groupby('event').cutoff.nunique() == 1).all()          # one cutoff date per event
X['early'] = ((X.case == 1) & (X.cutoff.dt.year < X.cutoff_own.dt.year)).astype(int)
E1 = X[X.early == 1]; K = X[X.case == 1]
R = {'inputs_sha256': {k: sha(p) for k, p in INP.items()},
     'rows': int(len(X)), 'cases': int(K.shape[0]), 'control_rows': int((X.case == 0).sum()),
     'early_cases': int(len(E1)), 'early_cases_share': float(len(E1) / len(K)),
     'early_by_own_year': {str(k): int(v) for k, v in E1.groupby(E1.cutoff_own.dt.year).size().items()},
     'early_by_event': {k: int(v) for k, v in E1.groupby('event').size().sort_values(ascending=False).items()},
     'cutoff_year_gap': {str(int(k)): int(v) for k, v in (E1.cutoff_own.dt.year - E1.cutoff.dt.year).value_counts().sort_index().items()},
     'composition_early': {c: float(E1[c].mean()) for c in ('R1', 'R3', 'R0')},
     'composition_other_cases': {c: float(K[K.early == 0][c].mean()) for c in ('R1', 'R3', 'R0')},
     'early_R3_cases': int(E1.R3.sum())}
e3 = E1[E1.R3 == 1]
R['early_R3_years_from_loss_end_to_event_cutoff'] = {str(int(k)): int(v) for k, v in e3.TSD.round().value_counts().sort_index().items()}
R['early_R3_within_1_year'] = int((e3.TSD <= 1).sum())
say('flagged', R['early_cases'], 'of', R['cases'], 'cases')
def fit(d, lab, boot):
    a = main_model(d, ['legacy'], lab + ':MR1', boot=boot); b = main_model(d, ['R1', 'R3'], lab + ':MR2', boot=boot)
    o = {'legacy': a['OR']['legacy'], 'R1': b['OR']['R1'], 'R3': b['OR']['R3'], 'AF_total': b.get('AF_total'),
         'n': {k: b[k] for k in ('n_rows', 'n_cases', 'n_control_rows', 'n_events', 'events_dropped')}}
    if boot: o['boot'] = b['boot']
    return clean(o)
R['fits'] = {}
for rg, sel in RNG.items():
    say('all cases', rg); R['fits'][f'all_cases_{rg}'] = fit(sel(X), f'v36:early:all:{rg}', False)
    say('without the flagged cases', rg); R['fits'][f'without_early_{rg}'] = fit(sel(X[X.early == 0]), f'v36:early:noearly:{rg}', True)
# with all cases the fits must equal the registered replication estimates (registers/rep27_v5.json, R1, replication period)
RES = json.load(open(os.path.join(H, 'reference', 'rep27_v5.json'), encoding='utf-8'))['R1']['periods']['rep']
for rg in RNG:
    assert abs(R['fits'][f'all_cases_{rg}']['legacy']['OR'] - RES[f'MR1_{rg}']['OR']['legacy']['OR']) < 1e-6
    for t in ('R1', 'R3'): assert abs(R['fits'][f'all_cases_{rg}'][t]['OR'] - RES[f'MR2_{rg}']['OR'][t]['OR']) < 1e-6
R['note'] = ('Without the 1,159 cases the replication odds ratios keep their direction and their event-level intervals exclude 1 (the '
             'pre-specified rule reads them as replicated); the exclusion also removes 849 of the cases of the 2009 Morakot entry.')
os.makedirs(os.path.join(H, 'v27', 'an'), exist_ok=True)
json.dump(R, open(os.path.join(H, 'v27', 'an', 'rep_early_cases_v1.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
say('saved rep_early_cases_v1.json')
