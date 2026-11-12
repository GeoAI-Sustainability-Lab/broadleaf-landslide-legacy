"""2009-2017 event-matched rows re-tagged with one cutoff date per event. Reads the yearly event inventories and the build files of stage s6; writes rep_rows_tagged_v5.csv and rep_rows_v5_register.json."""
import glob, re, json, os, sys, tarfile, numpy as np, pandas as pd, shapefile, shapely
from shapely.geometry import shape
from shapely.validation import make_valid
from shapely.strtree import STRtree
H = '@WORK@'
sys.path.insert(0, H + '/v11/unround/g1'); from lt_segments import summarize, segments_from_row
ROOT = '@SOURCE@/Landslide_Taiwan/2004_2023_annual_event_catalogs/raw/extracted'
G5 = H + '/g5'
OUT = H + '/v27/rep/'; os.makedirs(OUT, exist_ok=True)
BRO = '闊葉樹林型'; REG = {}
D8 = re.compile(r'(?<!\d)(\d{8})(?!\d)')
def dates(v):
    out = []
    for s in D8.findall(str(v)):
        t = pd.to_datetime(s, format='%Y%m%d', errors='coerce')
        if pd.notna(t) and 1990 <= t.year <= 2026: out.append(t)
    return out
recs = []; geoms = []
for y in range(2004, 2018):
    shp = sorted(glob.glob(f'{ROOT}/{y}/*.shp'))[0]
    r = shapefile.Reader(shp, encoding='utf-8', encodingErrors='replace')
    fs = [f[0] for f in r.fields[1:]]; ev = 'Events' if 'Events' in fs else 'Event_s_'
    for i in range(len(r)):
        try: g = shape(r.shape(i).__geo_interface__)
        except Exception: continue
        if g.is_empty: continue
        if not g.is_valid: g = make_valid(g)
        rec = r.record(i).as_dict(); bd = dates(rec.get('BeforeDate')); ad = dates(rec.get('AfterDate'))
        recs.append({'uid': f'r{y}_{i}', 'year': y, 'idx': i, 'cutoff': min(bd) if bd else pd.NaT, 'after_latest': max(ad) if ad else pd.NaT,
                     'events': re.sub(r'\s+', '', str(rec.get(ev, '')))})
        geoms.append(g)
A = pd.DataFrame(recs)
K0 = pd.read_csv(G5 + '/data/rep_cases_raw_v1.csv.gz', parse_dates=['cutoff', 'after_latest'])
chk = K0[['uid', 'cutoff', 'after_latest', 'events']].merge(A[['uid', 'cutoff', 'after_latest', 'events']], on='uid', suffixes=('_f', '_n'), how='left')
assert chk.cutoff_n.notna().sum() == chk.cutoff_f.notna().sum() and len(chk) == len(K0)
for c in ('cutoff', 'after_latest'):
    assert ((chk[c + '_f'] == chk[c + '_n']) | (chk[c + '_f'].isna() & chk[c + '_n'].isna())).all(), c
assert (chk.events_f.astype(str) == chk.events_n.astype(str)).all()
REG['catalogue'] = {'polygons_2004_2017': int(len(A)), 'case_polygons_checked_against_frozen_table': int(len(K0))}
EVT = pd.read_csv(G5 + '/data/rep_events_v1.csv', parse_dates=['cutoff'])
K = A[A.year >= 2009].copy(); K['event'] = K.year.astype(str) + '_' + K.events
ev_cut = K.groupby('event').cutoff.min()
e2 = EVT.set_index('event').cutoff
assert set(ev_cut.index) == set(e2.index) and ((ev_cut.reindex(e2.index) == e2) | (ev_cut.reindex(e2.index).isna() & e2.isna())).all()
D = pd.read_csv(G5 + '/data/rep_rows_v1.csv.gz', parse_dates=['cutoff', 'scar50_last_after'], low_memory=False)
D['cutoff_event'] = D.event.map(e2); assert D.cutoff_event.notna().all()
assert (D[D.case == 0].cutoff == D[D.case == 0].cutoff_event).all(), 'control cutoff differs from the event cutoff'
assert (D[D.case == 1].cutoff >= D[D.case == 1].cutoff_event).all()
D['cutoff_own'] = D.cutoff
tree = STRtree(geoms); uid_arr = A.uid.values; al = A.after_latest.values
def scars(sub, cut):
    pts = shapely.points(sub.x.values, sub.y.values); cutv = cut.values; uidv = sub.uid.values
    pi, gi = tree.query(pts, predicate='dwithin', distance=50.0)
    ok = (al[gi] <= cutv[pi]) & (uid_arr[gi] != uidv[pi]); pi, gi = pi[ok], gi[ok]
    n = np.zeros(len(sub), int); last = np.full(len(sub), np.datetime64('NaT'), dtype='datetime64[ns]')
    if len(pi):
        s_n = pd.Series(gi).groupby(pi).size(); s_l = pd.Series(al[gi]).groupby(pi).max()
        n[s_n.index.values] = s_n.values; last[s_l.index.values] = s_l.values
    pi0, gi0 = tree.query(pts, predicate='intersects'); ok0 = (al[gi0] <= cutv[pi0]) & (uid_arr[gi0] != uidv[pi0])
    n0 = np.zeros(len(sub), int)
    if ok0.any():
        s0 = pd.Series(gi0[ok0]).groupby(pi0[ok0]).size(); n0[s0.index.values] = s0.values
    return n, pd.to_datetime(last), n0
C = D[D.case == 1].copy()
n_own, last_own, n0_own = scars(C, C.cutoff_own)
assert (n_own == C.scar50_n.values).all() and (n0_own == C.scar0_n.values).all()
assert ((last_own == C.scar50_last_after.values) | (pd.isna(last_own) & C.scar50_last_after.isna().values)).all()
n_ev, last_ev, n0_ev = scars(C, C.cutoff_event)
C['scar50_n'] = n_ev; C['scar50_last_after'] = last_ev; C['scar0_n'] = n0_ev
C['cutoff'] = C.cutoff_event
D5 = pd.concat([D[D.case == 0], C]).sort_index()
D5['R1'] = (D5.scar50_n > 0).astype(int); D5['scar_age'] = (D5.cutoff - D5.scar50_last_after).dt.days / 365.25
D5['cutoff_year'] = D5.cutoff.dt.year; D5['cutoff_year_own'] = D5.cutoff_own.dt.year
REG['case_rows_scar_changes'] = {'cases': int(len(C)), 'R1_lost': int(((n_own > 0) & (n_ev == 0)).sum()), 'R1_kept_fewer_scars': int(((n_ev > 0) & (n_ev < n_own)).sum()),
                                 'broadleaf_R1_lost': int(((n_own > 0) & (n_ev == 0) & (C.TypeName == BRO).values).sum())}
LT = OUT + 'lt_batches_rep/'
if not glob.glob(LT + 'lt_*.jsonl'):
    os.makedirs(LT, exist_ok=True)
    for t in sorted(glob.glob(G5 + '/data/lt_batches_rep_v1_part*.tgz')):
        with tarfile.open(t) as tf:
            for m in tf.getmembers():
                if m.isfile() and m.name.endswith('.jsonl'):
                    m.name = os.path.basename(m.name); tf.extract(m, LT, filter='data')
rows = []; fits = []
for f in sorted(glob.glob(LT + 'lt_*.jsonl')):
    for line in open(f):
        r = json.loads(line); fits.append({'uid': r['uid'], 'cutoff_year': r['cutoff_year'], 'fit': r.get('fit', 0), 'rmse': r.get('rmse'),
                                           'end_fit': (r['fitted'][-1] if r.get('fit') else None), 'n_years': (len(r['years']) if r.get('fit') else 0)})
        rows += segments_from_row(r)
segs = pd.DataFrame(rows); fits = pd.DataFrame(fits)
REG['landtrendr'] = {'fits': int(len(fits)), 'segments': int(len(segs)), 'batches': len(glob.glob(LT + 'lt_*.jsonl'))}
S = summarize(segs, fits, mag_thr=100.0, max_dur=3)
B = D5[D5.TypeName == BRO].copy()
B = B.merge(S[['uid', 'cutoff_year', 'fit', 'dist_year', 'dist_mag', 'TSD', 'recovery_ratio']].rename(columns={'cutoff_year': 'cutoff_year_own'}), on=['uid', 'cutoff_year_own'], how='left')
T1 = pd.read_csv(G5 + '/data/rep_rows_tagged_v1.csv.gz', parse_dates=['cutoff'], low_memory=False)
J = T1[['uid', 'event', 'case', 'dist_year', 'TSD', 'lt_detected', 'R1', 'scar_age', 'Qt', 'score_terrain']].merge(
    B[['uid', 'event', 'dist_year', 'TSD', 'fit', 'cutoff_year', 'cutoff_year_own']], on=['uid', 'event'], suffixes=('_v1', '_own'), how='inner')
assert len(J) == len(T1) == len(B)
eqn = lambda a, b: ((a - b).abs() < 1e-9) | (a.isna() & b.isna())
assert eqn(J.dist_year_v1, J.dist_year_own).all() and eqn(J.TSD_v1, J.TSD_own).all()
earlier = B.cutoff_year < B.cutoff_year_own
q = segs[(segs.delta >= 100.0) & (segs.duration <= 3)].rename(columns={'cutoff_year': 'cutoff_year_own'})
q = q.merge(B.loc[earlier, ['uid', 'cutoff_year_own', 'cutoff_year']].drop_duplicates(), on=['uid', 'cutoff_year_own'])
q = q[q.end_year <= q.cutoff_year - 1].sort_values(['uid', 'cutoff_year_own', 'end_year']).groupby(['uid', 'cutoff_year_own']).tail(1)
adj = B.loc[earlier, ['uid', 'cutoff_year_own']].merge(q[['uid', 'cutoff_year_own', 'end_year', 'delta']], on=['uid', 'cutoff_year_own'], how='left')
B.loc[earlier, 'dist_year'] = adj.end_year.values; B.loc[earlier, 'dist_mag'] = adj.delta.values
B['TSD'] = B.cutoff_year - B.dist_year
B['lt_detected'] = B.dist_year.notna().astype(int); B['lt_missing'] = B.fit.isna().astype(int)
REG['earlier_event_year'] = {'broadleaf_case_rows': int((earlier & (B.case == 1)).sum()), 'by_own_year': {str(k): int(v) for k, v in B[earlier].groupby('cutoff_year_own').size().items()},
                             'loss_removed': int((earlier & J.set_index(B.index).dist_year_v1.notna() & B.dist_year.isna()).sum()) if False else None}
B['R3'] = ((B.lt_detected == 1) & (B.R1 == 0)).astype(int); B['legacy'] = ((B.R1 == 1) | (B.R3 == 1)).astype(int); B['R0'] = 1 - B.legacy
B['s20'] = B.slope - 20
B['scar_bin'] = np.select([B.R1 == 0, B.scar_age < 5, B.scar_age < 10], ['none', '0-5', '5-10'], '10+')
B['tsd_bin'] = np.select([B.R3 == 0, B.TSD < 5, B.TSD < 10, B.TSD < 20], ['none', '0-5', '5-10', '10-20'], '20+')
B = B.merge(T1[['uid', 'event', 'Qt', 'score_terrain']], on=['uid', 'event'], how='left'); assert B.Qt.notna().sum() == T1.Qt.notna().sum()
V1 = T1.set_index(['uid', 'event']); V5 = B.set_index(['uid', 'event']).reindex(V1.index)
cs = V1.case == 1
REG['broadleaf'] = {'rows': int(len(B)), 'cases': int(B.case.sum()), 'control_rows': int((B.case == 0).sum()),
                    'case_changes': {'R1_lost': int((cs & (V1.R1 == 1) & (V5.R1 == 0)).sum()), 'R3_lost': int((cs & (V1.R3 == 1) & (V5.R3 == 0)).sum()),
                                     'R3_gained': int((cs & (V1.R3 == 0) & (V5.R3 == 1)).sum()), 'legacy_lost': int((cs & (V1.legacy == 1) & (V5.legacy == 0)).sum()),
                                     'legacy_gained': int((cs & (V1.legacy == 0) & (V5.legacy == 1)).sum()),
                                     'scar_bin_changed': int((cs & (V1.scar_bin.astype(str) != V5.scar_bin.astype(str))).sum()),
                                     'tsd_bin_changed': int((cs & (V1.tsd_bin.astype(str) != V5.tsd_bin.astype(str))).sum())},
                    'controls_unchanged': bool(((~cs) & ((V1.R1 != V5.R1) | (V1.R3 != V5.R3) | (V1.scar_bin.astype(str) != V5.scar_bin.astype(str)))).sum() == 0)}
assert REG['broadleaf']['controls_unchanged']
B.drop(columns=['cutoff_year_own']).to_csv(OUT + 'rep_rows_tagged_v5.csv', index=False)
json.dump(REG, open(OUT + 'rep_rows_v5_register.json', 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(REG, ensure_ascii=False, indent=1, default=str))
