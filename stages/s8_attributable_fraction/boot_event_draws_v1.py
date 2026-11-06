"""Event-bootstrap draws and the events that drive the interval."""
import json, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, '@WORK@/v11/code')
from afcore_v1 import *
OUT = H + '/v11/an/boot_event_draws_register_v1.json'
REG = json.load(open(H + '/v11/an/af_window_register_v2.json', encoding='utf-8'))['bootstrap']['full']
M, B = load_shared()
R = {'status': 'exploratory, after all results (2026-10-29)', 'nboot': NBOOT,
     'design': 'full catalogue, shared definitions of gate 5, event fixed effects, events resampled with replacement (af_window_v2.py)'}
T0 = time.time()
for k, d in (('main', M), ('rep', B)):
    R[k] = {}
    for rg, sel in RNG.items():
        x = sel(d).copy(); x['R1w'] = x.R1; x['R3w'] = ((x.lt_detected == 1) & (x.R1w == 0)).astype(int)
        f = FastFE(x, ['R1w', 'R3w']); bl, ok = f.fit_all(); assert ok
        d2, _ = drop_uninf(x); codes, uni = pd.factorize(d2.event)
        nc = d2.groupby('event').case.sum(); big_name = str(nc.idxmax()); big = list(uni).index(big_name)
        assert all(len(f.rows[j]) == int((codes == j).sum()) for j in range(f.K))
        rng = np.random.default_rng(seed_for(f'boot:full:{k}:{rg}')); tot, draws, fails = [], [], 0
        for _ in range(NBOOT):
            pick = rng.integers(0, f.K, f.K)
            rows = [f.rows[j] for j in pick]; idx = np.concatenate(rows)
            ev = np.concatenate([np.full(len(r), j) for j, r in enumerate(rows)])
            X = f.X[idx]; y = f.y[idx]; T = X[:, -f.kt:]
            if any(T[y == 1, j].sum() == 0 or T[y == 0, j].sum() == 0 for j in range(f.kt)): fails += 1; continue
            beta, it, okk = f.fit_rows(X, y, ev, f.K)
            if not okk: fails += 1; continue
            tot.append(float(f.af_point(X, y, beta[-f.kt:]).sum())); draws.append(int((pick == big).sum()))
        tot = np.array(tot); draws = np.array(draws)
        ci = [float(np.percentile(tot, 2.5)), float(np.percentile(tot, 97.5))]
        assert len(tot) == REG[k][rg]['n_ok'] and max(abs(a - b) for a, b in zip(ci, REG[k][rg]['ci_boot'])) < 1e-12, (k, rg, ci)
        grp = {'0': draws == 0, '1': draws == 1, '2+': draws >= 2}
        R[k][rg] = {'largest_event': big_name, 'largest_cases': int(nc.max()), 'cases_total': int(nc.sum()), 'largest_share': float(nc.max() / nc.sum()),
                    'events': int(f.K), 'n_ok': int(len(tot)), 'fails': fails, 'ci_boot': ci,
                    'draw_share': {g: float(m.mean()) for g, m in grp.items()},
                    'AF_median_by_draws': {g: float(np.median(tot[m])) for g, m in grp.items() if m.any()},
                    'AF_range_by_draws': {g: [float(tot[m].min()), float(tot[m].max())] for g, m in grp.items() if m.any()}}
        print(k, rg, big_name, round(nc.max() / nc.sum(), 4), {g: round(v, 3) for g, v in R[k][rg]['draw_share'].items()},
              {g: round(v, 4) for g, v in R[k][rg]['AF_median_by_draws'].items()}, round(time.time() - T0), 's', flush=True)
json.dump(R, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
print('saved', OUT)
