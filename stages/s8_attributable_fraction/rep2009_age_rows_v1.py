"""Scar-age rows of the 2009 replication events."""
import json, sys
sys.path.insert(0, '@WORK@/v11/code')
from afcore_v1 import load_shared, H, RNG, drop_uninf
M, B = load_shared(); R = {'note': 'rep_2009 group, R1 rows with scar_age >= 5 yr; counts in the fitted rows (events with case-control variation) and in all rows'}
for rg, sel in RNG.items():
    d = sel(B[B.y2009]); f, _ = drop_uninf(d)
    for lab, x in (('all_rows', d), ('fitted_rows', f)):
        m = (x.R1 == 1) & (x.scar_age >= 5)
        R.setdefault(rg, {})[lab] = {'cases': int((m & (x.case == 1)).sum()), 'control_rows': int((m & (x.case == 0)).sum()),
                                     'max_scar_age_controls': float(x.loc[m & (x.case == 0), 'scar_age'].max()) if (m & (x.case == 0)).any() else None}
print(R); json.dump(R, open(H + '/v11/an/rep2009_age_rows_register_v1.json', 'w'), indent=1)
