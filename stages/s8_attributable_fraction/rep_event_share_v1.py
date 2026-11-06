"""Share of replication cases by event."""
import json, sys
sys.path.insert(0, '@WORK@/v11/code')
from afcore_v1 import load_shared, H
M, B = load_shared()
out = {'note': 'broadleaf cases per event, shared definitions of gate 5; counts'}
for lab, d in (('main', M), ('rep', B)):
    o = {}
    for rg, s in (('all', d), ('le39.5', d[d.slope <= 39.5])):
        c = s[s.case == 1].groupby('event').size().sort_values(ascending=False)
        o[rg] = {'cases_total': int(c.sum()), 'n_events_with_cases': int((c > 0).sum()), 'largest_event': str(c.index[0]), 'largest_cases': int(c.iloc[0]),
                 'largest_share': float(c.iloc[0] / c.sum()), 'by_event': {str(k): int(v) for k, v in c.items()}}
    out[lab] = o
assert out['rep']['le39.5']['largest_event'] == '2009_莫拉克颱風、芭瑪颱風、1011豪雨'
json.dump(out, open(H + '/v11/an/rep_event_share_v1.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print({lab: {rg: (v['largest_event'], v['largest_cases'], v['cases_total'], round(v['largest_share'], 4)) for rg, v in o.items()} for lab, o in out.items() if lab != 'note'})
