"""Share of class 1 cases located inside (not only near) a prior scar polygon."""
import json, sys
H = '@WORK@'
src = open(H + '/v10/an/g4/stand_origin_v1.py', encoding='utf-8').read()
ns = {'__name__': 'stand_origin_v1_defs'}
exec(compile(src[:src.index('rng=np.random.default_rng(SEED)')], 'stand_origin_v1_defs', 'exec'), ns)
pd, np, EV, fitfe, lin, TERR = ns['pd'], ns['np'], ns['EV'], ns['fitfe'], ns['lin'], ns['TERR']
REG = json.load(open(H + '/v10/an/g4/stand_origin_register_v1.json', encoding='utf-8'))
x = EV[(EV.site_class4 == 1) & (EV.year >= 2020) & EV.chm.notna()].copy()
x['tss1_bin'] = pd.Categorical(x.tss1_bin.astype(str), categories=['0-5', '5-10', '10+'])
x['chm5'] = x.chm / 5.0; x['in_scar'] = (x.class1_0m == 1).astype(int)
R = {'status': 'exploratory, written 2026-10-13 after the main results; not in any charter', 'data': 'tags_event_v4 + g3/analysis_rows_v1 (chm)'}
for nm, y in (('all', x), ('le39.5', x[x.slope <= 39.5])):
    m, mr, dd, nd = fitfe(y, f"case~{TERR}+C(tss1_bin)+chm5")
    base = lin(m, mr, {'chm5': 1}); ref = REG['M10_class1'][nm]['chm_per_5m']
    assert abs(base['OR'] - ref['OR']) < 1e-4, (nm, base, ref)
    m2, mr2, dd2, nd2 = fitfe(y, f"case~{TERR}+C(tss1_bin)+chm5+in_scar")
    o = {'rows': int(len(dd)), 'cases': int(dd.case.sum()),
         'in_scar_cases': int(((dd.case == 1) & (dd.in_scar == 1)).sum()), 'in_scar_control_rows': int(((dd.case == 0) & (dd.in_scar == 1)).sum()),
         'chm_median': {g: round(float(dd[(dd.case == c) & (dd.in_scar == s)].chm.median()), 2) for g, c, s in
                        (('in_cases', 1, 1), ('in_controls', 0, 1), ('out_cases', 1, 0), ('out_controls', 0, 0))},
         'chm_per_5m_base': base, 'chm_per_5m_with_in_scar': lin(m2, mr2, {'chm5': 1}), 'in_scar': lin(m2, mr2, {'in_scar': 1})}
    for sub, yy in (('inside_only', y[y.in_scar == 1]), ('outside_only', y[y.in_scar == 0])):
        m3, mr3, dd3, nd3 = fitfe(yy, f"case~{TERR}+C(tss1_bin)+chm5")
        o[f'chm_per_5m_{sub}'] = lin(m3, mr3, {'chm5': 1}); o[f'rows_{sub}'] = int(len(dd3)); o[f'cases_{sub}'] = int(dd3.case.sum()); o[f'events_dropped_{sub}'] = nd3
    R[nm] = o
    print(nm, json.dumps({k: (v if not isinstance(v, dict) or 'OR' not in v else (v['OR'], v['ci'])) for k, v in o.items()}, ensure_ascii=False), flush=True)
json.dump(R, open(H + '/v10/an/g4/canopy_inscar_register_v1.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print('done')
