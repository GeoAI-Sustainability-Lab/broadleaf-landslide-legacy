"""Log(1 - AF) interval from the same Monte Carlo draws as the attributable fractions."""
import os, sys, json, math
import numpy as np
H = '@WORK@'
G1 = H + '/v10/an/g1'; os.chdir(G1); sys.path.insert(0, G1)
src = open(G1 + '/robust_v1.py', encoding='utf-8').read()
cut = src.index("R['attributable_fraction']=")
defs = src[:cut]
a = "    for k in pt: out[f'AF_{k}']={'point':round(float(pt[k]),4),'ci_mc':[round(float(np.percentile(sims[k],2.5)),4),round(float(np.percentile(sims[k],97.5)),4)]}\n    return out"
assert defs.count(a) == 1
defs = defs.replace(a, a.replace("    return out", "    out['_sims_total']=np.array(sims['total']); out['_pt_total']=float(pt['total'])\n    return out"))
ns = {'__name__': 'robust_v1_defs'}
exec(compile(defs, 'robust_v1_defs', 'exec'), ns)
paf, be = ns['paf'], ns['be']
FACT = json.load(open(H + '/reference/reference_values.json', encoding='utf-8'))
R = {'status': 'exploratory (written 2026-10-13 after all charter results); same draws as robust_v1.py', 'seed': 20260924, 'ndraw': 4000}
for lab, d, fk in (('all', be, 'AF4.event.all'), ('le39_5', be[be.slope <= 39.5], 'AF4.event.le39_5')):
    o = paf(d, 'event'); sims = o['_sims_total']; pt = o['_pt_total']
    ref = FACT[fk]['value']['total']
    assert o['AF_total']['point'] == ref['point'] and o['AF_total']['ci_mc'] == ref['ci_mc'], (lab, o['AF_total'], ref)
    lz = np.log(1 - sims); s = float(np.std(lz, ddof=1)); c = math.log(1 - pt)
    R[lab] = {'AF': round(pt, 4), 'ci_mc_percentile': o['AF_total']['ci_mc'], 'ci_log1mAF': [round(1 - math.exp(c + 1.96 * s), 4), round(1 - math.exp(c - 1.96 * s), 4)],
              'n_cases': o['n_cases']}
    print(lab, R[lab], flush=True)
os.makedirs(H + '/v10/an/methods', exist_ok=True)
json.dump(R, open(H + '/v10/an/methods/af_log_interval_register_v2.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('done')
