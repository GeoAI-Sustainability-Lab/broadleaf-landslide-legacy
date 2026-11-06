"""Recent canopy disturbance within the stand-origin models."""
import json
H='@WORK@'
src=open(H+'/v10/an/g4/stand_origin_v1.py',encoding='utf-8').read()
exec(compile(src[:src.index('rng=np.random.default_rng(SEED)')],'stand_origin_v1_defs','exec'))
CMP=['primary','modified','plantation']
R={'label':'descriptive, added after M8 (not in the charter): share of rows in class 3 with <= 10 years since the disturbance, by stand origin','source_script':'stand_origin_class3_recent_v1.py'}
for nm,d in (('all',EV),('le39.5',EV[EV.slope<=39.5])):
    o={}
    for s in CMP:
        x=d[d.src==s]; c=x[x.case==0]; k=x[x.case==1]
        rec=lambda y: float(((y.site_class4==3)&(y.tsd3_bin.astype(str).isin(['0-5','5-10']))).mean())
        o[s]={'n_control_rows':int(len(c)),'n_cases':int(len(k)),
              'class3_le10yr_share_controls':round(rec(c),4),'class3_le10yr_share_cases':round(rec(k),4),
              'class3_share_controls':round(float((c.site_class4==3).mean()),4),
              'class3_tsd_bins_controls':{b:int(((c.site_class4==3)&(c.tsd3_bin.astype(str)==b)).sum()) for b in ['0-5','5-10','10-20','20+']}}
    R[nm]=o
json.dump(R,open(H+'/v10/an/g4/stand_origin_class3_recent_register_v1.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
for nm in ('all','le39.5'):
    print(nm,{s:(R[nm][s]['class3_le10yr_share_controls'],R[nm][s]['class3_share_controls'],R[nm][s]['class3_tsd_bins_controls']) for s in CMP})
