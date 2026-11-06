"""Stand-origin sensitivity restricted to 2020-2025 events."""
import json, re
H='@WORK@'
src=open(H+'/v10/an/g4/stand_origin_v1.py',encoding='utf-8').read()
head=src[:src.index('rng=np.random.default_rng(SEED)')]
m11_def=src[src.index('POPS={'):src.index('M11={')]
exec(compile(head+m11_def,'stand_origin_v1_defs','exec'))
CMP=['primary','modified','plantation']
from scipy.stats import chi2
E20=EV[EV.year>=2020].copy()
R={'charter':'charter_addendum_4_stand_origin, section 5 pitfall 1 (sensitivity on 2020-2025 events)','n_rows':int(len(E20)),'n_cases':int(E20.case.sum()),
   'events':int(E20.event.nunique())}
M9={}
for nm,d in (('all',E20),('le39.5',E20[E20.slope<=39.5])):
    x=d[d.src.isin(CMP)].copy(); x['src']=pd.Categorical(x.src,categories=CMP)
    m,mr,dd,nd=fitfe(x,f"case~{TERR}+legacy+C(src)+legacy:C(src)")
    o={'n':int(len(dd)),'n_case':int(dd.case.sum()),'events_dropped':nd,
       'legacy_OR_within':{'primary':lin(m,mr,{'legacy':1}),'modified':lin(m,mr,{'legacy':1,'legacy:C(src)[T.modified]':1}),'plantation':lin(m,mr,{'legacy':1,'legacy:C(src)[T.plantation]':1})},
       'interaction_vs_primary':{'modified':lin(m,mr,{'legacy:C(src)[T.modified]':1}),'plantation':lin(m,mr,{'legacy:C(src)[T.plantation]':1})}}
    t=['legacy:C(src)[T.modified]','legacy:C(src)[T.plantation]']; b=mr.params[t].values; V=mr.cov_params().loc[t,t].values
    w=float(b@np.linalg.solve(V,b)); o['joint_interaction_wald']={'chi2':round(w,3),'df':2,'p':float(1-chi2.cdf(w,2))}
    M9[nm]=o
R['M9_2020plus']=M9
R['M11_event_2020plus']={k:m11(f(E20)) for k,f in POPS.items()}
json.dump(R,open(H+'/v10/an/g4/stand_origin_sens2020_register_v1.json','w',encoding='utf-8'),ensure_ascii=False,indent=1,default=float)
print('rows',R['n_rows'],'cases',R['n_cases'],'events',R['events'])
for k,v in R['M11_event_2020plus'].items():
    print(k, v['n_case'], {s:(v[s]['OR_at_20']['OR'],v[s]['OR_at_20']['ci']) for s in ('modified','plantation')}, v['n_cases_by_src'])
for k,v in M9.items(): print('M9',k,{s:v['legacy_OR_within'][s]['OR'] for s in CMP},{s:(v['interaction_vs_primary'][s]['OR'],v['interaction_vs_primary'][s]['ci']) for s in ('modified','plantation')},round(v['joint_interaction_wald']['p'],4))
