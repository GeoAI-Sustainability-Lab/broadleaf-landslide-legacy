"""Stand origin (forest function class) and canopy height models, revegetated-scar odds by canopy height, bamboo contrast with broadleaf restricted to intact stands."""
import json, math, warnings, numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
warnings.filterwarnings('ignore')
H='@WORK@'
SEED=20260930; NDRAW=4000; BRO='闊葉樹林型'; TERR="slope+slope2+north+east+planc+profc+elev"
SRC={'原生林':'primary','經改造天然林':'modified','保護性人工林':'plantation','生產性人工林':'plantation','半天然林':'other','未確認':'other'}
R={'charter':'charter_addendum_4_stand_origin (2026-09-30, written before any gate-4 fit); approved by the author 2026-10-10','seed':SEED,'ndraw':NDRAW}
A=pd.read_csv(H+'/g3/analysis_rows_v1.csv',dtype={'uid':str},low_memory=False)[['uid','stand_function','chm','chm_date']].drop_duplicates('uid')
def prep(df):
    d=df.merge(A,on='uid',how='left'); d['case']=d.case.astype(int); d['slope2']=d.slope**2; d['s20']=d.slope-20.0
    d['src']=d.stand_function.map(SRC).fillna('missing')
    d['legacy']=(d.site_class4>0).astype(int)
    for c in (1,2,3): d[f'sc{c}']=(d.site_class4==c).astype(int)
    d['tss1_bin']=pd.Categorical(np.where(d.site_class4==1,d.tss_bin.replace({'20+':'10+','10-20':'10+'}),'none'),categories=['none','0-5','5-10','10+'])
    d['tsd3_bin']=pd.Categorical(d.tsd3_bin,categories=['none','0-5','5-10','10-20','20+'])
    d['cluster']=np.where(d.kind=='case',d.event.astype(str),'controls')
    return d.dropna(subset=['slope','north','east','planc','profc','elev'])
EV=prep(pd.read_csv('@WORK@/v10/data/tags_event_v4.csv',dtype={'uid':str},low_memory=False).query('ft==@BRO'))
IS=prep(pd.read_csv(H+'/g1/tags_island_v3.csv',dtype={'uid':str},low_memory=False).query('ft==@BRO'))
def drop_uninf(d):
    g=d.groupby('event').case.agg(['sum','size']); keep=g[(g['sum']>0)&(g['sum']<g['size'])].index
    return d[d.event.isin(keep)].copy(), int(len(g)-len(keep))
def fitfe(d, form, design='event'):
    if design=='island':
        m=smf.logit(form,data=d).fit(disp=0,maxiter=200,method='newton')
        mr=smf.logit(form,data=d).fit(disp=0,maxiter=200,method='newton',cov_type='cluster',cov_kwds={'groups':pd.factorize(d.cluster)[0]})
        return m,mr,d,0
    d,nd=drop_uninf(d); f2=form+'+C(event)'
    m=smf.logit(f2,data=d).fit(disp=0,maxiter=100,method='newton')
    if not m.mle_retvals.get('converged',True):
        m0=m
        m=smf.logit(f2,data=d).fit(disp=0,maxiter=5000,method='bfgs',gtol=1e-8)
        mr=smf.logit(f2,data=d).fit(disp=0,maxiter=5000,method='bfgs',gtol=1e-8,cov_type='cluster',cov_kwds={'groups':pd.factorize(d.uid)[0]})
        ok=m.mle_retvals.get('converged',True) or abs(m.llf-m0.llf)<1e-5
        print('FALLBACK_BFGS',form,len(d),'bfgs_converged',m.mle_retvals.get('converged'),'llf_newton',round(m0.llf,6),'llf_bfgs',round(m.llf,6),flush=True)
        assert ok, form
        return m,mr,d,nd
    mr=smf.logit(f2,data=d).fit(disp=0,maxiter=100,method='newton',cov_type='cluster',cov_kwds={'groups':pd.factorize(d.uid)[0]})
    return m,mr,d,nd
def lin(m,mr,coefs):
    b=sum(w*m.params[t] for t,w in coefs.items())
    def se_of(mod):
        c=mod.cov_params(); return math.sqrt(sum(wi*wj*c.loc[ti,tj] for ti,wi in coefs.items() for tj,wj in coefs.items()))
    sr=se_of(mr); s=se_of(m)
    return {'OR':round(math.exp(b),4),'ci':[round(math.exp(b-1.96*sr),4),round(math.exp(b+1.96*sr),4)],'ci_model':[round(math.exp(b-1.96*s),4),round(math.exp(b+1.96*s),4)],
            'p':float(2*(1-norm.cdf(abs(b/sr)))),'beta':float(b),'se':float(sr)}
def af(d, terms, rng):
    m,mr,dd,nd=fitfe(d,f"case~{TERR}+"+"+".join(terms))
    cases=dd[dd.case==1]; n=len(cases)
    rho=np.array([float(cases[t].mean()) for t in terms]); full=np.append(rho,1-rho.sum())
    b=m.params[terms].values; V=mr.cov_params().loc[terms,terms].values; draws=rng.multivariate_normal(b,V,size=NDRAW)
    f=lambda bv,r:[r[i]*(1-math.exp(-bv[i])) for i in range(len(terms))]
    pt=f(b,rho); sims=np.array([f(bv,rng.multinomial(n,full)[:-1]/n) for bv in draws])
    out={'n_cases':n,'events_dropped':nd,'rho_cases':{t:round(float(r),4) for t,r in zip(terms,rho)},'OR':{t:lin(m,mr,{t:1}) for t in terms}}
    for i,t in enumerate(terms): out[f'AF_{t}']={'point':round(pt[i],4),'ci_mc':[round(float(np.percentile(sims[:,i],2.5)),4),round(float(np.percentile(sims[:,i],97.5)),4)]}
    out['AF_total']={'point':round(sum(pt),4),'ci_mc':[round(float(np.percentile(sims.sum(1),2.5)),4),round(float(np.percentile(sims.sum(1),97.5)),4)]}
    return out
rng=np.random.default_rng(SEED)
inv={}
for nm,d in (('all',EV),('le39.5',EV[EV.slope<=39.5])):
    inv[nm]={k:{'control_rows':int((v.case==0).sum()),'cases':int(v.case.sum())} for k,v in d.groupby(d.stand_function.fillna('缺值'))}
R['inventory']=inv
CMP=['primary','modified','plantation']
M8={}
for nm,d in (('all',EV),('le39.5',EV[EV.slope<=39.5])):
    o={}
    for s in CMP:
        x=d[d.src==s]
        o[s]={'n_cases':int(x.case.sum()),'n_control_rows':int((x.case==0).sum()),
              'share_cases':{str(c):round(float((x[x.case==1].site_class4==c).mean()),4) for c in (0,1,2,3)},
              'share_controls':{str(c):round(float((x[x.case==0].site_class4==c).mean()),4) for c in (0,1,2,3)},
              'legacy_share_controls':round(float(x[x.case==0].legacy.mean()),4),'legacy_share_cases':round(float(x[x.case==1].legacy.mean()),4),
              'AF':af(x,['sc1','sc2','sc3'],rng)}
    M8[nm]=o
R['M8']=M8
M9={}
for nm,d in (('all',EV),('le39.5',EV[EV.slope<=39.5])):
    x=d[d.src.isin(CMP)].copy(); x['src']=pd.Categorical(x.src,categories=CMP)
    m,mr,dd,nd=fitfe(x,f"case~{TERR}+legacy+C(src)+legacy:C(src)")
    o={'n':int(len(dd)),'n_case':int(dd.case.sum()),'events_dropped':nd,
       'legacy_OR_within':{'primary':lin(m,mr,{'legacy':1}),
                           'modified':lin(m,mr,{'legacy':1,'legacy:C(src)[T.modified]':1}),
                           'plantation':lin(m,mr,{'legacy':1,'legacy:C(src)[T.plantation]':1})},
       'interaction_vs_primary':{'modified':lin(m,mr,{'legacy:C(src)[T.modified]':1}),'plantation':lin(m,mr,{'legacy:C(src)[T.plantation]':1})}}
    t=['legacy:C(src)[T.modified]','legacy:C(src)[T.plantation]']; b=mr.params[t].values; V=mr.cov_params().loc[t,t].values
    from scipy.stats import chi2
    w=float(b@np.linalg.solve(V,b)); o['joint_interaction_wald']={'chi2':round(w,3),'df':2,'p':float(1-chi2.cdf(w,2))}
    M9[nm]=o
R['M9']=M9
POPS={'all':lambda d:d,'excl_1_2':lambda d:d[~d.site_class4.isin([1,2])],'excl_1_3':lambda d:d[d.site_class4==0],'excl_1_3_le39.5':lambda d:d[(d.site_class4==0)&(d.slope<=39.5)]}
def m11(d, design='event'):
    x=d[d.src.isin(CMP)].copy(); x['modified']=(x.src=='modified').astype(int); x['plantation']=(x.src=='plantation').astype(int)
    m,mr,dd,nd=fitfe(x,f"case~{TERR}+modified+plantation+modified:s20+plantation:s20",design)
    o={'n':int(len(dd)),'n_case':int(dd.case.sum()),'events_dropped':nd,
       'n_cases_by_src':{s:int(dd[(dd.src==s)].case.sum()) for s in CMP},'n_control_rows_by_src':{s:int(((dd.src==s)&(dd.case==0)).sum()) for s in CMP}}
    for s in ('modified','plantation'):
        o[s]={'per_degree':lin(m,mr,{f'{s}:s20':1})}
        for deg in (20,30,40):
            o[s][f'OR_at_{deg}']=lin(m,mr,{s:1,f'{s}:s20':deg-20})
    return o
M11={'event':{k:m11(f(EV)) for k,f in POPS.items()},'island':{k:m11(f(IS),'island') for k,f in POPS.items()}}
loeo={}
for k in ('all','excl_1_3'):
    base=POPS[k](EV); rows=[]
    for ev in sorted(base.event.unique()):
        try:
            o=m11(base[base.event!=ev]); rows.append({'event_left_out':str(ev),'modified_20':o['modified']['OR_at_20']['OR'],'plantation_20':o['plantation']['OR_at_20']['OR']})
        except Exception as e:
            rows.append({'event_left_out':str(ev),'error':str(e)[:80]})
    ok=[r for r in rows if 'error' not in r]
    loeo[k]={'n_fits':len(ok),'n_failed':len(rows)-len(ok),
             'modified_20_range':[min(r['modified_20'] for r in ok),max(r['modified_20'] for r in ok)],
             'plantation_20_range':[min(r['plantation_20'] for r in ok),max(r['plantation_20'] for r in ok)],'rows':rows}
M11['loeo_event']=loeo
R['M11']=M11
def m10(d, cls, bins, label):
    x=d[(d.site_class4==cls)&(d.year>=2020)&d.chm.notna()].copy(); x['chm5']=x.chm/5.0; x['chm12']=(x.chm>=12).astype(int)
    out={'label':label,'n_rows':int(len(x)),'n_cases':int(x.case.sum())}
    for nm,y in (('all',x),('le39.5',x[x.slope<=39.5])):
        m,mr,dd,nd=fitfe(y,f"case~{TERR}+C({bins})+chm5")
        o={'n':int(len(dd)),'n_case':int(dd.case.sum()),'events_dropped':nd,'chm_per_5m':lin(m,mr,{'chm5':1}),
           'bins':{t.split('T.')[1].rstrip(']'):lin(m,mr,{t:1}) for t in m.params.index if t.startswith(f'C({bins})')}}
        m2,mr2,dd2,nd2=fitfe(y,f"case~{TERR}+C({bins})+chm12"); o['chm_ge12']=lin(m2,mr2,{'chm12':1})
        m3,mr3,dd3,nd3=fitfe(y,f"case~{TERR}+chm5"); o['chm_per_5m_without_age']=lin(m3,mr3,{'chm5':1})
        o['chm_median_cases']=round(float(dd[dd.case==1].chm.median()),2); o['chm_median_controls']=round(float(dd[dd.case==0].chm.median()),2)
        out[nm]=o
    return out
x1=EV.copy(); x1['tss1_bin']=pd.Categorical(x1.tss1_bin.astype(str),categories=['0-5','5-10','10+'])
R['M10_class1']=m10(x1,1,'tss1_bin','class 1 (prior scar, revegetated at cutoff), 2020-2025 events: terrain + scar-age bins (0-5 ref) + canopy height per 5 m')
x3=EV.copy(); x3['tsd3_bin']=pd.Categorical(x3.tsd3_bin.astype(str),categories=['0-5','5-10','10-20','20+'])
R['M10_class3_exploratory']=m10(x3,3,'tsd3_bin','exploratory: class 3 (LandTrendr disturbance), 2020-2025 events: terrain + years-since-disturbance bins (0-5 ref) + canopy height per 5 m')
json.dump(R,open(H+'/v10/an/g4/stand_origin_register_v1.json','w',encoding='utf-8'),ensure_ascii=False,indent=1,default=float)
print('done')
x=EV[(EV.site_class4==1)&(EV.year>=2020)&EV.chm.notna()].copy()
x['chm_dt']=pd.to_datetime(x.chm_date,errors='coerce'); x['scar_dt']=pd.to_datetime(x.scar_50m_last_after,errors='coerce')
x=x[x.chm_dt>x.scar_dt].copy(); x['chm5']=x.chm/5.0; x['chm12']=(x.chm>=12).astype(int)
x['tss1_bin']=pd.Categorical(x.tss1_bin.astype(str),categories=['0-5','5-10','10+'])
o={'label':'exploratory: class 1, 2020-2025 events, canopy image dated after the most recent prior scar','n_rows':int(len(x)),'n_cases':int(x.case.sum())}
for nm,y in (('all',x),('le39.5',x[x.slope<=39.5])):
    m,mr,dd,nd=fitfe(y,f"case~{TERR}+C(tss1_bin)+chm5")
    o[nm]={'n':int(len(dd)),'n_case':int(dd.case.sum()),'events_dropped':nd,'chm_per_5m':lin(m,mr,{'chm5':1})}
    o[nm]['tss1_bin_counts']={str(k):int(v) for k,v in dd.groupby('tss1_bin',observed=True).case.agg(['size']).iloc[:,0].items()}
R['M10_class1_chm_after_scar_exploratory']=o
json.dump(R,open(H+'/v10/an/g4/stand_origin_register_v1.json','w',encoding='utf-8'),ensure_ascii=False,indent=1,default=float)
print('exploratory M10 added', o['n_rows'], o['n_cases'], o['all']['chm_per_5m'], o['le39.5']['chm_per_5m'], o['all']['tss1_bin_counts'])
