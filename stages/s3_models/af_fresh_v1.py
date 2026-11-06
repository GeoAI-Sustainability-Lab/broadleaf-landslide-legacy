"""Attributable fractions with scars younger than one year split out (five-level classes) and composition tables."""
import pandas as pd, numpy as np, json, math, warnings
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
BRO='闊葉樹林型'; TERR="slope+slope2+north+east+planc+profc+elev"
ISL=pd.read_csv('tags_island_v3.csv',dtype={'uid':str},low_memory=False); EV=pd.read_csv('@WORK@/v10/data/tags_event_v4.csv',dtype={'uid':str},low_memory=False)
rng=np.random.default_rng(20260924); R={}
def prep(df):
    d=df[df.ft==BRO].copy(); d['case']=d.case.astype(int); d['slope2']=d.slope**2
    d['cluster']=np.where(d.kind=='case',d.event.astype(str),'controls')
    fresh=(d.scar_50m_min_yrs<1)&(d.site_class4.isin([1,2]))
    d['lvl']=np.select([d.site_class4==0,fresh,d.site_class4==1,d.site_class4==2],['c0','fresh','c1','c2'],'c3')
    d['lvl']=pd.Categorical(d.lvl,categories=['c0','fresh','c1','c2','c3'])
    d['fresh_inside']=(fresh&(d.scar_0m_min_yrs<1)).astype(int)
    d['fresh_adj']=(fresh&~(d.scar_0m_min_yrs<1)).astype(int)
    for l in ('fresh','c1','c2','c3'): d[f'x_{l}']=(d.lvl==l).astype(int)
    return d.dropna(subset=['slope','north','east','planc','profc','elev'])
def drop_uninformative(d):
    g=d.groupby('event').case.agg(['sum','size']); keep=g[(g['sum']>0)&(g['sum']<g['size'])].index
    return d[d.event.isin(keep)].copy(), int(len(g)-len(keep))
def fit(d, form, design):
    if design=='event':
        d,nd=drop_uninformative(d); f=form+'+C(event)'
        m=smf.logit(f,data=d).fit(disp=0,maxiter=100,method='newton'); mr=smf.logit(f,data=d).fit(disp=0,maxiter=100,method='newton',cov_type='cluster',cov_kwds={'groups':pd.factorize(d.uid)[0]})
    else:
        nd=0; m=smf.logit(form,data=d).fit(disp=0,maxiter=100); mr=smf.logit(form,data=d).fit(disp=0,maxiter=100,cov_type='cluster',cov_kwds={'groups':pd.factorize(d.cluster)[0]})
    assert m.mle_retvals.get('converged',True)
    return m,mr,d,nd
def af_block(d, design, terms, ndraw=4000):
    m,mr,dd,nd=fit(d,f"case~{TERR}+"+"+".join(terms),design)
    cases=dd[dd.case==1]; n=len(cases); levels=[t.replace('x_','') for t in terms]
    rho0=np.array([float((cases.lvl==l).mean()) for l in levels]); rho_c0=1-rho0.sum()
    b=m.params[terms].values; V=mr.cov_params().loc[terms,terms].values; draws=rng.multivariate_normal(b,V,size=ndraw)
    def af(bv,rho):
        o=np.exp(bv); parts={l:rho[i]*(o[i]-1)/o[i] for i,l in enumerate(levels)}; parts['total']=sum(parts.values()); return parts
    pt=af(b,rho0); sims={k:[] for k in pt}
    full=np.append(rho0,rho_c0)
    for bv in draws:
        cnt=rng.multinomial(n,full); a=af(bv,cnt[:-1]/n)
        for k in a: sims[k].append(a[k])
    out={'design':design,'n':int(len(dd)),'n_cases':n,'events_dropped':nd,'rho_cases':{l:round(float(r),4) for l,r in zip(levels,rho0)},'rho_c0':round(float(rho_c0),4),
         'OR':{l:{'OR':round(float(math.exp(m.params[t])),3),'ci_cluster':[round(float(math.exp(m.params[t]-1.96*mr.bse[t])),3),round(float(math.exp(m.params[t]+1.96*mr.bse[t])),3)]} for l,t in zip(levels,terms)}}
    for k in pt: out[f'AF_{k}']={'point':round(float(pt[k]),4),'ci_mc':[round(float(np.percentile(sims[k],2.5)),4),round(float(np.percentile(sims[k],97.5)),4)]}
    if 'fresh' in levels:
        ex=[k for k in levels if k!='fresh']; s=np.array([sims[k] for k in ex]).sum(0); p=sum(pt[k] for k in ex)
        out['AF_excluding_fresh']={'point':round(float(p),4),'ci_mc':[round(float(np.percentile(s,2.5)),4),round(float(np.percentile(s,97.5)),4)]}
    return out
bi=prep(ISL); be=prep(EV)
R['definitions']={'fresh':'site_class4 in {1,2} and years since most recent prior catalog scar within 50 m < 1','c1':'site_class4==1 and scar age >= 1 yr','c2':'site_class4==2 without a fresh scar','c3':'site_class4==3 (LandTrendr)'}
R['composition']={}
for name,d in [('island',bi),('event',be)]:
    c=d[d.case==1]; c39=c[c.slope<=39.5]
    R['composition'][name]={'cases_by_level':c.lvl.value_counts().reindex(['c0','fresh','c1','c2','c3']).to_dict(),'cases_le39.5_by_level':c39.lvl.value_counts().reindex(['c0','fresh','c1','c2','c3']).to_dict(),
                            'fresh_inside_cases':int(c.fresh_inside.sum()),'fresh_adjacent_cases':int(c.fresh_adj.sum()),'fresh_inside_cases_le39.5':int(c39.fresh_inside.sum()),'fresh_adjacent_cases_le39.5':int(c39.fresh_adj.sum()),
                            'control_rows_by_level':d[d.case==0].lvl.value_counts().reindex(['c0','fresh','c1','c2','c3']).to_dict()}
T=['x_fresh','x_c1','x_c2','x_c3']
R['five_level']={'event_all_slopes':af_block(be,'event',T),'event_le39.5':af_block(be[be.slope<=39.5],'event',T),'island_all_slopes':af_block(bi,'island',T),'island_le39.5':af_block(bi[bi.slope<=39.5],'island',T)}
def split_block(d,design):
    d=d.copy(); d['x_fi']=d.fresh_inside; d['x_fa']=d.fresh_adj
    m,mr,dd,nd=fit(d,f"case~{TERR}+x_fi+x_fa+x_c1+x_c2+x_c3",design)
    cases=dd[dd.case==1]; out={'n_cases':int(len(cases))}
    for t,l in [('x_fi','fresh_inside'),('x_fa','fresh_adjacent'),('x_c1','c1'),('x_c2','c2'),('x_c3','c3')]:
        rho=float(cases[t].mean()); o=math.exp(m.params[t]); out[l]={'rho_cases':round(rho,4),'OR':round(o,3),'ci_cluster':[round(math.exp(m.params[t]-1.96*mr.bse[t]),3),round(math.exp(m.params[t]+1.96*mr.bse[t]),3)],'AF_point':round(rho*(o-1)/o,4)}
    return out
R['fresh_split']={'event_all_slopes':split_block(be,'event'),'event_le39.5':split_block(be[be.slope<=39.5],'event')}
T3=['x_c1','x_c2','x_c3']
R['drop_fresh']={'event_all_slopes':af_block(be[be.lvl!='fresh'],'event',T3),'event_le39.5':af_block(be[(be.lvl!='fresh')&(be.slope<=39.5)],'event',T3)}
json.dump(R,open('af_fresh_register_v1.json','w'),ensure_ascii=False,indent=1,default=str)
for k,v in R['five_level'].items(): print(k,'n_cases',v['n_cases'],'rho',v['rho_cases'],'OR',{l:o['OR'] for l,o in v['OR'].items()},'AF',{kk:vv['point'] for kk,vv in v.items() if kk.startswith('AF')})
print('split',json.dumps(R['fresh_split']['event_le39.5']))
for k,v in R['drop_fresh'].items(): print('drop',k,v['n_cases'],{kk:vv['point'] for kk,vv in v.items() if kk.startswith('AF')})
print(json.dumps(R['composition']['island'],ensure_ascii=False))
