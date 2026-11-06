"""Attributable fractions with Monte Carlo intervals, leave-one-event-out, small-event sensitivity, 20 km spatial block cross-validation and the specification curve."""
import pandas as pd, numpy as np, json, math, sys, warnings, itertools
import statsmodels.formula.api as smf
from scipy.stats import norm
warnings.filterwarnings('ignore')
sys.path.insert(0,'.'); from lt_segments import summarize
BRO='闊葉樹林型'; BAM='竹林'; TERR="slope+slope2+north+east+planc+profc+elev"; TERR_S="slope+slope2"
ISL=pd.read_csv('tags_island_v3.csv',dtype={'uid':str},low_memory=False); EV=pd.read_csv('@WORK@/v10/data/tags_event_v4.csv',dtype={'uid':str},low_memory=False)
rng=np.random.default_rng(20260924); R={}
def base(df):
    d=df.copy(); d['case']=d.case.astype(int); d['slope2']=d.slope**2; d['s20']=d.slope-20.0
    d['cluster']=np.where(d.kind=='case',d.event.astype(str),'controls'); d['bamboo']=(d.ft==BAM).astype(int)
    return d.dropna(subset=['slope','north','east','planc','profc','elev'])
def with_classes(d, r=50, thr_col='lt_detected'):
    d=d.copy()
    c1=d[f'class1_{r}m'].astype(int); c2=d[f'bare_at_cutoff_{0 if r==0 else 50}m'].astype(int)
    sc=np.where(c2==1,2,np.where(c1==1,1,0)); sc=np.where((sc==0)&(d[thr_col]==1),3,sc)
    d['sc4']=sc
    for c in (1,2,3): d[f'sc{c}']=(sc==c).astype(int)
    d['legacy_any4']=(sc>0).astype(int)
    return d
def drop_uninformative(d):
    g=d.groupby('event').case.agg(['sum','size']); keep=g[(g['sum']>0)&(g['sum']<g['size'])].index
    return d[d.event.isin(keep)].copy(), int(len(g)-len(keep))
def fit(d, form, design):
    if design=='event':
        d,_=drop_uninformative(d); f=form+'+C(event)'
        m=smf.logit(f,data=d).fit(disp=0,maxiter=100,method='newton'); mr=smf.logit(f,data=d).fit(disp=0,maxiter=100,method='newton',cov_type='cluster',cov_kwds={'groups':pd.factorize(d.uid)[0]})
    else:
        m=smf.logit(form,data=d).fit(disp=0,maxiter=100); mr=smf.logit(form,data=d).fit(disp=0,maxiter=100,cov_type='cluster',cov_kwds={'groups':pd.factorize(d.cluster)[0]})
    return m,mr,d
def ors(m,mr,terms):
    return {t:{'OR':round(math.exp(m.params[t]),4),'ci_cluster':[round(math.exp(m.params[t]-1.96*mr.bse[t]),4),round(math.exp(m.params[t]+1.96*mr.bse[t]),4)]} for t in terms}
def or_at(m,mr,main,inter,deg,center=20.0):
    b=m.params[main]+(deg-center)*m.params[inter]; c=mr.cov_params(); se=math.sqrt(c.loc[main,main]+(deg-center)**2*c.loc[inter,inter]+2*(deg-center)*c.loc[main,inter])
    return {'OR':round(math.exp(b),4),'ci_cluster':[round(math.exp(b-1.96*se),4),round(math.exp(b+1.96*se),4)]}
bi=with_classes(base(ISL[ISL.ft==BRO])); be=with_classes(base(EV[EV.ft==BRO]))
hi=with_classes(base(ISL[ISL.ft.isin([BRO,BAM])])); he=with_classes(base(EV[EV.ft.isin([BRO,BAM])]))
def paf(d, design, ndraw=4000):
    m,mr,dd=fit(d,f"case~{TERR}+sc1+sc2+sc3",design)
    cases=dd[dd.case==1]; rho={c:float((cases.sc4==c).mean()) for c in (0,1,2,3)}
    terms=['sc1','sc2','sc3']; b=m.params[terms].values; V=mr.cov_params().loc[terms,terms].values
    draws=rng.multivariate_normal(b,V,size=ndraw)
    def af(bv):
        o=np.exp(bv); parts={c:rho[c]*(o[c-1]-1)/o[c-1] for c in (1,2,3)}; parts['total']=1-(rho[0]+sum(rho[c]/o[c-1] for c in (1,2,3))); return parts
    pt=af(b); sims={k:[] for k in pt}
    n=len(cases); rho0=np.array([rho[c] for c in (0,1,2,3)])
    for bv in draws:
        cnt=rng.multinomial(n,rho0); rho={c:cnt[c]/n for c in (0,1,2,3)}
        a=af(bv)
        for k in a: sims[k].append(a[k])
    rho={c:float(rho0[c]) for c in (0,1,2,3)}
    out={'n_cases':int(len(cases)),'rho_cases_by_class':{str(k):round(v,4) for k,v in rho.items()},
         'OR':{t:round(float(math.exp(m.params[t])),3) for t in terms}}
    for k in pt: out[f'AF_{k}']={'point':round(float(pt[k]),4),'ci_mc':[round(float(np.percentile(sims[k],2.5)),4),round(float(np.percentile(sims[k],97.5)),4)]}
    return out
R['attributable_fraction']={'event_all_slopes':paf(be,'event'),'event_le39.5':paf(be[be.slope<=39.5],'event'),
                            'island_all_slopes':paf(bi,'island'),'island_le39.5':paf(bi[bi.slope<=39.5],'island')}
print('AF',json.dumps(R['attributable_fraction']['event_le39.5'],ensure_ascii=False))
ev=be.groupby(['year','event']).agg(cases=('case','sum'),rows=('case','size'),control_points=('uid',lambda s: s[be.loc[s.index,'case']==0].nunique())).reset_index()
ev['controls']=ev.rows-ev.cases; ev['year']=ev.year.astype(int)
for c in (1,2,3): ev[f'cases_class{c}']=be[be.case==1].groupby(['year','event'])[f'sc{c}'].sum().reindex(pd.MultiIndex.from_frame(ev[['year','event']])).values
ev=ev.sort_values(['year','cases']); ev.to_csv('events_v1.csv',index=False,encoding='utf-8-sig')
R['events']={'n_events':int(len(ev)),'events_lt5_cases':int((ev.cases<5).sum()),'events_lt10_cases':int((ev.cases<10).sum()),'min_controls_per_event':int(ev.controls.min()),
             'cases_in_events_lt5':int(ev[ev.cases<5].cases.sum()),'cases_in_events_lt10':int(ev[ev.cases<10].cases.sum())}
def main_quant(dB, dH, design):
    m,mr,_=fit(dB,f"case~{TERR}+sc1+sc2+sc3",design); o=ors(m,mr,['sc1','sc2','sc3'])
    m4,mr4,_=fit(dB,f"case~{TERR}+legacy_any4+legacy_any4:s20",design); o['legacy_x_slope']=ors(m4,mr4,['legacy_any4:s20'])['legacy_any4:s20']; o['legacy_OR_20']=or_at(m4,mr4,'legacy_any4','legacy_any4:s20',20); o['legacy_OR_40']=or_at(m4,mr4,'legacy_any4','legacy_any4:s20',40)
    h=dH[(dH.bamboo==1)|((dH.bamboo==0)&(dH.sc4==0))]
    mh,mrh,_=fit(h,f"case~{TERR}+bamboo+bamboo:slope",design); o['H4_class0_OR20']=or_at(mh,mrh,'bamboo','bamboo:slope',20,center=0.0); o['H4_interaction']=ors(mh,mrh,['bamboo:slope'])['bamboo:slope']
    hf=dH; mf,mrf,_=fit(hf,f"case~{TERR}+bamboo+bamboo:slope",design); o['H4_full_OR20']=or_at(mf,mrf,'bamboo','bamboo:slope',20,center=0.0)
    return o
R['reference_event']=main_quant(be,he,'event')
rows=[]
for e in sorted(be.event.unique()):
    q=main_quant(be[be.event!=e],he[he.event!=e],'event')
    rows.append({'event_left_out':e,'cases_left_out':int(be[(be.event==e)&(be.case==1)].shape[0]),'OR_sc1':q['sc1']['OR'],'OR_sc2':q['sc2']['OR'],'OR_sc3':q['sc3']['OR'],
                 'legacy_x_slope':q['legacy_x_slope']['OR'],'legacy_OR_20':q['legacy_OR_20']['OR'],'H4_class0_OR20':q['H4_class0_OR20']['OR'],'H4_full_OR20':q['H4_full_OR20']['OR']})
loeo=pd.DataFrame(rows); loeo.to_csv('loeo_v1.csv',index=False,encoding='utf-8-sig')
R['leave_one_event_out']={c:{'min':float(loeo[c].min()),'max':float(loeo[c].max()),'event_min':str(loeo.loc[loeo[c].idxmin(),'event_left_out']),'event_max':str(loeo.loc[loeo[c].idxmax(),'event_left_out'])} for c in ['OR_sc1','OR_sc2','OR_sc3','legacy_x_slope','legacy_OR_20','H4_class0_OR20','H4_full_OR20']}
print('LOEO',json.dumps(R['leave_one_event_out'],ensure_ascii=False))
R['small_event_sensitivity']={}
for k in (5,10):
    keep=set(ev[ev.cases>=k].event); R['small_event_sensitivity'][f'events_with_ge{k}_cases']={'n_events':len(keep),**main_quant(be[be.event.isin(keep)],he[he.event.isin(keep)],'event')}
def blocks(d): return (np.floor(d.x/20000).astype(int).astype(str)+'_'+np.floor(d.y/20000).astype(int).astype(str))
bi['block']=blocks(bi); be['block']=blocks(be); hi['block']=blocks(hi); he['block']=blocks(he)
ub=np.array(sorted(set(bi.block)|set(be.block))); fold_of={b:int(i) for b,i in zip(ub,rng.integers(0,5,size=len(ub)))}
for d in (bi,be,hi,he): d['fold']=d.block.map(fold_of)
def auc(y,p):
    y=np.asarray(y); p=np.asarray(p); pos=p[y==1]; neg=p[y==0]
    allv=np.concatenate([pos,neg]); ranks=pd.Series(allv).rank().values; return float((ranks[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))
cv={'n_blocks':int(len(ub)),'folds':[]}
for f in range(5):
    tr=bi[bi.fold!=f]; te=bi[bi.fold==f]
    m0=smf.logit(f"case~{TERR}",data=tr).fit(disp=0); m1=smf.logit(f"case~{TERR}+sc1+sc2+sc3",data=tr).fit(disp=0)
    cv['folds'].append({'fold':f,'n_test':int(len(te)),'cases_test':int(te.case.sum()),'auc_terrain':round(auc(te.case,m0.predict(te)),4),'auc_terrain_legacy':round(auc(te.case,m1.predict(te)),4),
                        'OR_sc_train':[round(float(math.exp(m1.params[t])),3) for t in ('sc1','sc2','sc3')]})
cv['auc_terrain_mean']=round(float(np.mean([x['auc_terrain'] for x in cv['folds']])),4); cv['auc_terrain_legacy_mean']=round(float(np.mean([x['auc_terrain_legacy'] for x in cv['folds']])),4)
cv['event_leave_fold_out']=[]
for f in range(5):
    q=main_quant(be[be.fold!=f],he[he.fold!=f],'event'); cv['event_leave_fold_out'].append({'fold':f,'OR_sc1':q['sc1']['OR'],'OR_sc2':q['sc2']['OR'],'OR_sc3':q['sc3']['OR'],'legacy_x_slope':q['legacy_x_slope']['OR'],'H4_class0_OR20':q['H4_class0_OR20']['OR']})
R['spatial_block_cv_20km']=cv; print('CV',cv['auc_terrain_mean'],cv['auc_terrain_legacy_mean'])
segs=pd.read_csv('lt_segments_all_v1.csv',dtype={'uid':str}); fits=pd.read_csv('lt_fits_all_v1.csv',dtype={'uid':str})
det={}
for thr in (75,100,125):
    S=summarize(segs,fits,mag_thr=float(thr),max_dur=3); det[thr]=S.set_index(['uid','cutoff_year']).dist_year.notna().astype(int)
def add_det(d):
    d=d.copy(); idx=pd.MultiIndex.from_frame(d[['uid','cutoff_year']])
    for thr in det: d[f'lt_{thr}']=det[thr].reindex(idx).fillna(0).astype(int).values
    return d
BI=add_det(base(ISL[ISL.ft==BRO])); BE=add_det(base(EV[EV.ft==BRO])); HI=add_det(base(ISL[ISL.ft.isin([BRO,BAM])])); HE=add_det(base(EV[EV.ft.isin([BRO,BAM])]))
specs=[]
for design,r,srange,thr,terr in itertools.product(('event','island'),(0,50,100),('all','le39.5'),(75,100,125),('full','slope_only')):
    dB=with_classes((BE if design=='event' else BI),r,f'lt_{thr}'); dH=with_classes((HE if design=='event' else HI),r,f'lt_{thr}')
    if srange=='le39.5': dB=dB[dB.slope<=39.5]; dH=dH[dH.slope<=39.5]
    T=TERR if terr=='full' else TERR_S
    m,mr,_=fit(dB,f"case~{T}+sc1+sc2+sc3",design); o=ors(m,mr,['sc1','sc2','sc3'])
    h=dH[(dH.bamboo==1)|((dH.bamboo==0)&(dH.sc4==0))]; mh,mrh,_=fit(h,f"case~{T}+bamboo+bamboo:slope",design); h4=or_at(mh,mrh,'bamboo','bamboo:slope',20,center=0.0)
    specs.append({'design':design,'radius_m':r,'slope_range':srange,'lt_threshold_x1000':thr,'terrain':terr,'n':int(len(dB)),'n_cases':int(dB.case.sum()),
                  'OR_sc1':o['sc1']['OR'],'lo_sc1':o['sc1']['ci_cluster'][0],'hi_sc1':o['sc1']['ci_cluster'][1],'OR_sc2':o['sc2']['OR'],'lo_sc2':o['sc2']['ci_cluster'][0],'hi_sc2':o['sc2']['ci_cluster'][1],
                  'OR_sc3':o['sc3']['OR'],'lo_sc3':o['sc3']['ci_cluster'][0],'hi_sc3':o['sc3']['ci_cluster'][1],'H4_class0_OR20':h4['OR'],'lo_H4':h4['ci_cluster'][0],'hi_H4':h4['ci_cluster'][1],'n_bamboo_cases':int(((h.case==1)&(h.bamboo==1)).sum())})
    print(specs[-1]['design'],r,srange,thr,terr,specs[-1]['OR_sc1'],specs[-1]['OR_sc2'],specs[-1]['OR_sc3'],specs[-1]['H4_class0_OR20'])
sc=pd.DataFrame(specs); sc.to_csv('spec_curve_v1.csv',index=False,encoding='utf-8-sig')
R['specification_curve']={'n_specs':int(len(sc)),'OR_sc1_range':[float(sc.OR_sc1.min()),float(sc.OR_sc1.max())],'OR_sc2_range':[float(sc.OR_sc2.min()),float(sc.OR_sc2.max())],'OR_sc3_range':[float(sc.OR_sc3.min()),float(sc.OR_sc3.max())],
                          'share_sc3_lower_ci_gt1':float((sc.lo_sc3>1).mean()),'H4_class0_OR20_range':[float(sc.H4_class0_OR20.min()),float(sc.H4_class0_OR20.max())],'share_H4_ci_includes_1':float(((sc.lo_H4<=1)&(sc.hi_H4>=1)).mean()),
                          'H4_class0_OR20_event_range':[float(sc[sc.design=='event'].H4_class0_OR20.min()),float(sc[sc.design=='event'].H4_class0_OR20.max())]}
json.dump(R,open('robust_register_v1.json','w'),ensure_ascii=False,indent=1,default=str)
print(json.dumps(R['specification_curve'],ensure_ascii=False))
