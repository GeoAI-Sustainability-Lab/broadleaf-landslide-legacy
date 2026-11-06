"""Replication tags and models with the main-analysis definitions, attributable fractions, expected-risk trend."""
import os, sys, json, math, glob, warnings, numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
warnings.filterwarnings('ignore')
H='@WORK@'
sys.path.insert(0,H+'/g1'); from lt_segments import summarize, segments_from_row
sys.path.insert(0,H+'/g5'); from terrain_v1 import terrain
SEED=20261010; NDRAW=4000; BRO='闊葉樹林型'; TERR="slope+slope2+north+east+planc+profc+elev"
SIX=['闊葉樹林型','針葉樹林型','針闊葉樹混淆','竹林','竹闊混淆林','待成林地']
RUN_REP=os.environ.get('RUN_REP','1')=='1'
R={'charter':'charter_addendum_5_replication (2026-10-10, frozen before fitting)','seed':SEED}
def drop_uninf(d):
    g=d.groupby('event').case.agg(['sum','size']); keep=g[(g['sum']>0)&(g['sum']<g['size'])].index
    return d[d.event.isin(keep)].copy(), int(len(g)-len(keep))
def fitfe(d, form):
    d,nd=drop_uninf(d); f2=form+'+C(event)'
    m=smf.logit(f2,data=d).fit(disp=0,maxiter=100,method='newton')
    mr=smf.logit(f2,data=d).fit(disp=0,maxiter=100,method='newton',cov_type='cluster',cov_kwds={'groups':pd.factorize(d.uid)[0]})
    assert m.mle_retvals.get('converged',True), form
    return m,mr,d,nd
def term(m,mr,t):
    b=m.params[t]; s=m.bse[t]; sr=mr.bse[t]
    return {'OR':round(math.exp(b),4),'ci':[round(math.exp(b-1.96*sr),4),round(math.exp(b+1.96*sr),4)],'ci_model':[round(math.exp(b-1.96*s),4),round(math.exp(b+1.96*s),4)],
            'p':float(2*(1-norm.cdf(abs(b/sr)))),'beta':float(b),'se':float(sr)}
def wald(mr,ta,tb):
    b=mr.params[ta]-(mr.params[tb] if tb else 0.0); c=mr.cov_params(); se=math.sqrt(c.loc[ta,ta]+(c.loc[tb,tb]-2*c.loc[ta,tb] if tb else 0.0))
    return {'ratio':round(math.exp(b),4),'ci':[round(math.exp(b-1.96*se),4),round(math.exp(b+1.96*se),4)],'p':float(2*(1-norm.cdf(abs(b/se)))),'beta':float(b),'se':float(se)}
def counts(d,col):
    return {str(k):{'cases':int(v.get(1,0)),'control_rows':int(v.get(0,0))} for k,v in d.groupby(col,observed=True).case.value_counts().unstack().fillna(0).iterrows()}
def af(d, terms, rng):
    m,mr,dd,nd=fitfe(d,f"case~{TERR}+"+"+".join(terms))
    cases=dd[dd.case==1]; n=len(cases)
    rho=np.array([float(cases[t].mean()) for t in terms]); full=np.append(rho,1-rho.sum())
    b=m.params[terms].values; V=mr.cov_params().loc[terms,terms].values; draws=rng.multivariate_normal(b,V,size=NDRAW)
    def f(bv,r): return [r[i]*(1-math.exp(-bv[i])) for i in range(len(terms))]
    pt=f(b,rho); sims=[]
    for bv in draws:
        cnt=rng.multinomial(n,full); sims.append(f(bv,cnt[:-1]/n))
    sims=np.array(sims)
    out={'n_cases':n,'events_dropped':nd,'rho_cases':{t:round(float(r),4) for t,r in zip(terms,rho)},
         'OR':{t:term(m,mr,t) for t in terms}}
    for i,t in enumerate(terms): out[f'AF_{t}']={'point':round(pt[i],4),'ci_mc':[round(float(np.percentile(sims[:,i],2.5)),4),round(float(np.percentile(sims[:,i],97.5)),4)]}
    out['AF_total']={'point':round(sum(pt),4),'ci_mc':[round(float(np.percentile(sims.sum(1),2.5)),4),round(float(np.percentile(sims.sum(1),97.5)),4)]}
    return out
def bright(B, qcol):
    out={}; B=B.copy(); B['Qc']=pd.Categorical(B[qcol].astype(int),categories=sorted(B[qcol].unique().astype(int))); qs=list(B.Qc.cat.categories); top=qs[-1]
    comp={}
    for q in qs:
        s=B[B[qcol]==q]
        comp[int(q)]={'n_cases':int(s.case.sum()),'n_control_rows':int((s.case==0).sum()),
                      'intact_share_cases':round(float((s[s.case==1].R0==1).mean()),4),'intact_share_controls':round(float((s[s.case==0].R0==1).mean()),4)}
    out['B1_composition']=comp
    m,mr,d,nd=fitfe(B,f"case~C(Qc)+C(Qc):legacy+{TERR}")
    out['B2_legacy_by_stratum']={int(q):term(m,mr,f'C(Qc)[{q}]:legacy') for q in qs}
    B['qrank']=B[qcol].astype(int)-1
    m,mr,d,nd=fitfe(B,f"case~C(Qc)+legacy+legacy:qrank+{TERR}")
    out['B2_trend_per_stratum_step']=term(m,mr,'legacy:qrank')
    B['cell']=B[qcol].astype(int).astype(str)+'_'+np.where(B.legacy==1,'legacy','intact')
    cats=[f'{int(q)}_{s}' for q in qs for s in ('intact','legacy')]; B['cell']=pd.Categorical(B.cell,categories=cats)
    m,mr,d,nd=fitfe(B,"case~C(cell)")
    cells={c:({'OR':1.0,'ci':[1.0,1.0]} if c==cats[0] else term(m,mr,f'C(cell)[T.{c}]')) for c in cats}
    cn=d.groupby('cell',observed=True).case.agg(['sum','size'])
    for c in cats: cells[c]['n_cases']=int(cn.loc[c,'sum']) if c in cn.index else 0; cells[c]['n_control_rows']=int(cn.loc[c,'size']-cn.loc[c,'sum']) if c in cn.index else 0
    out['B3_cells']=cells
    out['B3_contrasts']={f'{top}_intact_vs_1_legacy':wald(mr,f'C(cell)[T.{top}_intact]','C(cell)[T.1_legacy]'),
                         f'{top}_intact_vs_3_legacy':wald(mr,f'C(cell)[T.{top}_intact]','C(cell)[T.3_legacy]'),
                         f'{top}_legacy_vs_{top}_intact':wald(mr,f'C(cell)[T.{top}_legacy]',f'C(cell)[T.{top}_intact]')}
    return out
def block_folds(df, seed):
    blk=np.floor(df.x/20000).astype(int).astype(str)+'_'+np.floor(df.y/20000).astype(int).astype(str)
    ub=np.array(sorted(blk.unique())); rng=np.random.default_rng(seed); fo={b:int(f) for b,f in zip(ub,rng.integers(0,5,size=len(ub)))}
    return blk.map(fo).values, int(len(ub))
def auc(y,s):
    from scipy.stats import rankdata
    y=np.asarray(y); r=rankdata(s); n1=y.sum(); n0=len(y)-n1
    return float((r[y==1].sum()-n1*(n1+1)/2)/(n1*n0))
def period_block(B, label, quint_col, rng):
    o={'label':label,'n_rows':int(len(B)),'n_cases':int(B.case.sum()),'n_control_points':int(B[B.case==0].uid.nunique()),'n_events':int(B.event.nunique()),
       'n_cases_le39.5':int(((B.case==1)&(B.slope<=39.5)).sum()),'n_control_rows_le39.5':int(((B.case==0)&(B.slope<=39.5)).sum()),
       'composition_cases':{c:round(float(B[B.case==1][c].mean()),4) for c in ('R1','R3','R0')},
       'composition_controls':{c:round(float(B[B.case==0][c].mean()),4) for c in ('R1','R3','R0')},
       'composition_cases_le39.5':{c:round(float(B[(B.case==1)&(B.slope<=39.5)][c].mean()),4) for c in ('R1','R3','R0')},
       'composition_controls_le39.5':{c:round(float(B[(B.case==0)&(B.slope<=39.5)][c].mean()),4) for c in ('R1','R3','R0')}}
    L=B[B.slope<=39.5]
    for nm,d in (('all',B),('le39.5',L)):
        m,mr,dd,nd=fitfe(d,f"case~{TERR}+legacy"); o[f'MR1_{nm}']={'legacy':term(m,mr,'legacy'),'events_dropped':nd,'n':int(len(dd)),'n_case':int(dd.case.sum())}
        m,mr,dd,nd=fitfe(d,f"case~{TERR}+R1+R3"); o[f'MR2_{nm}']={'R1':term(m,mr,'R1'),'R3':term(m,mr,'R3'),'events_dropped':nd}
    m,mr,dd,nd=fitfe(B,f"case~{TERR}+C(scar_bin)+R3")
    o['MR3a_scar_age']={k:term(m,mr,f'C(scar_bin)[T.{k}]') for k in ('0-5','5-10','10+') if f'C(scar_bin)[T.{k}]' in m.params.index}
    o['MR3a_counts']=counts(dd,'scar_bin')
    if 'C(scar_bin)[T.10+]' in m.params.index: o['MR3a_0-5_vs_10+']=wald(mr,'C(scar_bin)[T.0-5]','C(scar_bin)[T.10+]')
    o['MR3a_0-5_vs_5-10']=wald(mr,'C(scar_bin)[T.0-5]','C(scar_bin)[T.5-10]')
    m,mr,dd,nd=fitfe(B,f"case~{TERR}+R1+C(tsd_bin)")
    o['MR3b_tsd']={k:term(m,mr,f'C(tsd_bin)[T.{k}]') for k in ('0-5','5-10','10-20','20+')}
    o['MR3b_counts']=counts(dd,'tsd_bin')
    o['MR3b_5-10_vs_20+']=wald(mr,'C(tsd_bin)[T.5-10]','C(tsd_bin)[T.20+]'); o['MR3b_0-5_vs_10-20']=wald(mr,'C(tsd_bin)[T.0-5]','C(tsd_bin)[T.10-20]')
    m,mr,dd,nd=fitfe(B,f"case~{TERR}+legacy+legacy:s20"); o['MR4_per_degree']=term(m,mr,'legacy:s20')
    o['MR4_OR_at']={}
    for deg in (10,20,30,40):
        b=m.params['legacy']+(deg-20)*m.params['legacy:s20']; c=mr.cov_params(); se=math.sqrt(c.loc['legacy','legacy']+(deg-20)**2*c.loc['legacy:s20','legacy:s20']+2*(deg-20)*c.loc['legacy','legacy:s20'])
        o['MR4_OR_at'][str(deg)]={'OR':round(math.exp(b),4),'ci':[round(math.exp(b-1.96*se),4),round(math.exp(b+1.96*se),4)]}
    o['AF_le39.5']=af(L,['R1','R3'],rng)
    o['bright_terrain_only']=bright(B,quint_col)
    return o
E=pd.read_csv('@WORK@/v10/data/tags_event_v4.csv',dtype={'uid':str},low_memory=False)
BR=pd.read_csv(H+'/v10/an/g3/bright_rows_v1.csv',dtype={'uid':str})
E=E.merge(BR[['uid','event','Qt','score_terrain']],on=['uid','event'],how='left')
M=E[E.ft==BRO].copy(); M['case']=M.case.astype(int); M['slope2']=M.slope**2; M['s20']=M.slope-20
M=M.dropna(subset=['slope','north','east','planc','profc','elev'])
M['R1']=(M.class1_50m==1).astype(int); M['R3']=((M.lt_detected==1)&(M.R1==0)).astype(int); M['legacy']=((M.R1==1)|(M.R3==1)).astype(int); M['R0']=1-M.legacy
M['scar_age']=M.scar_50m_min_yrs
M['scar_bin']=pd.Categorical(np.select([M.R1==0,M.scar_age<5,M.scar_age<10],['none','0-5','5-10'],'10+'),categories=['none','0-5','5-10','10+'])
M['tsd_bin']=pd.Categorical(np.select([M.R3==0,M.TSD<5,M.TSD<10,M.TSD<20],['none','0-5','5-10','10-20'],'20+'),categories=['none','0-5','5-10','10-20','20+'])
R['main_definitions']={'R1':'class1_50m (catalog scar <= 50 m, any bare state)','R3':'LandTrendr detected and not R1','R0':'neither (includes bare without a catalog scar and no LandTrendr detection)',
    'rows_by_class':{'R1':int(M.R1.sum()),'R3':int(M.R3.sum()),'R0':int(M.R0.sum())},'quintiles':'terrain-only out-of-fold expected risk (Qt) from the gate-3 run'}
rng=np.random.default_rng(SEED)
R['main']=period_block(M,'2018-2025 events, existing event-matched table, same definitions as the replication','Qt',rng)
t=terrain(M.x.values,M.y.values); M2=M.copy()
for k in ('slope','north','east','planc','profc','elev'): M2[k]=t[k]
M2['slope2']=M2.slope**2
for c in (1,2,3): M2[f'sc{c}']=(M2.site_class4==c).astype(int); M[f'sc{c}']=(M.site_class4==c).astype(int)
T1={}
for nm,(a,b) in {'all':(M,M2),'le39.5':(M[M.slope<=39.5],M2[M2.slope<=39.5])}.items():
    m,mr,_,_=fitfe(a,f"case~{TERR}+sc1+sc2+sc3"); m2,mr2,_,_=fitfe(b,f"case~{TERR}+sc1+sc2+sc3")
    T1[nm]={'table_terrain':{f'sc{c}':term(m,mr,f'sc{c}') for c in (1,2,3)},'dem_terrain':{f'sc{c}':term(m2,mr2,f'sc{c}') for c in (1,2,3)}}
T1['rows_terrain_differs_1e-3']=int((np.abs(M.slope.values-M2.slope.values)>1e-3).sum()); T1['n_rows']=int(len(M))
R['T1_main_terrain_recomputed']=T1
R['terrain_check']=json.load(open(H+'/g5/terrain_check_v1.json'))
if RUN_REP:
    D=pd.read_csv(H+'/g5/rep_rows_v1.csv',parse_dates=['cutoff','scar50_last_after'])
    rows=[]; fits=[]
    for f in sorted(glob.glob(H+'/g5/lt_batches_rep/lt_*.jsonl')):
        for line in open(f):
            r=json.loads(line); fits.append({'uid':r['uid'],'cutoff_year':r['cutoff_year'],'fit':r.get('fit',0),'rmse':r.get('rmse'),'end_fit':(r['fitted'][-1] if r.get('fit') else None),'n_years':(len(r['years']) if r.get('fit') else 0)})
            rows+=segments_from_row(r)
    segs=pd.DataFrame(rows); fits=pd.DataFrame(fits)
    S=summarize(segs,fits,mag_thr=100.0,max_dur=3)
    B=D[D.TypeName==BRO].copy()
    B=B.merge(S[['uid','cutoff_year','fit','dist_year','dist_mag','TSD','recovery_ratio']],on=['uid','cutoff_year'],how='left')
    B['lt_detected']=B.dist_year.notna().astype(int); B['lt_missing']=B.fit.isna().astype(int)
    B['R3']=((B.lt_detected==1)&(B.R1==0)).astype(int); B['legacy']=((B.R1==1)|(B.R3==1)).astype(int); B['R0']=1-B.legacy
    B['s20']=B.slope-20
    B['scar_bin']=pd.Categorical(np.select([B.R1==0,B.scar_age<5,B.scar_age<10],['none','0-5','5-10'],'10+'),categories=['none','0-5','5-10','10+'])
    B['tsd_bin']=pd.Categorical(np.select([B.R3==0,B.TSD<5,B.TSD<10,B.TSD<20],['none','0-5','5-10','10-20'],'20+'),categories=['none','0-5','5-10','10-20','20+'])
    A=D[D.TypeName.isin(SIX)].copy().reset_index(drop=True); A['fold'],nb=block_folds(A,SEED); A['score_terrain']=np.nan; folds=[]
    for f in range(5):
        tr=A[A.fold!=f]; te=A.fold==f
        mt=smf.logit(f"case~{TERR}",data=tr).fit(disp=0,maxiter=200); A.loc[te,'score_terrain']=np.asarray(mt.predict(A[te],which='linear'))
        folds.append({'fold':f,'n_test':int(te.sum()),'cases_test':int(A[te].case.sum()),'auc':round(auc(A[te].case,A[te].score_terrain),4)})
    cuts=np.quantile(A.loc[A.case==0,'score_terrain'],[0.2,0.4,0.6,0.8]); A['Qt']=np.searchsorted(cuts,A.score_terrain.values,side='right')+1
    R['rep_expected_risk']={'n_rows':int(len(A)),'n_cases':int(A.case.sum()),'n_blocks':nb,'folds':folds,'auc_oof':round(auc(A.case,A.score_terrain),4),'quintile_cuts':[round(float(c),4) for c in cuts]}
    B=B.merge(A[['uid','event','Qt','score_terrain']],on=['uid','event'],how='left')
    R['rep_lt']={'pairs_fitted':int(S.fit.sum()),'pairs_total':int(len(S)),'broadleaf_rows_lt_missing':int(B.lt_missing.sum()),'broadleaf_rows':int(len(B))}
    R['rep']=period_block(B,'2009-2017 events, new controls, DEM terrain, catalog and LandTrendr legacy','Qt',rng)
    def ratio(a,b):
        d=a['beta']-b['beta']; se=math.sqrt(a['se']**2+b['se']**2); return {'ratio':round(math.exp(d),4),'ci':[round(math.exp(d-1.96*se),4),round(math.exp(d+1.96*se),4)]}
    P={}
    for k in ('MR1_all','MR1_le39.5'): P[k]=ratio(R['rep'][k]['legacy'],R['main'][k]['legacy'])
    for k in ('MR2_all','MR2_le39.5'):
        for c in ('R1','R3'): P[f'{k}_{c}']=ratio(R['rep'][k][c],R['main'][k][c])
    for c in ('0-5','5-10','10+'):
        if c in R['rep']['MR3a_scar_age'] and c in R['main']['MR3a_scar_age']: P[f'scar_{c}']=ratio(R['rep']['MR3a_scar_age'][c],R['main']['MR3a_scar_age'][c])
    for c in ('0-5','5-10','10-20','20+'): P[f'tsd_{c}']=ratio(R['rep']['MR3b_tsd'][c],R['main']['MR3b_tsd'][c])
    P['MR4_per_degree']=ratio(R['rep']['MR4_per_degree'],R['main']['MR4_per_degree'])
    P['B2_trend']=ratio(R['rep']['bright_terrain_only']['B2_trend_per_stratum_step'],R['main']['bright_terrain_only']['B2_trend_per_stratum_step'])
    R['period_ratios_rep_over_main']=P
    B.to_csv(H+'/v10/an/g5/rep_rows_tagged_v1.csv',index=False)
    Z=pd.read_csv(H+'/g5/zoning_out_v1.csv',dtype={'uid':str})
    out={}
    for per,d in (('main',M),('rep',B)):
        d=d.merge(Z,on='uid',how='left'); d['zin']=d.gsmma_inside.fillna(0).astype(int)
        c=d[(d.case==1)&(d.slope<=39.5)]; k=d[(d.case==0)&(d.slope<=39.5)]
        o={'cases_le39.5_inside_share':round(float(c.zin.mean()),4),'control_rows_le39.5_inside_share':round(float(k.zin.mean()),4),
           'cases_le39.5_by_class_inside_share':{cl:round(float(c[c[cl]==1].zin.mean()),4) for cl in ('R1','R3','R0')},
           'legacy_cases_le39.5_outside_share':round(float(1-c[c.legacy==1].zin.mean()),4),
           'n_cases_le39.5':int(len(c)),'n_legacy_cases_le39.5':int(c.legacy.sum()),'n_legacy_cases_le39.5_outside':int(((c.legacy==1)&(c.zin==0)).sum()),
           'llp_rows_inside':int(d.llp_inside.fillna(0).sum())}
        L=d[d.slope<=39.5].copy()
        m,mr,dd,nd=fitfe(L,f"case~{TERR}+zin+legacy+legacy:zin")
        o['legacy_OR_outside']=term(m,mr,'legacy'); o['interaction_inside']=term(m,mr,'legacy:zin'); o['zone_OR_intact']=term(m,mr,'zin')
        b=m.params['legacy']+m.params['legacy:zin']; cv=mr.cov_params(); se=math.sqrt(cv.loc['legacy','legacy']+cv.loc['legacy:zin','legacy:zin']+2*cv.loc['legacy','legacy:zin'])
        o['legacy_OR_inside']={'OR':round(math.exp(b),4),'ci':[round(math.exp(b-1.96*se),4),round(math.exp(b+1.96*se),4)]}
        o['AF_outside_le39.5']=af(L[L.zin==0],['R1','R3'],rng)
        out[per]=o
    R['Z_gsmma']=out
json.dump(R,open(H+'/v10/an/g5/replication_register_v1.json','w'),ensure_ascii=False,indent=1,default=float)
print('written; RUN_REP',RUN_REP)
