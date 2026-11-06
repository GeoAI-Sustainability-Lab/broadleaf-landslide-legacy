"""Event-matched (event fixed effects) and island logistic models with seven terrain terms and point-clustered standard errors, all site-class contrasts and interactions."""
import pandas as pd, numpy as np, json, math, warnings
import statsmodels.formula.api as smf
from scipy.stats import norm
warnings.filterwarnings('ignore')
ISL=pd.read_csv('tags_island_v3.csv',dtype={'uid':str},low_memory=False); EV=pd.read_csv('@WORK@/v10/data/tags_event_v4.csv',dtype={'uid':str},low_memory=False)
BRO='闊葉樹林型'; BAM='竹林'; TERR="slope+slope2+north+east+planc+profc+elev"; R={}
def prep(df):
    d=df.copy(); d['case']=d.case.astype(int); d['slope2']=d.slope**2; d['s20']=d.slope-20.0
    d['legacy_any4']=(d.site_class4>0).astype(int); d['cluster']=np.where(d.kind=='case',d.event.astype(str),'controls')
    for c in (1,2,3): d[f'sc{c}']=(d.site_class4==c).astype(int)
    d['tsd3_bin']=pd.Categorical(d.tsd3_bin,categories=['none','0-5','5-10','10-20','20+'])
    d['tsd3_bin2']=pd.Categorical(np.select([d.class3==0,d.TSD<10,d.TSD<20],['none','0-10','10-20'],'20+'),categories=['none','0-10','10-20','20+'])
    d['tss1_bin']=pd.Categorical(np.where(d.site_class4==1,d.tss_bin.replace({'20+':'10+','10-20':'10+'}),'none'),categories=['none','0-5','5-10','10+'])
    yrs=d.scar_50m_min_yrs
    age=np.select([yrs<1,yrs<10],['lt1','1-10'],'10+')
    d['cell']=np.select([d.site_class4==0,d.site_class4==3,(d.site_class4==1)&(age=='lt1'),(d.site_class4==1)&(age=='1-10'),(d.site_class4==1),
                         (d.site_class4==2)&yrs.isna(),(d.site_class4==2)&(age=='lt1'),(d.site_class4==2)&(age=='1-10')],
                        ['c0','c3','s_lt1','s_1_10','s_10p','b_noscar','b_lt1','b_1_10'],'b_10p')
    d['cell']=pd.Categorical(d.cell,categories=['c0','s_lt1','s_1_10','s_10p','b_lt1','b_1_10','b_10p','b_noscar','c3'])
    d['rec_bin']=np.select([d.class3==0,d.recovery_ratio<0.5,d.recovery_ratio<1.0],['none','<0.5','0.5-1'],'>=1')
    d['rec_bin']=pd.Categorical(d.rec_bin,categories=['none','<0.5','0.5-1','>=1'])
    return d.dropna(subset=['slope','north','east','planc','profc','elev'])
def drop_uninformative_events(d):
    g=d.groupby('event').case.agg(['sum','size']); keep=g[(g['sum']>0)&(g['sum']<g['size'])].index
    return d[d.event.isin(keep)].copy(), int(len(g)-len(keep))
def fit(d, form, design, label):
    if design=='island':
        m=smf.logit(form,data=d).fit(disp=0,maxiter=200); mr=smf.logit(form,data=d).fit(disp=0,maxiter=200,cov_type='cluster',cov_kwds={'groups':pd.factorize(d.cluster)[0]})
    else:
        d,n_drop=drop_uninformative_events(d)
        f2=form+'+C(event)'; m=smf.logit(f2,data=d).fit(disp=0,maxiter=100,method='newton'); mr=smf.logit(f2,data=d).fit(disp=0,maxiter=100,method='newton',cov_type='cluster',cov_kwds={'groups':pd.factorize(d.uid)[0]})
    out={'label':label,'design':design,'n':int(len(d)),'n_case':int(d.case.sum()),'n_control_points':int(d[d.case==0].uid.nunique()),'n_events':int(d.event.nunique()) if design=='event' else None,'events_dropped_no_outcome_variation':(n_drop if design=='event' else 0),'converged':bool(m.mle_retvals.get('converged',True)),'llf':float(m.llf),'terms':{}}
    for t in m.params.index:
        if t.startswith('C(event)') or t=='Intercept' or t in ('slope','slope2','north','east','planc','profc','elev'): continue
        b=m.params[t]; s=m.bse[t]; sr=mr.bse[t]
        out['terms'][t]={'OR':round(math.exp(b),4),'ci_model':[round(math.exp(b-1.96*s),4),round(math.exp(b+1.96*s),4)],'ci_cluster':[round(math.exp(b-1.96*sr),4),round(math.exp(b+1.96*sr),4)],'p_cluster':float(2*(1-norm.cdf(abs(b/sr))))}
    return out,m,mr
def or_at(m,mr,main,inter,deg):
    res={}
    for tag,mod in [('model',m),('cluster',mr)]:
        b=mod.params[main]+(deg-20)*mod.params[inter]; c=mod.cov_params(); se=math.sqrt(c.loc[main,main]+(deg-20)**2*c.loc[inter,inter]+2*(deg-20)*c.loc[main,inter])
        res[tag]={'OR':round(math.exp(b),3),'ci':[round(math.exp(b-1.96*se),3),round(math.exp(b+1.96*se),3)]}
    return res
def wald_diff(mod,ta,tb):
    b=mod.params[ta]-mod.params[tb]; c=mod.cov_params(); se=math.sqrt(c.loc[ta,ta]+c.loc[tb,tb]-2*c.loc[ta,tb]); return {'OR_ratio':round(math.exp(b),3),'ci':[round(math.exp(b-1.96*se),3),round(math.exp(b+1.96*se),3)],'p':float(2*(1-norm.cdf(abs(b/se))))}
bi=prep(ISL[ISL.ft==BRO]); be=prep(EV[EV.ft==BRO])
R['composition']={'island_cases':bi[bi.case==1].site_class4.value_counts().sort_index().to_dict(),'island_controls':bi[bi.case==0].site_class4.value_counts().sort_index().to_dict(),
                  'island_cases_le39.5':bi[(bi.case==1)&(bi.slope<=39.5)].site_class4.value_counts().sort_index().to_dict(),'island_controls_le39.5':bi[(bi.case==0)&(bi.slope<=39.5)].site_class4.value_counts().sort_index().to_dict()}
M={}
for design,d in [('island',bi),('event',be)]:
    o,m,mr=fit(d,f"case~{TERR}+sc1+sc2+sc3",design,'M2c site classes 1-3 vs 0'); M[f'M2c_{design}']=o
    o['H2_bare_vs_scar']={'cluster':wald_diff(mr,'sc2','sc1')}
    o,m,mr=fit(d,f"case~{TERR}+C(tss1_bin)+sc2+sc3",design,'M1b class 1 by years since prior scar (0-5 / 5-10 / 10+), classes 2-3 in model'); M[f'M1b_{design}']=o
    o['H1_decay_0-5_vs_10+']={'cluster':wald_diff(mr,'C(tss1_bin)[T.0-5]','C(tss1_bin)[T.10+]')}; o['H1_0-5_vs_5-10']={'cluster':wald_diff(mr,'C(tss1_bin)[T.0-5]','C(tss1_bin)[T.5-10]')}
    o,m,mr=fit(d,f"case~{TERR}+C(cell)",design,'M2d exploratory: scar age (<1 / 1-10 / 10+ yr) x bare status; bare without catalog scar; class 3'); M[f'M2d_{design}']=o
    for a in ('lt1','1_10','10p'): o[f'H2_bare_vs_scar_age_{a}']={'cluster':wald_diff(mr,f'C(cell)[T.b_{a}]',f'C(cell)[T.s_{a}]')}
    o['cell_counts']={k:{'case':int(v.get(1,0)),'control_rows':int(v.get(0,0))} for k,v in d.groupby('cell',observed=True).case.value_counts().unstack().fillna(0).iterrows()}
    o,m,mr=fit(d,f"case~{TERR}+sc1+sc2+C(tsd3_bin)",design,'M3 class 3 by years since LandTrendr disturbance'); M[f'M3_{design}']=o
    o['H3_decay_0-5_vs_10-20']={'cluster':wald_diff(mr,'C(tsd3_bin)[T.0-5]','C(tsd3_bin)[T.10-20]')}
    o['H3_5-10_vs_10-20']={'cluster':wald_diff(mr,'C(tsd3_bin)[T.5-10]','C(tsd3_bin)[T.10-20]')}
    o['H3_5-10_vs_20+']={'cluster':wald_diff(mr,'C(tsd3_bin)[T.5-10]','C(tsd3_bin)[T.20+]')}
    o['H3_10-20_vs_20+']={'cluster':wald_diff(mr,'C(tsd3_bin)[T.10-20]','C(tsd3_bin)[T.20+]')}
    o2,m2,mr2=fit(d,f"case~{TERR}+sc1+sc2+C(tsd3_bin2)",design,'M3b class 3 by years since disturbance, pre-registered bins 0-10 / 10-20 / 20+'); M[f'M3b_{design}']=o2
    o2['H3_0-10_vs_10-20']={'cluster':wald_diff(mr2,'C(tsd3_bin2)[T.0-10]','C(tsd3_bin2)[T.10-20]')}
    o2['H3_0-10_vs_20+']={'cluster':wald_diff(mr2,'C(tsd3_bin2)[T.0-10]','C(tsd3_bin2)[T.20+]')}
    M[f'M3r_{design}']=fit(d,f"case~{TERR}+sc1+sc2+C(rec_bin)",design,'M3r class 3 by recovery ratio at cutoff')[0]
    o,m,mr=fit(d,f"case~{TERR}+legacy_any4+legacy_any4:s20",design,'M4 legacy_any (classes 1-3) x slope'); M[f'M4_{design}']=o
    o['OR_legacy_at']={str(deg):or_at(m,mr,'legacy_any4','legacy_any4:s20',deg) for deg in (10,20,30,40,50)}
    o,m,mr=fit(d,f"case~{TERR}+sc1+sc2+sc3+sc1:s20+sc2:s20+sc3:s20",design,'M4b classes x slope'); M[f'M4b_{design}']=o
    o['OR_at']={c:{str(deg):or_at(m,mr,f'sc{c}',f'sc{c}:s20',deg) for deg in (10,20,30,40)} for c in (1,2,3)}
    o,m,mr=fit(d[d.slope<=39.5],f"case~{TERR}+sc1+sc2+sc3",design,'M2c slope<=39.5'); M[f'M2c_{design}_le39.5']=o
    o['H2_bare_vs_scar']={'cluster':wald_diff(mr,'sc2','sc1')}
R['models']=M
def bb(d,label,design):
    d=d.copy(); d['bamboo']=(d.ft==BAM).astype(int); f=f"case~{TERR}+bamboo+bamboo:slope"+('+C(event)' if design=='event' else '')
    kw=dict(disp=0,maxiter=100,method='newton') if design=='event' else dict(disp=0)
    n_drop=0
    if design=='event': d,n_drop=drop_uninformative_events(d)
    m=smf.logit(f,data=d).fit(**kw); g=pd.factorize(d.uid)[0] if design=='event' else pd.factorize(d.cluster)[0]; mr=smf.logit(f,data=d).fit(cov_type='cluster',cov_kwds={'groups':g},**kw)
    out={'label':label,'design':design,'n':int(len(d)),'n_case':int(d.case.sum()),'n_bamboo_case':int(((d.case==1)&(d.bamboo==1)).sum()),'n_broadleaf_case':int(((d.case==1)&(d.bamboo==0)).sum()),'n_events':int(d.event.nunique()) if design=='event' else None,'events_dropped_no_outcome_variation':n_drop,'converged':bool(m.mle_retvals.get('converged',True)),'llf':float(m.llf)}
    b_=m.params['bamboo:slope']; s=m.bse['bamboo:slope']; sr=mr.bse['bamboo:slope']
    out['interaction']={'OR':round(math.exp(b_),4),'ci_model':[round(math.exp(b_-1.96*s),4),round(math.exp(b_+1.96*s),4)],'ci_cluster':[round(math.exp(b_-1.96*sr),4),round(math.exp(b_+1.96*sr),4)],'p_cluster':float(2*(1-norm.cdf(abs(b_/sr))))}
    for deg in (20,30,40):
        bbv=m.params['bamboo']+deg*m.params['bamboo:slope']; c=m.cov_params(); se=math.sqrt(c.loc['bamboo','bamboo']+deg*deg*c.loc['bamboo:slope','bamboo:slope']+2*deg*c.loc['bamboo','bamboo:slope'])
        cr=mr.cov_params(); ser=math.sqrt(cr.loc['bamboo','bamboo']+deg*deg*cr.loc['bamboo:slope','bamboo:slope']+2*deg*cr.loc['bamboo','bamboo:slope'])
        out[f'OR_at_{deg}']={'OR':round(math.exp(bbv),4),'ci_model':[round(math.exp(bbv-1.96*se),4),round(math.exp(bbv+1.96*se),4)],'ci_cluster':[round(math.exp(bbv-1.96*ser),4),round(math.exp(bbv+1.96*ser),4)]}
    return out
R['H4']={}
for design,src in [('island',ISL),('event',EV)]:
    sub=prep(src[src.ft.isin([BAM,BRO])])
    R['H4'][design]={'full':bb(sub,'all broadleaf',design),
        'broadleaf_class0_v2':bb(sub[(sub.ft==BAM)|((sub.ft==BRO)&(sub.site_class==0))],'broadleaf class 0 (catalog+bare only, as prelim)',design),
        'broadleaf_class0_v3':bb(sub[(sub.ft==BAM)|((sub.ft==BRO)&(sub.site_class4==0))],'broadleaf class 0 incl. LandTrendr',design),
        'both_class0_v3':bb(sub[sub.site_class4==0],'both types class 0 incl. LandTrendr',design)}
json.dump(R,open('final_register_v2.json','w'),ensure_ascii=False,indent=1,default=str)
def show(k):
    o=M[k]; print(f"== {k}: {o['label']} | n={o['n']} cases={o['n_case']}")
    for t,v in o['terms'].items(): print(f"   {t:24s} OR {v['OR']:.3f} cluster CI {v['ci_cluster']} p {v['p_cluster']:.3g}")
    for key in ('H3_decay_0-5_vs_10-20','OR_legacy_at','OR_at'):
        if key in o: print('  ',key,json.dumps(o[key])[:400])
for k in ['M2c_event','M3_event','M3r_event','M4_event','M4b_event','M2c_island','M3_island','M4_island','M2c_event_le39.5']: show(k)
print('composition',R['composition'])
for design in ('event','island'):
    for k,h in R['H4'][design].items(): print(design,k,'nb',h['n_bamboo_case'],'nbr',h['n_broadleaf_case'],'int',h['interaction']['OR'],h['interaction']['ci_cluster'],'OR20',h['OR_at_20']['OR'],h['OR_at_20']['ci_cluster'],'OR40',h['OR_at_40']['OR'])
