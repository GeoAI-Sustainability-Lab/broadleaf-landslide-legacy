"""Behaviour of the legacy effect in the 2009 extreme events."""
import json, math, warnings, numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
warnings.filterwarnings('ignore')
H='@WORK@'
BRO='闊葉樹林型'; TERR="slope+slope2+north+east+planc+profc+elev"
SIX=['闊葉樹林型','針葉樹林型','針闊葉樹混淆','竹林','竹闊混淆林','待成林地']
R={'status':'exploratory, written after the gate-5 replication results','definitions':'gate 5 (R1, R3, legacy)'}
def drop_uninf(d):
    g=d.groupby('event').case.agg(['sum','size']); keep=g[(g['sum']>0)&(g['sum']<g['size'])].index
    return d[d.event.isin(keep)].copy(), int(len(g)-len(keep))
def fitfe(d, form):
    d,nd=drop_uninf(d); f2=form+'+C(event)'
    m=smf.logit(f2,data=d).fit(disp=0,maxiter=100,method='newton')
    mr=smf.logit(f2,data=d).fit(disp=0,maxiter=100,method='newton',cov_type='cluster',cov_kwds={'groups':pd.factorize(d.uid)[0]})
    assert m.mle_retvals.get('converged',True), form
    return m,mr,d,nd
def lin(m,mr,coefs):
    b=sum(w*m.params[t] for t,w in coefs.items()); c=mr.cov_params()
    se=math.sqrt(sum(wi*wj*c.loc[ti,tj] for ti,wi in coefs.items() for tj,wj in coefs.items()))
    return {'OR':round(math.exp(b),4),'ci':[round(math.exp(b-1.96*se),4),round(math.exp(b+1.96*se),4)],'p':float(2*(1-norm.cdf(abs(b/se)))),'beta':float(b),'se':float(se)}
E=pd.read_csv('@WORK@/v10/data/tags_event_v4.csv',dtype={'uid':str},low_memory=False)
BR=pd.read_csv(H+'/v10/an/g3/bright_rows_v1.csv',dtype={'uid':str})
size_main=E[E.ft.isin(SIX)].groupby('event').case.sum().rename('n_cases_six')
E=E.merge(BR[['uid','event','max24','total']],on=['uid','event'],how='left')
M=E[E.ft==BRO].copy(); M['case']=M.case.astype(int); M['slope2']=M.slope**2; M['s20']=M.slope-20
M=M.dropna(subset=['slope','north','east','planc','profc','elev'])
M['R1']=(M.class1_50m==1).astype(int); M['R3']=((M.lt_detected==1)&(M.R1==0)).astype(int); M['legacy']=((M.R1==1)|(M.R3==1)).astype(int)
P=pd.read_csv(H+'/v10/an/g5/rep_rows_tagged_v1.csv',dtype={'uid':str},low_memory=False)
D=pd.read_csv(H+'/g5/rep_rows_v1.csv',dtype={'uid':str},low_memory=False)
size_rep=D[D.TypeName.isin(SIX)].groupby('event').case.sum().rename('n_cases_six')
E5a={}
for var in ('total','max24'):
    d=M.dropna(subset=[var]).copy()
    cuts=np.quantile(d.loc[d.case==0,var],[1/3,2/3]); d['rt']=np.searchsorted(cuts,d[var].values,side='right')+1
    for k in (1,2,3):
        d[f'rt{k}']=(d.rt==k).astype(int); d[f'L{k}']=d.legacy*d[f'rt{k}']; d[f'LS{k}']=d.legacy*d.s20*d[f'rt{k}']
    form=f"case~{TERR}+rt2+rt3+rt2:s20+rt3:s20+L1+L2+L3+LS1+LS2+LS3"
    m,mr,dd,nd=fitfe(d,form)
    o={'cuts_mm':[round(float(c),1) for c in cuts],'n':int(len(dd)),'n_case':int(dd.case.sum()),'events_dropped':nd,
       'cases_by_tercile':{str(k):int(dd[(dd.rt==k)].case.sum()) for k in (1,2,3)},
       'legacy_cases_by_tercile':{str(k):int(dd[(dd.rt==k)&(dd.legacy==1)].case.sum()) for k in (1,2,3)},
       'legacy_OR_at_20':{str(k):lin(m,mr,{f'L{k}':1}) for k in (1,2,3)},
       'legacy_per_degree':{str(k):lin(m,mr,{f'LS{k}':1}) for k in (1,2,3)},
       'per_degree_T3_over_T1':lin(m,mr,{'LS3':1,'LS1':-1}),'OR20_T3_over_T1':lin(m,mr,{'L3':1,'L1':-1})}
    E5a[var]=o
R['E5a_main_rain_terciles']=E5a
def e5b(d, size, label):
    d=d.merge(size,left_on='event',right_index=True,how='left'); d=d[d.n_cases_six>0].copy(); d['z']=np.log10(d.n_cases_six)
    zc=float(d.drop_duplicates('event').z.mean()); d['z']=d.z-zc
    m,mr,dd,nd=fitfe(d,f"case~{TERR}+legacy+legacy:s20+legacy:z+legacy:s20:z+s20:z")
    o={'label':label,'n':int(len(dd)),'n_case':int(dd.case.sum()),'events_dropped':nd,'z_centre_log10':round(zc,4),
       'legacy_per_degree_at_mean_size':lin(m,mr,{'legacy:s20':1}),'legacy_s20_x_log10size':lin(m,mr,{'legacy:s20:z':1}),
       'legacy_x_log10size':lin(m,mr,{'legacy:z':1})}
    ev=[]
    for e,g in dd.groupby('event'):
        if g.case.sum()<100: continue
        try:
            mm=smf.logit(f"case~{TERR}+legacy+legacy:s20",data=g).fit(disp=0,maxiter=200,method='newton')
            mmr=smf.logit(f"case~{TERR}+legacy+legacy:s20",data=g).fit(disp=0,maxiter=200,method='newton',cov_type='cluster',cov_kwds={'groups':pd.factorize(g.uid)[0]})
            r=lin(mm,mmr,{'legacy:s20':1}); r.update({'event':str(e),'broadleaf_cases':int(g.case.sum()),'cases_six':int(size.get(e,np.nan))})
            ev.append(r)
        except Exception as ex:
            ev.append({'event':str(e),'error':str(ex)[:60]})
    o['per_event_ge100']=ev
    return o
R['E5b_main']=e5b(M,size_main,'2018-2025 events')
R['E5b_rep']=e5b(P,size_rep,'2009-2017 events')
Mp=M.merge(size_main,left_on='event',right_index=True); Mp['period']='main'
Pp=P.merge(size_rep,left_on='event',right_index=True); Pp['period']='rep'
cols=['uid','event','case','slope','slope2','north','east','planc','profc','elev','s20','legacy','n_cases_six','period']
Q=pd.concat([Mp[cols],Pp[cols]],ignore_index=True); Q=Q[Q.n_cases_six>0].copy(); Q['uid']=Q.period+':'+Q.uid.astype(str); Q['event']=Q.period+':'+Q.event.astype(str)
Q['z']=np.log10(Q.n_cases_six); Q['z']=Q.z-float(Q.drop_duplicates('event').z.mean()); Q['rep']=(Q.period=='rep').astype(int)
m,mr,dd,nd=fitfe(Q,"case~slope+slope2+north+east+planc+profc+elev+rep:slope+rep:slope2+rep:north+rep:east+rep:planc+rep:profc+rep:elev+legacy+legacy:rep+legacy:s20+legacy:s20:rep+legacy:z+legacy:s20:z+s20:z")
R['E5b_pooled']={'n':int(len(dd)),'n_case':int(dd.case.sum()),'n_events':int(dd.event.nunique()),'legacy_s20_x_log10size':lin(m,mr,{'legacy:s20:z':1}),
                 'legacy_per_degree_main_at_mean_size':lin(m,mr,{'legacy:s20':1}),'legacy_per_degree_rep_at_mean_size':lin(m,mr,{'legacy:s20':1,'legacy:s20:rep':1})}
R['event_size']={'main':{'max_event':str(size_main.idxmax()),'max_cases_six':int(size_main.max()),'n_events':int((size_main>0).sum()),'median_cases_six':float(size_main[size_main>0].median())},
                 'rep':{'max_event':str(size_rep.idxmax()),'max_cases_six':int(size_rep.max()),'n_events':int((size_rep>0).sum()),'median_cases_six':float(size_rep[size_rep>0].median()),
                        'second_event':str(size_rep.sort_values().index[-2]),'second_cases_six':int(size_rep.sort_values().iloc[-2])}}
json.dump(R,open(H+'/v10/an/g6/extreme_register_v1.json','w',encoding='utf-8'),ensure_ascii=False,indent=1,default=float)
print('done')
