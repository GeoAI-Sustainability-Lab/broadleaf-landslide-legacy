"""Independent Sentinel-2 check of the LandTrendr disturbance class."""
import glob, json, math, sys, numpy as np, pandas as pd, warnings
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
G='@WORK@/g0/'
S=pd.concat([pd.read_csv(f,dtype={'uid':str}) for f in sorted(glob.glob('s2_batches/s2_*.csv'))],ignore_index=True).drop_duplicates('uid')
YEARS=list(range(2019,2025)); R={'n_points_with_s2':int(len(S))}
nbr=S[[f's2nbr_{y}' for y in YEARS]].values.astype(float); cnt=S[[f's2n_{y}' for y in YEARS]].values
valid=(cnt>=3)&(nbr>-32768); nbr=np.where(valid,nbr,np.nan)
R['share_valid_years']={str(y):round(float(valid[:,i].mean()),3) for i,y in enumerate(YEARS)}
R['median_obs_by_year']={str(y):float(np.median(cnt[:,i])) for i,y in enumerate(YEARS)}
drop=np.full((len(S),len(YEARS)),np.nan)
for i in range(1,len(YEARS)): drop[:,i]=nbr[:,i]-nbr[:,i-1]
pairs=pd.read_csv('lt_pairs_v1.csv',dtype={'uid':str}); LT=pd.read_csv('lt_summary_v1.csv',dtype={'uid':str})
segs=pd.read_csv('lt_segments_all_v1.csv',dtype={'uid':str}); segs=segs[(segs.delta>=100)&(segs.duration<=3)]
P=pairs[pairs.cutoff_year>=2021].merge(S[['uid']],on='uid')
idx={u:i for i,u in enumerate(S.uid)}
rows=[]
for r in P.itertuples():
    i=idx[r.uid]; Y=int(r.cutoff_year); yrs=[y for y in range(2020,Y) if y<=2024]
    d=[(y,drop[i,YEARS.index(y)]) for y in yrs if not np.isnan(drop[i,YEARS.index(y)])]
    n_eval=len(d); hits=[y for y,v in d if v<=-100]
    s2_year=max(hits) if hits else np.nan
    rows.append({'uid':r.uid,'cutoff_year':Y,'n_years_evaluable':n_eval,'s2_detected':int(bool(hits)),'s2_dist_year':s2_year})
D=pd.DataFrame(rows)
lt_in=segs[(segs.end_year>=2020)].groupby(['uid','cutoff_year']).end_year.max().rename('lt_dist_year_win').reset_index()
D=D.merge(lt_in,on=['uid','cutoff_year'],how='left'); D['lt_detected_win']=((D.lt_dist_year_win.notna())&(D.lt_dist_year_win<=D.cutoff_year-1)).astype(int)
D=D[D.n_years_evaluable>=1]
D.to_csv('s2_pairs_v1.csv',index=False)
T=pd.read_csv('tags_island_v3.csv',dtype={'uid':str},low_memory=False)[['uid','kind','ft','slope','site_class']]
D=D.merge(T,on='uid',how='left')
def agree(d):
    ct=pd.crosstab(d.lt_detected_win,d.s2_detected); a=int(ct.loc[1,1]) if (1 in ct.index and 1 in ct.columns) else 0
    n=len(d); po=(d.lt_detected_win==d.s2_detected).mean(); p1=d.lt_detected_win.mean(); p2=d.s2_detected.mean(); pe=p1*p2+(1-p1)*(1-p2); kappa=(po-pe)/(1-pe) if pe<1 else float('nan')
    return {'n_pairs':n,'lt_rate':round(float(p1),4),'s2_rate':round(float(p2),4),'both':a,'lt_only':int((d.lt_detected_win==1).sum()-a),'s2_only':int((d.s2_detected==1).sum()-a),
            'P(s2|lt)':round(float(d[d.lt_detected_win==1].s2_detected.mean()),3) if p1>0 else None,'P(lt|s2)':round(float(d[d.s2_detected==1].lt_detected_win.mean()),3) if p2>0 else None,'kappa':round(float(kappa),3),
            'same_year_share_among_both':round(float((d[(d.lt_detected_win==1)&(d.s2_detected==1)].lt_dist_year_win==d[(d.lt_detected_win==1)&(d.s2_detected==1)].s2_dist_year).mean()),3) if a>0 else None}
R['agreement']={'all':agree(D),'broadleaf':agree(D[D.ft=='闊葉樹林型']),'broadleaf_class0':agree(D[(D.ft=='闊葉樹林型')&(D.site_class==0)]),'broadleaf_le39.5':agree(D[(D.ft=='闊葉樹林型')&(D.slope<=39.5)])}
cat=pd.read_csv(G+'catalog_matches_within100m_v1.csv',dtype={'uid':str,'record_id':str},parse_dates=['after'])
own=T[T.kind=='case'].copy(); own['lid']=own.uid.str.split(':').str[1]; own['rid']=own.lid.str.split('_').str[0]+':'+own.lid.str.split('_').str[1]
cat['is_own']=cat.record_id.isin(set(own.rid))&(cat.uid.str[5:]==cat.cat_year.astype(str)+'_'+cat.record_index.astype(str))
pos=cat[(cat.inside==1)&(~cat.is_own)&cat.after.notna()].copy(); pos['scar_year']=pos.after.dt.year
pos=pos.merge(D[['uid','cutoff_year','s2_detected','s2_dist_year','n_years_evaluable']],on='uid'); pos=pos[(pos.scar_year>=2020)&(pos.scar_year<=pos.cutoff_year-1)]
pos=pos.sort_values('scar_year').groupby(['uid','cutoff_year']).tail(1)
def hit_s2(p): return ((p.s2_dist_year>=p.scar_year-1)&(p.s2_dist_year<=p.scar_year+1)).astype(int)
m=pos.merge(segs[['uid','cutoff_year','end_year']],on=['uid','cutoff_year'],how='left'); m['hit']=(m.end_year>=m.scar_year-1)&(m.end_year<=m.scar_year+1)
lt_hit=m.groupby(['uid','cutoff_year']).hit.max().fillna(False).astype(int)
pos['lt_hit']=lt_hit.reindex(pd.MultiIndex.from_frame(pos[['uid','cutoff_year']])).values
def s2_hit_row(r):
    i=idx[r.uid]; ok=False
    for y in range(int(r.scar_year)-1,int(r.scar_year)+2):
        if 2020<=y<=min(2024,int(r.cutoff_year)-1):
            v=drop[i,YEARS.index(y)]
            if not np.isnan(v) and v<=-100: ok=True
    return int(ok)
pos['s2_hit']=[s2_hit_row(r) for r in pos.itertuples()]
R['recall_catalog_scars_2020plus']={'n_pos_pairs':int(len(pos)),'recall_lt':round(float(pos.lt_hit.mean()),3),'recall_s2':round(float(pos.s2_hit.mean()),3),'recall_either':round(float(((pos.lt_hit==1)|(pos.s2_hit==1)).mean()),3),'recall_both':round(float(((pos.lt_hit==1)&(pos.s2_hit==1)).mean()),3),
    'by_scar_year':pos.groupby('scar_year').agg(n=('lt_hit','size'),recall_lt=('lt_hit','mean'),recall_s2=('s2_hit','mean')).round(3).reset_index().to_dict(orient='records')}
EV=pd.read_csv('@WORK@/v10/data/tags_event_v4.csv',dtype={'uid':str},low_memory=False); EV=EV[(EV.ft=='闊葉樹林型')&(EV.cutoff_year>=2021)].copy()
EV=EV.merge(D[['uid','cutoff_year','s2_detected','lt_detected_win']],on=['uid','cutoff_year'],how='left')
EV=EV[EV.s2_detected.notna()]
EV['case']=EV.case.astype(int); EV['slope2']=EV.slope**2
for c in (1,2): EV[f'sc{c}']=(EV.site_class==c).astype(int)
EV['c3_lt']=((EV.site_class==0)&(EV.lt_detected_win==1)).astype(int); EV['c3_s2']=((EV.site_class==0)&(EV.s2_detected==1)).astype(int)
EV['c3_both']=((EV.c3_lt==1)&(EV.c3_s2==1)).astype(int); EV['c3_either']=((EV.c3_lt==1)|(EV.c3_s2==1)).astype(int)
EV=EV.dropna(subset=['slope','north','east','planc','profc','elev'])
TERR="slope+slope2+north+east+planc+profc+elev"
def fit_event(d,form):
    g=d.groupby('event').case.agg(['sum','size']); keep=g[(g['sum']>0)&(g['sum']<g['size'])].index; d=d[d.event.isin(keep)]
    m=smf.logit(form+'+C(event)',data=d).fit(disp=0,maxiter=100,method='newton'); mr=smf.logit(form+'+C(event)',data=d).fit(disp=0,maxiter=100,method='newton',cov_type='cluster',cov_kwds={'groups':pd.factorize(d.uid)[0]})
    return m,mr,d
R['class3_alternatives_event_2021plus']={}
for lab in ['c3_lt','c3_s2','c3_both','c3_either']:
    m,mr,d=fit_event(EV,f"case~{TERR}+sc1+sc2+{lab}")
    R['class3_alternatives_event_2021plus'][lab]={'n':int(len(d)),'n_cases':int(d.case.sum()),'n_exposed_cases':int(d[(d.case==1)][lab].sum()),'n_exposed_control_rows':int(d[(d.case==0)][lab].sum()),
        'OR':round(math.exp(m.params[lab]),3),'ci_cluster':[round(math.exp(m.params[lab]-1.96*mr.bse[lab]),3),round(math.exp(m.params[lab]+1.96*mr.bse[lab]),3)]}
m,mr,d=fit_event(EV,f"case~{TERR}+sc1+sc2+c3_lt+c3_s2")
R['class3_alternatives_event_2021plus']['joint']={t:{'OR':round(math.exp(m.params[t]),3),'ci_cluster':[round(math.exp(m.params[t]-1.96*mr.bse[t]),3),round(math.exp(m.params[t]+1.96*mr.bse[t]),3)]} for t in ('c3_lt','c3_s2')}
R['s2_threshold_sensitivity']={}
for thr in (100,150,200):
    def s2_det_row(uid,Y):
        i=idx[uid]; yrs=[y for y in range(2020,int(Y)) if y<=2024]; ok=False
        for y in yrs:
            v=drop[i,YEARS.index(y)]
            if not np.isnan(v) and v<=-thr: ok=True
        return int(ok)
    def s2_hit_thr(r):
        i=idx[r.uid]; ok=False
        for y in range(int(r.scar_year)-1,int(r.scar_year)+2):
            if 2020<=y<=min(2024,int(r.cutoff_year)-1):
                v=drop[i,YEARS.index(y)]
                if not np.isnan(v) and v<=-thr: ok=True
        return int(ok)
    rec=float(np.mean([s2_hit_thr(r) for r in pos.itertuples()]))
    E2=EV.copy(); E2['s2d']=[s2_det_row(u,Y) for u,Y in zip(E2.uid,E2.cutoff_year)]; E2['c3_s2t']=((E2.site_class==0)&(E2.s2d==1)).astype(int)
    m,mr,d=fit_event(E2,f"case~{TERR}+sc1+sc2+c3_s2t")
    R['s2_threshold_sensitivity'][str(thr)]={'recall_catalog_2020plus':round(rec,3),'s2_rate_pairs':round(float(E2.drop_duplicates(['uid','cutoff_year']).s2d.mean()),4),'n_exposed_cases':int(d[d.case==1].c3_s2t.sum()),'n_exposed_control_rows':int(d[d.case==0].c3_s2t.sum()),
        'OR':round(math.exp(m.params['c3_s2t']),3),'ci_cluster':[round(math.exp(m.params['c3_s2t']-1.96*mr.bse['c3_s2t']),3),round(math.exp(m.params['c3_s2t']+1.96*mr.bse['c3_s2t']),3)]}
    print('thr',thr,R['s2_threshold_sensitivity'][str(thr)])
R['window_note']='S2 window 2019..cutoff_year-1; evaluable pairs need >=1 consecutive valid year pair; LandTrendr compared on the same window (loss segment ending 2020..Y-1); events with cutoff_year>=2021 only'
json.dump(R,open('s2_register_v1.json','w'),ensure_ascii=False,indent=1,default=str)
print(json.dumps(R,ensure_ascii=False,indent=1,default=str)[:6000])
