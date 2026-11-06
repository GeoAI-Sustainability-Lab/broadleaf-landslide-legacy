"""Assembles the point dates (cutoffs), the inventory matches within 100 m and the annual-state matches within 50 m."""
import pandas as pd, numpy as np, json, glob, os, hashlib, time
t0=time.time()
G='@WORK@/g0/'; OUT=os.path.expanduser('@WORK@/g0/')
P=pd.read_csv(G+'points.csv',dtype={'uid':str,'lid':str,'year':str,'event':str})
P['year']=P.year.replace('',np.nan)
def parse_dates(v):
    import re as _re
    s=str(v) if v is not None else ''
    s=s.replace('\u200b','').replace('\r',' ').replace('\n',' ')
    ds=[pd.Timestamp(x) for x in _re.findall(r'(?<!\d)(\d{8})(?!\d)',s)]
    if not ds: return (pd.NaT,pd.NaT)
    return (min(ds),max(ds))
def to_date(v):
    return parse_dates(v)[0]
cat=[]
for f in sorted(glob.glob(G+'cat/cat_*_matches.csv')):
    y=int(os.path.basename(f).split('_')[1]); m=pd.read_csv(f,dtype={'uid':str}); m=m[m.uid!='ERROR'].copy(); m['cat_year']=y
    m=m.rename(columns={'Event_s_':'Events'}); cat.append(m)
cat=pd.concat(cat,ignore_index=True)
cat['before']=cat.BeforeDate.apply(lambda v: parse_dates(v)[0]); cat['before_max']=cat.BeforeDate.apply(lambda v: parse_dates(v)[1]); cat['after']=cat.AfterDate.apply(lambda v: parse_dates(v)[1]); cat['after_min']=cat.AfterDate.apply(lambda v: parse_dates(v)[0])
cat['record_id']=cat.cat_year.astype(str)+':'+cat.record_index.astype(str)
cat.to_csv(OUT+'catalog_matches_within100m_v1.csv',index=False,encoding='utf-8-sig')
st=[]
for f in sorted(glob.glob(G+'state/state_*_matches.csv')):
    y=int(os.path.basename(f).split('_')[1]); m=pd.read_csv(f,dtype={'uid':str}); m=m[m.uid!='ERROR'].copy(); m['state_year']=y
    dcol=[c for c in m.columns if c.lower()=='date'][0]; acol=[c for c in m.columns if c.lower().startswith('area')][0]
    m=m.rename(columns={dcol:'DATE',acol:'Area'}); st.append(m[['uid','record_index','distance_m','inside','poly_area_m2','DATE','Area','state_year']])
st=pd.concat(st,ignore_index=True); st['img_date']=st.DATE.apply(to_date)
st.to_csv(OUT+'annual_state_matches_within50m_v1.csv',index=False,encoding='utf-8-sig')
P['yr']=np.where(P.kind=='case',P.lid.str.split('_').str[0],''); P['idx']=np.where(P.kind=='case',P.lid.str.split('_').str[1],'')
own=cat.merge(P[P.kind=='case'][['uid','yr','idx']],on='uid')
own=own[(own.cat_year.astype(str)==own.yr)&(own.record_index.astype(str)==own.idx)]
own=own[['uid','record_index','distance_m','inside','before','before_max','after','Events','Area_ha']].rename(columns={'record_index':'own_record_index','distance_m':'own_dist_m','inside':'own_inside','before':'own_before','before_max':'own_before_max','after':'own_after','Events':'own_event_label','Area_ha':'own_area_ha'})
D=P.merge(own,on='uid',how='left')
bb=pd.read_csv(G+'event_polygon_bboxes_2018_2025.csv',dtype={'Events':str})
miss=D[(D.kind=='case')&D.own_record_index.isna()]
for i,r in miss.iterrows():
    q=bb[(bb.year==int(r.yr))&(bb.record_index==int(r.idx))].iloc[0]
    D.loc[i,'own_before']=parse_dates(q.BeforeDate)[0]; D.loc[i,'own_before_max']=parse_dates(q.BeforeDate)[1]; D.loc[i,'own_after']=parse_dates(q.AfterDate)[1]; D.loc[i,'own_event_label']=q.Events; D.loc[i,'own_area_ha']=q.Area_ha; D.loc[i,'own_record_index']=int(r.idx); D.loc[i,'own_inside']=0
    assert len(miss)==1, ('unexpected number of cases without own match',len(miss))
D['own_event_matches']=np.where(D.kind=='case',(D.own_event_label==D.event).astype(int),np.nan)
C=D[D.kind=='case']; K=D[D.kind=='control']
cx=C.x.values; cy=C.y.values; cb=C.own_before.values; cu=C.uid.values
nn_uid=[]; nn_d=[]; nn_b=[]
kx=K.x.values; ky=K.y.values
for s in range(0,len(K),500):
    dx=kx[s:s+500,None]-cx[None,:]; dy=ky[s:s+500,None]-cy[None,:]; d=np.sqrt(dx*dx+dy*dy); j=d.argmin(1)
    nn_uid+=list(cu[j]); nn_d+=list(d[np.arange(len(j)),j]); nn_b+=list(cb[j])
D.loc[K.index,'isl_nearest_case']=nn_uid; D.loc[K.index,'isl_nearest_dist_m']=np.round(nn_d,1); D.loc[K.index,'isl_cutoff']=nn_b
D['cutoff_isl']=np.where(D.kind=='case',D.own_before,D.isl_cutoff); D['cutoff_isl']=pd.to_datetime(D.cutoff_isl)
ev=D[D.kind=='case'].groupby(['year','event']).size().reset_index()[['year','event']]
bb['year']=bb.year.astype(int); bb['before']=bb.BeforeDate.apply(lambda v: parse_dates(v)[0])
FP={}; EVB={}
for _,e in ev.iterrows():
    g=bb[(bb.year==int(e.year))&(bb.Events==e.event)]
    cells=set()
    for x0,y0,x1,y1 in g[['xmin','ymin','xmax','ymax']].values:
        for cx_ in range(int(np.floor(x0/5000)),int(np.floor(x1/5000))+1):
            for cy_ in range(int(np.floor(y0/5000)),int(np.floor(y1/5000))+1): cells.add((cx_,cy_))
    FP[(e.year,e.event)]=cells; EVB[(e.year,e.event)]=(g.before.min(),g.before.max(),len(g))
K=D[D.kind=='control'].copy(); K['cell']=list(zip(np.floor(K.x/5000).astype(int),np.floor(K.y/5000).astype(int)))
rows=[]
for (y,e),cells in FP.items():
    m=K[K.cell.isin(cells)]
    for u in m.uid: rows.append({'uid':u,'year':y,'event':e,'cutoff_event_min':EVB[(y,e)][0],'cutoff_event_max':EVB[(y,e)][1]})
EM=pd.DataFrame(rows); EM.to_csv(OUT+'control_event_footprint_membership_v1.csv',index=False,encoding='utf-8-sig')
json.dump({f'{y}|{e}':{'n_cells':len(c),'n_polygons':EVB[(y,e)][2],'before_min':str(EVB[(y,e)][0].date()),'before_max':str(EVB[(y,e)][1].date()),'n_controls_in_footprint':int((EM.year.eq(y)&EM.event.eq(e)).sum())} for (y,e),c in FP.items()},open(OUT+'event_footprints_v1.json','w'),ensure_ascii=False,indent=1)
D.drop(columns=['yr','idx']).to_csv(OUT+'point_dates_v1.csv',index=False,encoding='utf-8-sig')
R={'n_points':int(len(D)),'n_cases':int((D.kind=='case').sum()),'n_controls':int((D.kind=='control').sum()),
   'cases_own_inside':int(D.own_inside.eq(1).sum()),'cases_own_outside_within100m':int(((D.kind=='case')&(D.own_inside==0)&(D.own_dist_m<=100)).sum()),
   'cases_own_beyond100m':int(((D.kind=='case')&(D.own_dist_m>100)).sum()),'own_event_label_mismatch':int(((D.kind=='case')&(D.own_event_matches==0)).sum()),
   'cases_missing_own_before':int(((D.kind=='case')&D.own_before.isna()).sum()),'cases_before_is_range':int(((D.kind=='case')&(D.own_before_max>D.own_before)).sum()),
   'catalog_match_rows':int(len(cat)),'state_match_rows':int(len(st)),'n_events':int(len(FP)),'controls_in_any_footprint':int(EM.uid.nunique()),'control_event_rows':int(len(EM)),
   'isl_nearest_dist_median_m':float(np.median(nn_d)),'isl_nearest_dist_p90_m':float(np.percentile(nn_d,90))}
json.dump(R,open(OUT+'assembly_register_v1.json','w'),ensure_ascii=False,indent=1); print(json.dumps(R,ensure_ascii=False,indent=1)); print('elapsed',round(time.time()-t0,1))
