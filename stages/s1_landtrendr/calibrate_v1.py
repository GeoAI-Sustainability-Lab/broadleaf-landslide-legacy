"""Calibrates the LandTrendr magnitude threshold on catalogue-scar positives and scar-free negative proxies (Youden index), reports recall by period and slope band."""
import json, glob, sys, numpy as np, pandas as pd
sys.path.insert(0,'@WORK@/g1'); from lt_segments import *
G='@WORK@/g0/'
segs,fits=load_segments('lt_batches/lt_*.jsonl')
segs.to_csv('lt_segments_all_v1.csv',index=False); fits.to_csv('lt_fits_all_v1.csv',index=False)
cat=pd.read_csv(G+'catalog_matches_within100m_v1.csv',dtype={'uid':str,'record_id':str},parse_dates=['after'])
T=pd.read_csv(G+'tags_island_v2.csv',dtype={'uid':str},low_memory=False)[['uid','kind','ft','slope','north','elev']]; T['lid']=T.uid.str.split(':').str[1]
pairs=pd.read_csv('lt_pairs_v1.csv',dtype={'uid':str})
own=T[T.kind=='case'].assign(rid=lambda d: d.lid.str.split('_').str[0]+':'+d.lid.str.split('_').str[1])
cat['is_own']=cat.record_id.isin(set(own.rid))&(cat.uid.str[5:]==cat.cat_year.astype(str)+'_'+cat.record_index.astype(str))
pos=cat[(cat.inside==1)&(~cat.is_own)&cat.after.notna()].copy(); pos['scar_year']=pos.after.dt.year
pos=pos.merge(pairs[['uid','cutoff_year']],on='uid'); pos=pos[(pos.scar_year<=pos.cutoff_year-1)&(pos.scar_year>=1989)]
pos=pos.sort_values('scar_year').groupby(['uid','cutoff_year']).tail(1)
print('positive pairs (point inside prior catalog scar):',len(pos))
S=segs[segs.delta>0].copy()
def detected(pos_df, thr, max_dur=3, win=1):
    m=pos_df.merge(S[(S.delta>=thr)&(S.duration<=max_dur)][['uid','cutoff_year','end_year','delta']],on=['uid','cutoff_year'],how='left')
    m['hit']=(m.end_year>=m.scar_year-win)&(m.end_year<=m.scar_year+win)
    return m.groupby(['uid','cutoff_year']).hit.max().fillna(False)
tagsI=pd.read_csv(G+'tags_island_v2.csv',dtype={'uid':str},low_memory=False)
neg_uids=set(tagsI[(tagsI.kind=='control')&(tagsI.class1_100m==0)&(tagsI.bare_ever==0)].uid)
neg=pairs[pairs.uid.isin(neg_uids)][['uid','cutoff_year']].drop_duplicates()
def false_alarm(neg_df, thr, max_dur=3):
    m=neg_df.merge(S[(S.delta>=thr)&(S.duration<=max_dur)][['uid','cutoff_year']].drop_duplicates(),on=['uid','cutoff_year'],how='left',indicator=True)
    return (m._merge=='both')
rng=np.random.default_rng(20260923)
pos_u=pos.uid.unique(); cal_u=set(rng.choice(pos_u,size=len(pos_u)//2,replace=False)); pos['split']=np.where(pos.uid.isin(cal_u),'cal','val')
neg_u=neg.uid.unique(); cal_n=set(rng.choice(neg_u,size=len(neg_u)//2,replace=False)); neg['split']=np.where(neg.uid.isin(cal_n),'cal','val')
R={'n_pos_pairs':int(len(pos)),'n_pos_points':int(len(pos_u)),'n_neg_pairs':int(len(neg)),'n_neg_points':int(len(neg_u)),'thresholds':{}}
best=None
for thr in [50,75,100,125,150,175,200,250,300]:
    rc=detected(pos[pos.split=='cal'],thr).mean(); fa=false_alarm(neg[neg.split=='cal'],thr).mean()
    rv=detected(pos[pos.split=='val'],thr).mean(); fv=false_alarm(neg[neg.split=='val'],thr).mean()
    R['thresholds'][thr]={'recall_cal':round(float(rc),3),'falsealarm_cal':round(float(fa),3),'youden_cal':round(float(rc-fa),3),'recall_val':round(float(rv),3),'falsealarm_val':round(float(fv),3)}
    if best is None or rc-fa>best[1]: best=(thr,rc-fa)
R['chosen_threshold_x1000']=int(best[0]); print('chosen threshold (NBR x1000):',best[0]); print(pd.DataFrame(R['thresholds']).T)
thr=best[0]
pv=pos[pos.split=='val'].merge(T[['uid','slope','north','elev','kind','ft']],on='uid')
pv['hit']=detected(pv,thr).reindex(pd.MultiIndex.from_frame(pv[['uid','cutoff_year']])).values
pv['band']=pd.cut(pv.slope,[-0.1,10,20,30,39.5,90],labels=['<10','10-20','20-30','30-39.5','>=39.5']); pv['asp']=np.where(pv.north>0.5,'N',np.where(pv.north<-0.5,'S','EW')); pv['elevband']=pd.cut(pv.elev,[-1,500,1000,1500,4000],labels=['<500','500-1000','1000-1500','>1500'])
for key in ['band','asp','elevband','kind']:
    t=pv.groupby(key,observed=True).hit.agg(['size','mean']).rename(columns={'size':'n','mean':'recall'}).round(3); R[f'recall_val_by_{key}']=t.reset_index().to_dict(orient='records'); print(t)
for win in (0,1,2): R[f'recall_val_win{win}']=round(float(detected(pos[pos.split=='val'],thr,win=win).mean()),3)
pv['scar_period']=pd.cut(pv.scar_year,[1988,1999,2009,2017,2026],labels=['1989-1999','2000-2009','2010-2017','2018-2025'])
R['recall_val_by_scar_period']=pv.groupby('scar_period',observed=True).hit.agg(['size','mean']).round(3).reset_index().to_dict(orient='records'); print(pv.groupby('scar_period',observed=True).hit.agg(['size','mean']).round(3))
json.dump(R,open('calibration_register_v1.json','w'),ensure_ascii=False,indent=1,default=str)
