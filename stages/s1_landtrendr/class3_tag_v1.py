"""Adds class 3 (uncatalogued canopy disturbance) to the island tags."""
import json, sys, numpy as np, pandas as pd
sys.path.insert(0,'@WORK@/g1'); from lt_segments import summarize
G='@WORK@/g0/'
C=json.load(open('calibration_register_v1.json')); thr=float(C['chosen_threshold_x1000'])
segs=pd.read_csv('lt_segments_all_v1.csv',dtype={'uid':str}); fits=pd.read_csv('lt_fits_all_v1.csv',dtype={'uid':str})
S=summarize(segs,fits,mag_thr=thr,max_dur=3)
S.to_csv('lt_summary_v1.csv',index=False)
def attach(T):
    T=T.copy(); T['cutoff_year']=pd.to_datetime(T.cutoff).dt.year
    T=T.merge(S[['uid','cutoff_year','fit','rmse','n_loss_segments','dist_year','dist_mag','dist_dur','TSD','recovery_ratio','end_fit']],on=['uid','cutoff_year'],how='left')
    T['lt_detected']=T.dist_year.notna().astype(int)
    T['class3']=((T.site_class==0)&(T.lt_detected==1)).astype(int)
    T['site_class4']=np.where(T.class3==1,3,T.site_class)
    T['tsd3_bin']=np.select([T.class3==0,T.TSD<5,T.TSD<10,T.TSD<20],['none','0-5','5-10','10-20'],'20+')
    T['lt_missing']=T.fit.isna().astype(int)
    return T
I=attach(pd.read_csv(G+'tags_island_v2.csv',dtype={'uid':str},low_memory=False)); E=attach(pd.read_csv(G+'tags_event_v2.csv',dtype={'uid':str},low_memory=False))
I.to_csv('tags_island_v3.csv',index=False,encoding='utf-8-sig'); E.to_csv('tags_event_v3.csv',index=False,encoding='utf-8-sig')
b=I[I.ft=='闊葉樹林型']; bam=I[I.ft=='竹林']
R={'threshold_x1000':thr,'lt_pairs_fitted':int(S.fit.sum()),'lt_pairs_total':int(len(S)),
   'island_missing_pairs':int(I.lt_missing.sum()),'event_missing_rows':int(E.lt_missing.sum()),
   'broadleaf_island':{'cases':b[b.kind=='case'].site_class4.value_counts().sort_index().to_dict(),'controls':b[b.kind=='control'].site_class4.value_counts().sort_index().to_dict()},
   'bamboo_island':{'cases':bam[bam.kind=='case'].site_class4.value_counts().sort_index().to_dict(),'controls':bam[bam.kind=='control'].site_class4.value_counts().sort_index().to_dict()},
   'broadleaf_class3_TSD':{k:{'n':int(len(g)),'median_TSD':float(g.TSD.median()),'median_mag':float(g.dist_mag.median()),'median_recovery':float(g.recovery_ratio.median())} for k,g in b[b.class3==1].groupby('kind')},
   'lt_detected_share_by_class_broadleaf':{f'{k[0]}_class{k[1]}':float(v) for k,v in b.groupby(['kind','site_class']).lt_detected.mean().round(3).items()}}
json.dump(R,open('class3_register_v1.json','w'),ensure_ascii=False,indent=1,default=str); print(json.dumps(R,ensure_ascii=False,indent=1,default=str))
