"""Replication controls (new random points) and event-matched rows."""
import sys, json, pickle, numpy as np, pandas as pd, shapely
from shapely.strtree import STRtree
sys.path.insert(0,'@WORK@/g5'); from terrain_v1 import terrain
SIX=['闊葉樹林型','針葉樹林型','針闊葉樹混淆','竹林','竹闊混淆林','待成林地']; R={}
cat=pickle.load(open('catalog_polys_v1.pkl','rb')); A=cat['attrs']; GEO=cat['geoms']
A['cutoff']=pd.to_datetime(A.cutoff); A['after_latest']=pd.to_datetime(A.after_latest)
K=pd.read_csv('rep_cases_raw_v1.csv',parse_dates=['cutoff','after_latest']); EV=pd.read_csv('rep_events_v1.csv',parse_dates=['cutoff']); FP=json.load(open('rep_footprints_v1.json'))
C=pd.read_csv('rep_control_candidates_v1.csv')
L=pd.read_csv('forest_type_labels_v1.csv').sort_values(['uid','rec']).drop_duplicates('uid',keep='first')
R['labels']={'points_labelled':int(len(L)),'boundary_points_multiple_polygons':148,'rule':'lowest record index'}
K=K.merge(L[['uid','TypeName','Function']],on='uid',how='left')
R['cases_flow']={'polygons_2009_2017':int(len(K))}
K=K[~K.relist]; R['cases_flow']['after_relisting_excluded']=int(len(K))
K=K[K.cutoff.notna()]; R['cases_flow']['after_missing_cutoff_excluded']=int(len(K))
K=K[K.TypeName.isin(SIX)]; R['cases_flow']['in_six_forest_types']=int(len(K))
C=C[C.land].merge(L[['uid','TypeName','Function']],on='uid',how='inner'); C=C[C.TypeName.isin(SIX)]
R['controls_flow']={'candidates':80000,'with_land_dem':int(pd.read_csv('rep_control_candidates_v1.csv').land.sum()),'in_six_forest_types':int(len(C))}
P=pd.concat([K[['uid','x','y','TypeName','Function']].assign(kind='case'),C[['uid','x','y','TypeName','Function']].assign(kind='control')],ignore_index=True)
t=terrain(P.x.values,P.y.values)
for k,v in t.items(): P[k]=v
P['slope2']=P.slope**2
bad=P[['slope','north','east','planc','profc','elev']].isna().any(axis=1)
R['terrain_missing_points']={'case':int((bad&(P.kind=='case')).sum()),'control':int((bad&(P.kind=='control')).sum())}
P=P[~bad].reset_index(drop=True)
P.to_csv('rep_points_v1.csv',index=False)
EV=EV[EV.cutoff.notna()]
cell=(np.floor(P.x/5000).astype(int).astype(str)+'_'+np.floor(P.y/5000).astype(int).astype(str)); P['cell']=cell
ctrl=P[P.kind=='control']; rows=[]
Kc=K.set_index('uid')
tree=STRtree(GEO); uid_arr=A.uid.values
poly_event=(A.year.astype(str)+'_'+A.events).values
for e in EV.itertuples():
    cs=P[(P.kind=='case')&P.uid.isin(K[K.year.astype(str)+'_'+K.events==e.event].uid)]
    for r in cs.itertuples():
        rows.append({'uid':r.uid,'event':e.event,'case':1,'cutoff':Kc.loc[r.uid,'cutoff']})
    cells=set(FP[e.event]); cc=ctrl[ctrl.cell.isin(cells)]
    ev_idx=np.nonzero(poly_event==e.event)[0]
    if len(cc):
        pts=shapely.points(cc.x.values,cc.y.values)
        inside=np.zeros(len(cc),bool)
        sub=STRtree([GEO[i] for i in ev_idx]); hits=sub.query(pts,predicate='intersects')
        inside[np.unique(hits[0])]=True
        for r,ins in zip(cc.itertuples(),inside):
            if not ins: rows.append({'uid':r.uid,'event':e.event,'case':0,'cutoff':e.cutoff})
        R.setdefault('control_rows_inside_event_polygons',0); R['control_rows_inside_event_polygons']+=int(inside.sum())
D=pd.DataFrame(rows).merge(P,on='uid',how='left')
D['cutoff']=pd.to_datetime(D.cutoff); D['cutoff_year']=D.cutoff.dt.year
pts=shapely.points(D.x.values,D.y.values)
pi,gi=tree.query(pts,predicate='dwithin',distance=50.0)
al=A.after_latest.values; cut=D.cutoff.values
ok=(al[gi]<=cut[pi])&(uid_arr[gi]!=D.uid.values[pi])
pi,gi=pi[ok],gi[ok]
last=pd.Series(al[gi]).groupby(pi).max(); n=pd.Series(gi).groupby(pi).size()
D['scar50_n']=0; D.loc[n.index,'scar50_n']=n.values
D['scar50_last_after']=pd.NaT; D.loc[last.index,'scar50_last_after']=last.values
D['scar50_last_after']=pd.to_datetime(D.scar50_last_after)
D['R1']=(D.scar50_n>0).astype(int)
D['scar_age']=(D.cutoff-D.scar50_last_after).dt.days/365.25
pi0,gi0=tree.query(pts,predicate='intersects'); ok0=(al[gi0]<=cut[pi0])&(uid_arr[gi0]!=D.uid.values[pi0])
D['scar0_n']=0; s0=pd.Series(gi0[ok0]).groupby(pi0[ok0]).size(); D.loc[s0.index,'scar0_n']=s0.values
D.to_csv('rep_rows_v1.csv',index=False)
B=D[D.TypeName=='闊葉樹林型']
R['rows']={'events_with_cutoff':int(EV.shape[0]),'rows_all_types':int(len(D)),'cases_all_types':int(D.case.sum()),
           'broadleaf_rows':int(len(B)),'broadleaf_cases':int(B.case.sum()),'broadleaf_control_rows':int((B.case==0).sum()),'broadleaf_control_points':int(B[B.case==0].uid.nunique()),
           'broadleaf_cases_le39.5':int(((B.case==1)&(B.slope<=39.5)).sum()),'broadleaf_control_rows_le39.5':int(((B.case==0)&(B.slope<=39.5)).sum()),
           'R1_share_cases':round(float(B[B.case==1].R1.mean()),4),'R1_share_controls':round(float(B[B.case==0].R1.mean()),4)}
lt=B[['uid','x','y','cutoff_year']].drop_duplicates(['uid','cutoff_year'])
lt.to_csv('lt_pairs_rep_v1.csv',index=False)
R['lt_pairs']={'pairs':int(len(lt)),'by_cutoff_year':{int(k):int(v) for k,v in lt.cutoff_year.value_counts().sort_index().items()}}
json.dump(R,open('rep_build_register_v1.json','w'),ensure_ascii=False,indent=1,default=str)
print(json.dumps(R,ensure_ascii=False,indent=1,default=str))
