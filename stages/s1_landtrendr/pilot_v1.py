"""Pilot of the per-point Landsat observation counts."""
import os, sys, time, json, numpy as np, pandas as pd, ee
sys.path.insert(0,'@WORK@/g1'); from gee_common import *
init()
G='@WORK@/g0/'
T=pd.read_csv(G+'tags_island_v2.csv',dtype={'uid':str},low_memory=False)
rng=np.random.default_rng(20260923)
T['band']=pd.cut(T.slope,[-0.1,10,20,30,39.5,90],labels=['<10','10-20','20-30','30-39.5','>=39.5'])
T['asp']=np.where(T.north>0.5,'N',np.where(T.north<-0.5,'S','EW'))
parts=[]
for ft,kind,n in [('闊葉樹林型','case',100),('闊葉樹林型','control',100),('竹林','case',50),('竹林','control',50)]:
    d=T[(T.ft==ft)&(T.kind==kind)]
    g=d.groupby(['band','asp'],observed=True); alloc={k:max(2,int(round(n*len(v)/len(d)))) for k,v in g}
    for k,v in g:
        parts.append(v.sample(n=min(alloc[k],len(v)),random_state=int(rng.integers(1e9))))
P=pd.concat(parts).drop_duplicates('uid'); print('pilot points',len(P), P.groupby(['ft','kind']).size().to_dict())
P[['uid','kind','ft','slope','north','elev','x','y','band','asp']].to_csv('pilot_points_v1.csv',index=False)
region=ee.Geometry.Rectangle([119.9,21.8,122.1,25.4]); fc=to_fc(P)
t0=time.time(); img=annual_stack(1988,2024,region).addBands(s2_year_counts(2017,2025,region))
got=sample(img,fc,30); df=feats_to_df(got); print('sampled',len(df),'rows in',round(time.time()-t0,1),'s')
df.to_csv('pilot_nobs_v1.csv',index=False)
t0=time.time(); imgs=[]
for y in range(2013,2022):
    for tag in ('L7','L8'):
        m=landsat_year(y,region,sensors=[tag]); imgs.append(m.median().multiply(1000).toInt16().unmask(-32768).rename(f'{tag}_{y}')); imgs.append(m.count().toInt16().unmask(0).rename(f'{tag}n_{y}'))
got2=sample(ee.Image.cat(imgs),fc,30); df2=feats_to_df(got2); df2.to_csv('pilot_sensor_overlap_v1.csv',index=False); print('overlap sampled in',round(time.time()-t0,1),'s')
