"""Valid growing-season observations per point and year (Earth Engine)."""
import os, sys, time, json, numpy as np, pandas as pd, ee
sys.path.insert(0,'@WORK@/g1'); from gee_common import *
init(); region=ee.Geometry.Rectangle([119.9,21.8,122.1,25.4])
P=pd.read_csv('all_points_v1.csv',dtype={'uid':str}); B=800
os.makedirs('nobs_batches',exist_ok=True)
img=annual_stack(1988,2024,region).addBands(s2_year_counts(2017,2025,region))
for i in range(0,len(P),B):
    out=f'nobs_batches/nobs_{i:06d}.csv'
    if os.path.exists(out): continue
    t0=time.time()
    for attempt in range(3):
        try:
            df=feats_to_df(sample(img,to_fc(P.iloc[i:i+B]),30)); break
        except Exception as e:
            print('retry',i,attempt,str(e)[:100],flush=True); time.sleep(20+40*attempt); df=None
    if df is None: print('FAILED batch',i,flush=True); continue
    df.to_csv(out,index=False); print(f'{i+len(df):,}/{len(P):,} rows {len(df)} in {time.time()-t0:.0f}s',flush=True)
print('DONE')
