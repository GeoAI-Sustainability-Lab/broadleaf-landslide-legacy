"""Sentinel-2 NBR pairs for the independent disturbance check (Earth Engine)."""
import os, sys, time, numpy as np, pandas as pd, ee
sys.path.insert(0,'@WORK@/g1'); from gee_common import *
init(); region=ee.Geometry.Rectangle([119.9,21.8,122.1,25.4])
def s2_nbr(img):
    sr=img.select(['B8','B12']).multiply(0.0001)
    valid=sr.select('B8').gt(0).And(sr.select('B12').gt(0)).And(sr.select('B8').lt(1)).And(sr.select('B12').lt(1))
    return sr.normalizedDifference(['B8','B12']).rename('NBR').updateMask(valid).copyProperties(img,['system:time_start'])
def s2_annual(y0,y1,region,months=(4,10)):
    imgs=[]
    for y in range(y0,y1+1):
        c=(ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(region).filter(ee.Filter.calendarRange(y,y,'year'))
           .filter(ee.Filter.calendarRange(months[0],months[1],'month')).map(s2_mask).map(s2_nbr))
        imgs.append(c.median().multiply(1000).toInt16().unmask(-32768).rename(f's2nbr_{y}')); imgs.append(c.count().toInt16().unmask(0).rename(f's2n_{y}'))
    return ee.Image.cat(imgs)
P=pd.read_csv('all_points_v1.csv',dtype={'uid':str}); B=800
os.makedirs('s2_batches',exist_ok=True)
img=s2_annual(2019,2024,region)
for i in range(0,len(P),B):
    out=f's2_batches/s2_{i:06d}.csv'
    if os.path.exists(out): continue
    t0=time.time(); df=None
    for attempt in range(3):
        try:
            got=img.sampleRegions(collection=to_fc(P.iloc[i:i+B]),scale=20,geometries=False,tileScale=4).getInfo(); df=feats_to_df(got); break
        except Exception as e:
            print('retry',i,attempt,str(e)[:100],flush=True); time.sleep(20+40*attempt); df=None
    if df is None: print('FAILED batch',i,flush=True); continue
    got_uids=set(df.uid)
    missing=[u for u in P.iloc[i:i+B].uid if u not in got_uids]
    if missing: print('missing',len(missing),'in batch',i,flush=True)
    df.to_csv(out,index=False); print(f'{i+len(df):,}/{len(P):,} rows {len(df)} in {time.time()-t0:.0f}s',flush=True)
print('DONE')
