"""LandTrendr fits per point, series truncated at the year before each cutoff year, in 8 cutoff-year groups (Earth Engine)."""
import os, sys, time, json, numpy as np, pandas as pd, ee
sys.path.insert(0,'@WORK@/g1'); from gee_common import *
init(); region=ee.Geometry.Rectangle([119.9,21.8,122.1,25.4])
D=pd.read_csv('lt_pairs_v1.csv',dtype={'uid':str}); B=400
os.makedirs('lt_batches',exist_ok=True)
for Y in sorted(D.cutoff_year.unique()):
    G=D[D.cutoff_year==Y].reset_index(drop=True); y1=int(Y)-1
    lt=lt_image(1988,y1,region)
    for i in range(0,len(G),B):
        out=f'lt_batches/lt_{Y}_{i:06d}.jsonl'
        if os.path.exists(out): continue
        t0=time.time(); rows=None
        for attempt in range(3):
            try:
                got=lt.sampleRegions(collection=to_fc(G.iloc[i:i+B]),scale=30,geometries=False,tileScale=4).getInfo()
                rows=[]
                for f in got['features']:
                    p=f['properties']; arr=p.get('LandTrendr')
                    if arr is None: rows.append({'uid':p['uid'],'cutoff_year':int(Y),'fit':0}); continue
                    rows.append({'uid':p['uid'],'cutoff_year':int(Y),'fit':1,'rmse':p.get('rmse'),'years':arr[0],'source':arr[1],'fitted':arr[2],'isvertex':arr[3]})
                break
            except Exception as e:
                print('retry',Y,i,attempt,str(e)[:120],flush=True); time.sleep(30+60*attempt)
        if rows is None: print('FAILED',Y,i,flush=True); continue
        got_uids={r['uid'] for r in rows}
        for u in G.iloc[i:i+B].uid:
            if u not in got_uids: rows.append({'uid':u,'cutoff_year':int(Y),'fit':0,'note':'not returned by sampleRegions'})
        with open(out,'w') as f:
            for r in rows: f.write(json.dumps(r)+'\n')
        print(f'group {Y}: {min(i+B,len(G)):,}/{len(G):,} in {time.time()-t0:.0f}s (fitted {sum(r["fit"] for r in rows)})',flush=True)
print('DONE')
