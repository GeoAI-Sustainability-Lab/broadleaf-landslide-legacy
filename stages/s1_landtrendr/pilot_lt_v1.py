"""Pilot LandTrendr run on a small point sample."""
import os, sys, time, json, numpy as np, pandas as pd, ee
sys.path.insert(0,'@WORK@/g1'); from gee_common import *
init(); region=ee.Geometry.Rectangle([119.9,21.8,122.1,25.4])
P=pd.read_csv('pilot_points_v1.csv',dtype={'uid':str}); fc=to_fc(P)
t0=time.time(); rows=lt_sample(1988,2023,region,fc); dt=time.time()-t0
print('LandTrendr 1988-2023 at',len(P),'points:',round(dt,1),'s; fitted:',sum(r['fit'] for r in rows),'returned:',len(rows))
json.dump(rows,open('pilot_lt_1988_2023_v1.json','w'))
r=[x for x in rows if x['fit']][0]; print('example uid',r['uid'],'rmse',r['rmse']); print('years',r['years'][:8],'...'); print('fitted',[round(v) for v in r['fitted'][:8]]); print('isvertex',r['isvertex'][:8])
