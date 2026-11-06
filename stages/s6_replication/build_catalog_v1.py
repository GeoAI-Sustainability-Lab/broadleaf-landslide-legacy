"""Replication cases 2009-2017 from the event inventories (rainfall events only)."""
import glob, re, json, pickle, numpy as np, pandas as pd, shapefile, shapely
from shapely.geometry import shape
from shapely.validation import make_valid
from shapely.strtree import STRtree
ROOT='@SOURCE@/Landslide_Taiwan/2004_2023_annual_event_catalogs/raw/extracted'
D8=re.compile(r'(?<!\d)(\d{8})(?!\d)')
def dates(v):
    out=[]
    for s in D8.findall(str(v)):
        t=pd.to_datetime(s,format='%Y%m%d',errors='coerce')
        if pd.notna(t) and 1990<=t.year<=2026: out.append(t)
    return out
recs=[]; geoms=[]; EMPTY=[]
for y in range(2004,2018):
    shp=sorted(glob.glob(f'{ROOT}/{y}/*.shp'))[0]
    r=shapefile.Reader(shp,encoding='utf-8',encodingErrors='replace')
    fs=[f[0] for f in r.fields[1:]]; ev='Events' if 'Events' in fs else 'Event_s_'
    nrec=len(r); bad=0
    for i in range(nrec):
        try:
            g=shape(r.shape(i).__geo_interface__)
        except Exception:
            bad+=1; EMPTY.append((y,i)); continue
        if g.is_empty: bad+=1; EMPTY.append((y,i)); continue
        if not g.is_valid: g=make_valid(g)
        rec=r.record(i).as_dict()
        bd=dates(rec.get('BeforeDate')); ad=dates(rec.get('AfterDate'))
        c=g.centroid
        recs.append({'uid':f'r{y}_{i}','year':y,'idx':i,'cutoff':min(bd) if bd else pd.NaT,'after_latest':max(ad) if ad else pd.NaT,
                     'before_raw':str(rec.get('BeforeDate')),'after_raw':str(rec.get('AfterDate')),'events':re.sub(r'\s+','',str(rec.get(ev,''))),
                     'area_ha':float(rec.get('Area_ha') or np.nan),'area_m2_geom':float(g.area),'x':float(c.x),'y':float(c.y),
                     'minx':g.bounds[0],'miny':g.bounds[1],'maxx':g.bounds[2],'maxy':g.bounds[3]})
        geoms.append(g)
C=pd.DataFrame(recs)
print('empty or unreadable geometries (year, index):',len(EMPTY),EMPTY[:20])
print('polygons by year',C.groupby('year').size().to_dict())
assert C.x.between(140000,360000).all() and C.y.between(2400000,2820000).all(), 'coordinate range'
print('missing cutoff by year',C[C.cutoff.isna()].groupby('year').size().to_dict())
print('missing after by year',C[C.after_latest.isna()].groupby('year').size().to_dict())
tree=STRtree(geoms); garr=np.array(geoms,dtype=object)
dup=np.zeros(len(C),bool); dup_of=['']*len(C); maxiou=np.zeros(len(C))
for k in np.nonzero(C.year.values>=2009)[0]:
    g=geoms[k]
    for j in tree.query(g,predicate='intersects'):
        if C.year.values[j]>=C.year.values[k] or j==k: continue
        inter=g.intersection(geoms[j]).area; uni=g.union(geoms[j]).area
        iou=inter/uni if uni>0 else 0
        if iou>maxiou[k]: maxiou[k]=iou; dup_of[k]=C.uid.values[j]
    dup[k]=maxiou[k]>=0.9
C['relist_iou_max']=maxiou; C['relist']=dup; C['relist_of']=dup_of
print('re-listed (IoU>=0.9) by year',C[C.relist].groupby('year').size().to_dict())
K=C[C.year>=2009].copy()
K['event']=K.year.astype(str)+'_'+K.events
ev=K.groupby('event').agg(year=('year','first'),n_poly=('uid','size'),cutoff=('cutoff','min'),events=('events','first')).reset_index()
fp={}
for e,g in K.groupby('event'):
    cells=set()
    for r in g.itertuples():
        for cx in range(int(r.minx//5000),int(r.maxx//5000)+1):
            for cy in range(int(r.miny//5000),int(r.maxy//5000)+1): cells.add(f'{cx}_{cy}')
    fp[e]=sorted(cells)
ev['n_cells']=ev.event.map(lambda e: len(fp[e]))
print(ev[['event','n_poly','cutoff','n_cells']].to_string())
pickle.dump({'attrs':C,'geoms':geoms},open('catalog_polys_v1.pkl','wb'))
K.to_csv('rep_cases_raw_v1.csv',index=False); ev.to_csv('rep_events_v1.csv',index=False); json.dump(fp,open('rep_footprints_v1.json','w'))
