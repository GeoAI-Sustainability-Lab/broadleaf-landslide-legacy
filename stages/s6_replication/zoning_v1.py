"""Point-in-polygon of the points against the landslide geologically sensitive areas (GeoPackage, R-tree prefilter)."""
import sys, sqlite3, struct, csv, time
sys.path.insert(0,'.')
import numpy as np, shapely, shapefile
from shapely.geometry import shape
pts_csv, gpkg, llp, out = sys.argv[1:5]; t0=time.time()
P=np.genfromtxt(pts_csv,delimiter=',',names=True,dtype=None,encoding='utf-8')
uid=P['uid']; X=P['x'].astype(float); Y=P['y'].astype(float); n=len(uid)
con=sqlite3.connect(gpkg); cur=con.cursor()
tabs=[r[0] for r in cur.execute("select name from sqlite_master where name like 'rtree_%'")]
rt=[t for t in tabs if t.startswith('rtree_geological_sensitive_landslide') and not t.endswith(('_node','_parent','_rowid'))][0]
def gp_to_geom(b):
    flags=b[3]; env=(flags>>1)&7; L={0:0,1:32,2:48,3:48,4:64}[env]
    return shapely.from_wkb(bytes(b[8+L:]))
inside=np.zeros(n,bool); zone=['']*n
cache={}
for i in range(n):
    x,y=X[i],Y[i]
    ids=[r[0] for r in cur.execute(f"select id from {rt} where minx<=? and maxx>=? and miny<=? and maxy>=?",(x,x,y,y))]
    for fid in ids:
        if fid not in cache:
            row=cur.execute("select geom, source_zone_id from geological_sensitive_landslide where fid=?",(fid,)).fetchone()
            cache[fid]=(shapely.prepared.prep(gp_to_geom(row[0])) if False else gp_to_geom(row[0]), row[1])
        g,z=cache[fid]
        if g.covers(shapely.Point(x,y)): inside[i]=True; zone[i]=z; break
r=shapefile.Reader(llp,encoding='utf-8',encodingErrors='replace'); polys=[shape(s.__geo_interface__) for s in r.shapes()]
tree=shapely.STRtree(polys); hits=tree.query(shapely.points(X,Y),predicate='within')
ll=np.zeros(n,bool); ll[np.unique(hits[0])]=True
with open(out,'w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['uid','gsmma_inside','gsmma_zone','llp_inside'])
    for i in range(n): w.writerow([uid[i],int(inside[i]),zone[i],int(ll[i])])
print('points',n,'gsmma inside',int(inside.sum()),'llp inside',int(ll.sum()),'llp polygons',len(polys),'rtree',rt,'elapsed',round(time.time()-t0,1))
