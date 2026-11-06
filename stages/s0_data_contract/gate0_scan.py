"""Streams one inventory or annual-state shapefile record by record and writes the polygons within 100 m of each point (bounding-box prefilter, exact distance)."""
import sys, os, json, struct, time, csv
sys.path.insert(0,'.')
import numpy as np, shapefile, shapely
from shapely.geometry import shape as shp_shape
from shapely.validation import make_valid
shp_path, out_prefix, B, MAXS = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
fields = sys.argv[5].split('=')[1].split(',') if len(sys.argv)>5 else None
t0=time.time()
P=np.genfromtxt('@WORK@/g0/points.csv',delimiter=',',names=True,dtype=None,encoding='utf-8',usecols=('uid','x','y'))
uid=P['uid']; X=P['x'].astype(float); Y=P['y'].astype(float)
pts=shapely.points(X,Y)
prog_path=out_prefix+'_progress.json'; out_path=out_prefix+'_matches.csv'
prog=json.load(open(prog_path)) if os.path.exists(prog_path) else {'next':0,'done':False,'n_records':None,'n_candidates':0,'n_matches':0,'invalid_repaired':0,'range_ok':None}
if prog['done']: print('already done',prog); sys.exit(0)
shx=open(shp_path[:-4]+'.shx','rb'); shx.seek(24); flen=struct.unpack('>i',shx.read(4))[0]*2; nrec=(flen-100)//8
shx.seek(100); idx=np.frombuffer(shx.read(nrec*8),dtype='>i4').reshape(-1,2); offsets=idx[:,0]*2
prog['n_records']=int(nrec)
fh=open(shp_path,'rb')
fh.seek(36); fxmin,fymin,fxmax,fymax=struct.unpack('<4d',fh.read(32))
prog['file_bbox']=[fxmin,fymin,fxmax,fymax]; prog['range_ok']=bool(100000<fxmin<400000 and 2300000<fymin<2900000)
rd=shapefile.Reader(shp_path)
newfile=not os.path.exists(out_path)
w=open(out_path,'a',newline='',encoding='utf-8'); cw=csv.writer(w)
if newfile: cw.writerow(['uid','record_index','distance_m','inside','poly_area_m2']+(fields or []))
i=prog['next']; stopped=False
while i<nrec:
    if time.time()-t0>MAXS: stopped=True; break
    fh.seek(offsets[i]+8); st=struct.unpack('<i',fh.read(4))[0]
    if st in (5,15,25):
        xmin,ymin,xmax,ymax=struct.unpack('<4d',fh.read(32))
        m=np.nonzero((X>=xmin-B)&(X<=xmax+B)&(Y>=ymin-B)&(Y<=ymax+B))[0]
        if len(m):
            prog['n_candidates']+=int(len(m))
            try:
                g=shp_shape(rd.shape(i).__geo_interface__)
                if not g.is_valid: g=make_valid(g); prog['invalid_repaired']+=1
                d=shapely.distance(g,pts[m]); ins=shapely.covers(g,pts[m])
                rec=rd.record(i) if fields else None
                extra=[rec[f] for f in fields] if fields else []
                for j,k in enumerate(m):
                    if d[j]<=B:
                        cw.writerow([uid[k],i,round(float(d[j]),3),int(ins[j]),round(float(g.area),1)]+extra); prog['n_matches']+=1
            except Exception as e:
                cw.writerow(['ERROR',i,'',-1,'',str(e)[:60]]+['']*((len(fields) if fields else 0)-1 if fields else 0))
    i+=1
prog['next']=int(i); prog['done']=bool(i>=nrec); prog['elapsed_last']=round(time.time()-t0,1)
w.close(); json.dump(prog,open(prog_path,'w'))
print(json.dumps(prog))
