"""Forest type label of every replication point from the fourth national forest resource inventory map."""
import sys, os, json, struct, time, csv
sys.path.insert(0,'.')
import numpy as np, shapefile, shapely
from shapely.geometry import shape
pts_csv, shp_path, out_dir, MAXS = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
t0=time.time(); os.makedirs(out_dir,exist_ok=True)
prog_path=os.path.join(out_dir,'label_progress.json'); out_path=os.path.join(out_dir,'label_matches.csv')
prog=json.load(open(prog_path)) if os.path.exists(prog_path) else {'next':0,'done':False,'n_records':None,'n_parsed':0,'n_matches':0}
if prog['done']: print('already done',prog); sys.exit(0)
P=np.genfromtxt(pts_csv,delimiter=',',names=True,dtype=None,encoding='utf-8')
uid=P['uid']; X=P['x'].astype(float); Y=P['y'].astype(float)
G=1000.0; gx=np.floor(X/G).astype(np.int64); gy=np.floor(Y/G).astype(np.int64)
buckets={}
for i,(a,b) in enumerate(zip(gx,gy)): buckets.setdefault((int(a),int(b)),[]).append(i)
shx=open(shp_path[:-4]+'.shx','rb'); shx.seek(24); flen=struct.unpack('>i',shx.read(4))[0]*2; nrec=(flen-100)//8
shx.seek(100); idx=np.frombuffer(shx.read(nrec*8),dtype='>i4').reshape(-1,2); offs=idx[:,0].astype(np.int64)*2
prog['n_records']=int(nrec)
rd=shapefile.Reader(shp_path,encoding='utf-8',encodingErrors='replace')
fh=open(shp_path,'rb'); newfile=not os.path.exists(out_path)
w=open(out_path,'a',newline='',encoding='utf-8'); cw=csv.writer(w)
if newfile: cw.writerow(['uid','rec','TypeName','FunctionTy','Function','ORIG_FID'])
BLK=20000
i=prog['next']
while i<nrec and time.time()-t0<MAXS:
    j=min(nrec,i+BLK)
    bb=np.empty((j-i,4))
    for k in range(i,j):
        fh.seek(offs[k]+12); bb[k-i]=struct.unpack('<4d',fh.read(32))
    for k in range(i,j):
        x0,y0,x1,y1=bb[k-i]
        cand=[]
        for a in range(int(np.floor(x0/G)),int(np.floor(x1/G))+1):
            for b in range(int(np.floor(y0/G)),int(np.floor(y1/G))+1):
                for p in buckets.get((a,b),()):
                    if x0<=X[p]<=x1 and y0<=Y[p]<=y1: cand.append(p)
        if not cand: continue
        g=shape(rd.shape(k).__geo_interface__); prog['n_parsed']+=1
        hit=[p for p in cand if g.covers(shapely.Point(X[p],Y[p]))]
        if hit:
            r=rd.record(k)
            for p in hit: cw.writerow([uid[p],k,r['TypeName'],r['FunctionTy'],r['Function'],r['ORIG_FID']]); prog['n_matches']+=1
    i=j; prog['next']=int(i); w.flush(); json.dump(prog,open(prog_path,'w'))
prog['done']=bool(i>=nrec); json.dump(prog,open(prog_path,'w')); w.close()
print(json.dumps(prog), 'elapsed', round(time.time()-t0,1))
