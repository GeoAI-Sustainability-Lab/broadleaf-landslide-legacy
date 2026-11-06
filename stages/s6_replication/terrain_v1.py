"""Terrain covariates of the replication points from the 20 m DEM."""
import numpy as np, rasterio
DEM='@SOURCE@/Taiwan_DEM_20m/不分幅_全台及澎湖DEM/dem_20m.tif'
_cache={}
def load():
    if 'z' not in _cache:
        with rasterio.open(DEM) as s:
            z=s.read(1).astype(np.float32); nod=s.nodata; T=s.transform
        z[z==nod]=np.nan; _cache['z']=z; _cache['T']=T
    return _cache['z'],_cache['T']
def terrain(x, y, algo='zt'):
    z,T=load(); x=np.asarray(x,float); y=np.asarray(y,float)
    col=np.floor((x-T.c)/T.a).astype(int); row=np.floor((y-T.f)/T.e).astype(int)
    n=len(x); out={k:np.full(n,np.nan) for k in ('elev','slope','north','east','planc','profc')}
    ok=(row>=1)&(row<z.shape[0]-1)&(col>=1)&(col<z.shape[1]-1)
    r=row[ok]; c=col[ok]; d=20.0
    W=[z[r+dr,c+dc] for dr in (-1,0,1) for dc in (-1,0,1)]
    z1,z2,z3,z4,z5,z6,z7,z8,z9=W
    if algo=='horn':
        dzdx=((z3+2*z6+z9)-(z1+2*z4+z7))/(8*d)
        dzdy=((z1+2*z2+z3)-(z7+2*z8+z9))/(8*d)
    else:
        dzdx=(z6-z4)/(2*d); dzdy=(z2-z8)/(2*d)
    slope=np.degrees(np.arctan(np.hypot(dzdx,dzdy)))
    G0=(z6-z4)/(2*d); H0=(z2-z8)/(2*d)
    asp=np.arctan2(H0,-G0)
    D=((z4+z6)/2-z5)/d**2; E=((z2+z8)/2-z5)/d**2; F=(-z1+z3+z7-z9)/(4*d**2); G=(z6-z4)/(2*d); H=(z2-z8)/(2*d)
    gh=G**2+H**2
    with np.errstate(invalid='ignore',divide='ignore'):
        prof=np.where(gh>0,-2*(D*G**2+E*H**2+F*G*H)/gh,0.0)
        plan=np.where(gh>0,2*(D*H**2+E*G**2-F*G*H)/gh,0.0)
    for k,v in (('elev',z5),('slope',slope),('north',np.cos(asp)),('east',np.sin(asp)),('planc',plan),('profc',prof)):
        out[k][ok]=v
    return out
