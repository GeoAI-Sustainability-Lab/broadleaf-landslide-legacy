"""Earth Engine helpers: Landsat Collection 2 Level 2 harmonisation, growing-season NBR, LandTrendr parameters. Credentials are read from the environment variables GEE_SERVICE_ACCOUNT_KEY and GEE_PROJECT."""
import os, json, time, ee, numpy as np, pandas as pd
from pyproj import Transformer
def init():
    key=os.environ['GEE_SERVICE_ACCOUNT_KEY']; proj=os.environ['GEE_PROJECT']
    sa=json.load(open(key))['client_email']; ee.Initialize(ee.ServiceAccountCredentials(sa,key),project=proj)
TR=Transformer.from_crs(3826,4326,always_xy=True)
def to_fc(df, idcol='uid'):
    feats=[]
    for r in df.itertuples():
        lon,lat=TR.transform(r.x,r.y); feats.append(ee.Feature(ee.Geometry.Point([lon,lat]),{'uid':getattr(r,idcol)}))
    return ee.FeatureCollection(feats)
SENSORS=[('LANDSAT/LT05/C02/T1_L2','SR_B4','SR_B7','L5'),('LANDSAT/LE07/C02/T1_L2','SR_B4','SR_B7','L7'),
         ('LANDSAT/LC08/C02/T1_L2','SR_B5','SR_B7','L8'),('LANDSAT/LC09/C02/T1_L2','SR_B5','SR_B7','L9')]
def mask_l(img):
    qa=img.select('QA_PIXEL')
    keep=(qa.bitwiseAnd(1<<1).eq(0).And(qa.bitwiseAnd(1<<2).eq(0)).And(qa.bitwiseAnd(1<<3).eq(0))
          .And(qa.bitwiseAnd(1<<4).eq(0)).And(qa.bitwiseAnd(1<<5).eq(0)).And(qa.bitwiseAnd(1<<7).eq(0)))
    return img.updateMask(keep)
def nbr_img(img, nir, swir2):
    sr=img.select([nir,swir2]).multiply(0.0000275).add(-0.2)
    valid=sr.select(nir).gt(0).And(sr.select(swir2).gt(0)).And(sr.select(nir).lt(1)).And(sr.select(swir2).lt(1))
    nbr=sr.normalizedDifference([nir,swir2]).rename('NBR').updateMask(valid)
    return nbr.copyProperties(img,['system:time_start'])
def landsat_year(y, region, months=(4,10), sensors=None):
    cols=[]
    for cid,nir,swir2,tag in SENSORS:
        if sensors and tag not in sensors: continue
        c=(ee.ImageCollection(cid).filterBounds(region).filter(ee.Filter.calendarRange(y,y,'year'))
           .filter(ee.Filter.calendarRange(months[0],months[1],'month')).map(mask_l).map(lambda im,n=nir,s=swir2: nbr_img(im,n,s)))
        cols.append(c)
    merged=cols[0]
    for c in cols[1:]: merged=merged.merge(c)
    return merged
def annual_stack(y0,y1,region,months=(4,10)):
    imgs=[]
    for y in range(y0,y1+1):
        m=landsat_year(y,region,months)
        imgs.append(m.median().multiply(1000).toInt16().unmask(-32768).rename(f'nbr_{y}')); imgs.append(m.count().toInt16().unmask(0).rename(f'nobs_{y}'))
    return ee.Image.cat(imgs)
def s2_mask(img):
    scl=img.select('SCL'); keep=scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11)).And(scl.neq(0)).And(scl.neq(1))
    return img.updateMask(keep)
def s2_year_counts(y0,y1,region,months=(4,10)):
    imgs=[]
    for y in range(y0,y1+1):
        c=(ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(region).filter(ee.Filter.calendarRange(y,y,'year'))
           .filter(ee.Filter.calendarRange(months[0],months[1],'month')).map(s2_mask).select('B8'))
        imgs.append(c.count().toInt16().unmask(0).rename(f's2obs_{y}'))
    return ee.Image.cat(imgs)
def sample(img, fc, scale):
    return img.sampleRegions(collection=fc,scale=scale,geometries=False,tileScale=4).getInfo()
def feats_to_df(got):
    return pd.DataFrame([f['properties'] for f in got['features']])

LT_PARAMS=dict(maxSegments=6, spikeThreshold=0.9, vertexCountOvershoot=3, preventOneYearRecovery=True,
               recoveryThreshold=0.25, pvalThreshold=0.05, bestModelProportion=0.75, minObservationsNeeded=6)
def lt_collection(y0, y1, region, months=(4,10)):
    imgs=[]
    for y in range(y0,y1+1):
        m=landsat_year(y,region,months)
        im=m.median().multiply(-1000).toFloat().rename('NBRneg').set('system:time_start',ee.Date.fromYMD(y,7,1).millis()).set('year',y)
        imgs.append(im)
    return ee.ImageCollection(imgs)
def lt_image(y0, y1, region):
    coll=lt_collection(y0,y1,region)
    lt=ee.Algorithms.TemporalSegmentation.LandTrendr(timeSeries=coll.select('NBRneg'), **LT_PARAMS)
    return lt
def lt_sample(y0, y1, region, fc, scale=30):
    lt=lt_image(y0,y1,region)
    got=lt.sampleRegions(collection=fc,scale=scale,geometries=False,tileScale=4).getInfo()
    rows=[]
    for f in got['features']:
        p=f['properties']; arr=p.get('LandTrendr')
        if arr is None: rows.append({'uid':p['uid'],'fit':0}); continue
        years,src,fit,isv=arr
        rows.append({'uid':p['uid'],'fit':1,'rmse':p.get('rmse'),'years':years,'source':src,'fitted':fit,'isvertex':isv})
    return rows
