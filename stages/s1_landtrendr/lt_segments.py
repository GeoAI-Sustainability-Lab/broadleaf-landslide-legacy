"""Parses LandTrendr vertices into loss segments and summarises them per point and cutoff year."""
import json, glob, numpy as np, pandas as pd
def segments_from_row(r):
    if not r.get('fit'): return []
    yrs=np.array(r['years']); fit=np.array(r['fitted'],dtype=float); isv=np.array(r['isvertex'])
    vi=np.nonzero(isv==1)[0]
    segs=[]
    for a,b in zip(vi[:-1],vi[1:]):
        segs.append({'uid':r['uid'],'cutoff_year':r['cutoff_year'],'start_year':int(yrs[a]),'end_year':int(yrs[b]),'duration':int(yrs[b]-yrs[a]),
                     'start_val':float(fit[a]),'end_val':float(fit[b]),'delta':float(fit[b]-fit[a]),'rmse':r.get('rmse')})
    return segs
def load_segments(pattern='lt_batches/lt_*.jsonl'):
    rows=[]; fits=[]
    for f in sorted(glob.glob(pattern)):
        for line in open(f):
            r=json.loads(line); fits.append({'uid':r['uid'],'cutoff_year':r['cutoff_year'],'fit':r.get('fit',0),'rmse':r.get('rmse'),
                                             'end_fit':(r['fitted'][-1] if r.get('fit') else None),'n_years':(len(r['years']) if r.get('fit') else 0)})
            rows+=segments_from_row(r)
    return pd.DataFrame(rows), pd.DataFrame(fits)
def summarize(segs, fits, mag_thr=100.0, max_dur=3):
    s=segs[(segs.delta>=mag_thr)&(segs.duration<=max_dur)].copy()
    s=s.sort_values(['uid','cutoff_year','end_year'])
    last=s.groupby(['uid','cutoff_year']).tail(1).set_index(['uid','cutoff_year'])
    out=fits.set_index(['uid','cutoff_year']).copy()
    out['n_loss_segments']=s.groupby(['uid','cutoff_year']).size().reindex(out.index).fillna(0).astype(int)
    out['dist_year']=last.end_year.reindex(out.index); out['dist_mag']=last.delta.reindex(out.index); out['dist_dur']=last.duration.reindex(out.index)
    out['dist_pre']=last.start_val.reindex(out.index); out['dist_post']=last.end_val.reindex(out.index)
    out['TSD']=out.index.get_level_values('cutoff_year')-out.dist_year
    out['recovery_ratio']=(out.dist_post-out.end_fit)/(out.dist_post-out.dist_pre)
    return out.reset_index()
if __name__=='__main__':
    segs,fits=load_segments('pilot_lt_1988_2023_v1.json'.replace('.json','*.json')) if False else (None,None)
