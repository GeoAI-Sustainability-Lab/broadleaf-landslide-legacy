"""Transfer to the replication without the 2009 inventory year."""
import json, sys
H='@WORK@'
src=open(H+'/v10/an/g4/stand_origin_v1.py',encoding='utf-8').read()
exec(compile(src[:src.index('rng=np.random.default_rng(SEED)')],'stand_origin_v1_defs','exec'))
P=H+'/v10/an/methods/methods_register_v1.json'; R=json.load(open(P,encoding='utf-8'))
TV=['slope','slope2','north','east','planc','profc','elev']
Mn=EV.copy(); Mn['R1']=(Mn.class1_50m==1).astype(int); Mn['R3']=((Mn.lt_detected==1)&(Mn.R1==0)).astype(int)
Rp=pd.read_csv(H+'/v10/an/g5/rep_rows_tagged_v1.csv',low_memory=False); Rp['case']=Rp.case.astype(int); Rp['slope2']=Rp.slope**2; Rp=Rp.dropna(subset=TV)
Rp=Rp[~Rp.event.astype(str).str.startswith('2009')].copy()
def strat_auc(d,s):
    num=den=0.0
    for ev,g in d.assign(s=s).groupby('event'):
        y=g.case.values; n1=y.sum(); n0=len(y)-n1
        if n1==0 or n0==0: continue
        r=pd.Series(g.s.values).rank().values; a=(r[y==1].sum()-n1*(n1+1)/2)/(n1*n0); num+=a*n1*n0; den+=n1*n0
    return num/den
o={}
for nm in ('all','le39.5'):
    A_=Mn if nm=='all' else Mn[Mn.slope<=39.5]; B_=Rp if nm=='all' else Rp[Rp.slope<=39.5]
    r={}
    for mod,f in (('terrain',''),('terrain+legacy','+R1+R3')):
        dtr,_=drop_uninf(A_); mm=smf.logit(f"case~{TERR}{f}+C(event)",data=dtr).fit(disp=0,maxiter=100,method='newton')
        cols=TV+(['R1','R3'] if f else []); r[mod]=round(strat_auc(B_,B_[cols].values@mm.params[cols].values),4)
    r['gain']=round(r['terrain+legacy']-r['terrain'],4); r['n_cases_test']=int(B_.case.sum()); r['events_test']=int(B_.event.nunique())
    o[nm]=r
R['C7b_transfer_main_to_rep_without_2009']=o
json.dump(R,open(P,'w',encoding='utf-8'),ensure_ascii=False,indent=1,default=float)
print(o)
