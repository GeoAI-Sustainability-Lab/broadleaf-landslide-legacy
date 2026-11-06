"""Exploratory checks of the replication (2009 inventory year, class-specific slope interactions)."""
import json, math, numpy as np, pandas as pd, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'@WORK@/g5')
import importlib.util
spec=importlib.util.spec_from_file_location('ra','@WORK@/g5/rep_analysis_v1.py')
src=open(spec.origin).read().split('# ------------------------------------------------------------------ main period, same definitions')[0]
exec(src)
B=pd.read_csv('rep_rows_tagged_v1.csv',low_memory=False); B['s20']=B.slope-20
H='@WORK@'
E=pd.read_csv('@WORK@/v10/data/tags_event_v4.csv',dtype={'uid':str},low_memory=False); BR=pd.read_csv(H+'/v10/an/g3/bright_rows_v1.csv',dtype={'uid':str})
E=E.merge(BR[['uid','event','Qt']],on=['uid','event'],how='left')
M=E[E.ft==BRO].copy(); M['case']=M.case.astype(int); M['slope2']=M.slope**2; M['s20']=M.slope-20
M=M.dropna(subset=['slope','north','east','planc','profc','elev'])
M['R1']=(M.class1_50m==1).astype(int); M['R3']=((M.lt_detected==1)&(M.R1==0)).astype(int); M['legacy']=((M.R1==1)|(M.R3==1)).astype(int)
X={}
def slope_int(d,lab):
    m,mr,dd,nd=fitfe(d,f"case~{TERR}+legacy+legacy:s20"); return {'per_degree':term(m,mr,'legacy:s20'),'n_case':int(dd.case.sum()),'label':lab}
def trend(d):
    d=d.copy(); d['qrank']=d.Qt.astype(int)-1; d['Qc']=pd.Categorical(d.Qt.astype(int))
    m,mr,dd,nd=fitfe(d,f"case~C(Qc)+legacy+legacy:qrank+{TERR}"); return term(m,mr,'legacy:qrank')
X['E1_rep_excl_2009']={'slope':slope_int(B[B.year!=2009] if 'year' in B.columns else B[~B.event.str.startswith('2009_')],'rep without 2009 events')}
B9=B[~B.event.str.startswith('2009_')]
X['E1_rep_excl_2009']={'slope':slope_int(B9,'rep without 2009 events'),'trend':trend(B9),'n_cases':int(B9.case.sum())}
X['E1_rep_2009_only']={'slope':slope_int(B[B.event.str.startswith('2009_')],'rep 2009 events only'),'n_cases':int(B[B.event.str.startswith('2009_')].case.sum())}
for nm,d in (('main',M),('rep',B)):
    m,mr,dd,nd=fitfe(d,f"case~{TERR}+R1+R3+R1:s20+R3:s20")
    X[f'E2_{nm}']={'R1_per_degree':term(m,mr,'R1:s20'),'R3_per_degree':term(m,mr,'R3:s20')}
Mr=M.copy(); old=(Mr.R1==1)&(Mr.scar_50m_min_yrs>=14); Mr=Mr[~old]
X['E3_main_scars_lt14yr']={'slope':slope_int(Mr,'main, scars older than 14 yr removed'),'rows_removed':int(old.sum())}
json.dump(X,open('rep_explore_register_v1.json','w'),ensure_ascii=False,indent=1,default=float)
def o(v,d=3): return f"{v['OR']:.{d}f} ({v['ci'][0]:.{d}f}–{v['ci'][1]:.{d}f})"
for k,v in X.items(): print(k, {kk:(o(vv['per_degree']) if isinstance(vv,dict) and 'per_degree' in vv else (o(vv) if isinstance(vv,dict) and 'OR' in vv else vv)) for kk,vv in v.items()})
B9=B[~B.event.str.startswith('2009_')].copy(); B9['R0']=1-B9.legacy
X['E4_rep_excl_2009_bright']=bright(B9,'Qt')
B09=B[B.event.str.startswith('2009_')].copy(); B09['R0']=1-B09.legacy
X['E4_rep_2009_only_bright']=bright(B09,'Qt')
json.dump(X,open('rep_explore_register_v1.json','w'),ensure_ascii=False,indent=1,default=float)
for k in ('E4_rep_excl_2009_bright','E4_rep_2009_only_bright'):
    b=X[k]; print(k,'B2',[round(b['B2_legacy_by_stratum'][q]['OR'],2) for q in (1,2,3,4,5)],'trend',o(b['B2_trend_per_stratum_step']),'Q5 intact share',b['B1_composition'][5]['intact_share_controls'],b['B1_composition'][5]['intact_share_cases'],
          'B3',{kk:f"{vv['ratio']:.2f} ({vv['ci'][0]:.2f}–{vv['ci'][1]:.2f})" for kk,vv in b['B3_contrasts'].items()})
