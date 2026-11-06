"""First legacy tagging (site classes 0-2) for the island and event-matched designs; superseded for the event design by stages/s2_event_tags/tag_event_v4.py."""
import pandas as pd, numpy as np, json, os
OUT=os.path.expanduser('@WORK@/g0/')
D=pd.read_csv(OUT+'point_dates_v1.csv',dtype={'uid':str,'lid':str},parse_dates=['own_before','own_before_max','own_after','cutoff_isl','isl_cutoff'])
cat=pd.read_csv(OUT+'catalog_matches_within100m_v1.csv',dtype={'uid':str,'record_id':str},parse_dates=['before','before_max','after','after_min'])
st=pd.read_csv(OUT+'annual_state_matches_within50m_v1.csv',dtype={'uid':str},parse_dates=['img_date'])
own_ids=set((D[D.kind=='case'].assign(rid=lambda d: d.lid.str.split('_').str[0]+':'+d.own_record_index.astype(int).astype(str))).rid)
cat['is_own']=cat.record_id.isin(own_ids)&(cat.uid.str.startswith('case:'))&(cat.uid.str[5:]==cat.cat_year.astype(str)+'_'+cat.record_index.astype(str))
def tag(points_cut, label):
    pc=points_cut.copy(); pc['Yc']=pc.cutoff.dt.year
    m=cat.merge(pc[['uid','cutoff','Yc']],on='uid'); m=m[(~m.is_own)&m.after.notna()&(m.after<=m.cutoff)]
    m['yrs_since']=(m.cutoff-m.after).dt.days/365.25
    out={}
    for B,name in [(0,'0m'),(50,'50m'),(100,'100m')]:
        mm=m[(m.inside==1)] if B==0 else m[m.distance_m<=B]
        g=mm.groupby('uid').agg(n_rec=('record_id','size'),n_years=('cat_year','nunique'),last_after=('after','max'),first_after=('after','min'),min_yrs=('yrs_since','min'),max_yrs=('yrs_since','max'),max_area_ha=('Area_ha','max'))
        out[name]=g
    s=st.merge(pc[['uid','cutoff','Yc']],on='uid'); s=s[s.img_date.notna()&(s.img_date<=s.cutoff)&(s.state_year>=s.Yc-1)]
    sb={}
    for B,name in [(0,'0m'),(50,'50m')]:
        ss=s[s.inside==1] if B==0 else s
        sb[name]=ss.groupby('uid').agg(bare_years=('state_year','nunique'),bare_last_img=('img_date','max'),bare_area_m2=('Area','max'))
    s2=st.merge(pc[['uid','cutoff','Yc']],on='uid'); s2=s2[s2.img_date.notna()&(s2.img_date<=s2.cutoff)&(s2.inside==1)]
    ever=s2.groupby('uid').agg(bare_ever_years=('state_year','nunique'),bare_ever_first=('state_year','min'),bare_ever_last=('state_year','max'))
    R=pc.copy()
    for name,g in out.items():
        R=R.merge(g.add_prefix(f'scar_{name}_'),left_on='uid',right_index=True,how='left'); R[f'class1_{name}']=R[f'scar_{name}_n_rec'].notna().astype(int)
    for name,g in sb.items():
        R=R.merge(g.add_prefix(f'bare_{name}_'),left_on='uid',right_index=True,how='left'); R[f'bare_at_cutoff_{name}']=R[f'bare_{name}_bare_years'].notna().astype(int)
    R=R.merge(ever,left_on='uid',right_index=True,how='left'); R['bare_ever']=R.bare_ever_years.notna().astype(int)
    R['design']=label
    return R
isl=tag(D[['uid','kind','ft','slope','year','event','cutoff_isl']].rename(columns={'cutoff_isl':'cutoff'}),'island')
isl.to_csv(OUT+'tags_island_v1.csv',index=False,encoding='utf-8-sig')
EM=pd.read_csv(OUT+'control_event_footprint_membership_v1.csv',dtype={'uid':str},parse_dates=['cutoff_event_min','cutoff_event_max'])
ctrl=EM.merge(D[['uid','kind','ft','slope']],on='uid').rename(columns={'cutoff_event_min':'cutoff'}); ctrl['year']=ctrl.year.astype(str)
cas=D[D.kind=='case'][['uid','kind','ft','slope','year','event','own_before']].rename(columns={'own_before':'cutoff'}); cas['year']=cas.year.astype(str)
ev=tag(pd.concat([cas,ctrl[['uid','kind','ft','slope','year','event','cutoff']]],ignore_index=True),'event')
ev.to_csv(OUT+'tags_event_v1.csv',index=False,encoding='utf-8-sig')
bands=[-0.1,10,20,30,39.5,90]; lab=['<10','10-20','20-30','30-39.5','>=39.5']
isl['band']=pd.cut(isl.slope,bands,labels=lab)
def summ(df):
    g=df.groupby(['ft','kind','band'],observed=True).agg(n=('uid','size'),class1_0m=('class1_0m','sum'),class1_50m=('class1_50m','sum'),class1_100m=('class1_100m','sum'),bare0=('bare_at_cutoff_0m','sum'),bare50=('bare_at_cutoff_50m','sum'),bare_ever=('bare_ever','sum'))
    for c in ['class1_0m','class1_50m','class1_100m','bare0','bare50','bare_ever']: g[c+'_pct']=(100*g[c]/g.n).round(1)
    return g.reset_index()
S=summ(isl); S.to_csv(OUT+'summary_island_by_ft_band_v1.csv',index=False,encoding='utf-8-sig')
print(S[S.ft.isin(['闊葉樹林型','竹林'])].to_string())
b=isl[isl.ft=='闊葉樹林型']
ct=pd.crosstab([b.kind,b.class1_0m],b.bare_at_cutoff_0m); print(ct)
bb=b[b.class1_0m==1]; print(bb.groupby('kind').scar_0m_min_yrs.describe())
P=pd.read_csv('@WORK@/pack/data/cases_annot.csv',dtype={'lid':str})[['lid','old_scar']]; P['uid']='case:'+P.lid
cmp=isl.merge(P[['uid','old_scar']],on='uid')
sub=cat[(cat.cat_year<=2017)&(cat.inside==1)].groupby('uid').size()
cmp['inside_2004_2017_any']=cmp.uid.isin(sub.index).astype(int)
print(pd.crosstab(cmp.old_scar,cmp.inside_2004_2017_any))
json.dump({'island_rows':int(len(isl)),'event_rows':int(len(ev)),'event_cases':int((ev.kind=='case').sum()),'event_control_rows':int((ev.kind=='control').sum()),
           'broadleaf_cases_class1_0m':int(((isl.ft=='闊葉樹林型')&(isl.kind=='case')&(isl.class1_0m==1)).sum()),
           'broadleaf_controls_class1_0m':int(((isl.ft=='闊葉樹林型')&(isl.kind=='control')&(isl.class1_0m==1)).sum()),
           'broadleaf_cases_bare0':int(((isl.ft=='闊葉樹林型')&(isl.kind=='case')&(isl.bare_at_cutoff_0m==1)).sum()),
           'broadleaf_controls_bare0':int(((isl.ft=='闊葉樹林型')&(isl.kind=='control')&(isl.bare_at_cutoff_0m==1)).sum()),
           'old_scar_vs_inside_2004_2017_crosstab':pd.crosstab(cmp.old_scar,cmp.inside_2004_2017_any).to_dict()},open(OUT+'tag_register_v1.json','w'),ensure_ascii=False,indent=1,default=str)
