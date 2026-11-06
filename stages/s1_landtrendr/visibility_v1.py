"""Landsat visibility diagnostics by slope, aspect, elevation and forest type."""
import glob, json, sys, numpy as np, pandas as pd
sys.path.insert(0,'@WORK@/pack/擾動史研究_資料啟動包_v1/code'); import palette as P
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
cjk=[f.name for f in fm.fontManager.ttflist if 'Noto Sans CJK' in f.name][0]
plt.rcParams.update({'font.family':[cjk,'DejaVu Sans'],'font.size':8.3,'axes.titlesize':9,'axes.labelsize':8.5,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7.2})
G='@WORK@/g0/'
N=pd.concat([pd.read_csv(f,dtype={'uid':str}) for f in sorted(glob.glob('nobs_batches/nobs_*.csv'))],ignore_index=True).drop_duplicates('uid')
T=pd.read_csv(G+'tags_island_v2.csv',dtype={'uid':str},low_memory=False)[['uid','kind','ft','slope','north','elev','site_class']]
D=T.merge(N,on='uid',how='left'); assert D.nobs_2000.notna().all(), 'missing points in stack'
nobs=[f'nobs_{y}' for y in range(1988,2025)]; M=D[nobs].values
D['obs_mean']=M.mean(1); D['years_lt3']=(M<3).sum(1); D['years_0']=(M==0).sum(1)
D['band']=pd.cut(D.slope,[-0.1,10,20,30,39.5,90],labels=['<10','10-20','20-30','30-39.5','>=39.5']); D['asp']=np.where(D.north>0.5,'N',np.where(D.north<-0.5,'S','EW')); D['elevband']=pd.cut(D.elev,[-1,500,1000,1500,4000],labels=['<500','500-1000','1000-1500','>1500'])
s2=[f's2obs_{y}' for y in range(2017,2026)]; D['s2_mean_2019_2025']=D[[f's2obs_{y}' for y in range(2019,2026)]].mean(1)
D.to_csv('visibility_points_v1.csv',index=False)
R={'n_points':int(len(D)),'point_years':int(M.size),'share_point_years_0':round(float((M==0).mean()*100),2),'share_point_years_lt3':round(float((M<3).mean()*100),2),
   'median_obs_by_year':{int(y):float(v) for y,v in zip(range(1988,2025),np.median(M,axis=0))},
   'median_s2_by_year':{int(y):float(v) for y,v in zip(range(2017,2026),np.median(D[s2].values,axis=0))}}
def tab(keys):
    g=D.groupby(keys,observed=True).agg(n=('uid','size'),obs_mean=('obs_mean','mean'),years_lt3=('years_lt3','mean'),years_0=('years_0','mean'),s2_mean=('s2_mean_2019_2025','mean')).round(2)
    return g.reset_index()
for keys,name in [(['asp'],'by_aspect'),(['band'],'by_slope_band'),(['elevband'],'by_elevation'),(['kind'],'by_kind'),(['ft'],'by_forest_type'),(['band','asp'],'by_band_aspect'),(['kind','asp'],'by_kind_aspect')]:
    t=tab(keys); t.to_csv(f'visibility_{name}_v1.csv',index=False); R[name]=t.to_dict(orient='records')
B=D[D.ft.isin(['闊葉樹林型','竹林'])]
R['broadleaf_bamboo_by_kind_asp']=B.groupby(['ft','kind','asp']).agg(n=('uid','size'),obs_mean=('obs_mean','mean'),years_lt3=('years_lt3','mean')).round(2).reset_index().to_dict(orient='records')
json.dump(R,open('visibility_register_v1.json','w'),ensure_ascii=False,indent=1)
fig,ax=plt.subplots(1,3,figsize=(183/25.4,64/25.4))
yrs=list(range(1988,2025)); ax[0].plot(yrs,[R['median_obs_by_year'][y] for y in yrs],'-o',ms=2.5,color='#0072B2'); ax[0].fill_between(yrs,np.percentile(M,25,axis=0),np.percentile(M,75,axis=0),color='#0072B2',alpha=0.2)
ax[0].set_xlabel('年'); ax[0].set_ylabel('每點每年有效觀測數'); ax[0].set_title('(a) 逐年中位數與四分位距',loc='left',fontsize=7.8)
for a,(key,lab,pl) in zip(ax[1:],[('asp','坡向','(b)'),('band','坡度帶 (°)','(c)')]):
    t=D.groupby([key,'kind'],observed=True).obs_mean.mean().unstack()[['case','control']]
    x=np.arange(len(t)); a.bar(x-0.2,t['case'],0.4,color=P.BROAD,label='病例'); a.bar(x+0.2,t['control'],0.4,color='#C9A98F',label='對照')
    a.set_xticks(x); a.set_xticklabels([{'N':'北向','S':'南向','EW':'東西向'}.get(str(i),str(i)) for i in t.index]); a.set_ylabel('1988–2024 平均年有效觀測數'); a.set_xlabel(lab); a.set_title(f'{pl} 依{lab.split(" ")[0]}與病例／對照',loc='left',fontsize=7.8); a.set_ylim(0,15)
ax[1].legend(frameon=False,loc='upper right',ncol=2)
fig.suptitle('圖 12（真實資料）　偵測可見度診斷：20,585 點的 Landsat 生長季（4–10 月）有效觀測數',x=0.01,ha='left',fontsize=8.4)
fig.tight_layout(); fig.savefig('F12_visibility_v1.png',dpi=400); fig.savefig('F12_visibility_v1.pdf')
print(json.dumps({k:R[k] for k in ['n_points','share_point_years_0','share_point_years_lt3']})); print(pd.DataFrame(R['by_aspect'])); print(pd.DataFrame(R['by_slope_band'])); print(pd.DataFrame(R['by_elevation'])); print(pd.DataFrame(R['by_kind']))
