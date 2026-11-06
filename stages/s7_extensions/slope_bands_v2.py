"""Legacy odds ratios by slope band."""
import json
H='@WORK@'
src=open(H+'/v10/an/g4/stand_origin_v1.py',encoding='utf-8').read()
exec(compile(src[:src.index('rng=np.random.default_rng(SEED)')],'stand_origin_v1_defs','exec'))
SEED_BANDS=20261011
rng=np.random.default_rng(SEED_BANDS)
BANDS=[('0-20',None,20.0),('20-30',20.0,30.0),('30-39.5',30.0,39.5),('39.5-50',39.5,50.0),('50-90',50.0,90.0)]
out={'label':'exploratory slope-band check, written 2026-10-11 after all charter results had been seen; not in any charter',
     'source_script':'slope_bands_v2.py','seed':SEED_BANDS,'ndraw':NDRAW,'bands_include_upper_bound':True,'results':{},'joint_all_slopes':{}}
def block(d,lab):
    a=af(d,['sc1','sc2','sc3'],rng)
    m,mr,dd,nd=fitfe(d,f"case~{TERR}+legacy")
    m3,mr3,_,_=fitfe(d,f"case~{TERR}+sc1+sc2+sc3")
    cases=dd[dd.case==1]; ctrl=dd[dd.case==0]
    o={'n_cases':int(len(cases)),'n_control_rows':int(len(ctrl)),'n_events':int(dd.event.nunique()),'events_dropped':nd,
       'legacy_share_cases':round(float(cases.legacy.mean()),4),'legacy_share_controls':round(float(ctrl.legacy.mean()),4),
       'OR_any_legacy':lin(m,mr,{'legacy':1}),'OR':{k:a['OR'][k] for k in ('sc1','sc2','sc3')},'rho_cases':a['rho_cases'],
       'bare_vs_scar':lin(m3,mr3,{'sc2':1,'sc1':-1}),
       'AF_total':a['AF_total'],'AF_by_class':{k:a[f'AF_{k}'] for k in ('sc1','sc2','sc3')}}
    assert abs(m3.params['sc2']-math.log(a['OR']['sc2']['OR']))<1e-3
    o['AF_total_point_6dp']=round(sum(float(cases[t].mean())*(1-1/a['OR'][t]['OR']) for t in ('sc1','sc2','sc3')),6)
    out['results'][lab]=o
    print(f"{lab:9s} events {o['n_events']:2d} cases {o['n_cases']:5d} ctrl {o['n_control_rows']:6d} | any {o['OR_any_legacy']['OR']:.2f} {o['OR_any_legacy']['ci']}"
          f" | sc1 {o['OR']['sc1']['OR']:.2f} sc2 {o['OR']['sc2']['OR']:.2f} sc3 {o['OR']['sc3']['OR']:.2f} | bare/scar {o['bare_vs_scar']['OR']:.2f} {o['bare_vs_scar']['ci']}"
          f" | AF {o['AF_total']['point']*100:.1f}% {[round(x*100,1) for x in o['AF_total']['ci_mc']]}")
block(EV,'all')
block(EV[EV.slope<=39.5],'le39.5')
block(EV[EV.slope>39.5],'gt39.5')
for lab,lo,hi in BANDS:
    block(EV[EV.slope<=hi] if lo is None else EV[(EV.slope>lo)&(EV.slope<=hi)],lab)
J=EV.copy(); J['above']=(J.slope>39.5).astype(int); J['hinge']=np.clip(J.slope-39.5,0,None)
m,mr,dd,nd=fitfe(J,f"case~{TERR}+legacy+legacy:s20+above+legacy:above")
out['joint_all_slopes']['J1_step']={'legacy_x_s20_per_degree':lin(m,mr,{'legacy:s20':1}),'legacy_x_above_step':lin(m,mr,{'legacy:above':1}),
                                     'legacy_at_20':lin(m,mr,{'legacy':1}),'n':int(len(dd)),'n_case':int(dd.case.sum()),'events_dropped':nd}
m,mr,dd,nd=fitfe(J,f"case~{TERR}+legacy+legacy:s20+hinge+legacy:hinge")
out['joint_all_slopes']['J2_hinge']={'legacy_x_s20_per_degree':lin(m,mr,{'legacy:s20':1}),'legacy_x_hinge_per_degree':lin(m,mr,{'legacy:hinge':1}),
                                      'n':int(len(dd)),'n_case':int(dd.case.sum()),'events_dropped':nd}
m,mr,dd,nd=fitfe(J,f"case~{TERR}+sc1+sc2+sc3+above+sc1:above+sc2:above+sc3:above")
out['joint_all_slopes']['J3_class']={'bare_vs_scar_le39_5':lin(m,mr,{'sc2':1,'sc1':-1}),
                                      'bare_vs_scar_gt39_5':lin(m,mr,{'sc2':1,'sc2:above':1,'sc1':-1,'sc1:above':-1}),
                                      'bare_vs_scar_change_above':lin(m,mr,{'sc2:above':1,'sc1:above':-1}),
                                      'sc_above_change':{f'sc{c}':lin(m,mr,{f'sc{c}:above':1}) for c in (1,2,3)},
                                      'n':int(len(dd)),'n_case':int(dd.case.sum()),'events_dropped':nd}
for k,v in out['joint_all_slopes'].items():
    print(k,{kk:(vv['OR'],vv['ci']) for kk,vv in v.items() if isinstance(vv,dict) and 'OR' in vv})
print('J3 class x above',{k:(v['OR'],v['ci']) for k,v in out['joint_all_slopes']['J3_class']['sc_above_change'].items()})
json.dump(out,open(H+'/v10/an/slope_disc/slope_bands_register_v2.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
print('written slope_disc/slope_bands_register_v2.json')
