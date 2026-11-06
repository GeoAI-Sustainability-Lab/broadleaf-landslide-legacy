# -*- coding: utf-8 -*-
"""Independent recomputation of the main event-matched estimates, written from the method description only.

Reads data/frozen/tags_event_v4.csv.gz, keeps broadleaf rows, fits a logistic regression of case status on the three legacy
classes (class 0, intact stand, is the reference), the seven terrain terms (slope, slope squared, the two aspect components,
plan and profile curvature, elevation) and event fixed effects, with standard errors clustered by point, after dropping events
without outcome variation. Bruzzi attributable fractions are the sum over classes of the case share times (1 - 1/OR).
The results are compared with registers/unrounded/final_register_v2.json (odds ratios and cluster-robust intervals) and
registers/unrounded/robust_af_loeo_v1.json (attributable fractions). Prints PASS when every value agrees within 1e-6."""
import json, math, os
import numpy as np, pandas as pd, statsmodels.formula.api as smf
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
t = pd.read_csv(os.path.join(ROOT, "data", "frozen", "tags_event_v4.csv.gz"), encoding="utf-8-sig", low_memory=False)
b = t[t.ft == "闊葉樹林型"].copy()
for k in range(4): b[f"c{k}"] = (b.site_class4 == k).astype(int)
TERR = "slope+slope2+north+east+planc+profc+elev"


def fit(d):
    g = d.groupby("event").case.agg(["sum", "size"]); keep = g[(g["sum"] > 0) & (g["sum"] < g["size"])].index
    d = d[d.event.isin(keep)].copy()
    m = smf.logit(f"case ~ c1+c2+c3 + {TERR} + C(event)", data=d).fit(disp=0, maxiter=200, method="newton",
                                                                  cov_type="cluster", cov_kwds={"groups": pd.factorize(d.uid)[0]})
    return m, d


FR = json.load(open(os.path.join(ROOT, "registers", "unrounded", "final_register_v2.json"), encoding="utf-8"))["models"]
AF = json.load(open(os.path.join(ROOT, "registers", "unrounded", "robust_af_loeo_v1.json"), encoding="utf-8"))["attributable_fraction"]
worst = 0.0; n = 0
for lab, d, mkey, akey in (("slopes <= 39.5 deg", b[b.slope <= 39.5], "M2c_event_le39.5", "event_le39.5"), ("all slopes", b, "M2c_event", "event_all_slopes")):
    m, dd = fit(d)
    cs = dd[dd.case == 1]; af_total = 0.0
    for k, sc in (("c1", "sc1"), ("c2", "sc2"), ("c3", "sc3")):
        orr = math.exp(m.params[k]); lo, hi = math.exp(m.params[k] - 1.96 * m.bse[k]), math.exp(m.params[k] + 1.96 * m.bse[k])
        ref = FR[mkey]["terms"][sc]
        for x, y in ((orr, ref["OR"]), (lo, ref["ci_cluster"][0]), (hi, ref["ci_cluster"][1])):
            worst = max(worst, abs(x - y) / max(1.0, abs(y))); n += 1
        af_total += float(cs[k].mean()) * (1 - 1 / orr)
        print(f"  {lab:18} class {k[1]}: OR {orr:.4f} ({lo:.4f}-{hi:.4f})")
    ref_af = AF[akey]["AF_total"]["point"]
    worst = max(worst, abs(af_total - ref_af)); n += 1
    print(f"  {lab:18} attributable fraction {af_total:.6f} (register {ref_af:.6f})")
status = "PASS" if worst <= 1e-6 else "FAIL"
print(f"{status:4}  {'independent recomputation (event fixed effects, Bruzzi AF)':58} {n:6} values  max rel. diff {worst:.1e}")
raise SystemExit(0 if status == "PASS" else 1)
