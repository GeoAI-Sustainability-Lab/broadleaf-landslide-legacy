# -*- coding: utf-8 -*-
"""Verification runner.

Rebuilds the working folder that the stage scripts expect (verify/_work) from the frozen tables in data/frozen, runs the
stage scripts in their original order and compares every output with the reference registers in registers/.

    python verify/run_verify.py          # quick set
    python verify/run_verify.py --all    # adds the attributable-fraction comparisons (stages/s8) and the event-level
                                         # bootstrap of stage s9 (2,000 resamples of events; about three hours on two cores)

The stage scripts are stored with the placeholder @WORK@ for the working folder; the runner writes a runnable copy of each
script into verify/_work with @WORK@ replaced by the absolute path of verify/_work. Checks marked "unrounded" run the stage
script with Python's built-in round() disabled, which is how the reference registers in registers/unrounded/ were produced.
Each check prints PASS or FAIL with the number of compared values and the largest relative difference. Values must agree
within a relative 1e-6, Monte Carlo intervals (keys ci_mc) within an absolute 0.005."""
import gzip, hashlib, json, math, os, shutil, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = os.path.join(ROOT, "verify", "_work")
FROZEN = os.path.join(ROOT, "data", "frozen")
STAGES = os.path.join(ROOT, "stages")
REG = os.path.join(ROOT, "registers")
TOL = 1e-6          # relative tolerance for every value except Monte Carlo intervals
TOL_MC = 0.005      # absolute tolerance for Monte Carlo intervals (keys ci_mc): the multivariate-normal draws depend on the
                    # linear-algebra library, so these intervals agree within Monte Carlo error rather than to machine precision
UNROUND = ("import builtins as _ur_b\n"
           "def round(x, n=None):\n"
           "    if n is None: return _ur_b.round(x)\n"
           "    try: return float(x)\n"
           "    except TypeError: return x\n")
MANIFEST = json.load(open(os.path.join(ROOT, "data", "frozen_manifest.json"), encoding="utf-8"))


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def unpack(name, *dests):
    """Decompress data/frozen/<name>.gz into each destination (relative to the working folder) and check its md5."""
    for d in dests:
        out = os.path.join(W, d)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with gzip.open(os.path.join(FROZEN, name + ".gz"), "rb") as f, open(out, "wb") as g: shutil.copyfileobj(f, g)
        assert md5_file(out) == MANIFEST[name]["md5"], f"md5 mismatch for {name}"


def place(stage, script, dest_dir, unrounded=False, name=None):
    """Copy a stage script into the working folder with @WORK@ replaced; return its path."""
    t = open(os.path.join(STAGES, stage, script), encoding="utf-8").read().replace("@WORK@", W)
    if unrounded: t = UNROUND + t
    os.makedirs(os.path.join(W, dest_dir), exist_ok=True)
    p = os.path.join(W, dest_dir, name or script)
    open(p, "w", encoding="utf-8").write(t)
    return p


def run(path, cwd, env=None, tag=""):
    t0 = time.time()
    e = dict(os.environ); e.update(env or {})
    r = subprocess.run([sys.executable, path], cwd=os.path.join(W, cwd), capture_output=True, text=True, env=e)
    open(path + tag + ".log", "w", encoding="utf-8").write(r.stdout + "\n--- stderr ---\n" + r.stderr)
    if r.returncode != 0:
        print(r.stderr[-2000:]); raise SystemExit(f"FAILED to run {os.path.relpath(path, W)}")
    return time.time() - t0


def leaves(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items(): yield from leaves(v, f"{p}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o): yield from leaves(v, f"{p}[{i}]")
    else: yield p, o


def compare(out_path, ref_rel, label, secs):
    a = dict(leaves(json.load(open(out_path, encoding="utf-8"))))
    b = dict(leaves(json.load(open(os.path.join(REG, ref_rel), encoding="utf-8"))))
    keys = set(a) | set(b); worst = 0.0; worst_mc = 0.0; bad = []
    for k in keys:
        x, y = a.get(k, "<missing>"), b.get(k, "<missing>")
        if isinstance(x, bool) or isinstance(y, bool) or not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
            if x != y and not (isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y)): bad.append(k)
            continue
        if math.isnan(x) and math.isnan(y): continue
        if "ci_mc" in k:
            dmc = abs(x - y); worst_mc = max(worst_mc, dmc)
            if dmc > TOL_MC: bad.append(k)
            continue
        d = abs(x - y) / max(1.0, abs(y)); worst = max(worst, d)
        if d > TOL: bad.append(k)
    nnum = sum(1 for v in b.values() if isinstance(v, (int, float)) and not isinstance(v, bool))
    status = "PASS" if not bad else "FAIL"
    mc = f"  (Monte Carlo intervals: max abs. diff {worst_mc:.4f})" if worst_mc > 0 else ""
    print(f"{status:4}  {label:58} {nnum:6} values  max rel. diff {worst:.1e}  {secs:6.0f} s{mc}")
    for k in sorted(bad)[:10]: print("      differs:", k, a.get(k, "<missing>"), b.get(k, "<missing>"))
    return not bad


def main(full):
    if os.path.exists(W): shutil.rmtree(W)
    os.makedirs(W)
    unpack("point_dates_v1.csv", "g0/point_dates_v1.csv")
    unpack("catalog_matches_within100m_v1.csv", "g0/catalog_matches_within100m_v1.csv")
    unpack("annual_state_matches_within50m_v1.csv", "g0/annual_state_matches_within50m_v1.csv")
    unpack("control_event_footprint_membership_v1.csv", "g0/control_event_footprint_membership_v1.csv")
    unpack("tags_event_v3.csv", "g1/tags_event_v3.csv")
    unpack("lt_summary_v1.csv", "g1/lt_summary_v1.csv", "v11/unround/g1/lt_summary_v1.csv")
    unpack("lt_fits_all_v1.csv", "v11/unround/g1/lt_fits_all_v1.csv")
    unpack("lt_segments_all_v1.csv", "v11/unround/g1/lt_segments_all_v1.csv")
    unpack("tags_island_v3.csv", "g1/tags_island_v3.csv", "v11/unround/g1/tags_island_v3.csv")
    unpack("analysis_rows_v1.csv", "g3/analysis_rows_v1.csv")
    unpack("bright_rows_v1.csv", "v10/an/g3/bright_rows_v1.csv", "v11/unround/g3/bright_rows_v1.csv")
    unpack("rep_rows_v1.csv", "g5/rep_rows_v1.csv")
    unpack("rep_rows_tagged_v1.csv", "v10/an/g5/rep_rows_tagged_v1.csv", "v11/data/rep_rows_tagged_v1.csv")
    os.makedirs(os.path.join(W, "reference"))
    shutil.copy(os.path.join(REG, "reference_values.json"), os.path.join(W, "reference", "reference_values.json"))
    for n in ("point_dates_v1.csv", "catalog_matches_within100m_v1.csv", "annual_state_matches_within50m_v1.csv",
              "control_event_footprint_membership_v1.csv", "lt_summary_v1.csv", "tags_event_v3.csv"):
        unpack(n, "v27/in/" + n)
    unpack("zoning_out_v1.csv", "g5/zoning_out_v1.csv")
    unpack("s2_pairs_v1.csv", "v11/unround/g1/s2_pairs_v1.csv")
    unpack("rep_rows_tagged_v5.csv", "v27/rep/rep_rows_tagged_v5.csv")
    for f in ("reference_values_s9.json", "gsa_gap_register_v2.json"): shutil.copy(os.path.join(REG, f), os.path.join(W, "reference", f))
    os.makedirs(os.path.join(W, "v11/unround/methods"))
    shutil.copy(os.path.join(REG, "unrounded", "methods_register_v1.json"), os.path.join(W, "v11/unround/methods", "methods_register_v1.json"))
    ok = []

    # 1. event-matched tags rebuilt from the matched layers: tags_event_v4.csv must be byte-identical to the frozen table
    p = place("s2_event_tags", "tag_event_v4.py", "v10/code")
    s = run(p, "v10/code")
    same = md5_file(os.path.join(W, "v10/data/tags_event_v4.csv")) == MANIFEST["tags_event_v4.csv"]["md5"]
    print(f"{'PASS' if same else 'FAIL':4}  {'tags_event_v4.csv rebuilt (byte-identical to data/frozen)':58} {'':6}         {'':17}  {s:6.0f} s")
    ok.append(same)
    ok.append(compare(os.path.join(W, "v10/data/tag_fix_register_v1.json"), "tag_fix_register_v1.json", "tag_fix_register_v1.json", s))
    unpack("tags_event_v4.csv", "v11/data/tags_event_v4.csv")

    # 2. main models and the fresh-scar attributable fractions (unrounded)
    shutil.copy(os.path.join(STAGES, "s1_landtrendr", "lt_segments.py"), os.path.join(W, "v11/unround/g1/lt_segments.py"))
    p = place("s3_models", "final_models_v2.py", "v11/unround/g1", unrounded=True)
    s = run(p, "v11/unround/g1")
    ok.append(compare(os.path.join(W, "v11/unround/g1/final_register_v2.json"), "unrounded/final_register_v2.json", "main models (final_register_v2, unrounded)", s))
    p = place("s3_models", "af_fresh_v1.py", "v11/unround/g1", unrounded=True)
    s = run(p, "v11/unround/g1")
    ok.append(compare(os.path.join(W, "v11/unround/g1/af_fresh_register_v1.json"), "unrounded/af_fresh_register_v1.json", "fresh-scar attributable fractions (unrounded)", s))

    # 3. stand origin and canopy height (as run), slope bands (unrounded), 2009 extreme events (as run)
    for f in ("stand_origin_v1.py", "stand_origin_sens2020_v1.py", "stand_origin_class3_recent_v1.py"):
        place("s5_stand_origin", f, "v10/an/g4")
    s = run(os.path.join(W, "v10/an/g4/stand_origin_v1.py"), "v10/an/g4")
    ok.append(compare(os.path.join(W, "v10/an/g4/stand_origin_register_v1.json"), "stand_origin_register_v1.json", "stand origin and canopy height", s))
    s = run(os.path.join(W, "v10/an/g4/stand_origin_sens2020_v1.py"), "v10/an/g4")
    ok.append(compare(os.path.join(W, "v10/an/g4/stand_origin_sens2020_register_v1.json"), "stand_origin_sens2020_register_v1.json", "stand origin, 2020-2025 events", s))
    s = run(os.path.join(W, "v10/an/g4/stand_origin_class3_recent_v1.py"), "v10/an/g4")
    ok.append(compare(os.path.join(W, "v10/an/g4/stand_origin_class3_recent_register_v1.json"), "stand_origin_class3_recent_register_v1.json", "stand origin, recent canopy disturbance", s))
    os.makedirs(os.path.join(W, "v10/an/slope_disc"), exist_ok=True)
    p = place("s7_extensions", "slope_bands_v2.py", "v10/an/slope_disc", unrounded=True)
    s = run(p, "v10/an/slope_disc")
    ok.append(compare(os.path.join(W, "v10/an/slope_disc/slope_bands_register_v2.json"), "unrounded/slope_bands_register_v2.json", "slope bands (unrounded)", s))
    os.makedirs(os.path.join(W, "v10/an/g6"), exist_ok=True)
    p = place("s7_extensions", "extreme_v1.py", "v10/an/g6")
    s = run(p, "v10/an/g6")
    ok.append(compare(os.path.join(W, "v10/an/g6/extreme_register_v1.json"), "extreme_register_v1.json", "2009 extreme events", s))

    # 4. independent recomputation written from the method description only
    t0 = time.time()
    r = subprocess.run([sys.executable, os.path.join(ROOT, "verify", "independent_recompute.py")], capture_output=True, text=True)
    print(r.stdout.rstrip()); ok.append(r.returncode == 0 and "PASS" in r.stdout)

    # 5. stage s9: one cutoff date per event
    place("s8_attributable_fraction", "afcore_v1.py", "v11/code")
    place("s3_models", "clr_exact.py", "v10/code")
    for f in ("tag_event_v5.py", "lib27.py", "an27_main.py", "an27_rep.py", "an27_rep_b2.py", "counts_s9.py"): place("s9_event_cutoff", f, "v27/code")
    os.makedirs(os.path.join(W, "v27/data"), exist_ok=True); os.makedirs(os.path.join(W, "v27/an"), exist_ok=True)
    s = run(os.path.join(W, "v27/code/tag_event_v5.py"), "v27/code", {"MODE": "own"}, ".own")
    print(f"{'PASS':4}  {'tags_event_v4.csv rebuilt by tag_event_v5.py (MODE=own)':58} {'':6}         {'':17}  {s:6.0f} s")
    s = run(os.path.join(W, "v27/code/tag_event_v5.py"), "v27/code", {"MODE": "common"}, ".common")
    same = md5_file(os.path.join(W, "v27/data/tags_event_v5.csv")) == MANIFEST["tags_event_v5.csv"]["md5"]
    print(f"{'PASS' if same else 'FAIL':4}  {'tags_event_v5.csv rebuilt (byte-identical to data/frozen)':58} {'':6}         {'':17}  {s:6.0f} s")
    ok.append(same)
    ok.append(compare(os.path.join(W, "v27/data/tag_register_v5_common.json"), "tag_register_v5_common.json", "tag_register_v5_common.json", s))
    s = run(os.path.join(W, "v27/code/counts_s9.py"), "v27/code")
    ok.append(compare(os.path.join(W, "v27/an/counts_s9.json"), "counts_s9.json", "counts_s9.json", s))
    # the earlier cutoff rule (TAGS=v4) without bootstrap: the scripts assert the registered values of stages s3 to s8
    for f, lab in (("an27_main.py", "2018-2025 events, earlier cutoff rule"), ("an27_rep.py", "both periods, earlier cutoff rule"),
                   ("an27_rep_b2.py", "susceptibility quintiles, earlier cutoff rule")):
        s = run(os.path.join(W, "v27/code", f), "v27/code", {"TAGS": "v4", "BOOT": "0"}, ".v4")
        reg = f.replace("an27_", "").replace(".py", "").replace("main", "main27").replace("rep", "rep27") + "_v4.json"
        ok.append(compare(os.path.join(W, "v27/an", reg), reg, lab, s))

    if full:
        # 6. stage s9 with one cutoff date per event and the event-level bootstrap (long)
        for f, lab in (("an27_main.py", "2018-2025 events, one cutoff per event, bootstrap"), ("an27_rep.py", "both periods, one cutoff per event, bootstrap"),
                       ("an27_rep_b2.py", "susceptibility quintiles, one cutoff per event")):
            s = run(os.path.join(W, "v27/code", f), "v27/code", {"TAGS": "v5", "BOOT": "1"}, ".v5")
            reg = f.replace("an27_", "").replace(".py", "").replace("main", "main27").replace("rep", "rep27") + "_v5.json"
            ok.append(compare(os.path.join(W, "v27/an", reg), reg, lab, s))
        # 7. attributable-fraction comparisons (event bootstrap; long)
        place("s3_models", "clr_exact.py", "v11/code")
        os.makedirs(os.path.join(W, "v11/an"), exist_ok=True)
        for f, reg, lab in [("af_robust_v2.py", "af_robust_register_v2.json", "attributable fractions: radii, estimators, bootstrap"),
                            ("af_rings_v3.py", "af_rings_register_v3.json", "attributable fractions by distance ring"),
                            ("af_window_v2.py", "af_window_register_v2.json", "attributable fractions with matched look-back windows"),
                            ("boot_event_draws_v1.py", "boot_event_draws_register_v1.json", "event-bootstrap draws"),
                            ("rep2009_age_rows_v1.py", "rep2009_age_rows_register_v1.json", "scar-age rows of the 2009 events"),
                            ("rep_event_share_v1.py", "rep_event_share_v1.json", "replication cases by event")]:
            p = place("s8_attributable_fraction", f, "v11/code")
            s = run(p, "v11/code")
            ok.append(compare(os.path.join(W, "v11/an", reg), reg, lab, s))
    print(("ALL PASS" if all(ok) else "SOME CHECKS FAILED") + f" ({sum(ok)} of {len(ok)})")
    return all(ok)


if __name__ == "__main__":
    sys.exit(0 if main("--all" in sys.argv) else 1)
