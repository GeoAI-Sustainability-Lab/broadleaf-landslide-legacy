# Stage scripts

The scripts are those that produced the registers, in run order. Module docstrings and comments were replaced by a one-line
description; code is unchanged except for the folder roots, written as placeholders:

- `@WORK@` the working folder; `verify/run_verify.py` rebuilds it from `data/frozen` and substitutes the placeholder;
- `@SOURCE@` the folder of the source layers as downloaded from the providers (see the main README);
- `@BAMBOO_REPO@` a clone of https://github.com/GeoAI-Sustainability-Lab/bamboo-landslide-terrain (event rainfall table).

Paths below `@WORK@` follow the original working layout: `g0` matching tables, `g1` tags and LandTrendr tables, `g3` per-point
attributes, `g5` replication rows, `v10/data` event-matched tags, `v10/an/<stage>` and `v11/an` stage outputs.

| Folder | Stage | Scripts |
|---|---|---|
| `s0_data_contract` | Matching of every point with the inventory polygons (within 100 m) and annual-map polygons (within 50 m), cutoff dates, event footprints. Reads the source layers. | `assemble_v1.py`, `bbox_scan.py`, `gate0_scan.py`, `run_catalogs.sh`, `run_state.sh`, `tag_v1.py` |
| `s1_landtrendr` | Landsat observation counts, LandTrendr fits and segments, threshold calibration, class 3 tags, Sentinel-2 check. Earth Engine. | `calibrate_v1.py`, `class3_tag_v1.py`, `gee_common.py`, `lt_segments.py`, `pilot_lt_v1.py`, `pilot_v1.py`, `run_lt_all.py`, `run_nobs_all.py`, `run_s2_all.py`, `s2_check_v1.py`, `visibility_v1.py` |
| `s2_event_tags` | Event-matched legacy tags evaluated per point and event (`tags_event_v4.csv`). | `tag_event_v4.py` |
| `s3_models` | Main event-matched and island models, attributable fractions, robustness. | `af_fresh_v1.py`, `clr_exact.py`, `final_models_v2.py`, `robust_v1.py` |
| `s4_unfailed_high_risk` | Expected-risk model and the composition of failed and unfailed sites. | `bright_spots_v1.py` |
| `s5_stand_origin` | Stand origin, canopy height and the bamboo contrast. | `canopy_inscar_v1.py`, `stand_origin_class3_recent_v1.py`, `stand_origin_sens2020_v1.py`, `stand_origin_v1.py` |
| `s6_replication` | Independent replication on 2009-2017 events: sample build (source layers, Earth Engine) and analysis. | `build_catalog_v1.py`, `build_rows_v1.py`, `label_forest_type_v1.py`, `rep_analysis_v1.py`, `rep_explore_v1.py`, `run_lt_rep.py`, `terrain_v1.py`, `zoning_v1.py` |
| `s7_extensions` | 2009 extreme events, slope bands, geologically sensitive areas, comparison methods. | `af_log_interval_v2.py`, `extreme_v1.py`, `gsa_gap_v2.py`, `methods_cmp_v1.py`, `methods_cmp_v1b.py`, `slope_bands_v2.py` |
| `s8_attributable_fraction` | Attributable-fraction comparisons: look-back windows, radii, estimators, distance rings, event bootstrap. | `af_rings_v3.py`, `af_robust_v2.py`, `af_window_v2.py`, `afcore_v1.py`, `boot_event_draws_v1.py`, `rep2009_age_rows_v1.py`, `rep_event_share_v1.py` |

Original run order after the event tags (`s2_event_tags/tag_event_v4.py`): `s3_models` final_models_v2, af_fresh_v1, robust_v1;
`s1_landtrendr/s2_check_v1`; `s4_unfailed_high_risk/bright_spots_v1`; `s5_stand_origin` stand_origin_v1, stand_origin_sens2020_v1,
stand_origin_class3_recent_v1; `s6_replication` rep_analysis_v1, rep_explore_v1; `s7_extensions` extreme_v1, slope_bands_v2, then
the remaining `s7_extensions` and `s8_attributable_fraction` scripts.
