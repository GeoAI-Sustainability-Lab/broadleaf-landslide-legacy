# Registers

Reference values written by the stage scripts. Files in `unrounded/` were produced with Python's built-in round() disabled. The registers of stage s9 are written without rounding.

| Register | Written by | Checked by verify/run_verify.py |
|---|---|---|
| `af_rings_register_v3.json` | s8_attributable_fraction/af_rings_v3.py (--all) | yes |
| `af_robust_register_v2.json` | s8_attributable_fraction/af_robust_v2.py (--all) | yes |
| `af_window_register_v2.json` | s8_attributable_fraction/af_window_v2.py (--all) | yes |
| `analysis_plan_checksums.json` | SHA-256 checksums, dates and sizes of the analysis plan and its three supplements (the files are not part of the repository) | no |
| `assembly_register_v1.json` | s0_data_contract/assemble_v1.py | no |
| `boot_event_draws_register_v1.json` | s8_attributable_fraction/boot_event_draws_v1.py (--all) | yes |
| `calibration_register_v1.json` | s1_landtrendr/calibrate_v1.py | no |
| `canopy_inscar_register_v1.json` | s5_stand_origin/canopy_inscar_v1.py | no |
| `class3_register_v1.json` | s1_landtrendr/class3_tag_v1.py | no |
| `counts_s9.json` | s9_event_cutoff/counts_s9.py | yes |
| `events_v1.csv` | s3_models/robust_v1.py | no |
| `extreme_register_v1.json` | s7_extensions/extreme_v1.py | yes |
| `gsa_gap_register_v2.json` | s7_extensions/gsa_gap_v2.py | no |
| `loeo_v1.csv` | s3_models/robust_v1.py | no |
| `main27_v4.json` | s9_event_cutoff/an27_main.py (TAGS=v4, BOOT=0) | yes |
| `main27_v5.json` | s9_event_cutoff/an27_main.py (TAGS=v5, BOOT=1; --all) | yes |
| `reference_values.json` | values that the attributable-fraction scripts assert against (from the replication and robustness registers) | no |
| `reference_values_s9.json` | values of stages s3 to s8 that the scripts of stage s9 assert against when run with TAGS=v4 | no |
| `rep2009_age_rows_register_v1.json` | s8_attributable_fraction/rep2009_age_rows_v1.py (--all) | yes |
| `rep27_b2_v4.json` | s9_event_cutoff/an27_rep_b2.py (TAGS=v4) | yes |
| `rep27_b2_v5.json` | s9_event_cutoff/an27_rep_b2.py (TAGS=v5; --all) | yes |
| `rep27_v4.json` | s9_event_cutoff/an27_rep.py (TAGS=v4, BOOT=0) | yes |
| `rep27_v5.json` | s9_event_cutoff/an27_rep.py (TAGS=v5, BOOT=1; --all) | yes |
| `rep_build_register_v1.json` | s6_replication/build_rows_v1.py | no |
| `rep_event_share_v1.json` | s8_attributable_fraction/rep_event_share_v1.py (--all) | yes |
| `rep_explore_register_v1.json` | s6_replication/rep_explore_v1.py | no |
| `rep_rows_v5_register.json` | s9_event_cutoff/rep_rows_v5.py (source layers) | no |
| `replication_register_v1.json` | s6_replication/rep_analysis_v1.py | no |
| `robust_register_v1.json` | s3_models/robust_v1.py | no |
| `spec_curve_v1.csv` | s3_models/robust_v1.py | no |
| `stand_origin_class3_recent_register_v1.json` | s5_stand_origin/stand_origin_class3_recent_v1.py | yes |
| `stand_origin_register_v1.json` | s5_stand_origin/stand_origin_v1.py | yes |
| `stand_origin_sens2020_register_v1.json` | s5_stand_origin/stand_origin_sens2020_v1.py | yes |
| `tag_fix_register_v1.json` | s2_event_tags/tag_event_v4.py | yes |
| `tag_register_v1.json` | s0_data_contract/tag_v1.py | no |
| `tag_register_v5_common.json` | s9_event_cutoff/tag_event_v5.py (MODE=common) | yes |
| `terrain_check_v1.json` | s6_replication/terrain_v1.py | no |
| `visibility_register_v1.json` | s1_landtrendr/visibility_v1.py | no |
| `unrounded/af_fresh_register_v1.json` | s3_models/af_fresh_v1.py | yes |
| `unrounded/af_log_interval_register_v2.json` | s7_extensions/af_log_interval_v2.py | no |
| `unrounded/bright_register_v1.json` | s4_unfailed_high_risk/bright_spots_v1.py | no |
| `unrounded/final_register_v2.json` | s3_models/final_models_v2.py | yes |
| `unrounded/methods_register_v1.json` | s7_extensions/methods_cmp_v1.py, methods_cmp_v1b.py | no |
| `unrounded/robust_af_loeo_v1.json` | s3_models/robust_v1.py (attributable fractions and leave-one-event-out, unrounded) | no |
| `unrounded/s2_register_v1.json` | s1_landtrendr/s2_check_v1.py | no |
| `unrounded/slope_bands_register_v2.json` | s7_extensions/slope_bands_v2.py | yes |

In the registers of stage s9 the key `ci_boot` holds the 95% percentile interval of the event-level bootstrap, `ci` the cluster-robust interval, `ci_mc` the Monte Carlo interval, `boot` the number of usable resamples (`n_ok`) and of resamples skipped because a model could not be fitted (`fails`), and `fallback` the models refitted with BFGS after Newton iterations did not converge.
