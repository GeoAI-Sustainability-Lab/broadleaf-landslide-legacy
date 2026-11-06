# Registers

Reference values written by the stage scripts. Files in `unrounded/` were produced with Python's built-in round() disabled.

| Register | Written by | Checked by verify/run_verify.py |
|---|---|---|
| `af_rings_register_v3.json` | s8_attributable_fraction/af_rings_v3.py (--all) | yes |
| `af_robust_register_v2.json` | s8_attributable_fraction/af_robust_v2.py (--all) | yes |
| `af_window_register_v2.json` | s8_attributable_fraction/af_window_v2.py (--all) | yes |
| `assembly_register_v1.json` | s0_data_contract/assemble_v1.py | no |
| `boot_event_draws_register_v1.json` | s8_attributable_fraction/boot_event_draws_v1.py (--all) | yes |
| `calibration_register_v1.json` | s1_landtrendr/calibrate_v1.py | no |
| `canopy_inscar_register_v1.json` | s5_stand_origin/canopy_inscar_v1.py | no |
| `class3_register_v1.json` | s1_landtrendr/class3_tag_v1.py | no |
| `events_v1.csv` | s3_models/robust_v1.py | no |
| `extreme_register_v1.json` | s7_extensions/extreme_v1.py | yes |
| `gsa_gap_register_v2.json` | s7_extensions/gsa_gap_v2.py | no |
| `loeo_v1.csv` | s3_models/robust_v1.py | no |
| `reference_values.json` | values that the attributable-fraction scripts assert against (from the replication and robustness registers) | no |
| `rep2009_age_rows_register_v1.json` | s8_attributable_fraction/rep2009_age_rows_v1.py (--all) | yes |
| `rep_build_register_v1.json` | s6_replication/build_rows_v1.py | no |
| `rep_event_share_v1.json` | s8_attributable_fraction/rep_event_share_v1.py (--all) | yes |
| `rep_explore_register_v1.json` | s6_replication/rep_explore_v1.py | no |
| `replication_register_v1.json` | s6_replication/rep_analysis_v1.py | no |
| `robust_register_v1.json` | s3_models/robust_v1.py | no |
| `spec_curve_v1.csv` | s3_models/robust_v1.py | no |
| `stand_origin_class3_recent_register_v1.json` | s5_stand_origin/stand_origin_class3_recent_v1.py | yes |
| `stand_origin_register_v1.json` | s5_stand_origin/stand_origin_v1.py | yes |
| `stand_origin_sens2020_register_v1.json` | s5_stand_origin/stand_origin_sens2020_v1.py | yes |
| `tag_fix_register_v1.json` | s2_event_tags/tag_event_v4.py | yes |
| `tag_register_v1.json` | s0_data_contract/tag_v1.py | no |
| `terrain_check_v1.json` | s6_replication/terrain_v1.py | no |
| `unrounded/af_fresh_register_v1.json` | s3_models/af_fresh_v1.py | yes |
| `unrounded/af_log_interval_register_v2.json` | s7_extensions/af_log_interval_v2.py | no |
| `unrounded/bright_register_v1.json` | s4_unfailed_high_risk/bright_spots_v1.py | no |
| `unrounded/final_register_v2.json` | s3_models/final_models_v2.py | yes |
| `unrounded/methods_register_v1.json` | s7_extensions/methods_cmp_v1.py, methods_cmp_v1b.py | no |
| `unrounded/robust_af_loeo_v1.json` | s3_models/robust_v1.py (attributable fractions and leave-one-event-out, unrounded) | no |
| `unrounded/s2_register_v1.json` | s1_landtrendr/s2_check_v1.py | no |
| `unrounded/slope_bands_register_v2.json` | s7_extensions/slope_bands_v2.py | yes |
| `visibility_register_v1.json` | s1_landtrendr/visibility_v1.py | no |
