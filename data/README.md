# Frozen analysis tables

All tables are UTF-8 CSV compressed with gzip. `data/frozen_manifest.json` lists the md5 of every uncompressed table, its size,
rows and columns; `verify/run_verify.py` checks the md5 before use.

| File | Rows | Content | Produced by |
|---|---|---|---|
| `tags_event_v4.csv.gz` | 40,559 | Event-matched rows (one row per point and event, with its own cutoff): case status, forest type, terrain, legacy tags, site classes 0-3, LandTrendr summary | stages/s2_event_tags/tag_event_v4.py |
| `tags_island_v3.csv.gz` | 20,585 | Island design (one row per point): case status, forest type, terrain, legacy tags, site classes | stages/s1_landtrendr/class3_tag_v1.py (from stages/s0_data_contract/tag_v1.py) |
| `rep_rows_tagged_v1.csv.gz` | 76,553 | Replication 2009-2017: event-matched rows with legacy tags and expected-risk quantiles | stages/s6_replication/rep_analysis_v1.py |
| `rep_rows_v1.csv.gz` | 117,020 | Replication 2009-2017: event-matched rows before tagging | stages/s6_replication/build_rows_v1.py |
| `analysis_rows_v1.csv.gz` | 15,219 | Per-point attributes: canopy height (Meta/WRI) with image date, forest function class, distances to roads, rivers and debris-flow torrents | point extraction of the source layers |
| `bright_rows_v1.csv.gz` | 40,559 | Expected-risk scores and quantiles per event-matched row | stages/s4_unfailed_high_risk/bright_spots_v1.py |
| `catalog_matches_within100m_v1.csv.gz` | 18,342 | Event-inventory polygons within 100 m of each point, with dates and distance | stages/s0_data_contract/gate0_scan.py, assemble_v1.py |
| `annual_state_matches_within50m_v1.csv.gz` | 25,465 | Annual landslide-state polygons within 50 m of each point, with map year and image date | stages/s0_data_contract/gate0_scan.py, assemble_v1.py |
| `point_dates_v1.csv.gz` | 20,585 | Cutoff dates of every point (case: own pre-event image date, control: earliest pre-event date of the matched event) | stages/s0_data_contract/assemble_v1.py |
| `control_event_footprint_membership_v1.csv.gz` | 33,105 | Events whose 5 km footprint contains each control point, with the event cutoffs | stages/s0_data_contract/assemble_v1.py |
| `tags_event_v3.csv.gz` | 40,559 | Earlier event-matched tags (summaries pooled per point); read by tag_event_v4.py for point-level columns and as the check that unaffected rows are unchanged | stages/s0_data_contract/tag_v1.py, stages/s1_landtrendr/class3_tag_v1.py |
| `lt_fits_all_v1.csv.gz` | 35,580 | LandTrendr fit per point and cutoff year (RMSE, vertices) | stages/s1_landtrendr/run_lt_all.py, lt_segments.py |
| `lt_segments_all_v1.csv.gz` | 86,572 | LandTrendr segments per point and cutoff year (start, end, magnitude, duration) | stages/s1_landtrendr/run_lt_all.py, lt_segments.py |
| `lt_summary_v1.csv.gz` | 35,580 | Largest qualifying loss segment per point and cutoff year (disturbance year, magnitude, recovery) | stages/s1_landtrendr/lt_segments.py |

Forest type labels are those of the national forest type map (闊葉樹林型 broadleaf, 針葉樹林型 conifer, 針闊葉樹混淆 mixed
conifer-broadleaf, 竹林 bamboo, 竹闊混淆林 mixed bamboo-broadleaf, 待成林地 unstocked). `case` is 1 for landslide cases and 0 for
controls; `site_class4` is the site class (0 intact stand, 1 prior scar, 2 still bare at the cutoff, 3 uncatalogued disturbance).
