# Landslide and disturbance legacies in the broadleaf forests of Taiwan

Data and code for an event-matched case-control analysis of rainfall-induced landslides in the broadleaf forests of
Taiwan. Every case and control point is described by its history before the event: a landslide scar of the event-based
landslide inventories, ground mapped as bare in the annual landslide maps, or a canopy disturbance detected in the Landsat
series (LandTrendr) that the inventories do not record. The repository contains the frozen analysis tables, the scripts of
the analysis stages, the registers with the reference values these scripts wrote and a runner that recomputes them from the
tables.

Author: Yu-Chun Hsu, Department of Forestry, National Chung Hsing University, Taichung, Taiwan
(ORCID https://orcid.org/0000-0002-6616-6906). Citation metadata is in `CITATION.cff`. Version 1.1.0 (2026-11-13).

## What the analysis does

Each landslide case (typhoons, tropical depressions and heavy-rain events of 2018-2025) is compared with the control points
inside the affected area of the same event, in a logistic regression with event fixed effects and seven terrain terms
(slope, slope squared, two aspect components, plan and profile curvature, elevation). Legacies are determined only from
data before the cutoff date. Four mutually exclusive site classes are used:

| Class | Name | Criterion before the cutoff date |
|---|---|---|
| 0 | stand with no recorded disturbance | none of the criteria below |
| 1 | revegetated landslide scar | inventory polygon within 50 m with an establishment date not later than the cutoff, not mapped as bare |
| 2 | bare landslide scar | annual-map polygon within 50 m with an image date not later than the cutoff |
| 3 | canopy disturbance not recorded in the inventories | LandTrendr loss segment (magnitude at least 0.10, duration at most 3 years), not in class 1 or 2 |

Attributable fractions follow Bruzzi et al. (1985). The same definitions are applied to non-overlapping events of
2009-2017 with new control points.

Two rules for the cutoff date are implemented:

- **One cutoff date per event** (stage `s9_event_cutoff`). The cases and the controls of an event share the earliest
  pre-event image date among all polygons of the event. Intervals come from an event-level bootstrap (events resampled with
  replacement 2,000 times, percentile intervals).
- **Earlier cutoff rule** (stages `s2` to `s8`). A case takes the pre-event image date of its own polygon and a control the
  earliest date of its event. Standard errors are clustered by point, and the intervals of the attributable fractions are
  Monte Carlo intervals.

Main quantities (broadleaf forest, event-matched design, 39 events, 5,744 cases,
6,796 control points counted as 22,415 controls over the events):

| Quantity | One cutoff date per event (`registers/main27_v5.json`) | Earlier cutoff rule (`registers/main27_v4.json`) |
|---|---|---|
| Legacy sites as share of cases / controls | 49.5% / 17.8% | 52.3% / 17.8% |
| Odds ratios against class 0, all slopes | revegetated landslide scars 3.50 (2.24 to 4.45), bare landslide scars 9.09 (5.81 to 12.06), canopy disturbance 2.45 (1.97 to 2.89) | revegetated landslide scars 4.07 (3.49 to 4.76), bare landslide scars 10.57 (9.08 to 12.29), canopy disturbance 2.44 (2.13 to 2.80) |
| Odds ratios against class 0, slopes up to 39.5 degrees | revegetated landslide scars 4.73 (2.86 to 6.28), bare landslide scars 14.83 (8.09 to 22.38), canopy disturbance 2.69 (2.06 to 3.44) | revegetated landslide scars 4.78 (3.86 to 5.91), bare landslide scars 16.88 (13.70 to 20.79), canopy disturbance 2.65 (2.20 to 3.18) |
| Attributable fraction, all slopes | 38.2% (28.4 to 45.1) | 41.6% (40.1 to 43.0) |
| Attributable fraction, slopes up to 39.5 degrees | 43.7% (30.8 to 54.0) | 45.5% (43.5 to 47.5) |

Values in parentheses are 95% intervals (event-level bootstrap in the first column, cluster-robust for the odds ratios and
Monte Carlo for the attributable fractions in the second). The values of the second column equal those of
`registers/unrounded/final_register_v2.json` and `robust_af_loeo_v1.json`, apart from the Monte Carlo intervals, which agree
within Monte Carlo error. For the events of 2009-2017 (6,033 broadleaf cases, `registers/rep27_v5.json`) the
odds ratio of any legacy against class 0 under the definitions shared by the two periods is
3.58 (3.00 to 4.81) on all slopes,
against 3.87 (2.78 to 4.69) for the events of 2018-2025.

## Repository layout

```
data/frozen/            frozen analysis tables (gzip-compressed CSV), described in data/README.md
data/frozen_manifest.json   md5, size, rows and columns of every frozen table
data/source_inventories/    versions and checksums of the 2024 and 2025 event-based inventories used
stages/                 scripts of every analysis stage in the order they were run (stages/README.md)
registers/              reference registers (JSON/CSV) written by the stage scripts (registers/README.md)
registers/unrounded/    registers of stages s3 to s7 computed with rounding disabled (full precision)
verify/run_verify.py    rebuilds the working folder from data/frozen, runs the stage scripts and compares the results
verify/independent_recompute.py   recomputes the main odds ratios and attributable fractions of the earlier cutoff rule from the method description only
```

## Quick start

```
pip install -r requirements.txt
python verify/run_verify.py          # tags rebuild, models of stages s3 to s7, stage s9 without bootstrap (about 10 minutes)
python verify/run_verify.py --all    # adds the event-level bootstrap of stage s9 and the comparisons of stage s8 (about three hours on two cores)
```

Each check prints PASS or FAIL, the number of compared values and the largest relative difference. Values must agree within
a relative 1e-6. Monte Carlo intervals (keys `ci_mc`) must agree within an absolute 0.005, because the multivariate-normal
draws depend on the linear-algebra library. The event-level bootstrap is seeded per model, so its intervals are reproduced
to the same tolerance as the point estimates. The runner rebuilds `tags_event_v4.csv` and `tags_event_v5.csv`
(40,559 rows each) byte-identically from the matched inventory, annual-map and LandTrendr tables.

## Stages that need the source layers

The stage scripts are kept as they were run, with module docstrings and comments replaced by a one-line description and the
working folder written as the placeholder `@WORK@` (the runner substitutes it). Stages 0 and 1, the build of the 2009-2017
sample and `stages/s9_event_cutoff/rep_rows_v5.py` read the source layers (`@SOURCE@`) or Google Earth Engine. Their outputs
are the frozen tables, so the later stages run without them. Earth Engine credentials are read only from the environment
variables `GEE_SERVICE_ACCOUNT_KEY` and `GEE_PROJECT`, and no credential is stored in the repository. The susceptibility
stage reads the event rainfall table of the public repository
https://github.com/GeoAI-Sustainability-Lab/bamboo-landslide-terrain (`@BAMBOO_REPO@`).

## Names used in code and tables

Scripts, columns and register keys keep the names under which they were first written. They correspond to the terms of this
documentation as follows.

| In code, columns and registers | Term |
|---|---|
| `site_class4` 0, `intact`, `R0` | stand with no recorded disturbance |
| `site_class4` 1, `sc1`, `scar`, `prior scar` | revegetated landslide scar |
| `site_class4` 2, `sc2`, `bare`, `still bare` | bare landslide scar |
| `site_class4` 3, `sc3`, `class3`, `uncatalogued disturbance`, `lt` | canopy disturbance not recorded in the inventories |
| `R1`, `catalogue scar` | inventory landslide scar within 50 m, bare or not (definition shared by the two periods) |
| `legacy`, `legacy_any4` | any of classes 1 to 3 |
| `expected risk`, `score`, `Q`, `bright` | landslide susceptibility predicted from terrain and event rainfall, and its quintile among the controls |
| `island` | island-wide design in which every control point enters once |
| `rep`, `replication` | events of 2009-2017, which do not overlap the events of 2018-2025 |
| `unfailed`, `failed` | controls, cases |
| `tags v4`, `own` | earlier cutoff rule |
| `tags v5`, `common` | one cutoff date per event |
| `gsa`, `zoning` | landslide geologically sensitive areas (2026 layer) |
| `chm` | canopy height (m) |

## Data provenance and licences

The frozen tables are derived per-point attributes. The source layers are not redistributed here. They are credited to, and
remain subject to the terms of, their providers:

| Source | Provider | Access | Licence |
|---|---|---|---|
| Event-based landslide inventories 2004-2025 | Agency of Rural Development and Soil and Water Conservation (ARDSWC), Ministry of Agriculture, with image interpretation by National Cheng Kung University | Government Open Data Platform, https://data.gov.tw (one dataset per year, for example https://data.gov.tw/dataset/173374 for 2024 and https://data.gov.tw/dataset/177455 for 2025) | Open Government Data License, version 1.0 |
| Annual landslide maps 2017-2024 | ARDSWC | https://data.gov.tw/dataset/166573, 166574, 166575, 166576, 166577, 166846, 168661 and 176015 | Open Government Data License, version 1.0 |
| Forest type map, fourth national forest resource inventory | Forestry and Nature Conservation Agency, Ministry of Agriculture | https://data.gov.tw/dataset/57873 | Open Government Data License, version 1.0 |
| 20 m digital elevation model | Ministry of the Interior | https://data.gov.tw/dataset/35430 | Open Government Data License, version 1.0 |
| Landslide geologically sensitive areas (2026 layer, 168,714 polygons) | Geological Survey and Mining Management Agency, Ministry of Economic Affairs | area packages L0001-L0017 on https://data.gov.tw (one dataset per area, for example https://data.gov.tw/dataset/77886) | Open Government Data License, version 1.0 |
| Landsat 5, 7, 8 and 9 Collection 2 Level 2 | U.S. Geological Survey | Google Earth Engine | no restrictions on use or redistribution |
| Sentinel-2 Level-2A, 2019-2024 | European Space Agency, Copernicus | Google Earth Engine | Copernicus Sentinel data licence; contains modified Copernicus Sentinel data 2019-2024 |
| GPM IMERG V07 half-hourly precipitation | NASA Goddard Earth Sciences Data and Information Services Center (Huffman et al., 2023, https://doi.org/10.5067/GPM/IMERG/3B-HH/07) | event rainfall table of the bamboo-landslide-terrain repository | NASA open data policy |
| High Resolution Canopy Height Maps (Tolan et al., 2024) | Meta and World Resources Institute | AWS Open Data registry, dataforgood-fb-forests | CC BY 4.0 |
| Case and control points, forest-type labels, event definitions | Hsu, Y.-C. (2026), https://doi.org/10.5281/zenodo.22702156 | Zenodo | MIT (as listed in the Zenodo record) |

The government data are used under the Open Government Data License, version 1.0 (https://data.gov.tw/license), and the
providers named above are the licensors. The tables in `data/frozen` are value-added data derived from these sources by the
stage scripts.

## Versions

- 1.1.0 (2026-11-13): stage `s9_event_cutoff` (one cutoff date per event for cases and controls, event-level bootstrap),
  its frozen tables (`tags_event_v5`, `rep_rows_tagged_v5`, `zoning_out_v1`, `s2_pairs_v1`) and registers, the checksums of
  the analysis plan (`registers/analysis_plan_checksums.json`) and the terms of this documentation.
- 1.0.1: event-based inventories of 2024 and 2025 identified by version and checksum.
- 1.0.0: stages `s0` to `s8`.

## Funding

National Science and Technology Council, Taiwan: Grant NSTC 114-2634-F-005-002 (Smart Sustainable New Agriculture
Research Center, SMARTer) and Grant NSTC 114-2121-M-005-007-MY2.

## License

Code is released under the MIT License (see LICENSE). The derived data files are provided for reproducibility with
attribution to the providers listed above.

## References

Bruzzi, P., Green, S.B., Byar, D.P., Brinton, L.A., Schairer, C., 1985. Estimating the population attributable risk for
multiple risk factors using case-control data. American Journal of Epidemiology 122, 904-914.
https://doi.org/10.1093/oxfordjournals.aje.a114174

Huffman, G.J., Stocker, E.F., Bolvin, D.T., Nelkin, E.J., Tan, J., 2023. GPM IMERG Final Precipitation L3 Half Hourly 0.1
degree x 0.1 degree V07. NASA Goddard Earth Sciences Data and Information Services Center. https://doi.org/10.5067/GPM/IMERG/3B-HH/07

Kennedy, R.E., Yang, Z., Cohen, W.B., 2010. Detecting trends in forest disturbance and recovery using yearly Landsat time
series: 1. LandTrendr - Temporal segmentation algorithms. Remote Sensing of Environment 114, 2897-2910.
https://doi.org/10.1016/j.rse.2010.07.008

Tolan, J., Yang, H.-I., Nosarzewski, B., et al., 2024. Very high resolution canopy height maps from RGB imagery using
self-supervised vision transformer and convolutional decoder trained on aerial lidar. Remote Sensing of Environment 300,
113888. https://doi.org/10.1016/j.rse.2023.113888
