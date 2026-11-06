# Landslide and disturbance legacies in the broadleaf forests of Taiwan

Data and code for an event-matched case-control analysis of rainfall-triggered landslides in the broadleaf forests of
Taiwan. Every case and control point is described by its history before the event: a prior landslide scar from the
event-based landslide inventories, ground still bare in the annual landslide maps at the cutoff date, or a canopy
disturbance detected in the Landsat record (LandTrendr) but absent from the inventories. The repository contains the
frozen analysis tables, the scripts of the analysis stages, the registers with the reference values these scripts wrote
and a runner that recomputes them from the tables.

Author: Yu-Chun Hsu, Department of Forestry, National Chung Hsing University, Taichung, Taiwan
(ORCID https://orcid.org/0000-0002-6616-6906). Citation metadata is in `CITATION.cff`.

## What the analysis does

Each landslide case (2018-2025 typhoons, tropical depressions and heavy-rain events) is compared with control points
inside the affected area of the same event, in a logistic regression with event fixed effects, seven terrain terms (slope,
slope squared, two aspect components, plan and profile curvature, elevation) and standard errors clustered by point. Legacies
are evaluated before the cutoff date of each row (the earliest pre-event image date of the case polygon or, for a control,
of the matched event). Four mutually exclusive site classes are used: 0 intact stand, 1 prior scar within 50 m, 2 still bare at the cutoff, 3 uncatalogued canopy disturbance.
Attributable fractions follow Bruzzi et al. (1985). The same definitions are applied to independent events of 2009-2017
with new control points.

Main quantities (broadleaf, event-matched design; registers/unrounded/final_register_v2.json and robust_af_loeo_v1.json):

- slopes up to 39.5 degrees, odds ratios against intact stands: prior scar 4.78 (95% CI 3.86 to 5.91), still bare
  16.88 (95% CI 13.70 to 20.79), uncatalogued disturbance 2.65 (95% CI 2.20 to 3.18);
- share of cases attributable to legacy sites: 45.5% (Monte Carlo 95% interval 43.4% to 47.5%) up to 39.5 degrees, 41.6% (Monte Carlo 95% interval 40.1% to 43.0%) on all slopes.

`verify/run_verify.py` rebuilds the event-matched analysis table (`tags_event_v4.csv`, 40,559 rows) byte-identically from
the matched inventory, annual-map and LandTrendr tables, refits the models and recomputes the odds ratios and the point
estimates of the attributable fractions; the Monte Carlo intervals come from `stages/s3_models/robust_v1.py`.

## Repository layout

```
data/frozen/            frozen analysis tables (gzip-compressed CSV), described in data/README.md
data/frozen_manifest.json   md5, size, rows and columns of every frozen table
data/source_inventories/    versions and checksums of the 2024 and 2025 event-based inventories used
stages/                 scripts of every analysis stage in the order they were run (stages/README.md)
registers/              reference registers (JSON/CSV) written by the stage scripts (registers/README.md)
registers/unrounded/    the same registers computed with rounding disabled (full precision)
verify/run_verify.py    rebuilds the working folder from data/frozen, runs the stage scripts and compares the results
verify/independent_recompute.py   recomputes the main odds ratios and attributable fractions from the method description only
```

## Quick start

```
pip install -r requirements.txt
python verify/run_verify.py          # tags rebuild, main models, stand origin, slope bands, 2009 events (about 2 minutes)
python verify/run_verify.py --all    # adds the attributable-fraction comparisons with event bootstrap (about 40 minutes)
```

Each check prints PASS or FAIL, the number of compared values and the largest relative difference. Values must agree within
a relative 1e-6; Monte Carlo intervals (keys `ci_mc`) within an absolute 0.005, because the multivariate-normal draws depend
on the linear-algebra library.

## Stages that need the source layers

The stage scripts are kept as they were run, with module docstrings and comments replaced by a one-line description and the
working folder written as the placeholder `@WORK@` (the runner substitutes it). Stages 0 and 1 and the build of the
replication sample read the source layers (`@SOURCE@`) and Google Earth Engine; their outputs are the frozen tables, so the
later stages run without them. Earth Engine credentials are read only from the environment variables `GEE_SERVICE_ACCOUNT_KEY`
and `GEE_PROJECT`; no credential is stored in the repository. The expected-risk stage reads the event rainfall table of the
public repository https://github.com/GeoAI-Sustainability-Lab/bamboo-landslide-terrain (`@BAMBOO_REPO@`).

## Data provenance and licences

The frozen tables are derived per-point attributes. The source layers are credited to, and remain subject to the terms
of, their providers:

| Source | Provider | Access | Licence |
|---|---|---|---|
| Event-based landslide inventories 2004-2025 | Agency of Rural Development and Soil and Water Conservation (ARDSWC), Ministry of Agriculture, with image interpretation by National Cheng Kung University | Government Open Data Platform, data.gov.tw (one dataset per year, e.g. 173374 for 2024 and 177455 for 2025) | Open Government Data License, version 1.0 |
| Annual landslide maps 2017-2024 | ARDSWC | data.gov.tw 166573, 166574, 166575, 166576, 166577, 166846, 168661, 176015 | Open Government Data License, version 1.0 |
| Forest type map, fourth national forest resource inventory | Forestry and Nature Conservation Agency, Ministry of Agriculture | data.gov.tw 57873 | Open Government Data License, version 1.0 |
| 20 m digital elevation model | Ministry of the Interior | data.gov.tw 35430 | Open Government Data License, version 1.0 |
| Landslide geologically sensitive areas (2026) | Geological Survey and Mining Management Agency, Ministry of Economic Affairs | area packages L0001-L0017; data.gov.tw (one dataset per area, e.g. 77886) | Open Government Data License, version 1.0 |
| Landsat 5, 7, 8 and 9 Collection 2 Level 2 | U.S. Geological Survey | Google Earth Engine | no restrictions on use or redistribution |
| Sentinel-2 Level-2A, 2019-2024 | European Space Agency, Copernicus | Google Earth Engine | Copernicus Sentinel data licence; contains modified Copernicus Sentinel data 2019-2024 |
| GPM IMERG V07 half-hourly precipitation | NASA Goddard Earth Sciences Data and Information Services Center (Huffman et al., 2023, https://doi.org/10.5067/GPM/IMERG/3B-HH/07) | event rainfall table of the bamboo-landslide-terrain repository | NASA open data policy |
| High Resolution Canopy Height Maps (Tolan et al., 2024) | Meta and World Resources Institute | AWS Open Data registry, dataforgood-fb-forests | CC BY 4.0 |
| Case and control points, forest-type labels, event definitions | Hsu, Y.-C. (2026), https://doi.org/10.5281/zenodo.22702156 | Zenodo | MIT (as listed in the Zenodo record) |

The government data are used under the Open Government Data License, version 1.0 (https://data.gov.tw/license); the
providers named above are the licensors.

## Funding

National Science and Technology Council, Taiwan: Grant NSTC 114-2634-F-005-002 (Smart Sustainable New Agriculture
Research Center, SMARTer) and Grant NSTC 114-2121-M-005-007-MY2.

## License

Code is released under the MIT License (see LICENSE). The derived data files are provided for research reproducibility with
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
