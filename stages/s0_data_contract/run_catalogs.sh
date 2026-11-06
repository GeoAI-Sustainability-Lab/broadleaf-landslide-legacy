#!/bin/bash
# Runs gate0_scan.py over the yearly event inventories 2004-2025.
R="@SOURCE@/Hazard_Events/Landslide_Taiwan"
mkdir -p @WORK@/g0/cat
T0=$(date +%s)
for y in $(seq 2004 2023); do
  shp=$(ls "$R/2004_2023_annual_event_catalogs/raw/extracted/$y/"*.shp | head -1)
  if [ -f @WORK@/g0/cat/cat_${y}_progress.json ] && grep -q '"done": true' @WORK@/g0/cat/cat_${y}_progress.json; then continue; fi
  left=$((160 - ($(date +%s) - T0))); if [ $left -lt 15 ]; then echo "time budget exhausted before $y"; exit 0; fi
  f=$(python3 -c "
import sys; sys.path.insert(0,'.'); import shapefile
r=shapefile.Reader('$shp'); fs=[x[0] for x in r.fields[1:]]
ev='Events' if 'Events' in fs else ('Event_s_' if 'Event_s_' in fs else fs[4]); print('BeforeDate,AfterDate,'+ev+',Area_ha')")
  echo "== $y $(basename "$shp") fields=$f"; python3 @WORK@/g0/gate0_scan.py "$shp" @WORK@/g0/cat/cat_$y 100 $left fields=$f
done
for y in 2024 2025; do
  if [ $y = 2024 ]; then shp="$R/2024_113/Event_Inventory_2024_ARDSWC_V7.shp"; else shp="$R/2025_114/20260526_114年事件型崩塌目錄判釋成果/20260526_114年事件型崩塌目錄判釋成果/Event_Inventory_2025_ARDSWC_V2.shp"; fi
  if [ -f @WORK@/g0/cat/cat_${y}_progress.json ] && grep -q '"done": true' @WORK@/g0/cat/cat_${y}_progress.json; then continue; fi
  left=$((160 - ($(date +%s) - T0))); if [ $left -lt 15 ]; then echo "time budget exhausted before $y"; exit 0; fi
  echo "== $y"; python3 @WORK@/g0/gate0_scan.py "$shp" @WORK@/g0/cat/cat_$y 100 $left fields=BeforeDate,AfterDate,Events,Area_ha
done
echo ALL_DONE
