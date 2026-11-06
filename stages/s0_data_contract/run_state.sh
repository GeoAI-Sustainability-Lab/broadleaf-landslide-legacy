#!/bin/bash
# Runs gate0_scan.py over the annual landslide-state maps 2017-2024.
y=$1; R="@SOURCE@/Hazard_Events/Landslide_Taiwan/Annual_State_2017_2024_20260912"
mkdir -p @WORK@/g0/state @WORK@/state/$y
if [ -f @WORK@/g0/state/state_${y}_progress.json ] && grep -q '"done": true' @WORK@/g0/state/state_${y}_progress.json; then echo "$y already done"; exit 0; fi
T0=$(date +%s)
if [ ! -f @WORK@/state/$y/*.shx ]; then
  python3 - <<PY
import zipfile
z=zipfile.ZipFile('$R/$y/official_payload.zip')
for n in z.namelist():
    if n.lower().endswith(('.shp','.shx','.dbf')): z.extract(n,'@WORK@/state/$y'); print('extracted',n)
PY
fi
shp=$(ls @WORK@/state/$y/*.shp | head -1)
f=$(python3 -c "
import sys; sys.path.insert(0,'.'); import shapefile
fs=[x[0] for x in shapefile.Reader('$shp').fields[1:]]
d=[x for x in fs if x.lower()=='date'][0]; a=[x for x in fs if x.lower().startswith('area')][0]; print(d+','+a)")
left=$((165 - ($(date +%s) - T0))); echo "== $y $(basename "$shp") fields=$f budget=${left}s"
python3 @WORK@/g0/gate0_scan.py "$shp" @WORK@/g0/state/state_$y 50 $left fields=$f
if grep -q '"done": true' @WORK@/g0/state/state_${y}_progress.json; then rm -rf @WORK@/state/$y; echo "cleaned $y"; fi
df -h /tmp | tail -1
