"""Writes the bounding box of every event-inventory polygon, used to build the 5 km event footprints."""
import sys, os, glob, struct, csv
sys.path.insert(0,'.'); import shapefile
R=os.path.expanduser('@SOURCE@/Hazard_Events/Landslide_Taiwan')
out=csv.writer(open('@WORK@/g0/event_polygon_bboxes_2018_2025.csv','w',newline='',encoding='utf-8'))
out.writerow(['year','record_index','xmin','ymin','xmax','ymax','Events','BeforeDate','AfterDate','Area_ha'])
for y in range(2018,2026):
    shp={2024:f'{R}/2024_113/Event_Inventory_2024_ARDSWC_V7.shp',2025:f'{R}/2025_114/20260526_114年事件型崩塌目錄判釋成果/20260526_114年事件型崩塌目錄判釋成果/Event_Inventory_2025_ARDSWC_V2.shp'}.get(y) or glob.glob(f'{R}/2004_2023_annual_event_catalogs/raw/extracted/{y}/*.shp')[0]
    rd=shapefile.Reader(shp); fs=[x[0] for x in rd.fields[1:]]
    shx=open(shp[:-4]+'.shx','rb'); shx.seek(24); flen=struct.unpack('>i',shx.read(4))[0]*2; n=(flen-100)//8; shx.seek(100)
    fh=open(shp,'rb')
    for i in range(n):
        off=struct.unpack('>i',shx.read(4))[0]*2; shx.read(4)
        fh.seek(off+8); st=struct.unpack('<i',fh.read(4))[0]
        if st in (5,15,25):
            xmin,ymin,xmax,ymax=struct.unpack('<4d',fh.read(32))
        else: xmin=ymin=xmax=ymax=''
        rec=rd.record(i)
        out.writerow([y,i,xmin,ymin,xmax,ymax,rec[fs.index('Events')],rec[fs.index('BeforeDate')],rec[fs.index('AfterDate')],rec[fs.index('Area_ha')]])
    print(y,n,flush=True)
