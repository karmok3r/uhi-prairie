"""17a_buildings.py

Extracts building footprints for the two study boxes from Microsoft's Canadian Building
Footprints (Saskatchewan file, data/drivers/raw/ms_buildings_sk.zip) and writes one row
per building with its centroid (UTM 13N) and footprint area.

Run with a Python that has geopandas (not ArcGIS Pro's Python).
Output: data/drivers/<city>_buildings.csv
"""
import os
import geopandas as gpd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "drivers", "raw", "ms_buildings_sk.zip")
BOXES = {"saskatoon": (-106.98, 51.93, -106.36, 52.33),     # lon/lat, generous
         "regina": (-104.91, 50.26, -104.31, 50.64)}

for city, bbox in BOXES.items():
    g = gpd.read_file(f"zip://{RAW}!Saskatchewan.geojson", bbox=bbox)
    g = g.to_crs(32613)
    c = g.geometry.centroid
    out = gpd.pd.DataFrame({"x": c.x, "y": c.y, "area_m2": g.geometry.area})
    out.to_csv(os.path.join(ROOT, "data", "drivers", f"{city}_buildings.csv"), index=False)
    print(city, len(out), "buildings, median area", round(out.area_m2.median(), 1), "m2")
