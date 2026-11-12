"""15_hot_spot_clusters.py

Groups the baseline summer intensifying and persistent hot-spot cells (90 m) of each city
into clusters of touching cells (8-neighbour) and writes, for every cluster of at least
MIN_CELLS cells, its size, centre (UTM and WGS84), bounding box, mean summer anomaly in
the first (1984-1993) and last (2017-2026) decade, and the share of cells that are
intensifying.

Run with ArcGIS Pro's Python.
Outputs: analysis/hot_spot_cells_<city>.csv, analysis/hot_spot_clusters.csv
"""
import os
import numpy as np
import pandas as pd
import netCDF4
import arcpy
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GDB = os.path.join(ROOT, "analysis", "uhi_prairie.gdb")
MIN_CELLS = 10          # 10 cells of 90 m = 8.1 ha
KEEP = {"Intensifying Hot Spot", "Persistent Hot Spot"}
sr_utm = arcpy.SpatialReference(32613)
sr_wgs = arcpy.SpatialReference(4326)

rows = []
for city in ["saskatoon", "regina"]:
    with netCDF4.Dataset(os.path.join(ROOT, "analysis", f"{city}_summer_anom_90m.nc")) as ds:
        a = np.ma.filled(ds["anom"][:].astype("f8"), np.nan)
        a[a == -9999] = np.nan
        xs, ys = np.asarray(ds["x"][:]), np.asarray(ds["y"][:])
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    grid = np.zeros(a.shape[1:], np.int8)      # 1 intensifying, 2 persistent
    cells = []
    fc = os.path.join(GDB, f"{city}_summer_ehsa")
    for (x, y), p in arcpy.da.SearchCursor(fc, ["SHAPE@XY", "PATTERN"]):
        if p in KEEP:
            c, r = int(round((x - xs[0]) / dx)), int(round((y - ys[0]) / dy))
            grid[r, c] = 1 if p.startswith("Intensifying") else 2
            cells.append((x, y, p, r, c))
    cdf = pd.DataFrame(cells, columns=["x", "y", "pattern", "row", "col"])
    lab, n = ndimage.label(grid > 0, structure=np.ones((3, 3)))
    cdf["cluster"] = lab[cdf.row, cdf.col]
    early = np.nanmean(a[:10], axis=0)     # 1984-1993
    late = np.nanmean(a[-10:], axis=0)     # 2017-2026
    cdf["anom_early"] = early[cdf.row, cdf.col]
    cdf["anom_late"] = late[cdf.row, cdf.col]
    cdf.to_csv(os.path.join(ROOT, "analysis", f"hot_spot_cells_{city}.csv"), index=False)
    for k, g in cdf.groupby("cluster"):
        if len(g) < MIN_CELLS:
            continue
        pt = arcpy.PointGeometry(arcpy.Point(g.x.mean(), g.y.mean()), sr_utm).projectAs(sr_wgs)
        lo = arcpy.PointGeometry(arcpy.Point(g.x.min() - 45, g.y.min() - 45), sr_utm).projectAs(sr_wgs)
        hi = arcpy.PointGeometry(arcpy.Point(g.x.max() + 45, g.y.max() + 45), sr_utm).projectAs(sr_wgs)
        rows.append(dict(city=city, cluster=int(k), cells=len(g), area_ha=len(g) * 0.81,
                         share_intensifying=(g.pattern == "Intensifying Hot Spot").mean(),
                         x=g.x.mean(), y=g.y.mean(),
                         lat=pt.firstPoint.Y, lon=pt.firstPoint.X,
                         south=lo.firstPoint.Y, west=lo.firstPoint.X,
                         north=hi.firstPoint.Y, east=hi.firstPoint.X,
                         anom_early=g.anom_early.mean(), anom_late=g.anom_late.mean()))
    print(city, len(cdf), "hot-spot cells in", n, "clusters")

out = pd.DataFrame(rows).sort_values(["city", "cells"], ascending=[False, False])
out.to_csv(os.path.join(ROOT, "analysis", "hot_spot_clusters.csv"), index=False)
pd.set_option("display.width", 220)
print(out[["city", "cluster", "cells", "area_ha", "share_intensifying", "lat", "lon",
           "anom_early", "anom_late"]].round(3).to_string(index=False))
