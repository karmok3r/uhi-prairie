"""17_driver_variables.py

Builds candidate driver variables on the 90 m analysis grid of each city and joins them
to the LST responses.

Drivers (per 90 m cell):
  bld_cover     building footprint cover, %          Microsoft Canadian Building Footprints
  bld_size      mean footprint area of buildings whose centroid is in the cell, m2
  built_2020    built-up surface, %                  GHSL GHS-BUILT-S R2023A, epoch 2020
  built_change  built-up surface 2020 minus 1985, percentage points (GHS-BUILT-S)
  height        average net building height, m       GHSL GHS-BUILT-H ANBH R2023A, 2018
  tree          tree cover, %                        ESA WorldCover 2021 (class 10)
  herb          grassland + cropland + bare, %       ESA WorldCover 2021 (30, 40, 60)
  water_dist    distance from the cell centre to WorldCover water (class 80), m

Responses:
  summer_late    mean summer anomaly 2017-2026, C   (analysis/<city>_summer_anom_90m.nc)
  summer_change  mean 2017-2026 minus mean 1984-1993, C
  cold_snow      anomaly of the snow-covered cold-season composite, C
  cold_nosnow    anomaly of the snow-free cold-season composite, C
  ehsa           summer emerging hot-spot pattern (baseline run)

Run with ArcGIS Pro's Python. Output: analysis/drivers_<city>.csv
"""
import os
import glob
import zipfile
import warnings
import numpy as np
import pandas as pd
import netCDF4
import arcpy
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "drivers", "raw")
WORK = os.path.join(ROOT, "data", "drivers", "work")
LST = os.path.join(ROOT, "data", "lst_composites")
GDB = os.path.join(ROOT, "analysis", "uhi_prairie.gdb")
os.makedirs(WORK, exist_ok=True)
arcpy.env.overwriteOutput = True
UTM = arcpy.SpatialReference(32613)
WC_TILE = {"saskatoon": "wc_N51W108.tif", "regina": "wc_N48W105.tif"}
FINE = 10            # metres, working resolution for resampling
AGG = 9              # 9 x 9 cells of 10 m = 90 m


def unzip(name):
    z = os.path.join(RAW, name)
    tif = [n for n in zipfile.ZipFile(z).namelist() if n.endswith(".tif")][0]
    out = os.path.join(WORK, tif)
    if not os.path.exists(out):
        zipfile.ZipFile(z).extract(tif, WORK)
    return out


def to_fine(src, city, tag, box):
    """Clip src around the box, project to UTM 13N at 10 m (nearest), return array."""
    xmin, ymin, xmax, ymax = box
    out = os.path.join(WORK, f"{city}_{tag}_10m.tif")
    nr, nc = int(round((ymax - ymin) / FINE)), int(round((xmax - xmin) / FINE))
    if not os.path.exists(out):
        sr = arcpy.Describe(src).spatialReference
        poly = arcpy.Polygon(arcpy.Array([arcpy.Point(xmin - 2000, ymin - 2000),
                                          arcpy.Point(xmin - 2000, ymax + 2000),
                                          arcpy.Point(xmax + 2000, ymax + 2000),
                                          arcpy.Point(xmax + 2000, ymin - 2000)]), UTM)
        e = poly.projectAs(sr).extent
        t = arcpy.Describe(src).extent
        if e.XMin >= t.XMax or e.XMax <= t.XMin or e.YMin >= t.YMax or e.YMax <= t.YMin:
            return np.full((nr, nc), np.nan)          # tile does not reach the box
        clip = os.path.join(WORK, f"{city}_{tag}_clip.tif")
        arcpy.management.Clip(src, f"{e.XMin} {e.YMin} {e.XMax} {e.YMax}", clip)
        with arcpy.EnvManager(outputCoordinateSystem=UTM,
                              extent=arcpy.Extent(xmin, ymin, xmax, ymax),
                              snapRaster=os.path.join(LST, f"{city}_summer_anom.tif")):
            arcpy.management.ProjectRaster(clip, out, UTM, "NEAREST", FINE)
        arcpy.management.Delete(clip)
    a = arcpy.RasterToNumPyArray(arcpy.Raster(out), arcpy.Point(xmin, ymin), nc, nr,
                                 nodata_to_value=-1).astype("f8")
    return a


def block(a, func=np.nanmean):
    nr, nc = a.shape[0] // AGG, a.shape[1] // AGG
    b = a[:nr * AGG, :nc * AGG].reshape(nr, AGG, nc, AGG)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return func(b, axis=(1, 3))


def read90(tif, band=None):
    """Read a 30 m composite and average to 90 m (same scheme as 03_emerging_hot_spots)."""
    a = arcpy.RasterToNumPyArray(arcpy.Raster(tif), nodata_to_value=np.nan).astype("f8")
    if a.ndim == 3:
        a = a[band]
    nr, nc = a.shape[0] // 3, a.shape[1] // 3
    b = a[:nr * 3, :nc * 3].reshape(nr, 3, nc, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmean(b, axis=(1, 3))


for city in ["saskatoon", "regina"]:
    ref = arcpy.Raster(os.path.join(LST, f"{city}_summer_anom.tif"))
    ex = ref.extent
    nr90 = int(ref.height) // 3
    nc90 = int(ref.width) // 3
    box = (ex.XMin, ex.YMax - nr90 * 90, ex.XMin + nc90 * 90, ex.YMax)
    xs = ex.XMin + 90 * (np.arange(nc90) + 0.5)
    ys = ex.YMax - 90 * (np.arange(nr90) + 0.5)
    print(city, "grid", nr90, "x", nc90)

    # ---- WorldCover ----
    wc = to_fine(os.path.join(RAW, WC_TILE[city]), city, "worldcover", box)
    valid = wc > 0
    frac = lambda cls: block(np.where(valid, np.isin(wc, cls), np.nan)) * 100
    tree, herb = frac([10]), frac([30, 40, 60])
    water = wc == 80
    dist10 = ndimage.distance_transform_edt(~water) * FINE
    water_dist = block(dist10)

    # ---- GHSL built-up surface and height (two tiles, combined) ----
    def ghsl(prefix, tag):
        arrs = []
        for t in ["R3_C11", "R4_C11"]:
            a = to_fine(unzip(f"{prefix}_{t}.zip"), city, f"{tag}_{t}", box)
            a[(a < 0) | (a > 65000)] = np.nan
            arrs.append(a)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.nanmax(np.stack(arrs), axis=0)
    bs85 = ghsl("built_s_1985", "bs1985") / 100.0     # m2 per 100 m cell -> % of cell
    bs20 = ghsl("built_s_2020", "bs2020") / 100.0
    h18 = ghsl("built_h_2018", "h2018")
    built_2020 = block(bs20)
    built_change = block(bs20 - bs85)
    w = np.where(np.isfinite(h18) & np.isfinite(bs20), bs20, 0.0)
    height = block(np.nan_to_num(h18) * w, np.nansum) / np.maximum(block(w, np.nansum), 1e-9)
    height[block(w, np.nansum) <= 0] = 0.0

    # ---- Buildings ----
    b = pd.read_csv(os.path.join(ROOT, "data", "drivers", f"{city}_buildings.csv"))
    c = np.floor((b.x - ex.XMin) / 90).astype(int)
    r = np.floor((ex.YMax - b.y) / 90).astype(int)
    ok = (c >= 0) & (c < nc90) & (r >= 0) & (r < nr90)
    area = np.zeros((nr90, nc90))
    count = np.zeros((nr90, nc90))
    np.add.at(area, (r[ok], c[ok]), b.area_m2[ok])
    np.add.at(count, (r[ok], c[ok]), 1)
    bld_cover = np.minimum(area / 8100 * 100, 100)
    bld_size = np.where(count > 0, area / np.maximum(count, 1), 0.0)

    # ---- Responses ----
    with netCDF4.Dataset(os.path.join(ROOT, "analysis", f"{city}_summer_anom_90m.nc")) as ds:
        a = np.ma.filled(ds["anom"][:].astype("f8"), np.nan)
        a[a == -9999] = np.nan
        nx, ny = np.asarray(ds["x"][:]), np.asarray(ds["y"][:])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        early, late = np.nanmean(a[:10], axis=0), np.nanmean(a[-10:], axis=0)
    cols = np.rint((nx - xs[0]) / 90).astype(int)
    rows = np.rint((ys[0] - ny) / 90).astype(int)
    assert np.allclose(xs[cols], nx) and np.allclose(ys[rows], ny), "grid mismatch"
    snow = read90(os.path.join(LST, f"{city}_cold_snow.tif"), band=1)
    nosnow = read90(os.path.join(LST, f"{city}_cold_nosnow.tif"), band=1)

    pat = {}
    for (x, y), p in arcpy.da.SearchCursor(os.path.join(GDB, f"{city}_summer_ehsa"),
                                           ["SHAPE@XY", "PATTERN"]):
        pat[(int(round((x - xs[0]) / 90)), int(round((ys[0] - y) / 90)))] = p

    recs = []
    for i, j in zip(*np.where(np.isfinite(late))):
        R, C = rows[i], cols[j]
        recs.append(dict(x=nx[j], y=ny[i], summer_late=late[i, j],
                         summer_change=late[i, j] - early[i, j],
                         cold_snow=snow[R, C], cold_nosnow=nosnow[R, C],
                         ehsa=pat.get((C, R), "No Pattern Detected"),
                         bld_cover=bld_cover[R, C], bld_size=bld_size[R, C],
                         built_2020=built_2020[R, C], built_change=built_change[R, C],
                         height=height[R, C], tree=tree[R, C], herb=herb[R, C],
                         water_dist=water_dist[R, C]))
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(ROOT, "analysis", f"drivers_{city}.csv"), index=False)
    print(df.describe().round(2).T.to_string())
