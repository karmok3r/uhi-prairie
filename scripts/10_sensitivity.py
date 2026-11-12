"""10_sensitivity.py

Sensitivity of the emerging hot-spot results to the analysis settings.
Run with ArcGIS Pro's Python:  python scripts/10_sensitivity.py [variant ...]
With no arguments it runs every variant for both cities and both seasons, skipping
results that already exist.

Variants (one setting changed from the baseline each time):
  base       90 m cells, 200 m neighbourhood, 1-year steps, INDIVIDUAL_TIME_STEP window
  cell60     60 m cells
  cell30     30 m cells
  nbr400     400 m neighbourhood
  nbr600     600 m neighbourhood
  win_nbr    global window NEIGHBORHOOD_TIME_STEP
  win_cube   global window ENTIRE_CUBE
  step2      2-year time steps (means of consecutive pairs of years)

Outputs: analysis/sensitivity.gdb/<city>_<season>_<variant>   (EHSA features)
         analysis/sensitivity/<city>_<season>_<variant>.csv    (cell centre, category)
"""
import os
import sys
import time
import datetime as dt
import importlib.util
import warnings

import numpy as np
import pandas as pd
import netCDF4
import arcpy

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ehsa", os.path.join(HERE, "03_emerging_hot_spots.py"))
ehsa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ehsa)

ROOT = ehsa.ROOT
OUT = os.path.join(ROOT, "analysis", "sensitivity")
GDB = os.path.join(ROOT, "analysis", "sensitivity.gdb")
os.makedirs(OUT, exist_ok=True)
if not arcpy.Exists(GDB):
    arcpy.management.CreateFileGDB(os.path.dirname(GDB), os.path.basename(GDB))
arcpy.env.overwriteOutput = True

BASE = dict(agg=3, nbr=200, window="INDIVIDUAL_TIME_STEP", step=1)
VARIANTS = {
    "base": {},
    "cell60": dict(agg=2),
    "cell30": dict(agg=1),
    "nbr400": dict(nbr=400),
    "nbr600": dict(nbr=600),
    "win_nbr": dict(window="NEIGHBORHOOD_TIME_STEP"),
    "win_cube": dict(window="ENTIRE_CUBE"),
    "step2": dict(step=2),
}
_cache = {}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def read_stack(city, season):
    """30 m anomaly stack and its georeferencing, cached per city and season."""
    key = (city, season)
    if key not in _cache:
        ras = arcpy.Raster(os.path.join(ehsa.LST_DIR, f"{city}_{season}_anom.tif"))
        arr = arcpy.RasterToNumPyArray(ras, nodata_to_value=np.nan).astype("float32")
        arr[~np.isfinite(arr)] = np.nan
        _cache.clear()
        _cache[key] = (arr, ras.meanCellWidth, ras.extent.XMin, ras.extent.YMax)
    return _cache[key]


def build_netcdf(city, season, p, path):
    arr, cell, x0, y1 = read_stack(city, season)
    nb, nr, nc = arr.shape
    agg = p["agg"]
    nr2, nc2 = nr // agg, nc // agg
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = np.nanmean(arr[:, :nr2 * agg, :nc2 * agg].reshape(nb, nr2, agg, nc2, agg),
                          axis=(2, 4)).astype("float32")
    size = cell * agg
    xs = x0 + size * (np.arange(nc2) + 0.5)
    ys = y1 - size * (np.arange(nr2) + 0.5)
    years = ehsa.FIRST_YEAR[season] + np.arange(nb)

    if p["step"] == 2:
        # Drop the first year if the count is odd, then average consecutive pairs.
        if nb % 2:
            grid, years = grid[1:], years[1:]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid = np.nanmean(grid.reshape(-1, 2, nr2, nc2), axis=1).astype("float32")
        years = years[1::2]          # label each pair by its second year

    mask = ehsa.polygon_mask(ehsa.analysis_area(city), xs, ys)
    grid[:, ~mask] = np.nan
    enough = np.isfinite(grid).mean(axis=0) >= ehsa.MIN_YEARS_FRAC
    grid[:, ~enough] = np.nan
    keep = mask & enough
    rows = np.where(keep.any(axis=1))[0]
    cols = np.where(keep.any(axis=0))[0]
    grid = grid[:, rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    xs, ys = xs[cols[0]:cols[-1] + 1], ys[rows[0]:rows[-1] + 1]

    month, day = ehsa.MONTH_DAY[season]
    if os.path.exists(path):
        os.remove(path)
    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.Conventions = "CF-1.6"
        ds.createDimension("time", len(years))
        ds.createDimension("y", len(ys))
        ds.createDimension("x", len(xs))
        t = ds.createVariable("time", "f8", ("time",))
        t.units = "days since 1970-01-01 00:00:00"
        t.calendar = "standard"
        t.standard_name = "time"
        t[:] = [(dt.date(int(y), month, day) - dt.date(1970, 1, 1)).days for y in years]
        y = ds.createVariable("y", "f8", ("y",))
        y.units = "m"; y.standard_name = "projection_y_coordinate"; y.axis = "Y"
        y[:] = ys[::-1]
        x = ds.createVariable("x", "f8", ("x",))
        x.units = "m"; x.standard_name = "projection_x_coordinate"; x.axis = "X"
        x[:] = xs
        crs = ds.createVariable("crs", "i4")
        wkt = ehsa.SR.exportToString().split(";")[0]
        crs.grid_mapping_name = "transverse_mercator"
        crs.longitude_of_central_meridian = -105.0
        crs.latitude_of_projection_origin = 0.0
        crs.scale_factor_at_central_meridian = 0.9996
        crs.false_easting = 500000.0
        crs.false_northing = 0.0
        crs.semi_major_axis = 6378137.0
        crs.inverse_flattening = 298.257223563
        crs.spatial_ref = wkt
        crs.esri_pe_string = wkt
        v = ds.createVariable("anom", "f4", ("time", "y", "x"), fill_value=-9999.0, zlib=True)
        v.grid_mapping = "crs"
        flipped = grid[:, ::-1, :]
        v[:] = np.where(np.isfinite(flipped), flipped, -9999.0)
    return int(keep.sum()), size, len(years)


def run(city, season, variant):
    name = f"{city}_{season}_{variant}"
    csv = os.path.join(OUT, name + ".csv")
    if os.path.exists(csv):
        log(f"skip {name} (exists)")
        return
    p = dict(BASE, **VARIANTS[variant])
    t0 = time.time()
    nc = os.path.join(OUT, name + "_in.nc")
    n_cells, size, n_steps = build_netcdf(city, season, p, nc)
    layer = name + "_md"
    arcpy.md.MakeMultidimensionalRasterLayer(nc, layer, ["anom"])
    cube = os.path.join(OUT, name + "_cube.nc")
    if os.path.exists(cube):
        os.remove(cube)
    arcpy.stpm.CreateSpaceTimeCubeMDRasterLayer(layer, cube, "SPACE_TIME_NEIGHBORS")
    with netCDF4.Dataset(cube) as ds:
        variable = [n for n, var in ds.variables.items()
                    if n.upper().startswith("ANOM") and "time" in var.dimensions][0]
    out_fc = os.path.join(GDB, name)
    arcpy.stpm.EmergingHotSpotAnalysis(cube, variable, out_fc, f"{p['nbr']} Meters", 1, None,
                                       "FIXED_DISTANCE", None, p["window"])
    rows = [(s[0], s[1], pat) for s, pat in arcpy.da.SearchCursor(out_fc, ["SHAPE@XY", "PATTERN"])]
    pd.DataFrame(rows, columns=["x", "y", "pattern"]).to_csv(csv, index=False)
    arcpy.management.Delete(layer)
    for f in (nc, cube):
        try:
            os.remove(f)
        except OSError:
            pass
    log(f"done {name}: {n_cells} cells at {size:.0f} m, {n_steps} steps, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    wanted = sys.argv[1:] or list(VARIANTS)
    for variant in wanted:
        for city in ["saskatoon", "regina"]:
            for season in ["summer", "cold"]:
                try:
                    run(city, season, variant)
                except Exception as e:          # keep going; report at the end
                    log(f"FAILED {city} {season} {variant}: {e}")
