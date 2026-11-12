"""03_emerging_hot_spots.py

Pilot step 3: space-time cube and emerging hot-spot analysis in ArcGIS Pro.

Run with ArcGIS Pro's Python, for example:
  "C:\\Program Files\\ArcGIS\\Pro\\bin\\Python\\envs\\arcgispro-py3\\python.exe" ^
      scripts\\03_emerging_hot_spots.py saskatoon summer

For one city and season it:
  1. builds the analysis area: municipal boundary (OpenStreetMap) plus a buffer;
  2. reads the yearly LST anomaly GeoTIFF, averages it to a coarser grid, and blanks
     everything outside the analysis area;
  3. writes a CF-style netCDF with a time dimension;
  4. builds an ArcGIS space-time cube from it;
  5. runs Emerging Hot Spot Analysis and writes the result to a file geodatabase.
"""
import os
import sys
import time

import numpy as np
import netCDF4
import arcpy
from matplotlib.path import Path

# ---------------- Settings ----------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LST_DIR = os.path.join(ROOT, "data", "lst_composites")
BND_DIR = os.path.join(ROOT, "data", "boundaries")
OUT_DIR = os.path.join(ROOT, "analysis")
GDB = os.path.join(OUT_DIR, "uhi_prairie.gdb")

EPSG = 32613            # UTM zone 13N, same as the GeoTIFFs
BUFFER_M = 1000         # buffer around the municipal boundary
AGG = 3                 # 3 x 3 cells of 30 m -> 90 m analysis cells
MIN_YEARS_FRAC = 0.8    # a cell needs data in at least 80% of the years
NEIGHBORHOOD = "200 Meters"
TIME_NEIGHBORS = 1
GLOBAL_WINDOW = "INDIVIDUAL_TIME_STEP"   # each year's hot spots are relative to that year

FIRST_YEAR = {"summer": 1984, "cold": 2000}
MONTH_DAY = {"summer": (7, 1), "cold": (1, 15)}   # nominal date for each time step

arcpy.env.overwriteOutput = True
SR = arcpy.SpatialReference(EPSG)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def analysis_area(city):
    """Municipal boundary + buffer as a feature class; returns its path."""
    os.makedirs(OUT_DIR, exist_ok=True)
    if not arcpy.Exists(GDB):
        arcpy.management.CreateFileGDB(OUT_DIR, os.path.basename(GDB))
    raw = os.path.join(BND_DIR, f"{city}_osm_raw.geojson")
    wgs = os.path.join(GDB, f"{city}_boundary_wgs84")
    utm = os.path.join(GDB, f"{city}_boundary")
    buf = os.path.join(GDB, f"{city}_analysis_area")
    if not arcpy.Exists(buf):
        arcpy.conversion.JSONToFeatures(raw, wgs, "POLYGON")
        arcpy.management.Project(wgs, utm, SR)
        arcpy.analysis.PairwiseBuffer(utm, buf, f"{BUFFER_M} Meters", "ALL")
        arcpy.management.Delete(wgs)
    return buf


def polygon_mask(fc, xs, ys):
    """Boolean grid (len(ys), len(xs)): True where the cell centre is inside fc."""
    xx, yy = np.meshgrid(xs, ys)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    inside = np.zeros(len(pts), dtype=bool)
    with arcpy.da.SearchCursor(fc, ["SHAPE@"]) as cur:
        for (geom,) in cur:
            for part in geom:
                ring, rings = [], []
                for p in part:
                    if p is None:           # separator between outer ring and holes
                        rings.append(ring)
                        ring = []
                    else:
                        ring.append((p.X, p.Y))
                rings.append(ring)
                part_in = Path(rings[0]).contains_points(pts)
                for hole in rings[1:]:
                    part_in &= ~Path(hole).contains_points(pts)
                inside |= part_in
    return inside.reshape(xx.shape)


def build_netcdf(city, season, area_fc):
    """Aggregate, mask and write the anomaly stack as netCDF; returns its path."""
    tif = os.path.join(LST_DIR, f"{city}_{season}_anom.tif")
    ras = arcpy.Raster(tif)
    cell = ras.meanCellWidth
    x0, y1 = ras.extent.XMin, ras.extent.YMax
    arr = arcpy.RasterToNumPyArray(ras, nodata_to_value=np.nan).astype("float32")
    arr[~np.isfinite(arr)] = np.nan
    nb, nr, nc = arr.shape
    log(f"read {os.path.basename(tif)}: {nb} bands, {nr} x {nc}, {cell:.0f} m")

    # Aggregate AGG x AGG blocks with a mean that ignores missing cells.
    nr2, nc2 = nr // AGG, nc // AGG
    blocks = arr[:, :nr2 * AGG, :nc2 * AGG].reshape(nb, nr2, AGG, nc2, AGG)
    with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        agg = np.nanmean(blocks, axis=(2, 4)).astype("float32")
    del arr, blocks
    size = cell * AGG
    xs = x0 + size * (np.arange(nc2) + 0.5)
    ys = y1 - size * (np.arange(nr2) + 0.5)          # north to south

    mask = polygon_mask(area_fc, xs, ys)
    agg[:, ~mask] = np.nan
    enough = np.isfinite(agg).mean(axis=0) >= MIN_YEARS_FRAC
    agg[:, ~enough] = np.nan
    n_cells = int((mask & enough).sum())
    log(f"analysis cells: {n_cells} at {size:.0f} m "
        f"({n_cells * size * size / 1e6:.1f} km2); dropped for missing years: "
        f"{int((mask & ~enough).sum())}")

    # Crop to the bounding box of the analysis cells.
    rows = np.where((mask & enough).any(axis=1))[0]
    cols = np.where((mask & enough).any(axis=0))[0]
    agg = agg[:, rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    xs, ys = xs[cols[0]:cols[-1] + 1], ys[rows[0]:rows[-1] + 1]

    years = FIRST_YEAR[season] + np.arange(nb)
    month, day = MONTH_DAY[season]
    path = os.path.join(OUT_DIR, f"{city}_{season}_anom_{int(size)}m.nc")
    if os.path.exists(path):
        os.remove(path)
    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.Conventions = "CF-1.6"
        ds.createDimension("time", nb)
        ds.createDimension("y", len(ys))
        ds.createDimension("x", len(xs))
        t = ds.createVariable("time", "f8", ("time",))
        t.units = "days since 1970-01-01 00:00:00"
        t.calendar = "standard"
        t.standard_name = "time"
        import datetime as dt
        t[:] = [(dt.date(int(y), month, day) - dt.date(1970, 1, 1)).days for y in years]
        y = ds.createVariable("y", "f8", ("y",))
        y.units = "m"; y.standard_name = "projection_y_coordinate"; y.axis = "Y"
        y[:] = ys[::-1]                                # south to north
        x = ds.createVariable("x", "f8", ("x",))
        x.units = "m"; x.standard_name = "projection_x_coordinate"; x.axis = "X"
        x[:] = xs
        crs = ds.createVariable("crs", "i4")
        wkt = SR.exportToString().split(";")[0]
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
        v.units = "degC"
        v.long_name = "median LST anomaly relative to scene mean"
        v.grid_mapping = "crs"
        v[:] = np.where(np.isfinite(agg[:, ::-1, :]), agg[:, ::-1, :], -9999.0)
    log(f"wrote {os.path.relpath(path, ROOT)}")
    return path


def run(city, season):
    area = analysis_area(city)
    nc = build_netcdf(city, season, area)

    layer = f"{city}_{season}_md"
    arcpy.md.MakeMultidimensionalRasterLayer(nc, layer, ["anom"])
    cube = os.path.join(OUT_DIR, f"{city}_{season}_cube.nc")
    if os.path.exists(cube):
        os.remove(cube)
    log("creating space-time cube ...")
    arcpy.stpm.CreateSpaceTimeCubeMDRasterLayer(layer, cube, "SPACE_TIME_NEIGHBORS")
    print(arcpy.GetMessages(), flush=True)

    # The analysis variable is the one cube variable that has a time dimension.
    with netCDF4.Dataset(cube) as ds:
        variable = [n for n, var in ds.variables.items()
                    if n.upper().startswith("ANOM") and "time" in var.dimensions][0]

    out_fc = os.path.join(GDB, f"{city}_{season}_ehsa")
    log(f"running Emerging Hot Spot Analysis on {variable} ...")
    arcpy.stpm.EmergingHotSpotAnalysis(
        cube, variable, out_fc, NEIGHBORHOOD, TIME_NEIGHBORS, None,
        "FIXED_DISTANCE", None, GLOBAL_WINDOW)
    print(arcpy.GetMessages(), flush=True)
    log(f"done: {os.path.relpath(out_fc, ROOT)}")


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
