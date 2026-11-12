"""06_snow_split_hot_spots.py

Tests whether the cold-season urban hot spot exists both with and without snow cover.
For each city it reads the snow-covered and snow-free composites, runs ArcGIS Hot Spot
Analysis (Getis-Ord Gi*) on each, and compares the two results.

Run with ArcGIS Pro's Python after the files from gee/03_cold_snow_split.js are in
data/lst_composites. Outputs:
  analysis/uhi_prairie.gdb/<city>_cold_<class>_gi   hot-spot points per class
  analysis/snow_split_summary.csv                    comparison statistics
  figures/snow_split_maps.png
"""
import importlib.util
import os
import warnings

import numpy as np
import pandas as pd
import arcpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ehsa", os.path.join(HERE, "03_emerging_hot_spots.py"))
ehsa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ehsa)

ROOT, GDB, AGG = ehsa.ROOT, ehsa.GDB, ehsa.AGG
BANDS = ["lst", "anom", "count", "snowfreq"]
DISTANCE = 200          # metres, same neighbourhood as the emerging hot-spot runs
arcpy.env.overwriteOutput = True


def read_composite(city, cls):
    """Returns dict of 90 m grids plus cell-centre coordinates."""
    tif = os.path.join(ehsa.LST_DIR, f"{city}_cold_{cls}.tif")
    ras = arcpy.Raster(tif)
    cell = ras.meanCellWidth
    arr = arcpy.RasterToNumPyArray(ras, nodata_to_value=np.nan).astype("float32")
    arr[~np.isfinite(arr)] = np.nan
    nb, nr, nc = arr.shape
    nr2, nc2 = nr // AGG, nc // AGG
    blocks = arr[:, :nr2 * AGG, :nc2 * AGG].reshape(nb, nr2, AGG, nc2, AGG)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        agg = np.nanmean(blocks, axis=(2, 4))
    size = cell * AGG
    xs = ras.extent.XMin + size * (np.arange(nc2) + 0.5)
    ys = ras.extent.YMax - size * (np.arange(nr2) + 0.5)
    return dict(zip(BANDS, agg)), xs, ys, size


def gi_star(city, cls, values, xs, ys, mask):
    """ArcGIS Hot Spot Analysis on the masked cells; returns grids of Gi_Bin and z-score."""
    rows, cols = np.where(mask & np.isfinite(values))
    pts = np.zeros(len(rows), dtype=[("x", "f8"), ("y", "f8"), ("anom", "f8"),
                                     ("row", "i4"), ("col", "i4")])
    pts["x"], pts["y"] = xs[cols], ys[rows]
    pts["anom"], pts["row"], pts["col"] = values[rows, cols], rows, cols
    in_fc = os.path.join(GDB, f"{city}_cold_{cls}_pts")
    out_fc = os.path.join(GDB, f"{city}_cold_{cls}_gi")
    arcpy.da.NumPyArrayToFeatureClass(pts, in_fc, ("x", "y"), ehsa.SR)
    arcpy.stats.HotSpots(in_fc, "anom", out_fc, "FIXED_DISTANCE_BAND",
                         "EUCLIDEAN_DISTANCE", "NONE", DISTANCE)
    # Join results back to grid cells by coordinates, which does not rely on row order.
    res = arcpy.da.FeatureClassToNumPyArray(out_fc, ["SHAPE@X", "SHAPE@Y", "GiZScore", "Gi_Bin"])
    c = np.round((res["SHAPE@X"] - xs[0]) / (xs[1] - xs[0])).astype(int)
    r = np.round((ys[0] - res["SHAPE@Y"]) / (ys[0] - ys[1])).astype(int)
    bins = np.full(values.shape, np.nan)
    z = np.full(values.shape, np.nan)
    bins[r, c] = res["Gi_Bin"]
    z[r, c] = res["GiZScore"]
    arcpy.management.Delete(in_fc)
    return bins, z


summary, panels = [], {}
for city in ["saskatoon", "regina"]:
    area_fc = ehsa.analysis_area(city)
    data = {}
    for cls in ["snow", "nosnow"]:
        grids, xs, ys, size = read_composite(city, cls)
        mask = ehsa.polygon_mask(area_fc, xs, ys)
        bins, z = gi_star(city, cls, grids["anom"], xs, ys, mask)
        data[cls] = dict(grids=grids, bins=bins, z=z)
        rural = ~mask & np.isfinite(grids["anom"])
        urban = mask & np.isfinite(grids["anom"])
        summary.append({
            "city": city, "class": cls,
            "obs_per_cell_median": float(np.nanmedian(grids["count"][urban])),
            "snow_index_share_rural": float(np.nanmean(grids["snowfreq"][rural])),
            "snow_index_share_city": float(np.nanmean(grids["snowfreq"][urban])),
            "lst_city": float(np.nanmean(grids["lst"][urban])),
            "lst_rural": float(np.nanmean(grids["lst"][rural])),
            "city_minus_rural": float(np.nanmean(grids["lst"][urban]) - np.nanmean(grids["lst"][rural])),
            "anom_sd_city": float(np.nanstd(grids["anom"][urban])),
            "hot95_share": float(np.nanmean(bins[urban] >= 2)),
            "cold95_share": float(np.nanmean(bins[urban] <= -2)),
        })
        print(city, cls, "done", flush=True)

    hs, hn = data["snow"]["bins"] >= 2, data["nosnow"]["bins"] >= 2
    cs, cn = data["snow"]["bins"] <= -2, data["nosnow"]["bins"] <= -2
    ok = np.isfinite(data["snow"]["bins"]) & np.isfinite(data["nosnow"]["bins"])
    a_s, a_n = data["snow"]["grids"]["anom"][ok], data["nosnow"]["grids"]["anom"][ok]
    summary.append({
        "city": city, "class": "comparison",
        "hot_in_both_share_of_snow_hot": float((hs & hn)[ok].sum() / max(hs[ok].sum(), 1)),
        "hot_in_both_share_of_nosnow_hot": float((hs & hn)[ok].sum() / max(hn[ok].sum(), 1)),
        "hot_jaccard": float((hs & hn)[ok].sum() / max((hs | hn)[ok].sum(), 1)),
        "cold_jaccard": float((cs & cn)[ok].sum() / max((cs | cn)[ok].sum(), 1)),
        "snow_hot_but_nosnow_cold": float((hs & cn)[ok].sum() / max(hs[ok].sum(), 1)),
        "anom_correlation": float(np.corrcoef(a_s, a_n)[0, 1]),
        "z_correlation": float(np.corrcoef(data["snow"]["z"][ok], data["nosnow"]["z"][ok])[0, 1]),
    })
    # Agreement classes: 0 none, 1 hot in both, 2 hot with snow only,
    # 3 hot without snow only, 4 cold in both, 5 cold in one.
    agree = np.full(ok.shape, np.nan)
    agree[ok] = 0
    agree[ok & (cs | cn)] = 5
    agree[ok & cs & cn] = 4
    agree[ok & hs & ~hn] = 2
    agree[ok & hn & ~hs] = 3
    agree[ok & hs & hn] = 1
    rr, cc = np.where(ok)
    sl = (slice(rr.min(), rr.max() + 1), slice(cc.min(), cc.max() + 1))
    panels[city] = (np.where(ok, data["snow"]["grids"]["anom"], np.nan)[sl],
                    np.where(ok, data["nosnow"]["grids"]["anom"], np.nan)[sl], agree[sl])

df = pd.DataFrame(summary)
df.to_csv(os.path.join(ROOT, "analysis", "snow_split_summary.csv"), index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(df[df["class"] != "comparison"].dropna(axis=1).round(2).to_string(index=False))
print(df[df["class"] == "comparison"].dropna(axis=1).round(2).to_string(index=False))

colors = ["#f0f0f0", "#a80000", "#ff9d5c", "#ffd37f", "#004da8", "#bed2ff"]
labels = ["No hot or cold spot", "Hot spot in both", "Hot spot with snow only",
          "Hot spot without snow only", "Cold spot in both", "Cold spot in one"]
fig, axes = plt.subplots(2, 3, figsize=(17, 11))
for i, city in enumerate(["saskatoon", "regina"]):
    a_s, a_n, agree = panels[city]
    for j, (img, title) in enumerate([(a_s, "snow-covered scenes"), (a_n, "snow-free scenes")]):
        lim = np.nanpercentile(np.abs(img), 99)
        h = axes[i, j].imshow(img, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
        axes[i, j].set_title(f"{city.title()}: anomaly, {title}")
        fig.colorbar(h, ax=axes[i, j], fraction=0.04, label="deg C vs scene mean")
    axes[i, 2].imshow(agree, cmap=ListedColormap(colors), vmin=-0.5, vmax=5.5,
                      interpolation="nearest")
    axes[i, 2].set_title(f"{city.title()}: hot-spot agreement (95% confidence)")
    for ax in axes[i]:
        ax.set_xticks([])
        ax.set_yticks([])
fig.legend(handles=[Patch(facecolor=c, edgecolor="0.5", label=l) for c, l in zip(colors, labels)],
           loc="lower center", ncol=6, fontsize=10, frameon=False)
fig.suptitle("Cold season (Nov-Mar, 2000-2026): surface temperature anomaly with and "
             "without snow cover", fontsize=13)
fig.tight_layout(rect=[0, 0.05, 1, 0.96])
fig.savefig(os.path.join(ROOT, "figures", "snow_split_maps.png"), dpi=100)
