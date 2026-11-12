"""09_proposal_figure.py

Builds the single summary figure used in the research proposal.
Run with ArcGIS Pro's Python. Output: figures/proposal_figure.png
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
CELL = 90.0

# Emerging hot-spot categories grouped into five classes for a small map.
GROUPS = {
    "Intensifying Hot Spot": 1, "Persistent Hot Spot": 1, "Consecutive Hot Spot": 1, "New Hot Spot": 1,
    "Sporadic Hot Spot": 2, "Oscillating Hot Spot": 2, "Diminishing Hot Spot": 2, "Historical Hot Spot": 2,
    "Intensifying Cold Spot": 4, "Persistent Cold Spot": 4, "Consecutive Cold Spot": 4, "New Cold Spot": 4,
    "Sporadic Cold Spot": 3, "Oscillating Cold Spot": 3, "Diminishing Cold Spot": 3, "Historical Cold Spot": 3,
}
GROUP_COLORS = ["#eeeeee", "#a80000", "#f4b6a6", "#bcd0ee", "#004da8"]
GROUP_LABELS = ["No pattern", "Hot: persistent, intensifying or new", "Hot: sporadic or oscillating",
                "Cold: sporadic or oscillating", "Cold: persistent, intensifying or new"]


def ehsa_grid(city, season):
    fc = os.path.join(GDB, f"{city}_{season}_ehsa")
    rows = [(s[0], s[1], GROUPS.get(p, 0)) for s, p in arcpy.da.SearchCursor(fc, ["SHAPE@XY", "PATTERN"])]
    d = pd.DataFrame(rows, columns=["x", "y", "g"])
    c = ((d.x - d.x.min()) / CELL).round().astype(int)
    r = ((d.y.max() - d.y) / CELL).round().astype(int)
    grid = np.full((r.max() + 1, c.max() + 1), np.nan)
    grid[r, c] = d.g
    return grid


def anomaly_grid(city, cls):
    ras = arcpy.Raster(os.path.join(ehsa.LST_DIR, f"{city}_cold_{cls}.tif"))
    arr = arcpy.RasterToNumPyArray(ras, nodata_to_value=np.nan).astype("float32")[1]   # band 2 = anom
    arr[~np.isfinite(arr)] = np.nan
    nr, nc = arr.shape[0] // AGG, arr.shape[1] // AGG
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        agg = np.nanmean(arr[:nr * AGG, :nc * AGG].reshape(nr, AGG, nc, AGG), axis=(1, 3))
    size = ras.meanCellWidth * AGG
    xs = ras.extent.XMin + size * (np.arange(nc) + 0.5)
    ys = ras.extent.YMax - size * (np.arange(nr) + 0.5)
    mask = ehsa.polygon_mask(ehsa.analysis_area(city), xs, ys)
    agg[~mask] = np.nan
    rr, cc = np.where(mask)
    return agg[rr.min():rr.max() + 1, cc.min():cc.max() + 1]


scenes = pd.read_csv(os.path.join(ROOT, "analysis", "scene_urban_rural.csv"), parse_dates=["date"])

plt.rcParams.update({"font.size": 8})
fig = plt.figure(figsize=(7.5, 8.0))
gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 0.16, 0.95], hspace=0.22, wspace=0.05,
                      left=0.085, right=0.985, top=0.955, bottom=0.045)
letters = iter("abcdefgh")
for i, city in enumerate(["saskatoon", "regina"]):
    ax = fig.add_subplot(gs[i, 0])
    ax.imshow(ehsa_grid(city, "summer"), cmap=ListedColormap(GROUP_COLORS), vmin=-0.5, vmax=4.5,
              interpolation="nearest", aspect="equal")
    ax.set_title(f"({next(letters)}) {city.title()}: summer hot spots\n1984-2026", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    for j, (cls, label) in enumerate([("snow", "snow-covered scenes"), ("nosnow", "snow-free scenes")]):
        ax = fig.add_subplot(gs[i, j + 1])
        im = ax.imshow(anomaly_grid(city, cls), cmap="RdBu_r", vmin=-3.5, vmax=3.5,
                       interpolation="nearest", aspect="equal")
        ax.set_title(f"({next(letters)}) {city.title()}: cold season\n{label}, 2000-2026", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

lax = fig.add_subplot(gs[2, 0])
lax.axis("off")
lax.legend(handles=[Patch(facecolor=c, edgecolor="0.5", label=l) for c, l in zip(GROUP_COLORS, GROUP_LABELS)],
           loc="upper left", bbox_to_anchor=(-0.12, 1.25), ncol=1, fontsize=6, frameon=False,
           handlelength=1.2, labelspacing=0.25)
cbox = fig.add_subplot(gs[2, 1:])
cbox.axis("off")
cax = cbox.inset_axes([0.12, 0.55, 0.76, 0.22])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.set_label("LST anomaly relative to scene mean (deg C)", fontsize=7)

sub = gs[3, :].subgridspec(1, 2, wspace=0.2)
for k, city in enumerate(["saskatoon", "regina"]):
    ax = fig.add_subplot(sub[0, k])
    a = scenes[scenes.city == city]
    for cls, col, lab in [("nosnow", "#d95f02", "no snow"), ("mixed", "0.6", "partial snow"),
                          ("snow", "#1f78b4", "snow-covered")]:
        v = a[a.snow_class == cls]
        ax.scatter(v.date.dt.dayofyear, v.dT, s=5, color=col, label=lab, alpha=0.75, linewidths=0)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlim(0, 366)
    ax.set_xticks([15, 74, 135, 196, 258, 319])
    ax.set_xticklabels(["Jan", "Mar", "May", "Jul", "Sep", "Nov"])
    ax.set_ylabel("built-up minus rural LST (deg C)")
    ax.set_title(f"({next(letters)}) {city.title()}: one point per Landsat scene (n = {len(a)})",
                 fontsize=8)
    if k == 0:
        ax.legend(frameon=False, fontsize=7, loc="upper right", markerscale=2)

fig.savefig(os.path.join(ROOT, "figures", "proposal_figure.png"), dpi=220)
print("saved")
