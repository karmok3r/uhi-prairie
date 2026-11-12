"""12_manuscript_figures.py

Builds the manuscript figures from existing results. Run with ArcGIS Pro's Python.
Outputs (figures/ms_*.png, 300 dpi):
  ms_fig1_study_area.png        study areas and scene availability
  ms_fig3_summer_trend.png      summer anomaly series and the 2013 sensor change
  ms_fig2_ehsa.png              emerging hot-spot categories, both seasons and cities
  ms_fig4_snow_maps.png         cold-season anomaly with and without snow, and agreement
  ms_fig5_scenes.png            built-up minus rural LST for individual scenes
"""
import os
import importlib.util
import warnings

import numpy as np
import pandas as pd
import netCDF4
import arcpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Polygon as MplPolygon
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ehsa", os.path.join(HERE, "03_emerging_hot_spots.py"))
ehsa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ehsa)
ROOT, GDB = ehsa.ROOT, ehsa.GDB
FIG = os.path.join(ROOT, "figures")
CITIES = ["saskatoon", "regina"]
NAMES = {"saskatoon": "Saskatoon", "regina": "Regina"}
CELL = 90.0
warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "font.family": "DejaVu Sans"})


# ---------------- helpers ----------------
def outline(city, which):
    """List of (N, 2) arrays: rings of the municipal boundary or the analysis area (UTM)."""
    fc = os.path.join(GDB, f"{city}_{'boundary' if which == 'boundary' else 'analysis_area'}")
    rings = []
    with arcpy.da.SearchCursor(fc, ["SHAPE@"]) as cur:
        for (g,) in cur:
            for part in g:
                ring = []
                for p in part:
                    if p is None:
                        rings.append(np.array(ring)); ring = []
                    else:
                        ring.append((p.X, p.Y))
                rings.append(np.array(ring))
    return [r for r in rings if len(r) > 2]


def draw_outline(ax, city, which, **kw):
    for r in outline(city, which):
        ax.add_patch(MplPolygon(r, closed=True, fill=False, **kw))


def scalebar(ax, km=5, loc=(0.05, 0.05), color="k"):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    x = x0 + loc[0] * (x1 - x0); y = y0 + loc[1] * (y1 - y0)
    ax.plot([x, x + km * 1000], [y, y], color=color, lw=2.5, solid_capstyle="butt")
    ax.text(x + km * 500, y + 0.015 * (y1 - y0), f"{km} km", ha="center", va="bottom",
            fontsize=7, color=color)


def north(ax, loc=(0.93, 0.86), color="k"):
    ax.annotate("N", xy=(loc[0], loc[1] + 0.08), xytext=(loc[0], loc[1]),
                xycoords="axes fraction", ha="center", fontsize=8, fontweight="bold", color=color,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2))


def letter(ax, s, x=0.01, y=0.99, **kw):
    ax.text(x, y, f"({s})", transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="top", ha="left", **kw)


def mapax(ax):
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6)


def read_tif(path, band=None):
    ras = arcpy.Raster(path)
    arr = arcpy.RasterToNumPyArray(ras, nodata_to_value=np.nan).astype("float32")
    arr[~np.isfinite(arr)] = np.nan
    if band is not None and arr.ndim == 3:
        arr = arr[band]
    ext = [ras.extent.XMin, ras.extent.XMax, ras.extent.YMin, ras.extent.YMax]
    return arr, ext, ras.meanCellWidth


def aggregate(arr, k=3):
    nr, nc = arr.shape[-2] // k, arr.shape[-1] // k
    a = arr[..., :nr * k, :nc * k]
    return np.nanmean(a.reshape(*a.shape[:-2], nr, k, nc, k), axis=(-3, -1))


def points_to_grid(xy, values):
    xs = np.array([p[0] for p in xy]); ys = np.array([p[1] for p in xy])
    c = np.round((xs - xs.min()) / CELL).astype(int)
    r = np.round((ys.max() - ys) / CELL).astype(int)
    g = np.full((r.max() + 1, c.max() + 1), np.nan)
    g[r, c] = values
    ext = [xs.min() - CELL / 2, xs.max() + CELL / 2, ys.min() - CELL / 2, ys.max() + CELL / 2]
    return g, ext


def season_counts(city):
    d = pd.read_csv(os.path.join(ROOT, "data", "scene_stats", f"{city}_scenes_used.csv"),
                    parse_dates=["date"])
    m, y = d.date.dt.month, d.date.dt.year
    summer = d[m.isin([6, 7, 8]) & (d.mean_lst >= 10)].groupby(y).size().reindex(range(1984, 2027), fill_value=0)
    cy = y + (m >= 11).astype(int)
    cold = d[m.isin([11, 12, 1, 2, 3])].groupby(cy[m.isin([11, 12, 1, 2, 3])]).size()
    cold = cold.reindex(range(2000, 2027), fill_value=0)
    return summer, cold


# ---------------- Figure 1: study areas and scene availability ----------------
def fig1():
    fig = plt.figure(figsize=(7.48, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 0.75], hspace=0.3, wspace=0.08,
                          left=0.06, right=0.99, top=0.97, bottom=0.07)
    for j, city in enumerate(CITIES):
        lst, ext, _ = read_tif(os.path.join(ehsa.LST_DIR, f"{city}_summer_lst.tif"))
        mean = aggregate(np.nanmean(lst[-10:], axis=0)[None], 3)[0]   # 2017-2026, 90 m
        ax = fig.add_subplot(gs[0, j])
        lo, hi = np.nanpercentile(mean, [2, 98])
        im = ax.imshow(mean, extent=ext, cmap="inferno", vmin=lo, vmax=hi, interpolation="nearest")
        draw_outline(ax, city, "area", edgecolor="#7fdbff", lw=0.9, ls="--")
        draw_outline(ax, city, "boundary", edgecolor="white", lw=1.0)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        mapax(ax); scalebar(ax, 10, color="white"); north(ax, color="white")
        ax.set_title(f"{NAMES[city]}: mean summer LST, 2017–2026")
        letter(ax, "ab"[j], color="white")
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.045, pad=0.02)
        cb.set_label("LST (°C)")
        if j == 1:
            handles = [Line2D([], [], color="white", lw=1.0, label="Municipal boundary"),
                       Line2D([], [], color="#7fdbff", lw=0.9, ls="--", label="Analysis area (boundary + 1 km)")]
            leg = ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=6.5)
            leg.get_frame().set_facecolor("0.15"); leg.get_frame().set_edgecolor("none")
            for t in leg.get_texts():
                t.set_color("white")

    ax = fig.add_subplot(gs[1, :])
    years = np.arange(1984, 2027)
    w = 0.2
    cols = {"saskatoon": ("#c0392b", "#2c7fb8"), "regina": ("#f1948a", "#9ecae1")}
    for k, city in enumerate(CITIES):
        s, c = season_counts(city)
        ax.bar(years + (k - 1.5) * w, s.values, w, color=cols[city][0], label=f"{NAMES[city]}, summer")
        ax.bar(c.index + (k + 0.5) * w, c.values, w, color=cols[city][1], label=f"{NAMES[city]}, cold season")
    ax.axvline(2012.5, color="0.4", lw=0.8, ls=":")
    ax.text(2012.6, ax.get_ylim()[1] * 0.92, "Landsat 8 from 2013", fontsize=7, color="0.3")
    ax.set_xlim(1983.3, 2026.7); ax.set_ylabel("Retained scenes per season")
    ax.legend(ncol=4, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.13))
    letter(ax, "c", x=-0.055, y=1.12)
    fig.savefig(os.path.join(FIG, "ms_fig1_study_area.png"), dpi=300)
    plt.close(fig)


# ---------------- Figure 2: summer anomaly series ----------------
def fig2():
    fig, axes = plt.subplots(1, 2, figsize=(7.48, 2.7), sharey=True)
    for j, city in enumerate(CITIES):
        with netCDF4.Dataset(os.path.join(ROOT, "analysis", f"{city}_summer_anom_90m.nc")) as ds:
            a = np.ma.filled(ds["anom"][:].astype("f8"), np.nan); a[a == -9999] = np.nan
            xs, ys = ds["x"][:], ds["y"][:]
        years = 1984 + np.arange(a.shape[0])
        hot = np.zeros(a.shape[1:], bool)
        for (x, y), p in arcpy.da.SearchCursor(os.path.join(GDB, f"{city}_summer_ehsa"), ["SHAPE@XY", "PATTERN"]):
            if p == "Intensifying Hot Spot":
                hot[int(round((y - ys[0]) / (ys[1] - ys[0]))), int(round((x - xs[0]) / (xs[1] - xs[0])))] = True
        ax = axes[j]
        ax.axvspan(2012.5, 2026.5, color="0.92", zorder=0)
        for sel, col, lab in [(np.isfinite(a).any(axis=0), "0.35", "Whole analysis area"),
                              (hot, "#b2182b", "Intensifying hot-spot cells")]:
            s = np.nanmean(a[:, sel], axis=1)
            ax.plot(years, s, "-o", color=col, ms=2.5, lw=1, label=lab)
            for lo, hi in [(1984, 2012), (2013, 2026)]:
                k = (years >= lo) & (years <= hi)
                b = np.polyfit(years[k], s[k], 1)
                ax.plot(years[k], np.polyval(b, years[k]), color=col, lw=1.6, alpha=0.6)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlim(1983, 2027); ax.set_title(NAMES[city])
        ax.set_xlabel("Year")
        ax.text(2013.2, ax.get_ylim()[0] + 0.3 if j else -1.2, "Landsat 8/9", fontsize=7, color="0.35")
        ax.text(1984.5, -1.2, "Landsat 5/7", fontsize=7, color="0.35")
        letter(ax, "ab"[j])
    axes[0].set_ylabel("Summer LST anomaly (°C)")
    axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.06, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "ms_fig3_summer_trend.png"), dpi=300)
    plt.close(fig)


# ---------------- Figure 3: emerging hot-spot categories ----------------
GROUPS = {
    "New Hot Spot": 1, "Consecutive Hot Spot": 1, "Intensifying Hot Spot": 1,
    "Persistent Hot Spot": 2,
    "Sporadic Hot Spot": 3, "Oscillating Hot Spot": 3, "Diminishing Hot Spot": 3, "Historical Hot Spot": 3,
    "Sporadic Cold Spot": 4, "Oscillating Cold Spot": 4, "Diminishing Cold Spot": 4, "Historical Cold Spot": 4,
    "Persistent Cold Spot": 5,
    "New Cold Spot": 6, "Consecutive Cold Spot": 6, "Intensifying Cold Spot": 6,
}
GCOL = ["#f0f0f0", "#67000d", "#d6604d", "#fbcfbf", "#c6dbef", "#4393c3", "#08306b"]
GLAB = ["No pattern", "Intensifying, new or consecutive hot spot", "Persistent hot spot",
        "Sporadic, oscillating or diminishing hot spot", "Sporadic, oscillating or diminishing cold spot",
        "Persistent cold spot", "Intensifying, new or consecutive cold spot"]


def fig3():
    fig, axes = plt.subplots(2, 2, figsize=(7.48, 7.2))
    k = iter("abcd")
    for i, city in enumerate(CITIES):
        for j, (season, label) in enumerate([("summer", "summer, 1984–2026"), ("cold", "cold season, 2000–2026")]):
            rows = [(xy, GROUPS.get(p, 0)) for xy, p in
                    arcpy.da.SearchCursor(os.path.join(GDB, f"{city}_{season}_ehsa"), ["SHAPE@XY", "PATTERN"])]
            g, ext = points_to_grid([r[0] for r in rows], [r[1] for r in rows])
            ax = axes[i, j]
            ax.imshow(g, extent=ext, cmap=ListedColormap(GCOL), vmin=-0.5, vmax=6.5, interpolation="nearest")
            draw_outline(ax, city, "boundary", edgecolor="0.2", lw=0.6)
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
            mapax(ax); scalebar(ax, 5); north(ax)
            ax.set_title(f"{NAMES[city]}, {label}")
            letter(ax, next(k))
    fig.legend(handles=[Patch(facecolor=c, edgecolor="0.5", lw=0.4, label=l) for c, l in zip(GCOL, GLAB)],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.12, hspace=0.12, wspace=0.03)
    fig.savefig(os.path.join(FIG, "ms_fig2_ehsa.png"), dpi=300)
    plt.close(fig)


# ---------------- Figure 4: snow-covered vs snow-free ----------------
def fig4():
    fig, axes = plt.subplots(2, 3, figsize=(7.48, 5.4))
    agree_col = ["#f0f0f0", "#a50f15", "#fc9272", "#fdd49e", "#08519c", "#bdd7e7"]
    agree_lab = ["Neither hot nor cold spot", "Hot spot in both", "Hot spot with snow only",
                 "Hot spot without snow only", "Cold spot in both", "Cold spot in one only"]
    k = iter("abcdef")
    for i, city in enumerate(CITIES):
        mask_fc = ehsa.analysis_area(city)
        for j, cls in enumerate(["snow", "nosnow"]):
            arr, ext, cell = read_tif(os.path.join(ehsa.LST_DIR, f"{city}_cold_{cls}.tif"), band=1)
            a90 = aggregate(arr, 3)
            xs = ext[0] + cell * 3 * (np.arange(a90.shape[1]) + 0.5)
            ys = ext[3] - cell * 3 * (np.arange(a90.shape[0]) + 0.5)
            m = ehsa.polygon_mask(mask_fc, xs, ys)
            a90[~m] = np.nan
            rr, cc = np.where(m)
            sub = a90[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
            sext = [xs[cc.min()] - 45, xs[cc.max()] + 45, ys[rr.max()] - 45, ys[rr.min()] + 45]
            ax = axes[i, j]
            im = ax.imshow(sub, extent=sext, cmap="RdBu_r", vmin=-3.5, vmax=3.5, interpolation="nearest")
            draw_outline(ax, city, "boundary", edgecolor="0.2", lw=0.5)
            ax.set_xlim(sext[0], sext[1]); ax.set_ylim(sext[2], sext[3])
            mapax(ax); scalebar(ax, 5)
            ax.set_title(f"{NAMES[city]}: {'snow-covered' if cls == 'snow' else 'snow-free'} scenes")
            letter(ax, next(k))
        # agreement map from the Gi* results
        def bins(cls):
            return {xy: b for xy, b in arcpy.da.SearchCursor(
                os.path.join(GDB, f"{city}_cold_{cls}_gi"), ["SHAPE@XY", "Gi_Bin"])}
        bs, bn = bins("snow"), bins("nosnow")
        keys = [xy for xy in bs if xy in bn]
        vals = []
        for xy in keys:
            hs, hn, cs, cn = bs[xy] >= 2, bn[xy] >= 2, bs[xy] <= -2, bn[xy] <= -2
            v = 0
            if cs or cn: v = 5
            if cs and cn: v = 4
            if hs and not hn: v = 2
            if hn and not hs: v = 3
            if hs and hn: v = 1
            vals.append(v)
        g, gext = points_to_grid(keys, vals)
        ax = axes[i, 2]
        ax.imshow(g, extent=gext, cmap=ListedColormap(agree_col), vmin=-0.5, vmax=5.5, interpolation="nearest")
        draw_outline(ax, city, "boundary", edgecolor="0.2", lw=0.5)
        ax.set_xlim(gext[0], gext[1]); ax.set_ylim(gext[2], gext[3])
        mapax(ax); scalebar(ax, 5)
        ax.set_title(f"{NAMES[city]}: hot-spot agreement")
        letter(ax, next(k))
    cax = fig.add_axes([0.08, 0.075, 0.5, 0.018])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Median LST anomaly relative to the scene mean (°C)")
    fig.legend(handles=[Patch(facecolor=c, edgecolor="0.5", lw=0.4, label=l) for c, l in zip(agree_col, agree_lab)],
               loc="lower right", bbox_to_anchor=(0.995, 0.0), ncol=2, frameon=False, fontsize=6.5)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.17, hspace=0.14, wspace=0.04)
    fig.savefig(os.path.join(FIG, "ms_fig4_snow_maps.png"), dpi=300)
    plt.close(fig)


# ---------------- Figure 5: per-scene urban minus rural ----------------
def fig5():
    df = pd.read_csv(os.path.join(ROOT, "analysis", "scene_urban_rural.csv"), parse_dates=["date"])
    cold = df[df.month.isin([11, 12, 1, 2, 3])]
    fig, axes = plt.subplots(2, 3, figsize=(7.48, 5.0), gridspec_kw=dict(width_ratios=[1.1, 0.8, 1.3]))
    k = iter("abcdef")
    for i, city in enumerate(CITIES):
        g = cold[cold.city == city]
        ax = axes[i, 0]
        sc = ax.scatter(g.snow_rural, g.dT, c=g.lst_rural, cmap="coolwarm", s=10, edgecolor="0.3", lw=0.2,
                        vmin=-30, vmax=15)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("Share of rural land under snow"); ax.set_ylabel("Built-up − rural LST (°C)")
        ax.set_title(f"{NAMES[city]}, cold season (n = {len(g)})")
        letter(ax, next(k))
        ax = axes[i, 1]
        data, labs = [], []
        for mo, mname in [(11, "Nov"), (3, "Mar")]:
            for cls, cl in [("nosnow", "no snow"), ("snow", "snow")]:
                v = g[(g.month == mo) & (g.snow_class == cls)].dT
                data.append(v); labs.append(f"{mname}\n{cl}\n({len(v)})")
        bp = ax.boxplot(data, labels=labs, patch_artist=True, widths=0.6,
                        medianprops=dict(color="k", lw=1), flierprops=dict(ms=2))
        for p, c in zip(bp["boxes"], ["#fdd49e", "#9ecae1"] * 2):
            p.set_facecolor(c)
        ax.axhline(0, color="k", lw=0.5)
        ax.tick_params(axis="x", labelsize=6.5)
        ax.set_title("Same month")
        letter(ax, next(k))
        ax = axes[i, 2]
        a = df[df.city == city]
        for cls, col, lab in [("nosnow", "#d95f02", "No snow"), ("mixed", "0.6", "Partial snow"),
                              ("snow", "#1f78b4", "Snow-covered")]:
            v = a[a.snow_class == cls]
            ax.scatter(v.date.dt.dayofyear, v.dT, s=4, color=col, label=lab, alpha=0.75, linewidths=0)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks([15, 74, 135, 196, 258, 319]); ax.set_xticklabels(["Jan", "Mar", "May", "Jul", "Sep", "Nov"])
        ax.set_xlim(0, 366)
        ax.set_title(f"All scenes, 1984–2026 (n = {len(a)})")
        letter(ax, next(k))
        if i == 0:
            ax.legend(frameon=False, markerscale=2.5, loc="upper right")
    cax = fig.add_axes([0.08, 0.085, 0.25, 0.014])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label("Rural LST of the scene (°C), panels a and d", fontsize=7)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.95, bottom=0.22, hspace=0.62, wspace=0.3)
    fig.savefig(os.path.join(FIG, "ms_fig5_scenes.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    todo = sys.argv[1:] or ["fig1", "fig2", "fig3", "fig4", "fig5"]
    for f in todo:
        globals()[f]()
        print("done", f, flush=True)
