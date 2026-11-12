"""20_rural_greenness_test.py

Does change in the farmland around each city explain part of the summer anomaly trend?

The summer anomaly of a pixel is its LST minus the mean LST of the 40 km study box, and
most of the box is cropland. If Prairie farmland became greener (less summerfallow, more
canola and pulses; Campbell et al. 2002; Betts et al. 2013), the box mean cools and city
anomalies rise without any change in the city.

Inputs (gee/05_rural_greenness_crops.js, copied to data/rural_greenness/):
  <city>_scene_ndvi.csv    per scene: clear-sky NDVI and LST sums for four zones
                           (1 built-up in city, 2 rural, 3 other city land, 0 rest of box)
  <city>_aci_classes.csv   per year 2009+: AAFC Annual Crop Inventory class counts in the
                           rural zone and in the whole box
and data/scene_stats/<city>_scenes_used.csv (the scenes that entered the composites).

Tests, per city, on the retained June-August scenes:
  1. yearly rural NDVI: Sen slope and Mann-Kendall for 1984-2012 (Landsat 5/7) and
     2013-2026 (Landsat 8/9), and OLS trend with a 2013 step;
  2. per-scene OLS  anomaly ~ year + step2013 + month  with and without rural NDVI
     (standard errors clustered by year): how much of the trend does rural NDVI absorb?
     anomaly = city LST (zones 1+3) minus box LST; also built-up minus rural LST;
  3. crop composition of the rural zone, 2009-2025 (shares of cropland classes).

Run with ArcGIS Pro's Python. Outputs: analysis/rural_greenness_test.txt,
analysis/rural_greenness_yearly.csv, analysis/rural_crop_shares.csv,
figures/ms_fig8_rural_greenness.png
"""
import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "rural_greenness")
CITIES = ["saskatoon", "regina"]
NAMES = {"saskatoon": "Saskatoon", "regina": "Regina"}
MIN_RURAL_CLEAR = 0.5
out = io.StringIO()

# AAFC ACI class groups (cropland classes only; codes from the ACI legend)
GROUPS = {
    "fallow": [131],
    "cereals": [133, 134, 135, 136, 137, 138, 139, 140, 142, 145, 146, 147],
    "canola and oilseeds": [150, 151, 152, 153, 154, 155, 156, 157, 158],
    "pulses": [160, 161, 162, 163, 167, 168, 174],
    "forage and pasture": [122],
    "grassland": [110],
}


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    out.write(line + "\n")


def mann_kendall(y):
    y = np.asarray(y, float)
    n = len(y)
    s = sum(np.sign(y[j] - y[i]) for i in range(n - 1) for j in range(i + 1, n))
    var = n * (n - 1) * (2 * n + 5) / 18.0
    z = (s - np.sign(s)) / np.sqrt(var) if s != 0 else 0.0
    return 2 * (1 - stats.norm.cdf(abs(z)))


def sen_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(y)
    return np.median([(y[j] - y[i]) / (x[j] - x[i])
                      for i in range(n - 1) for j in range(i + 1, n) if x[j] != x[i]])


def load_scenes(city):
    d = pd.read_csv(os.path.join(DATA, f"{city}_scene_ndvi.csv"))
    used = pd.read_csv(os.path.join(ROOT, "data", "scene_stats", f"{city}_scenes_used.csv"))
    d = d[d.scene_id.isin(used.scene_id)].copy()
    d["date"] = pd.to_datetime(d["date"])
    d["year"], d["month"] = d.date.dt.year, d.date.dt.month
    d = d[d.month.between(6, 8)]
    # One scene per date, as in the composites (keep the clearer one)
    d = d.sort_values("z2_clear", ascending=False).drop_duplicates(["date"])
    m = lambda v, z: d[f"z{z}_{v}"] / d[f"z{z}_clear"]
    d["box_lst"] = sum(d[f"z{z}_lst"] for z in range(4)) / sum(d[f"z{z}_clear"] for z in range(4))
    d["city_lst"] = (d.z1_lst + d.z3_lst) / (d.z1_clear + d.z3_clear)
    d = d[d.box_lst >= 10]                       # same cold-cloud filter as the composites
    d["anom"] = d.city_lst - d.box_lst
    d["dT"] = m("lst", 1) - m("lst", 2)
    d["ndvi_rural"] = m("ndvi", 2)
    d["ndvi_city"] = (d.z1_ndvi + d.z3_ndvi) / (d.z1_clear + d.z3_clear)
    d["lst_rural"] = m("lst", 2)
    d = d[d.z2_clear / d.z2_n >= MIN_RURAL_CLEAR]
    d["step"] = (d.year >= 2013).astype(int)
    d["yr10"] = (d.year - 2000) / 10.0
    return d.sort_values("date")


def crop_shares(city):
    a = pd.read_csv(os.path.join(DATA, f"{city}_aci_classes.csv"))
    rows = []
    for _, r in a.iterrows():
        h = {int(k): v for k, v in json.loads(r["rural"]).items()}
        crop_total = sum(v for k, v in h.items() if 110 <= k < 200 or k == 122)
        row = {"city": city, "year": int(r["year"])}
        for g, codes in GROUPS.items():
            row[g] = 100 * sum(h.get(c, 0) for c in codes) / crop_total
        row["other crops"] = 100 - sum(row[g] for g in GROUPS)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("year")


yearly_all, crops_all, fits = [], [], {}
for city in CITIES:
    d = load_scenes(city)
    say(f"\n===== {NAMES[city]}: {len(d)} summer scenes, {d.year.nunique()} years "
        f"(rural zone at least {MIN_RURAL_CLEAR:.0%} clear)")

    y = d.groupby("year")[["anom", "dT", "ndvi_rural", "ndvi_city", "lst_rural"]].mean()
    y["city"] = city
    yearly_all.append(y.reset_index())

    say("\n1. Rural NDVI, June-August yearly means")
    for lab, lo, hi in [("1984-2012 (Landsat 5/7)", 1984, 2012), ("2013-2026 (Landsat 8/9)", 2013, 2026)]:
        s = y.loc[lo:hi, "ndvi_rural"]
        say(f"  {lab}: mean {s.mean():.3f}; Sen slope {10 * sen_slope(s.index, s.values):+.3f} per decade, "
            f"Mann-Kendall p = {mann_kendall(s.values):.3g}")
    yy = y.reset_index()
    yy["step"] = (yy.year >= 2013).astype(int)
    f = smf.ols("ndvi_rural ~ year + step", yy).fit()
    say(f"  OLS 1984-2026 with 2013 step: {10 * f.params['year']:+.3f} per decade (p = {f.pvalues['year']:.3g}); "
        f"step {f.params['step']:+.3f} (p = {f.pvalues['step']:.3g})")
    e1, e2 = y.loc[1984:1993, "ndvi_rural"].mean(), y.loc[2003:2012, "ndvi_rural"].mean()
    say(f"  1984-93 mean {e1:.3f}, 2003-12 mean {e2:.3f} (change {e2 - e1:+.3f})")

    say("\n2. Per-scene models, OLS with month fixed effects, SE clustered by year")
    fits[city] = {}
    for resp, lab in [("anom", "city minus box LST (anomaly)"), ("dT", "built-up minus rural LST")]:
        res = {}
        for name, form in [("base", f"{resp} ~ yr10 + step + C(month)"),
                           ("+ rural NDVI", f"{resp} ~ yr10 + step + C(month) + ndvi_rural")]:
            m = smf.ols(form, d).fit(cov_type="cluster", cov_kwds={"groups": d.year})
            res[name] = m
        b, g = res["base"], res["+ rural NDVI"]
        share = 1 - g.params["yr10"] / b.params["yr10"]
        ci = g.conf_int().loc["ndvi_rural"] / 10
        say(f"  {lab}:")
        say(f"    trend without NDVI {b.params['yr10']:+.3f} C/decade (p = {b.pvalues['yr10']:.3g}), "
            f"with NDVI {g.params['yr10']:+.3f} (p = {g.pvalues['yr10']:.3g}): "
            f"{100 * share:.0f}% of the trend absorbed")
        say(f"    rural NDVI effect {g.params['ndvi_rural'] / 10:+.2f} C per +0.1 NDVI "
            f"(95% CI {ci[0]:+.2f} to {ci[1]:+.2f}, p = {g.pvalues['ndvi_rural']:.3g}); "
            f"R2 {b.rsquared:.2f} -> {g.rsquared:.2f}")
        fits[city][resp] = (b, g, share)

    # Landsat 5/7 period alone (no sensor change inside it)
    p1 = d[d.year <= 2012]
    b = smf.ols("anom ~ yr10 + C(month)", p1).fit(cov_type="cluster", cov_kwds={"groups": p1.year})
    g = smf.ols("anom ~ yr10 + C(month) + ndvi_rural", p1).fit(cov_type="cluster", cov_kwds={"groups": p1.year})
    say(f"  anomaly, 1984-2012 only: trend {b.params['yr10']:+.3f} -> {g.params['yr10']:+.3f} C/decade "
        f"with rural NDVI ({100 * (1 - g.params['yr10'] / b.params['yr10']):.0f}% absorbed); "
        f"NDVI effect {g.params['ndvi_rural'] / 10:+.2f} C per +0.1 (p = {g.pvalues['ndvi_rural']:.3g})")
    fits[city]["p1"] = (b, g)

    say("\n3. Rural crop composition (AAFC Annual Crop Inventory), % of agricultural land")
    c = crop_shares(city)
    crops_all.append(c)
    first, last = c.iloc[0], c.iloc[-1]
    for col in list(GROUPS) + ["other crops"]:
        sl = 10 * sen_slope(c.year, c[col])
        say(f"  {col:22s} {int(first.year)}: {first[col]:5.1f}   {int(last.year)}: {last[col]:5.1f}   "
            f"Sen slope {sl:+.1f} pp/decade (MK p = {mann_kendall(c[col].values):.2g})")
    cy = c.set_index("year").join(y[["ndvi_rural", "anom"]], how="inner")
    for col in ["fallow", "canola and oilseeds", "pulses", "cereals"]:
        r, p = stats.pearsonr(cy[col], cy["ndvi_rural"])
        say(f"  yearly r({col}, rural NDVI) = {r:+.2f} (p = {p:.2g}, n = {len(cy)})")

    # 4. Hot-spot cells relative to the analysis-area mean. The rural reference shifts every
    #    cell of a year by the same amount, so it cancels in this difference.
    import netCDF4
    import arcpy
    with netCDF4.Dataset(os.path.join(ROOT, "analysis", f"{city}_summer_anom_90m.nc")) as ds:
        a = np.ma.filled(ds["anom"][:].astype("f8"), np.nan)
        a[a == -9999] = np.nan
        xs, ys = ds["x"][:], ds["y"][:]
    years = 1984 + np.arange(a.shape[0])
    fc = os.path.join(ROOT, "analysis", "uhi_prairie.gdb", f"{city}_summer_ehsa")
    hot = np.zeros(a.shape[1:], bool)
    for (x, yv), p in arcpy.da.SearchCursor(fc, ["SHAPE@XY", "PATTERN"]):
        if p == "Intensifying Hot Spot":
            c_, r_ = int(round((x - xs[0]) / (xs[1] - xs[0]))), int(round((yv - ys[0]) / (ys[1] - ys[0])))
            if 0 <= r_ < hot.shape[0] and 0 <= c_ < hot.shape[1]:
                hot[r_, c_] = True
    area = np.nanmean(a[:, np.isfinite(a).any(axis=0)], axis=1)
    rel = np.nanmean(a[:, hot], axis=1) - area
    dd = pd.DataFrame({"year": years, "area": area, "rel": rel, "step": (years >= 2013).astype(int)})
    dd = dd.merge(y[["ndvi_rural"]].reset_index(), on="year", how="left")
    say("\n4. Composite anomaly series (as in scripts/11), with yearly rural NDVI")
    for col, lab in [("area", "analysis-area mean"), ("rel", "intensifying hot spots minus area mean")]:
        b = smf.ols(f"{col} ~ year + step", dd).fit()
        g = smf.ols(f"{col} ~ year + step + ndvi_rural", dd).fit()
        say(f"  {lab}: trend {10 * b.params['year']:+.3f} C/decade (p = {b.pvalues['year']:.3g}); "
            f"with rural NDVI {10 * g.params['year']:+.3f} (p = {g.pvalues['year']:.3g}); "
            f"NDVI effect {g.params['ndvi_rural'] / 10:+.2f} C per +0.1 (p = {g.pvalues['ndvi_rural']:.3g})")
        e1 = dd[dd.year <= 2012]
        say(f"    1984-2012 Sen slope {10 * sen_slope(e1.year, e1[col]):+.3f} C/decade "
            f"(MK p = {mann_kendall(e1[col].values):.3g})")

yearly = pd.concat(yearly_all)
yearly.to_csv(os.path.join(ROOT, "analysis", "rural_greenness_yearly.csv"), index=False)
pd.concat(crops_all).round(2).to_csv(os.path.join(ROOT, "analysis", "rural_crop_shares.csv"), index=False)
with open(os.path.join(ROOT, "analysis", "rural_greenness_test.txt"), "w", encoding="utf-8") as fh:
    fh.write(out.getvalue())

# ---------- Figure ----------
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
C_NDVI, C_ANOM = "#2a7d4f", "#c0392b"
for j, city in enumerate(CITIES):
    y = yearly[yearly.city == city]
    ax = axes[0, j]
    ax.axvspan(2012.5, 2026.5, color="0.92", zorder=0)
    ax.plot(y.year, y.ndvi_rural, "o-", color=C_NDVI, ms=3, lw=1)
    ax.set_ylabel("Rural NDVI (June–August)", color=C_NDVI)
    ax.tick_params(axis="y", colors=C_NDVI)
    ax2 = ax.twinx()
    ax2.plot(y.year, y.anom, "s--", color=C_ANOM, ms=3, lw=1)
    ax2.set_ylabel("City minus box LST (°C)", color=C_ANOM)
    ax2.tick_params(axis="y", colors=C_ANOM)
    ax.set_xlim(1983, 2027)
    ax.set_title(f"({'ab'[j]}) {NAMES[city]}", loc="left")

    c = crops_all[j]
    ax = axes[1, j]
    cols = ["fallow", "cereals", "canola and oilseeds", "pulses", "forage and pasture", "grassland", "other crops"]
    colors = ["#8c6d31", "#e6ab02", "#f2d43d", "#66a61e", "#1b9e77", "#a6d96a", "#bdbdbd"]
    ax.stackplot(c.year, *[c[k] for k in cols], labels=cols, colors=colors, lw=0)
    ax.set_xlim(c.year.min(), c.year.max())
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of rural farmland (%)")
    ax.set_title(f"({'cd'[j]}) {NAMES[city]}, crop inventory", loc="left")
h, l = axes[1, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.09))
fig.savefig(os.path.join(ROOT, "figures", "ms_fig8_rural_greenness.png"), dpi=300, bbox_inches="tight")
print("\nsaved figure")
