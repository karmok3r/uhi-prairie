"""08_snow_test_per_scene.py

Per-scene test of whether the city-minus-rural surface temperature difference depends on
snow cover, holding month, the day's temperature and sun angle fixed.

Input : data/scene_stats/<city>_scene_zones.csv (from gee/04_scene_urban_rural.js)
Output: analysis/scene_urban_rural.csv         cleaned per-scene table
        analysis/snow_test_summary.txt         all numbers printed below
        figures/snow_test_per_scene.png
"""
import os
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_CLEAR = 0.8          # both the built-up and the rural zone must be at least 80% clear
SNOW_MIN, NOSNOW_MAX = 0.8, 0.1
COLD = [11, 12, 1, 2, 3]
out = io.StringIO()


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    out.write(line + "\n")


frames = []
for city in ["saskatoon", "regina"]:
    d = pd.read_csv(os.path.join(ROOT, "data", "scene_stats", f"{city}_scene_zones.csv"))
    d = d.drop(columns=[c for c in d.columns if c in ("system:index", ".geo")])
    d["date"] = pd.to_datetime(d["date"])
    for z, name in [(1, "built"), (2, "rural"), (3, "cityopen")]:
        d[f"clear_{name}"] = d[f"z{z}_clear"] / d[f"z{z}_n"]
        d[f"lst_{name}"] = d[f"z{z}_lst"] / d[f"z{z}_clear"].where(d[f"z{z}_clear"] > 0)
        d[f"snow_{name}"] = d[f"z{z}_snow"] / d[f"z{z}_clear"].where(d[f"z{z}_clear"] > 0)
    d = d[(d.clear_built >= MIN_CLEAR) & (d.clear_rural >= MIN_CLEAR)].copy()
    d = d.sort_values("clear_rural", ascending=False).drop_duplicates("date").sort_values("date")
    d["month"] = d.date.dt.month
    d["year"] = d.date.dt.year
    # Undetected cloud decks: implausibly cold scenes in the warm months.
    d = d[~(d.month.between(5, 9) & (d.lst_rural < 5))]
    d["dT"] = d.lst_built - d.lst_rural            # built-up minus rural
    d["dT_open"] = d.lst_cityopen - d.lst_rural    # other city land minus rural
    d["snow_class"] = np.select([d.snow_rural >= SNOW_MIN, d.snow_rural <= NOSNOW_MAX],
                                ["snow", "nosnow"], "mixed")
    frames.append(d)

df = pd.concat(frames, ignore_index=True)
keep = ["city", "scene_id", "sensor", "date", "year", "month", "sun_elevation", "lst_built",
        "lst_rural", "lst_cityopen", "snow_built", "snow_rural", "snow_cityopen",
        "clear_built", "clear_rural", "dT", "dT_open", "snow_class"]
df[keep].to_csv(os.path.join(ROOT, "analysis", "scene_urban_rural.csv"), index=False)
cold = df[df.month.isin(COLD)].copy()

say("Scenes kept (both zones at least 80% clear, one per date):")
say(df.groupby("city").size().to_string())
say("\nCold-season scenes (Nov-Mar, all years) by snow class:")
say(pd.crosstab(cold.city, cold.snow_class).to_string())

say("\n=== 1. Built-up minus rural LST (deg C) by snow class, cold season ===")
for city, g in cold.groupby("city"):
    t = g.groupby("snow_class").dT.agg(["count", "mean", "median", "std"]).round(2)
    say(f"\n{city}\n{t.to_string()}")
    a, b = g[g.snow_class == "snow"].dT, g[g.snow_class == "nosnow"].dT
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    say(f"snow minus no-snow: {a.mean() - b.mean():+.2f} C, Mann-Whitney p = {u.pvalue:.2g}")

say("\n=== 2. Month-matched: same comparison within single months ===")
rows = []
for city, g in cold.groupby("city"):
    for m in COLD:
        gm = g[g.month == m]
        a, b = gm[gm.snow_class == "snow"].dT, gm[gm.snow_class == "nosnow"].dT
        p = stats.mannwhitneyu(a, b).pvalue if len(a) >= 3 and len(b) >= 3 else np.nan
        rows.append(dict(city=city, month=m, n_snow=len(a), n_nosnow=len(b),
                         dT_snow=a.mean(), dT_nosnow=b.mean(), diff=a.mean() - b.mean(), p=p,
                         rural_lst_snow=gm[gm.snow_class == "snow"].lst_rural.mean(),
                         rural_lst_nosnow=gm[gm.snow_class == "nosnow"].lst_rural.mean()))
say(pd.DataFrame(rows).round(3).to_string(index=False))

say("\n=== 3. Regression on all cold-season scenes ===")
say("dT ~ rural snow share + rural LST + sun elevation + month (robust standard errors)")
for city, g in cold.groupby("city"):
    m = smf.ols("dT ~ snow_rural + lst_rural + sun_elevation + C(month)", data=g).fit(cov_type="HC3")
    ci = m.conf_int().loc["snow_rural"]
    say(f"\n{city}: n = {int(m.nobs)}, R2 = {m.rsquared:.2f}")
    say(f"  snow share 0 -> 1 changes dT by {m.params['snow_rural']:+.2f} C "
        f"(95% CI {ci[0]:+.2f} to {ci[1]:+.2f}, p = {m.pvalues['snow_rural']:.2g})")
    say(f"  rural LST: {m.params['lst_rural']:+.3f} C per C (p = {m.pvalues['lst_rural']:.2g}); "
        f"sun elevation: {m.params['sun_elevation']:+.3f} C per degree (p = {m.pvalues['sun_elevation']:.2g})")
    m0 = smf.ols("dT ~ lst_rural + sun_elevation + C(month)", data=g).fit()
    say(f"  R2 without the snow term: {m0.rsquared:.2f}")

say("\n=== 4. Same regression for other city land (parks, fields) minus rural ===")
for city, g in cold.dropna(subset=["dT_open"]).groupby("city"):
    m = smf.ols("dT_open ~ snow_rural + lst_rural + sun_elevation + C(month)", data=g).fit(cov_type="HC3")
    ci = m.conf_int().loc["snow_rural"]
    say(f"{city}: snow effect {m.params['snow_rural']:+.2f} C (95% CI {ci[0]:+.2f} to {ci[1]:+.2f})")

say("\n=== 5. Is the built-up zone itself less snowy? (snow-class scenes only) ===")
s = cold[cold.snow_class == "snow"].groupby("city")[["snow_built", "snow_cityopen", "snow_rural"]].mean()
say(s.round(3).to_string())

say("\n=== 6. Monthly mean built-up minus rural LST, all years ===")
monthly = df.groupby(["city", "month"]).dT.agg(["mean", "count"]).round(2).unstack(0)
say(monthly.to_string())

with open(os.path.join(ROOT, "analysis", "snow_test_summary.txt"), "w", encoding="utf-8") as f:
    f.write(out.getvalue())

# ---------------- Figure ----------------
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
for i, city in enumerate(["saskatoon", "regina"]):
    g = cold[cold.city == city]
    ax = axes[i, 0]
    sc = ax.scatter(g.snow_rural, g.dT, c=g.lst_rural, cmap="coolwarm", s=22, edgecolor="0.3", lw=0.3)
    fig.colorbar(sc, ax=ax, label="rural LST that day (deg C)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("share of rural land under snow")
    ax.set_ylabel("built-up minus rural LST (deg C)")
    ax.set_title(f"{city.title()}: cold-season scenes (n = {len(g)})")

    ax = axes[i, 1]
    data, labels = [], []
    for m, name in [(11, "Nov"), (3, "Mar")]:
        for cls, lab in [("nosnow", "no snow"), ("snow", "snow")]:
            v = g[(g.month == m) & (g.snow_class == cls)].dT
            data.append(v)
            labels.append(f"{name}\n{lab}\n(n={len(v)})")
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, col in zip(bp["boxes"], ["#fdd49e", "#9ecae1", "#fdd49e", "#9ecae1"]):
        patch.set_facecolor(col)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("built-up minus rural LST (deg C)")
    ax.set_title(f"{city.title()}: same month, with and without snow")

    ax = axes[i, 2]
    a = df[df.city == city]
    for cls, col, lab in [("nosnow", "#d95f02", "no snow"), ("mixed", "0.6", "partial snow"),
                          ("snow", "#1f78b4", "snow")]:
        v = a[a.snow_class == cls]
        ax.scatter(v.date.dt.dayofyear, v.dT, s=10, color=col, label=lab, alpha=0.7)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("day of year")
    ax.set_ylabel("built-up minus rural LST (deg C)")
    ax.set_title(f"{city.title()}: all scenes, 1984-2026 (n = {len(a)})")
    ax.legend(frameon=False, fontsize=9)
fig.suptitle("Does the city's daytime surface warmth depend on snow cover? One point per Landsat scene",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(ROOT, "figures", "snow_test_per_scene.png"), dpi=100)
