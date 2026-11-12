"""16_air_temp_check.py

Compares the satellite surface contrast (LST of built-up land minus LST of rural
cropland, dT from 08_snow_test_per_scene.py) with the air-temperature contrast measured by
ECCC stations on the same dates (data from 14_fetch_station_air_temp.py).

Air-temperature contrast, per scene date:
  dTa_day   = urban station minus rural station(s), mean of 11:00 and 12:00 local standard
              time (the Landsat overpass is about 11:45 CST at these longitudes)
  dTa_night = the same for 00:00-05:00 of the same date (the night before the overpass)
Urban station: the international airport of each city (merged across station numbers,
with the co-located RCS station filling gaps). Rural: Bratt's Lake (25 km S of Regina);
for Saskatoon the mean of Outlook, Rosetown East and Watrous East (76-105 km away), because
no rural hourly station is closer.
Also reported: mean of urban air temperature at overpass vs study-area rural LST, as a
sanity check that LST tracks air temperature.

Outputs: analysis/air_temp_check.csv, analysis/air_temp_check.txt,
         figures/ms_fig7_air_temp.png
"""
import os
import io
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AIR = os.path.join(ROOT, "data", "air_temp")
URBAN = {"saskatoon": ["4057120", "4057152", "4057165"],
         "regina": ["4016560", "4016566", "4016699"]}
RURAL = {"saskatoon": ["4055736", "4046884", "4018642"], "regina": ["4010811"]}
COLD = [11, 12, 1, 2, 3]
out = io.StringIO()


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.write(s + "\n")


def load(cid):
    d = pd.read_csv(os.path.join(AIR, f"{cid}.csv"))
    d["date"] = pd.to_datetime(d.LOCAL_DATE).dt.normalize()
    d = d.dropna(subset=["TEMP"])
    day = d[d.LOCAL_HOUR.isin([11, 12])].groupby("date").TEMP.mean()
    night = d[d.LOCAL_HOUR.between(0, 5)].groupby("date").TEMP.mean()
    wind = d[d.LOCAL_HOUR.isin([11, 12])].groupby("date").WIND_SPEED.mean()
    return pd.DataFrame({"day": day, "night": night, "wind": wind})


def first_valid(frames):
    """Combine stations in priority order: take the first non-missing value per date."""
    res = frames[0]
    for f in frames[1:]:
        res = res.combine_first(f)
    return res


scenes = pd.read_csv(os.path.join(ROOT, "analysis", "scene_urban_rural.csv"),
                     parse_dates=["date"])
rows = []
for city in ["saskatoon", "regina"]:
    urb = first_valid([load(c) for c in URBAN[city]])
    rur_list = [load(c) for c in RURAL[city]]
    rur = pd.concat(rur_list, axis=1, keys=range(len(rur_list)))
    rur = pd.DataFrame({k: rur.xs(k, axis=1, level=1).mean(axis=1) for k in ["day", "night"]})
    s = scenes[scenes.city == city].set_index("date")
    j = s.join(urb.add_prefix("urb_")).join(rur.add_prefix("rur_"))
    j["dTa_day"] = j.urb_day - j.rur_day
    j["dTa_night"] = j.urb_night - j.rur_night
    j = j.dropna(subset=["dTa_day"])
    j["season"] = np.where(j.month.isin(COLD), "cold", np.where(j.month.isin([6, 7, 8]), "summer", "other"))
    j["city"] = city
    rows.append(j.reset_index())
df = pd.concat(rows, ignore_index=True)
df.to_csv(os.path.join(ROOT, "analysis", "air_temp_check.csv"), index=False)

for city in ["saskatoon", "regina"]:
    d = df[df.city == city]
    say(f"\n===== {city}: {len(d)} scenes with station data, "
        f"{d.date.min():%Y-%m-%d} to {d.date.max():%Y-%m-%d}")
    r = np.corrcoef(d.urb_day, d.lst_rural)[0, 1]
    say(f"  urban air T at overpass vs rural LST, all scenes: r = {r:.2f}")
    for season in ["summer", "cold"]:
        e = d[d.season == season]
        say(f"  {season}: n={len(e)}  LST dT {e.dT.mean():+.2f}  air dT day {e.dTa_day.mean():+.2f} "
            f"(sd {e.dTa_day.std():.2f})  air dT night {e.dTa_night.mean():+.2f}  "
            f"r(LST dT, air dT day) = {np.corrcoef(e.dT, e.dTa_day)[0, 1]:.2f}")
    c = d[d.season == "cold"]
    for cls in ["snow", "nosnow"]:
        e = c[c.snow_class == cls]
        if len(e) == 0:
            continue
        say(f"  cold {cls:6s}: n={len(e):3d}  LST dT {e.dT.mean():+.2f}  "
            f"air dT day {e.dTa_day.mean():+.2f} (se {e.dTa_day.std() / np.sqrt(len(e)):.2f})  "
            f"air dT night {e.dTa_night.mean():+.2f} (se {e.dTa_night.std() / np.sqrt(len(e)):.2f})")
    cc = c[c.snow_class.isin(["snow", "nosnow"])].copy()
    cc["snow"] = (cc.snow_class == "snow").astype(int)
    for v in ["dTa_day", "dTa_night"]:
        m = smf.ols(f"{v} ~ snow + C(month)", data=cc.dropna(subset=[v])).fit(cov_type="HC3")
        lo, hi = m.conf_int().loc["snow"]
        say(f"  {v} ~ snow + month: snow effect {m.params['snow']:+.2f} C "
            f"(95% CI {lo:+.2f} to {hi:+.2f}, p = {m.pvalues['snow']:.2g}, n = {int(m.nobs)})")

with open(os.path.join(ROOT, "analysis", "air_temp_check.txt"), "w", encoding="utf-8") as f:
    f.write(out.getvalue())

# ---------------- figure ----------------
plt.rcParams.update({"font.family": "Arial", "font.size": 9})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
groups = [("summer", None, "Summer"), ("cold", "nosnow", "Cold,\nno snow"), ("cold", "snow", "Cold,\nsnow")]
for ax, city, letter in zip(axes, ["saskatoon", "regina"], "ab"):
    d = df[df.city == city]
    pos = np.arange(len(groups))
    for k, (var, col, lab) in enumerate([("dT", "#c0504d", "Surface (LST)"),
                                         ("dTa_day", "#4f81bd", "Air, 11:00–12:00 (overpass)"),
                                         ("dTa_night", "#7f7f7f", "Air, 00:00–05:00 (night)")]):
        m, e = [], []
        for season, cls, _ in groups:
            g = d[(d.season == season) & ((d.snow_class == cls) if cls else True)][var].dropna()
            m.append(g.mean())
            e.append(1.96 * g.std() / np.sqrt(len(g)))
        ax.bar(pos + (k - 1) * 0.26, m, 0.26, yerr=e, color=col, label=lab, capsize=2,
               error_kw={"lw": 0.8})
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(pos)
    ns = [len(d[(d.season == s) & ((d.snow_class == c) if c else True)]) for s, c, _ in groups]
    ax.set_xticklabels([f"{g[2]}\n(n = {n})" for g, n in zip(groups, ns)])
    ax.set_title(f"({letter}) {city.capitalize()}", loc="left", fontsize=9)
axes[0].set_ylabel("Urban minus rural (°C)")
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "figures", "ms_fig7_air_temp.png"), dpi=300)
print("saved figure")
