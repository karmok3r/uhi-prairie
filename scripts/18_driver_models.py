"""18_driver_models.py

Research question 3: which urban features explain the summer hot spots and the
cold-season pattern? Uses analysis/drivers_<city>.csv from 17_driver_variables.py.

1. Mean driver values by summer hot-spot group (intensifying, persistent, other hot,
   no pattern, cold spots).
2. For each response (summer_late, summer_change, cold_snow, cold_nosnow) an OLS model on
   standardised drivers. Standard errors are clustered in blocks of 12 x 12 cells
   (1.08 km) to allow for spatial autocorrelation of the residuals.
   Variance inflation factors are reported for the driver set.

Run with ArcGIS Pro's Python (needs statsmodels).
Outputs: analysis/driver_group_means.csv, analysis/driver_models.csv,
         analysis/driver_models.txt, figures/ms_fig6_drivers.png,
         manuscript/tab_drivers.tex
"""
import os
import io
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVERS = ["bld_cover", "log_bld_size", "built_change", "height", "tree", "herb", "log_water_dist"]
LABEL = {"bld_cover": "Building cover", "log_bld_size": "Building size (log)",
         "built_change": "Built-up gain 1985–2020", "height": "Building height",
         "tree": "Tree cover", "herb": "Grass and crop cover",
         "log_water_dist": "Distance to water (log)"}
RESP = {"summer_late": "Summer anomaly\n2017–2026",
        "summer_change": "Summer change\nsince 1984–1993",
        "cold_snow": "Cold season\nwith snow", "cold_nosnow": "Cold season\nwithout snow"}
GROUP = {"Intensifying Hot Spot": "Intensifying hot", "Persistent Hot Spot": "Persistent hot",
         "No Pattern Detected": "No pattern"}
out = io.StringIO()


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.write(s + "\n")


def group(p):
    if p in GROUP:
        return GROUP[p]
    return "Other hot" if "Hot" in p else "Cold spot"


coef_rows, mean_rows = [], []
for city in ["saskatoon", "regina"]:
    d = pd.read_csv(os.path.join(ROOT, "analysis", f"drivers_{city}.csv"))
    d["log_bld_size"] = np.log1p(d.bld_size)
    d["log_water_dist"] = np.log1p(d.water_dist)
    d["group"] = d.ehsa.map(group)
    d["block"] = (np.floor(d.x / 1080).astype(int).astype(str) + "_" +
                  np.floor(d.y / 1080).astype(int).astype(str))
    say(f"\n===== {city}: {len(d)} cells")

    g = d.groupby("group")[["bld_cover", "bld_size", "built_2020", "built_change", "height",
                            "tree", "herb", "water_dist"]].mean()
    g["cells"] = d.groupby("group").size()
    g["share"] = 100 * g.cells / len(d)
    g.insert(0, "city", city)
    mean_rows.append(g.reset_index())
    say(g.round(1).to_string())

    X = d[DRIVERS].astype(float)
    ok_x = X.notna().all(axis=1)
    Z = (X - X[ok_x].mean()) / X[ok_x].std()
    vif = [variance_inflation_factor(sm.add_constant(Z[ok_x]).values, i + 1)
           for i in range(len(DRIVERS))]
    say("VIF:", ", ".join(f"{k} {v:.1f}" for k, v in zip(DRIVERS, vif)))
    for resp in RESP:
        ok = ok_x & d[resp].notna()
        y = d.loc[ok, resp]
        m = sm.OLS(y, sm.add_constant(Z[ok])).fit(cov_type="cluster",
                                                  cov_kwds={"groups": d.loc[ok, "block"]})
        say(f"\n{resp}: n = {int(m.nobs)}, R2 = {m.rsquared:.2f}")
        ci = m.conf_int()
        for k in DRIVERS:
            say(f"  {k:15s} {m.params[k]:+.3f}  ({ci.loc[k, 0]:+.3f}, {ci.loc[k, 1]:+.3f})  "
                f"p = {m.pvalues[k]:.2g}")
            coef_rows.append(dict(city=city, response=resp, driver=k, coef=m.params[k],
                                  lo=ci.loc[k, 0], hi=ci.loc[k, 1], p=m.pvalues[k],
                                  r2=m.rsquared, n=int(m.nobs)))

coef = pd.DataFrame(coef_rows)
coef.to_csv(os.path.join(ROOT, "analysis", "driver_models.csv"), index=False)
means = pd.concat(mean_rows, ignore_index=True)
means.to_csv(os.path.join(ROOT, "analysis", "driver_group_means.csv"), index=False)
with open(os.path.join(ROOT, "analysis", "driver_models.txt"), "w", encoding="utf-8") as f:
    f.write(out.getvalue())

# ---------------- Figure: standardised coefficients ----------------
plt.rcParams.update({"font.family": "Arial", "font.size": 8})
fig, axes = plt.subplots(1, 4, figsize=(7.4, 3.6), sharey=True)
yy = np.arange(len(DRIVERS))[::-1]
r2s = {}
for ax, (resp, title), letter in zip(axes, RESP.items(), "abcd"):
    for city, col, off in [("saskatoon", "#1f77b4", 0.15), ("regina", "#d62728", -0.15)]:
        c = coef[(coef.city == city) & (coef.response == resp)].set_index("driver").loc[DRIVERS]
        ax.errorbar(c.coef, yy + off, xerr=[c.coef - c.lo, c.hi - c.coef], fmt="o", ms=3,
                    color=col, lw=0.9, capsize=0,
                    label=city.capitalize())
        r2s.setdefault(resp, []).append(f"{c.r2.iloc[0]:.2f}")
    ax.axvline(0, color="k", lw=0.6)
    ax.set_title(f"({letter}) {title}\nR² = " + " / ".join(r2s[resp]), fontsize=8, loc="left")
    ax.set_xlabel("°C per SD")
    ax.grid(axis="x", lw=0.3, color="0.85")
axes[0].set_yticks(yy)
axes[0].set_yticklabels([LABEL[k] for k in DRIVERS])
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=8)
fig.tight_layout(w_pad=0.6, rect=(0, 0, 1, 0.94))
fig.savefig(os.path.join(ROOT, "figures", "ms_fig6_drivers.png"), dpi=330)

# ---------------- Table: driver means by summer hot-spot group ----------------
order = ["Intensifying hot", "Persistent hot", "Other hot", "No pattern", "Cold spot"]
cols = [("share", ("Area", "(\\%)"), 1), ("bld_cover", ("Building", "cover (\\%)"), 1),
        ("bld_size", ("Building", "size (m$^2$)"), 0), ("built_change", ("Built-up", "gain (pp)"), 1),
        ("height", ("Height", "(m)"), 1), ("tree", ("Tree", "(\\%)"), 1),
        ("herb", ("Grass, crop", "(\\%)"), 1)]
lines = [r"\begin{tabular}{ll" + "r" * len(cols) + "}", r"\toprule",
         " & & " + " & ".join(c[1][0] for c in cols) + r" \\",
         "City & Summer group & " + " & ".join(c[1][1] for c in cols) + r" \\", r"\midrule"]
for i, city in enumerate(["saskatoon", "regina"]):
    if i:
        lines.append(r"\addlinespace")
    m = means[means.city == city].set_index("group")
    for j, gname in enumerate(order):
        if gname not in m.index:
            continue
        vals = [f"{m.loc[gname, k]:.{dec}f}" for k, _, dec in cols]
        lines.append((city.capitalize() if j == 0 else "") + f" & {gname} & " +
                     " & ".join(vals) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
with open(os.path.join(ROOT, "manuscript", "tab_drivers.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("wrote figure and table")
