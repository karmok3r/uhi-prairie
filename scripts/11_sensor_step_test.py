"""11_sensor_step_test.py

Checks whether the summer trends are an artefact of the 2013 change of thermal sensor
(Landsat 5/7 to Landsat 8/9). For each city it takes the yearly summer anomaly averaged
over (a) the whole analysis area and (b) the baseline intensifying hot-spot cells, and
  1. fits  anomaly ~ year + step(year >= 2013)  by OLS,
  2. runs the Mann-Kendall test separately on 1984-2012 and 2013-2026,
  3. reports the Sen slope for each part.
If the trend survives the step term and appears within both sensor eras, it is not a
sensor artefact.

Run with ArcGIS Pro's Python. Output: analysis/sensor_step_test.txt
"""
import os
import io
import numpy as np
import pandas as pd
import netCDF4
import arcpy
from scipy import stats
import statsmodels.formula.api as smf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GDB = os.path.join(ROOT, "analysis", "uhi_prairie.gdb")
out = io.StringIO()


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    out.write(line + "\n")


def mann_kendall(y):
    """Two-sided Mann-Kendall test (no tie correction needed for continuous data)."""
    y = np.asarray(y, float)
    n = len(y)
    s = sum(np.sign(y[j] - y[i]) for i in range(n - 1) for j in range(i + 1, n))
    var = n * (n - 1) * (2 * n + 5) / 18.0
    z = (s - np.sign(s)) / np.sqrt(var) if s != 0 else 0.0
    return z, 2 * (1 - stats.norm.cdf(abs(z)))


def sen_slope(y):
    y = np.asarray(y, float)
    n = len(y)
    return np.median([(y[j] - y[i]) / (j - i) for i in range(n - 1) for j in range(i + 1, n)])


for city in ["saskatoon", "regina"]:
    with netCDF4.Dataset(os.path.join(ROOT, "analysis", f"{city}_summer_anom_90m.nc")) as ds:
        a = np.ma.filled(ds["anom"][:].astype("f8"), np.nan)
        a[a == -9999] = np.nan
        xs, ys = ds["x"][:], ds["y"][:]
    years = 1984 + np.arange(a.shape[0])

    # Baseline intensifying hot-spot cells, matched to the grid by coordinates.
    fc = os.path.join(GDB, f"{city}_summer_ehsa")
    pts = [s for s, p in arcpy.da.SearchCursor(fc, ["SHAPE@XY", "PATTERN"])
           if p == "Intensifying Hot Spot"]
    hot = np.zeros(a.shape[1:], bool)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    for x, y in pts:
        c = int(round((x - xs[0]) / dx))
        r = int(round((y - ys[0]) / dy))
        if 0 <= r < hot.shape[0] and 0 <= c < hot.shape[1]:
            hot[r, c] = True

    say(f"\n===== {city} summer, {len(pts)} intensifying hot-spot cells "
        f"({hot.sum()} matched to grid)")
    for label, sel in [("whole analysis area", np.isfinite(a).any(axis=0)),
                       ("intensifying hot spots", hot)]:
        series = np.nanmean(a[:, sel], axis=1)
        df = pd.DataFrame({"year": years, "anom": series, "step": (years >= 2013).astype(int)})
        m = smf.ols("anom ~ year + step", data=df).fit()
        m0 = smf.ols("anom ~ year", data=df).fit()
        e1, e2 = df[df.year <= 2012], df[df.year >= 2013]
        z1, p1 = mann_kendall(e1.anom)
        z2, p2 = mann_kendall(e2.anom)
        say(f"\n{label}:")
        say(f"  trend without step term: {10 * m0.params['year']:+.3f} C per decade "
            f"(p = {m0.pvalues['year']:.2g})")
        say(f"  trend with step term:    {10 * m.params['year']:+.3f} C per decade "
            f"(p = {m.pvalues['year']:.2g}); step at 2013: {m.params['step']:+.3f} C "
            f"(p = {m.pvalues['step']:.2g})")
        say(f"  1984-2012 (Landsat 5/7): Sen slope {10 * sen_slope(e1.anom):+.3f} C per decade, "
            f"Mann-Kendall p = {p1:.2g}")
        say(f"  2013-2026 (Landsat 8/9): Sen slope {10 * sen_slope(e2.anom):+.3f} C per decade, "
            f"Mann-Kendall p = {p2:.2g}")
        say(f"  era means: 1984-2012 {e1.anom.mean():+.3f} C, 2013-2026 {e2.anom.mean():+.3f} C")

with open(os.path.join(ROOT, "analysis", "sensor_step_test.txt"), "w", encoding="utf-8") as f:
    f.write(out.getvalue())
