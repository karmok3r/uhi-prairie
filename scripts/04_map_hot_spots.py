"""04_map_hot_spots.py

Maps the Emerging Hot Spot Analysis categories for both cities and seasons, and writes
a summary table. Run with ArcGIS Pro's Python after 03_emerging_hot_spots.py.
Outputs: figures/ehsa_pilot_maps.png and analysis/ehsa_category_summary.csv
"""
import os
import numpy as np
import pandas as pd
import arcpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GDB = os.path.join(ROOT, "analysis", "uhi_prairie.gdb")
CELL = 90.0

# Category names as written by ArcGIS, with a hot (red) and cold (blue) colour ramp.
COLORS = {
    "New Hot Spot": "#ff00c5", "Consecutive Hot Spot": "#e60000",
    "Intensifying Hot Spot": "#730000", "Persistent Hot Spot": "#a80000",
    "Diminishing Hot Spot": "#ff7f7f", "Sporadic Hot Spot": "#ffbebe",
    "Oscillating Hot Spot": "#ffe6e0", "Historical Hot Spot": "#d7b09e",
    "New Cold Spot": "#00c5ff", "Consecutive Cold Spot": "#0070ff",
    "Intensifying Cold Spot": "#002673", "Persistent Cold Spot": "#004da8",
    "Diminishing Cold Spot": "#73b2ff", "Sporadic Cold Spot": "#bed2ff",
    "Oscillating Cold Spot": "#e6eeff", "Historical Cold Spot": "#9eaad7",
    "No Pattern Detected": "#f0f0f0",
}
ORDER = list(COLORS)


def read(city, season):
    fc = os.path.join(GDB, f"{city}_{season}_ehsa")
    rows = [(s[0], s[1], p) for s, p in arcpy.da.SearchCursor(fc, ["SHAPE@XY", "PATTERN"])]
    return pd.DataFrame(rows, columns=["x", "y", "pattern"])


def to_rgb_grid(df):
    cols = ((df.x - df.x.min()) / CELL).round().astype(int)
    rows = ((df.y.max() - df.y) / CELL).round().astype(int)
    img = np.ones((rows.max() + 1, cols.max() + 1, 3))
    rgb = np.array([matplotlib.colors.to_rgb(COLORS.get(p, "#f0f0f0")) for p in df.pattern])
    img[rows, cols] = rgb
    extent = [df.x.min() - CELL / 2, df.x.max() + CELL / 2,
              df.y.min() - CELL / 2, df.y.max() + CELL / 2]
    return img, extent


fig, axes = plt.subplots(2, 2, figsize=(15, 13))
summary = {}
titles = {"summer": "Summer (Jun-Aug), 1984-2026", "cold": "Cold season (Nov-Mar), 2000-2026"}
for i, city in enumerate(["saskatoon", "regina"]):
    for j, season in enumerate(["summer", "cold"]):
        df = read(city, season)
        summary[f"{city}_{season}"] = df.pattern.value_counts()
        img, extent = to_rgb_grid(df)
        ax = axes[i, j]
        ax.imshow(img, extent=extent, interpolation="nearest")
        ax.set_title(f"{city.title()}: {titles[season]}", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        # 5 km scale bar
        x0, y0 = extent[0] + 800, extent[2] + 800
        ax.plot([x0, x0 + 5000], [y0, y0], color="k", lw=3)
        ax.text(x0 + 2500, y0 + 300, "5 km", ha="center", fontsize=9)

handles = [Patch(facecolor=COLORS[k], edgecolor="0.5", label=k) for k in ORDER]
fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9, frameon=False)
fig.suptitle("Emerging hot-spot categories of the land surface temperature anomaly "
             "(pilot: 90 m cells, 200 m neighbourhood)", fontsize=13)
fig.tight_layout(rect=[0, 0.07, 1, 0.97])
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
fig.savefig(os.path.join(ROOT, "figures", "ehsa_pilot_maps.png"), dpi=110)

table = pd.DataFrame(summary).reindex(ORDER).fillna(0).astype(int)
pct = (100 * table / table.sum()).round(1)
out = table.join(pct.add_suffix("_pct"))
out.to_csv(os.path.join(ROOT, "analysis", "ehsa_category_summary.csv"))
print(pct.to_string())
