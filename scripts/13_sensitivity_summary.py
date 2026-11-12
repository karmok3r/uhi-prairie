"""13_sensitivity_summary.py

Compares every sensitivity variant with the baseline (output of 10_sensitivity.py).
For each city and season it reports
  - the share of the analysis area in each grouped hot-spot class, and
  - agreement with the baseline at the baseline cell centres: the share of baseline
    intensifying hot spots, persistent hot spots and persistent cold spots that keep the
    same class, and Cohen's kappa over all seven classes.
Variants at a different cell size are sampled at the baseline cell centres.

Outputs: analysis/sensitivity_summary.csv and analysis/sensitivity_summary.txt
Runs with any Python that has pandas and numpy.
"""
import os
import io
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEN = os.path.join(ROOT, "analysis", "sensitivity")
GROUPS = {
    "New Hot Spot": 1, "Consecutive Hot Spot": 1, "Intensifying Hot Spot": 1,
    "Persistent Hot Spot": 2,
    "Sporadic Hot Spot": 3, "Oscillating Hot Spot": 3, "Diminishing Hot Spot": 3, "Historical Hot Spot": 3,
    "Sporadic Cold Spot": 4, "Oscillating Cold Spot": 4, "Diminishing Cold Spot": 4, "Historical Cold Spot": 4,
    "Persistent Cold Spot": 5,
    "New Cold Spot": 6, "Consecutive Cold Spot": 6, "Intensifying Cold Spot": 6,
}
NAMES = ["none", "hot_int", "hot_pers", "hot_weak", "cold_weak", "cold_pers", "cold_int"]
VARIANTS = ["base", "cell60", "cell30", "nbr400", "nbr600", "win_nbr", "win_cube", "step2"]
out = io.StringIO()


def load(city, season, variant):
    f = os.path.join(SEN, f"{city}_{season}_{variant}.csv")
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    d["g"] = d.pattern.map(GROUPS).fillna(0).astype(int)
    return d


def sample(base, other):
    """Group of `other` at each base cell centre (nearest cell of the other grid)."""
    xs = np.sort(other.x.unique()); ys = np.sort(other.y.unique())
    dx = np.median(np.diff(xs)); dy = np.median(np.diff(ys))
    key = {(int(round((x - xs[0]) / dx)), int(round((y - ys[0]) / dy))): g
           for x, y, g in zip(other.x, other.y, other.g)}
    return np.array([key.get((int(round((x - xs[0]) / dx)), int(round((y - ys[0]) / dy))), -1)
                     for x, y in zip(base.x, base.y)])


def kappa(a, b):
    ok = (a >= 0) & (b >= 0)
    a, b = a[ok], b[ok]
    po = np.mean(a == b)
    pe = sum(np.mean(a == k) * np.mean(b == k) for k in range(7))
    return (po - pe) / (1 - pe)


rows = []
for city in ["saskatoon", "regina"]:
    for season in ["summer", "cold"]:
        base = load(city, season, "base")
        if base is None:
            continue
        for v in VARIANTS:
            d = load(city, season, v)
            if d is None:
                continue
            r = dict(city=city, season=season, variant=v, cells=len(d))
            for k, n in enumerate(NAMES):
                r[f"pct_{n}"] = 100 * np.mean(d.g == k)
            s = sample(base, d) if v != "base" else base.g.values
            for k, n in [(1, "hot_int"), (2, "hot_pers"), (5, "cold_pers")]:
                sel = base.g.values == k
                r[f"keep_{n}"] = 100 * np.mean(s[sel] == k) if sel.any() else np.nan
            # intensifying or persistent hot spots that stay in either of those two classes
            sel = np.isin(base.g.values, [1, 2])
            r["keep_hot_strong"] = 100 * np.mean(np.isin(s[sel], [1, 2]))
            r["kappa"] = kappa(base.g.values, s)
            rows.append(r)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, "analysis", "sensitivity_summary.csv"), index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
cols = ["city", "season", "variant", "pct_hot_int", "pct_hot_pers", "pct_cold_pers",
        "keep_hot_int", "keep_hot_pers", "keep_hot_strong", "keep_cold_pers", "kappa"]
txt = df[cols].round(2).to_string(index=False)
print(txt)
out.write(txt + "\n")
with open(os.path.join(ROOT, "analysis", "sensitivity_summary.txt"), "w", encoding="utf-8") as f:
    f.write(out.getvalue())

# ---------------- LaTeX table for the manuscript ----------------
LABELS = {"cell60": r"60\,m cells", "cell30": r"30\,m cells", "nbr400": r"400\,m neighbourhood",
          "nbr600": r"600\,m neighbourhood", "step2": "Two-year time steps",
          "win_nbr": "Window: neighbourhood time step", "win_cube": "Window: entire cube"}
combos = [("saskatoon", "summer"), ("saskatoon", "cold"), ("regina", "summer"), ("regina", "cold")]
lines = [r"\begin{tabular}{l" + "rr" * 4 + "}", r"\toprule",
         r" & \multicolumn{4}{c}{Saskatoon} & \multicolumn{4}{c}{Regina} \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         r" & \multicolumn{2}{c}{Summer} & \multicolumn{2}{c}{Cold} & \multicolumn{2}{c}{Summer} & \multicolumn{2}{c}{Cold} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
         r"Setting changed & Kept & $\kappa$ & Kept & $\kappa$ & Kept & $\kappa$ & Kept & $\kappa$ \\",
         r"\midrule"]
for v in ["cell60", "cell30", "nbr400", "nbr600", "step2", "win_nbr", "win_cube"]:
    if v == "win_cube":
        lines.append(r"\addlinespace")
    cells = []
    for city, season in combos:
        r = df[(df.city == city) & (df.season == season) & (df.variant == v)]
        cells += ["--", "--"] if r.empty else [f"{r.keep_hot_strong.iloc[0]:.0f}", f"{r.kappa.iloc[0]:.2f}"]
    lines.append(LABELS[v] + " & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
with open(os.path.join(ROOT, "manuscript", "tab_sensitivity.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("wrote manuscript/tab_sensitivity.tex")
