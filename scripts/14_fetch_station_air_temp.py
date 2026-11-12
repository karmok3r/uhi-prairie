"""14_fetch_station_air_temp.py

Downloads hourly air temperature from Environment and Climate Change Canada (ECCC)
climate stations through the MSC GeoMet OGC API (api.weather.gc.ca, collection
climate-hourly). Only the hours needed are kept:
  - local hours 10, 11 and 12 (Landsat overpass is about 11:00-11:30 local standard time;
    Saskatchewan keeps Central Standard Time, UTC-6, all year), and
  - local hours 0 to 5 (night, for the nocturnal air-temperature contrast).

Stations (climate identifiers; the urban station is the airport, which the city now
surrounds on most sides):
  Saskatoon urban: 4057120 (Diefenbaker Int'l A, to 2012-03), 4057152 (Intl A, 2012-),
                   4057165 (RCS, 2008-)
  Saskatoon rural: 4055736 (Outlook PFRA), 4046884 (Rosetown East), 4018642 (Watrous East)
  Regina urban:    4016560 (Int'l A, to 2013-10), 4016566 (Intl A, 2013-), 4016699 (RCS)
  Regina rural:    4010811 (Bratt's Lake Climate, 2006-)

Output: data/air_temp/<climate_id>.csv
"""
import os
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "air_temp")
os.makedirs(OUT, exist_ok=True)
API = "https://api.weather.gc.ca/collections/climate-hourly/items"
STATIONS = {
    "4057120": (1994, 2012), "4057152": (2012, 2026), "4057165": (2008, 2026),
    "4055736": (1996, 2026), "4046884": (1994, 2026), "4018642": (1994, 2026),
    "4016560": (1994, 2013), "4016566": (2013, 2026), "4016699": (1996, 2026),
    "4010811": (2006, 2026),
}
HOURS = [0, 1, 2, 3, 4, 5, 10, 11, 12]
KEEP = ["CLIMATE_IDENTIFIER", "LOCAL_DATE", "LOCAL_HOUR", "TEMP", "WIND_SPEED",
        "RELATIVE_HUMIDITY"]


def get(url, tries=5):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.load(r)
        except Exception:
            time.sleep(3 * (k + 1))
    raise RuntimeError(url)


def fetch_year(cid, year):
    rows, offset = [], 0
    while True:
        url = (f"{API}?CLIMATE_IDENTIFIER={cid}&LOCAL_YEAR={year}&f=json&limit=10000"
               f"&offset={offset}&properties={','.join(KEEP)}")
        d = get(url)
        feats = d.get("features", [])
        rows += [f["properties"] for f in feats]
        if len(feats) < 10000:
            break
        offset += 10000
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df[df.LOCAL_HOUR.isin(HOURS)]


for cid, (y0, y1) in STATIONS.items():
    f = os.path.join(OUT, f"{cid}.csv")
    if os.path.exists(f):
        continue
    with ThreadPoolExecutor(6) as ex:
        parts = list(ex.map(lambda y: fetch_year(cid, y), range(y0, y1 + 1)))
    df = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    df.to_csv(f, index=False)
    print(cid, len(df), "rows", df.LOCAL_DATE.min(), df.LOCAL_DATE.max())
