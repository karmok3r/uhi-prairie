# Intra-urban heat in prairie cities: Saskatoon and Regina

Working folder for a study of where surface heat is intensifying inside Saskatoon and
Regina, using Landsat land surface temperature (LST), a space-time cube, and emerging
hot-spot analysis. The method builds on Jayasooriya & Adams (2024).

## Folder layout

| Folder | Contents |
|---|---|
| `literature/` | The seven background papers, renamed as `Author_Year_topic.pdf`. |
| `gee/` | Google Earth Engine scripts (JavaScript, for the Code Editor). |
| `data/scene_stats/` | Per-scene tables exported from GEE. |
| `data/scene_stats/v1_first_run/` | The first Saskatoon-only scene count. Superseded; kept for the record. |
| `data/lst_composites/` | Yearly seasonal LST GeoTIFFs, one band per year. |
| `figures/` | Quick-look maps and, later, paper figures. |
| `docs/` | Notes, proposal and manuscript drafts (empty so far). |

The original GEE exports also remain in Google Drive under `My Drive/UHI_Prairie`.

## GEE scripts

1. `gee/01_scene_count.js` counts usable Landsat scenes per city, year and season.
   Output: `<city>_landsat_scene_stats_v2.csv`.
2. `gee/02_lst_composites.js` builds the yearly seasonal composites.
   Output: the GeoTIFFs below and `<city>_scenes_used.csv`.

## Composite files

Named `<city>_<season>_<variable>.tif`. All are 30 m, EPSG:32613 (UTM zone 13N), covering
a 40 km x 40 km box centred on each city. Bands are named `y<year>`.

| Part | Values |
|---|---|
| city | `saskatoon`, `regina` |
| season | `summer` = June to August, bands 1984 to 2026 (43 bands). `cold` = November to March, bands 2000 to 2026 (27 bands); the band year is the year the season ends in. |
| variable | `lst` = median surface temperature, deg C. `anom` = median of (scene LST minus that scene's study-area mean), deg C. `count` = number of clear observations per pixel. |

## Processing choices

- Source: Landsat 5, 7, 8 and 9, Collection 2 Level 2 surface temperature.
- Landsat 7 is used only until March 2013, to bridge Landsat 5 and 8.
- A scene is used when it covers at least 70% of the study area and at least 80% of the
  covered part is free of cloud and shadow.
- One scene per date is kept. Summer scenes with a study-area mean below 10 deg C are
  dropped as undetected cloud.
- Snow is not masked in the cold season.

## Known data limits

- The Level 2 product does not exist when the sun is lower than 14 degrees, so December
  and January are almost absent, especially for Saskatoon. This is why the cold season is
  November to March and not December to February.
- Thin years, where most pixels rest on a single scene: Saskatoon cold 2009, 2011, 2012
  and 2015; Regina summer 1997. Saskatoon cold 2012 and 2015 also have gaps (73% and 55%
  coverage).
- Cropland around both cities varies strongly from year to year, so hot-spot analysis
  should be limited to the urban area.

## Analysis scripts

Run in order. "Pro" = ArcGIS Pro 3.4 Python (`arcgispro-py3`); "Py" = standard Python
with the packages in `requirements.txt`.

| Script | Env | What it does | Main outputs |
|---|---|---|---|
| `gee/03_cold_snow_split.js`, `gee/04_scene_urban_rural.js` | GEE | Snow-split cold composites; per-scene zone statistics | `data/lst_composites/*_cold_(no)snow.tif`, `data/scene_stats/*_scene_zones.csv` |
| `03_emerging_hot_spots.py` | Pro | Space-time cube and Emerging Hot Spot Analysis | `analysis/uhi_prairie.gdb` |
| `04_map_hot_spots.py` | Pro | Quick-look maps | `figures/` |
| `06_snow_split_hot_spots.py` | Pro | Gi* on snow / no-snow composites | `analysis/snow_split_summary.csv` |
| `08_snow_test_per_scene.py` | Pro | Scene-by-scene urban-rural test and OLS | `analysis/scene_urban_rural.csv`, `snow_test_summary.txt` |
| `10_sensitivity.py`, `13_sensitivity_summary.py` | Pro | Robustness runs and Table 6 | `analysis/sensitivity_summary.*`, `manuscript/tab_sensitivity.tex` |
| `11_sensor_step_test.py` | Pro | 2013 sensor-step test | `analysis/sensor_step_test.txt` |
| `12_manuscript_figures.py` | Pro | Figures 1-5 | `figures/ms_fig1-5_*.png` |
| `14_fetch_station_air_temp.py` | Py | ECCC hourly air temperature | `data/air_temp/*.csv` |
| `15_hot_spot_clusters.py` | Pro | Clusters of summer hot-spot cells | `analysis/hot_spot_clusters.csv` |
| `16_air_temp_check.py` | Pro | Surface vs air urban-rural contrast, Figure 7 | `analysis/air_temp_check.*`, `figures/ms_fig7_air_temp.png` |
| `17a_buildings.py` | Py | Building footprints for both boxes | `data/drivers/*_buildings.csv` |
| `17_driver_variables.py` | Pro | Drivers on the 90 m grid (WorldCover, GHSL, buildings) | `analysis/drivers_<city>.csv` |
| `18_driver_models.py` | Pro | Driver regressions, Figure 6, Table 7 | `analysis/driver_models.*`, `manuscript/tab_drivers.tex` |

Raw driver inputs (`data/drivers/raw/`) are downloaded from: ESA WorldCover 2021 tiles
N51W108 and N48W105 (AWS open data); GHSL R2023A GHS-BUILT-S (1985, 2020) and GHS-BUILT-H
ANBH (2018) tiles R3_C11 and R4_C11 (JRC); Microsoft Canadian Building Footprints,
`Saskatchewan.zip`.
