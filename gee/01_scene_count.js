// 01_scene_count.js  (version 2: Saskatoon + Regina, Landsat 7 stripe fix)
// Pilot step 1: how many usable Landsat scenes does each city have, per year and season?
// Paste into https://code.earthengine.google.com/ and press Run.
// Output: one CSV export per city in the Tasks tab (run each task).

// ---------------- Settings ----------------
var CITIES = {
  saskatoon: ee.Geometry.Point([-106.67, 52.13]),
  regina: ee.Geometry.Point([-104.61, 50.45])
};
var BUFFER_M = 20000;  // 40 km x 40 km box: city plus rural ring
var START = '1984-01-01';
var END = '2026-11-01';
var SCALE = 120;       // coarse scale is enough for counting, and much faster than 30 m

// ---------------- Landsat Collection 2 Level 2, all sensors ----------------
function loadAll(aoi) {
  function load(id) {
    return ee.ImageCollection(id)
      .filterBounds(aoi)
      .filterDate(START, END)
      .select('QA_PIXEL');
  }
  return load('LANDSAT/LT05/C02/T1_L2')
    .merge(load('LANDSAT/LE07/C02/T1_L2'))
    .merge(load('LANDSAT/LC08/C02/T1_L2'))
    .merge(load('LANDSAT/LC09/C02/T1_L2'));
}

var SEASONS = ee.List(['winter', 'winter', 'spring', 'spring', 'spring', 'summer',
                       'summer', 'summer', 'fall', 'fall', 'fall', 'winter']);

// ---------------- Per-scene statistics ----------------
// valid_frac : share of the study area that has data (not fill). Landsat 7 after
//              May 2003 tops out near 0.78 because of its missing-data stripes.
// clear_frac : share of the study area that has data AND is free of cloud/shadow.
// snow_frac  : share of the study area that is clear and flagged as snow.
// A scene is usable when clear_frac / valid_frac is high and valid_frac is not tiny;
// that test is applied later in Python so the thresholds can be changed freely.
function makeSceneStats(aoi, city) {
  return function (img) {
    var qa = img.select('QA_PIXEL');
    // QA_PIXEL bits: 0 fill, 1 dilated cloud, 2 cirrus, 3 cloud, 4 cloud shadow, 5 snow.
    var valid = qa.bitwiseAnd(1).eq(0).unmask(0).rename('valid_frac');
    var bad = qa.bitwiseAnd(31).neq(0);                      // any of bits 0-4
    var clear = bad.not().unmask(0).rename('clear_frac');
    var snow = qa.bitwiseAnd(32).neq(0).and(bad.not()).unmask(0).rename('snow_frac');

    var stats = valid.addBands(clear).addBands(snow).reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: aoi,
      scale: SCALE,
      maxPixels: 1e9,
      tileScale: 2
    });

    var date = img.date();
    var month = date.get('month');
    var year = date.get('year');
    // December belongs to the following winter (Dec 2020 -> winter 2021).
    var seasonYear = ee.Number(year).add(ee.Number(month).eq(12));

    return ee.Feature(null, {
      city: city,
      scene_id: img.get('LANDSAT_PRODUCT_ID'),
      sensor: img.get('SPACECRAFT_ID'),
      date: date.format('YYYY-MM-dd'),
      year: year,
      month: month,
      season: SEASONS.get(ee.Number(month).subtract(1)),
      season_year: seasonYear,
      wrs_path: img.get('WRS_PATH'),
      wrs_row: img.get('WRS_ROW'),
      cloud_cover_scene: img.get('CLOUD_COVER'),
      sun_elevation: img.get('SUN_ELEVATION'),
      valid_frac: stats.get('valid_frac'),
      clear_frac: stats.get('clear_frac'),
      snow_frac: stats.get('snow_frac')
    });
  };
}

// ---------------- One export per city ----------------
Object.keys(CITIES).forEach(function (city) {
  var aoi = CITIES[city].buffer(BUFFER_M).bounds();
  var landsat = loadAll(aoi);
  var scenes = ee.FeatureCollection(landsat.map(makeSceneStats(aoi, city)));

  Map.addLayer(aoi, {color: 'red'}, 'Study area: ' + city, true, 0.4);
  print('Scenes touching ' + city + ':', landsat.size());

  Export.table.toDrive({
    collection: scenes,
    description: city + '_landsat_scene_stats_v2',
    folder: 'UHI_Prairie',
    fileFormat: 'CSV',
    selectors: ['city', 'scene_id', 'sensor', 'date', 'year', 'month', 'season',
                'season_year', 'wrs_path', 'wrs_row', 'cloud_cover_scene', 'sun_elevation',
                'valid_frac', 'clear_frac', 'snow_frac']
  });
});

Map.setCenter(-105.6, 51.3, 7);
