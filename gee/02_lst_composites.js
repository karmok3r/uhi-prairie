// 02_lst_composites.js
// Pilot step 2: yearly seasonal land surface temperature (LST) composites at 30 m
// for Saskatoon and Regina, ready for the space-time cube in ArcGIS Pro.
//
// Paste into https://code.earthengine.google.com/ , press Run, then run every task
// in the Tasks tab (14 tasks). Files land in the Drive folder UHI_Prairie.
//
// For each city and season it exports three GeoTIFFs with ONE BAND PER YEAR:
//   <city>_<season>_lst.tif    median LST in deg C
//   <city>_<season>_anom.tif   median LST anomaly in deg C: each scene's LST minus that
//                              scene's study-area mean, so hot/cold days cancel out
//   <city>_<season>_count.tif  number of clear observations behind each pixel
// plus <city>_scenes_used.csv listing every scene that went in.

// ---------------- Settings ----------------
var CITIES = {
  saskatoon: ee.Geometry.Point([-106.67, 52.13]),
  regina: ee.Geometry.Point([-104.61, 50.45])
};
var BUFFER_M = 20000;          // 40 km x 40 km study area
var CRS = 'EPSG:32613';        // UTM zone 13N covers both cities
var SCALE = 30;

var SEASONS = {
  // startMonth/endMonth are inclusive. crossesYear: season starts in the previous year.
  // minMeanLst drops scenes whose study-area mean is implausibly cold for the season:
  // these are cloud decks the quality mask missed (e.g. Saskatoon 1984-06-25, 1992-07-01).
  summer: {firstYear: 1984, lastYear: 2026, startMonth: 6, endMonth: 8, crossesYear: false,
           minMeanLst: 10},
  cold:   {firstYear: 2000, lastYear: 2026, startMonth: 11, endMonth: 3, crossesYear: true,
           minMeanLst: -100}
};

var MIN_VALID = 0.7;           // scene must cover at least 70% of the study area
var MIN_CLEAR_REL = 0.8;       // and at least 80% of the covered part must be clear
var L7_END = '2013-04-01';     // Landsat 7 only bridges the gap to Landsat 8 (stripes, orbit drift later)

// ---------------- Helpers ----------------
function loadLandsat(aoi) {
  function load(id, thermalBand, start, end) {
    return ee.ImageCollection(id)
      .filterBounds(aoi)
      .filterDate(start, end)
      .select([thermalBand, 'QA_PIXEL'], ['ST', 'QA_PIXEL']);
  }
  return load('LANDSAT/LT05/C02/T1_L2', 'ST_B6', '1984-01-01', '2012-06-01')
    .merge(load('LANDSAT/LE07/C02/T1_L2', 'ST_B6', '1999-01-01', L7_END))
    .merge(load('LANDSAT/LC08/C02/T1_L2', 'ST_B10', '2013-01-01', '2027-01-01'))
    .merge(load('LANDSAT/LC09/C02/T1_L2', 'ST_B10', '2021-01-01', '2027-01-01'));
}

// Convert to deg C, mask cloud and shadow, and attach per-scene statistics.
function makePrep(aoi) {
  return function (img) {
    var qa = img.select('QA_PIXEL');
    var st = img.select('ST');
    // QA_PIXEL bits: 0 fill, 1 dilated cloud, 2 cirrus, 3 cloud, 4 cloud shadow.
    // Snow (bit 5) is deliberately kept: it is real surface in the cold season.
    var valid = qa.bitwiseAnd(1).eq(0);
    var clear = qa.bitwiseAnd(31).eq(0);
    var lst = st.multiply(0.00341802).add(149.0).subtract(273.15)
      .updateMask(clear).updateMask(st.neq(0))
      .rename('lst').toFloat();

    var stats = valid.unmask(0).rename('valid')
      .addBands(clear.unmask(0).rename('clear'))
      .addBands(lst)
      .reduceRegion({
        reducer: ee.Reducer.mean(),
        geometry: aoi,
        scale: 120,
        maxPixels: 1e9,
        tileScale: 2
      });
    var validFrac = ee.Number(stats.get('valid'));
    var clearFrac = ee.Number(stats.get('clear'));

    return lst.set({
      'system:time_start': img.get('system:time_start'),
      scene_id: img.get('LANDSAT_PRODUCT_ID'),
      sensor: img.get('SPACECRAFT_ID'),
      date: img.date().format('YYYY-MM-dd'),
      valid_frac: validFrac,
      clear_frac: clearFrac,
      clear_rel: clearFrac.divide(validFrac.max(0.0001)),
      mean_lst: stats.get('lst')
    });
  };
}

// Add the anomaly band: LST minus the scene's study-area mean.
function addAnomaly(img) {
  img = ee.Image(img);
  var sceneMean = ee.Image.constant(ee.Number(img.get('mean_lst')));
  var anom = img.select('lst').subtract(sceneMean).rename('anom').toFloat();
  return img.addBands(anom);
}

// A fully masked placeholder so that years with no scenes still produce a band.
var EMPTY = ee.Image.constant([0, 0]).rename(['lst', 'anom']).toFloat()
  .updateMask(ee.Image.constant(0));

function seasonWindow(year, s) {
  var start = ee.Date.fromYMD(s.crossesYear ? year - 1 : year, s.startMonth, 1);
  var end = ee.Date.fromYMD(year, s.endMonth, 1).advance(1, 'month');
  return [start, end];
}

// ---------------- Main loop ----------------
Object.keys(CITIES).forEach(function (city) {
  var aoi = CITIES[city].buffer(BUFFER_M).bounds();

  // sort() and distinct() hand back a generic collection, which has no .median(),
  // so the result is cast back to an ImageCollection explicitly.
  var scenes = ee.ImageCollection(loadLandsat(aoi)
    .map(makePrep(aoi))
    .filter(ee.Filter.notNull(['mean_lst']))
    .filter(ee.Filter.gte('valid_frac', MIN_VALID))
    .filter(ee.Filter.gte('clear_rel', MIN_CLEAR_REL))
    // Two overlapping scenes from the same pass share a date: keep the fuller one only,
    // so that no single day counts twice in the median.
    .sort('valid_frac', false)
    .distinct('date'))
    .map(addAnomaly);

  // Table of every scene that passed the filters (all months; filter by date later).
  Export.table.toDrive({
    collection: ee.FeatureCollection(scenes.map(function (img) {
      return ee.Feature(null, ee.Image(img).toDictionary(
        ['scene_id', 'sensor', 'date', 'valid_frac', 'clear_frac', 'clear_rel', 'mean_lst']))
        .set('city', city);
    })),
    description: city + '_scenes_used',
    folder: 'UHI_Prairie',
    fileFormat: 'CSV',
    selectors: ['city', 'scene_id', 'sensor', 'date', 'valid_frac', 'clear_frac',
                'clear_rel', 'mean_lst']
  });

  Object.keys(SEASONS).forEach(function (seasonName) {
    var s = SEASONS[seasonName];
    var lstBands = [], anomBands = [], countBands = [], names = [];

    for (var year = s.firstYear; year <= s.lastYear; year++) {
      var w = seasonWindow(year, s);
      var col = scenes.filterDate(w[0], w[1])
        .filter(ee.Filter.gte('mean_lst', s.minMeanLst));
      var withEmpty = ee.ImageCollection(col.merge(ee.ImageCollection([EMPTY])));
      var med = withEmpty.median();
      lstBands.push(med.select('lst'));
      anomBands.push(med.select('anom'));
      countBands.push(withEmpty.select('lst').count().unmask(0));
      names.push('y' + year);
    }

    var stacks = {
      lst: ee.ImageCollection(lstBands).toBands().rename(names).toFloat(),
      anom: ee.ImageCollection(anomBands).toBands().rename(names).toFloat(),
      count: ee.ImageCollection(countBands).toBands().rename(names).toInt16()
    };

    Object.keys(stacks).forEach(function (variable) {
      var name = city + '_' + seasonName + '_' + variable;
      Export.image.toDrive({
        image: stacks[variable],
        description: name,
        folder: 'UHI_Prairie',
        fileNamePrefix: name,
        region: aoi,
        scale: SCALE,
        crs: CRS,
        maxPixels: 1e10
      });
    });

    // Quick look: the most recent year's anomaly for each city and season.
    Map.addLayer(stacks.anom.select('y' + s.lastYear).clip(aoi),
      {min: -6, max: 8, palette: ['#2166ac', '#67a9cf', '#f7f7f7', '#ef8a62', '#b2182b']},
      city + ' ' + seasonName + ' ' + s.lastYear + ' anomaly', false);
  });
});

Map.setCenter(-106.67, 52.13, 11);
