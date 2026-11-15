// FloodLens - Google Earth Engine export (paste into code.earthengine.google.com)
// Produces, in your Drive:   <EVENT>_stack.tif   (9 bands, 30 m, EPSG:32642)
//                            <EVENT>_scenes.csv  (one row per Sentinel-1 scene used)
//
// STEP 1: run with DISCOVER = true, read the orbit table in the Console, pick ORBIT + PASS
//         that appear in BOTH the pre and the post window and cover Dadu AND Larkana.
// STEP 2: set ORBIT/PASS, DISCOVER = false, run, then press "Run" on the two tasks in the Tasks tab.

var EVENT = 'sindh2022';
var AOI   = ee.Geometry.Rectangle([67.2, 26.3, 68.6, 28.0]);       // Dadu + Larkana
var PRE   = ['2022-05-01', '2022-06-30'];
var POST  = ['2022-08-26', '2022-09-03'];
var DISCOVER = true;
var ORBIT = 0, PASS = 'ASCENDING';                                   // <-- fill after STEP 1

var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(AOI)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'));

function orbitTable(win) {
  var c = s1.filterDate(win[0], win[1]);
  return ee.List(c.aggregate_array('relativeOrbitNumber_start'))
    .zip(c.aggregate_array('orbitProperties_pass'));
}

if (DISCOVER) {
  print('PRE  (orbit, pass) pairs', orbitTable(PRE));
  print('POST (orbit, pass) pairs', orbitTable(POST));
  print('Pick a pair present in BOTH lists, then set DISCOVER = false.');
} else {
  var base = s1.filter(ee.Filter.eq('relativeOrbitNumber_start', ORBIT))
               .filter(ee.Filter.eq('orbitProperties_pass', PASS));
  var preC  = base.filterDate(PRE[0],  PRE[1]);
  var postC = base.filterDate(POST[0], POST[1]);
  print('pre scenes', preC.size(), 'post scenes', postC.size());

  var pre  = preC.select(['VV', 'VH']).median();        // already dB in this collection
  var post = postC.select(['VV', 'VH']).median();

  // NOTE: COPERNICUS/DEM/GLO30 is an ImageCollection of tiles -> mosaic it.
  var demCol = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM');
  var dem = demCol.mosaic().setDefaultProjection(demCol.first().select(0).projection());
  var slope = ee.Terrain.slope(dem);

  var jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').unmask(0);

  var pop = ee.ImageCollection('WorldPop/GP/100m/pop')
    .filter(ee.Filter.eq('country', 'PAK')).filter(ee.Filter.eq('year', 2020)).first();
  var popCellArea = ee.Image.pixelArea().reproject(pop.projection());

  var wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map');

  var stack = ee.Image.cat([
      pre.select('VV').rename('vv_pre'),   pre.select('VH').rename('vh_pre'),
      post.select('VV').rename('vv_post'), post.select('VH').rename('vh_post'),
      slope.rename('slope_deg'), jrc.rename('jrc_occurrence'),
      pop.rename('pop_count'), popCellArea.rename('pop_cell_m2'),
      wc.rename('worldcover')
    ]).toFloat().clip(AOI);

  // sanity number for the Python side: total people in the AOI according to GEE
  print('GEE total population in AOI (compare with summary.json people_in_aoi)',
        pop.reduceRegion({reducer: ee.Reducer.sum(), geometry: AOI, scale: 100, maxPixels: 1e10}));

  Export.image.toDrive({image: stack, description: EVENT + '_stack', fileNamePrefix: EVENT + '_stack',
                        region: AOI, scale: 30, crs: 'EPSG:32642', maxPixels: 1e10});

  function rows(col, role) {
    return col.map(function (i) {
      return ee.Feature(null, {
        scene_id: i.get('system:index'),
        relativeOrbitNumber_start: i.get('relativeOrbitNumber_start'),
        orbitProperties_pass: i.get('orbitProperties_pass'),
        time_start_ms: i.get('system:time_start'),
        transmitterReceiverPolarisation: ee.List(i.get('transmitterReceiverPolarisation')).join('+'),
        role: role});
    });
  }
  Export.table.toDrive({collection: rows(preC, 'pre').merge(rows(postC, 'post')),
                        description: EVENT + '_scenes', fileNamePrefix: EVENT + '_scenes',
                        fileFormat: 'CSV',
                        selectors: ['scene_id', 'relativeOrbitNumber_start', 'orbitProperties_pass',
                                    'time_start_ms', 'transmitterReceiverPolarisation', 'role']});
}
