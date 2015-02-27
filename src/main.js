$(function () {
  'use strict';

  function updateDataLayer() {
  }

  function aggregateByLocation() {
    var dataGroupedByLocation, key, locationBin = {}, aggData;
    if (mdata) {
      aggData = mdata.map(function(item) {
        key = item.field7 + '|' + item.field6;
        if (item.binCount === undefined) {
          item.binCount = 1;
          locationBin[key] = 1;
          return item;
        } else if (key in locationBin) {
          locationBin[key] += 1;
          item.binCount = locationBin[key];
        }
      });
    }
  }

  var map = geo.map({
    node: '#map',
    center: {
      x: -98.0,
      y: 39.5
    },
    zoom: 1
  }), locationBin = null, mdata = null, aggData = null;

  map.createLayer(
    'osm',
    {
      baseUrl: 'http://c.tile.stamen.com/terrain-labels/'
    }
  );

  $.ajax( "/api/v1/data?limit=10000" )
  .done(function(data) {
    // Cache the data for later
    mdata = data;
    aggregateByLocation();
    map
      .createLayer('feature')
        .createFeature('point')
          .data(aggData)
          .position(function(d) {return {x:d.field7, y:d.field6}})
          .style('stroke', false)
          .style('fillOpacity', 0.05)
          .style('fillColor', "orange");
    map.draw();
  })
  .fail(function() {
    alert( "error" );
  });
});