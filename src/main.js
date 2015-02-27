$(function () {
  'use strict';

  function updateDataLayer() {
  }

  function aggregateByLocation() {
    var dataGroupedByLocation, key, locationBin = {}, min = 0, max = 1;
    if (mdata) {
      mdata.forEach(function(item) {
        key = item.field7 + '|' + item.field6;
        if (key in locationBin) {
          locationBin[key].binCount = 1 + locationBin[key].binCount;
          if (locationBin[key].binCount > max) {
            max = locationBin[key].binCount
          }
        } else {
          item.binCount = 1;
          locationBin[key] = item;
          aggData.push(item);
        }
      });
    }

    scale = d3.scale.linear().domain([min, max])
              .range([2, 100]);
  }

  var map = geo.map({
    node: '#map',
    center: {
      x: -98.0,
      y: 39.5
    },
    zoom: 1
  }), locationBin = null, mdata = null, aggData = [], scale = null;

  map.createLayer(
    'osm',
    {
      baseUrl: 'http://c.tile.stamen.com/terrain-labels/'
    }
  );

  $.ajax( "/api/v1/data?limit=500000" )
  .done(function(data) {
    // Cache the data for later
    mdata = data;
    aggregateByLocation();
    console.log(aggData.length);
    map
      .createLayer('feature')
        .createFeature('point')
          .data(aggData)
          .position(function (d) { return { x:d.field7, y:d.field6 } })
          .style('radius', function (d) { return scale(d.binCount); })
          .style('stroke', false)
          .style('fillOpacity', 0.4)
          .style('fillColor', "orange");
    map.draw();
  })
  .fail(function() {
    alert( "error" );
  });
});