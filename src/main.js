$(function () {
  'use strict';

  // Aggregate data
  function aggregateByLocation(data) {
    var dataGroupedByLocation, key, locationBin = {}, min = 0, max = 1, newdata = [];
    if (data) {
      data.forEach(function(item) {
        key = item.field7 + '|' + item.field6;
        if (key in locationBin) {
          locationBin[key].binCount = 1 + locationBin[key].binCount;
          if (locationBin[key].binCount > max) {
            max = locationBin[key].binCount
          }
        } else {
          item.binCount = 1;
          locationBin[key] = item;
          newdata.push(item);
        }
      });
    }

    return {"data": newdata, "min": min, "max": max};
  }

  // Query given a time duration
  function queryData(timerange, callback) {
    console.log("querying data");
    console.log("/api/v1/data?limit=1000&duration=["+timerange+"]");

    $.ajax("/api/v1/data?limit=1000&duration=["+timerange+"]")
      .done(function(data) {
        console.log(data);
        if (callback !== undefined) {
          callback(data);
        }
      })
      .fail(function(err) {
        console.log(err);
      })
  }

  // Create geovis
  function createVis(data) {
    var aggdata = aggregateByLocation(data);
    console.log(aggdata);
    scale = d3.scale.linear().domain([aggdata.min, aggdata.max])
              .range([2, 100]);
    if (pointFeatureLayer === undefined) {

    }
    map
      .createLayer('feature')
        .createFeature('point')
          .data(aggdata.data)
          .position(function (d) { return { x:d.field7, y:d.field6 } })
          .style('radius', function (d) { return scale(d.binCount); })
          .style('stroke', false)
          .style('fillOpacity', 0.4)
          .style('fillColor', "orange");
    map.draw();
  }

  // Globals
  var map = geo.map({
    node: '#map',
    center: {
      x: -98.0,
      y: 39.5
    },
    zoom: 1
  }), locationBin = null, scale = null, pointFeatureLayer;

  map.createLayer(
    'osm',
    {
      baseUrl: 'http://c.tile.stamen.com/terrain-labels/'
    }
  );

  $.ajax( "/api/v1/data" )
    .done(function(range) {
    // return;

    // // Cache the data for later
    // mdata = data;
    // aggregateByLocation();
    // console.log(aggData);

    // Compute min and max for time
    var format = d3.time.format("%Y-%m-%d %H:%M:%S");
    // var minMax = [d3.min(aggData, function(d) {
    //     var date = format.parse(d.field4);
    //     if (date) {
    //       return date.getTime() / 1000
    //     }
    //   }), d3.max(aggData, function(d) {
    //     var date = format.parse(d.field4);
    //     if (date) {
    //       return date.getTime() / 1000
    //     }
    //   })];

    console.log(format.parse(range.duration.start.field4));
    console.log(format.parse(range.duration.end.field4));

    var min = format.parse(range.duration.start.field4);
    var max = format.parse(range.duration.end.field4);

    // Set the date range
    $( "#slider" ).slider({
      range: true,
      min: min.getTime()/1000,
      max: max.getTime()/1000,
      values: [ min.getTime()/1000, min.getTime()/1000 + 24 * 3600 * 180 ],
      slide: function( event, ui ) {
        queryData($("#slider").slider("values"), createVis);
      }
    });

    // Now query data
    queryData($("#slider").slider("values"), createVis);
  })
  .fail(function() {
    alert( "error" );
  });
});
