$(function () {
  'use strict';

  // Globals
  var map = geo.map({
    node: '#map',
    center: {
      x: -98.0,
      y: 39.5
    },
    zoom: 1
  }), locationBin = null, scale = null, pointFeature, ready = false, startTime = null;

  map.createLayer(
    'osm',
    {
      baseUrl: 'http://c.tile.stamen.com/terrain-labels/'
    }
  );

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
    console.log("/api/v1/data?limit=100000&duration=["+timerange+"]");

    $.ajax("/api/v1/data?limit=100000&duration=["+timerange+"]")
      .done(function(data) {
        console.log(data.length);
        if (callback !== undefined) {
          callback(data);
        }
      })
      .fail(function(err) {
        console.log(err);
      })
  }

  // Create geovis
  function createVis(data, callback) {
    var aggdata = aggregateByLocation(data);
    console.log(aggdata);
    scale = d3.scale.linear().domain([aggdata.min, aggdata.max])
              .range([2, 100]);
    if (pointFeature === undefined) {
      pointFeature = map
                       .createLayer('feature')
                       .createFeature('point');
    }
    pointFeature
      .data(aggdata.data)
      .position(function (d) { return { x:d.field7, y:d.field6 } })
      .style('radius', function (d) { return scale(d.binCount); })
      .style('stroke', false)
      .style('fillOpacity', 0.4)
      .style('fillColor', "orange");

    map.draw();

    if (callback) {
      callback();
    }
  }

  // Create animation
  function runAnimation(timestamp) {
    if (ready) {
      // First get the values from the slider
      var range = $( "#slider" ).slider( "values" ),
          min = $( "#slider" ).slider( "option", "min" ),
          max = $( "#slider" ).slider( "option", "max" ),
          delta = range[1] - range[0],
          stopAnimation = false,
          newRange = [ range[ 0 ] + delta, range[ 1 ] + delta ];

      if (newRange[0] >= max) {
        newRange[0] = max;
        stopAnimation = true;
      }
      if (newRange[0] <= min) {
        newRange[0] = min;
        stopAnimation = true;
      }
      if (newRange[1] >= max) {
        newRange[1] = max;
        stopAnimation = true;
      }
      if (newRange[1] <= min) {
        newRange[1] = min;
        stopAnimation = true;
      }

      // Set the slider value
      $( "#slider" ).slider( "option", "values", newRange );

      // Query the data and create vis again
      queryData( newRange, function(data) {
        createVis(data, function() {
          if (!stopAnimation) {
            window.requestAnimationFrame(runAnimation);
          }
        });
      });
    }
  }

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

    ready = true;

    // Testing
    window.requestAnimationFrame(runAnimation);
  })
  .fail(function() {
    console.log('failed');
  });

  $( "#slider" ).slider();
});
