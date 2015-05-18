// Globals
var myApp = {};

$(function () {
  'use strict';

  myApp.map = null,
  myApp.locationBin = null,
  myApp.scale = null,
  myApp.pointFeature,
  myApp.ready = false,
  myApp.startTime = null;
  myApp.animationState = 0;
  myApp.timeRange = 0;
  myApp.location = null;
  myApp.locationType="address";
  myApp.visibleDialogs = [];
  myApp.prevTimestamp = 0;

  $( window ).resize(function() {
    myApp.resize();
  });

  myApp.map = geo.map({
          node: '#map',
          center: {
            x: -98.0,
            y: 39.5
          },
          zoom: 2,
          autoResize: false
        });
  myApp.map.createLayer(
    'osm',
    {
      baseUrl: 'http://otile1.mqcdn.com/tiles/1.0.0/map/'
    }
  );

  myApp.resize = function() {
    var height = $(window).height(),
        width  = $("#map").width();
    myApp.map.resize(0, 0, width, height);
  }
  myApp.resize();

  // Bind plugins
  //--------------------------------------------------------------------------
  $("#search").geocomplete();

  // Bind events
  //--------------------------------------------------------------------------
  $("#search").on('keydown', function() {
    if(event.keyCode == 13) {
        myApp.setLocationTypeToAddress();
        myApp.runQuery();
    }
  });

  $("#spinner .gs-spinner-icon").hide();
  $("#spinner .gs-spinner-text").hide();

  // Aggregate data
  //--------------------------------------------------------------------------
  function aggregateByLocation(data) {
    var dataGroupedByLocation, key, min = 0, max = 1, newdata = [];
    myApp.locationBin = {};

    if (data) {
      data.forEach(function(item) {
        key = item.loc[0] + '|' + item.loc[1];
        if (key in myApp.locationBin) {
          myApp.locationBin[key].binCount = 1 + myApp.locationBin[key].binCount;
          myApp.locationBin[key].urls.push(item.url);
          if (myApp.locationBin[key].binCount > max) {
            max = myApp.locationBin[key].binCount
          }
        } else {
          item.binCount = 1;
          item.urls = [item.url];
          myApp.locationBin[key] = item;
          newdata.push(item);
        }
      });
    }

    return {"data": newdata, "min": min, "max": max};
  }

  // Scrap a URL for images and return list of images URL
  //--------------------------------------------------------------------------
  function scrapUrl(url, callback) {
    $.ajax("api/v1/scrape?url="+url+"")
      .done(function(data) {
        if (callback !== undefined) {
          callback(data);
        }
      })
      .fail(function(err) {
        console.log(err);
      })
  }

  // Create visualization given a dataset
  //--------------------------------------------------------------------------
  function render(data, callback) {
    var aggdata = aggregateByLocation(data);
    myApp.scale = d3.scale.linear().domain([aggdata.min, aggdata.max])
              .range([2, 50]);
    if (myApp.pointFeature === undefined) {
      myApp.pointFeature = myApp.map
                       .createLayer('feature')
                       .createFeature('point', {selectionAPI: true});
    }
    if (myApp.pointFeature) {
      myApp.pointFeature.geoOff(geo.event.feature.mouseclick);
    }

    myApp.pointFeature
      .data(aggdata.data)
      .position(function (d) { return { x:d.loc[0], y:d.loc[1] } })
      .style('radius', function (d) { return myApp.scale(d.binCount); })
      .style('stroke', false)
      .style('fillOpacity', 0.4)
      .style('fillColor', 'orange')
      .geoOn(geo.event.feature.mouseclick, function (evt) {
        var i = 0, anchor = null;

        var currLocation = $("#search").val();
        var clickedLocation = evt.data["place"];
        if (!currLocation || currLocation.length < 1 || currLocation !== myApp.location ||
          clickedLocation !== myApp.location) {
          $("#search").val(evt.data["place"])
          $("#search").trigger("geocode");
          myApp.location = evt.data["loc"];
          myApp.setLocationTypeToLatLon();
        }
        for (i = 0; i < evt.data.urls.length; ++i) {
          // Scrap the URL
          scrapUrl(evt.data.urls[i], function(images) {
            myApp.displaySearchResults({
              "data": evt.data,
              "images": images});
          });
        }

        if (myApp.location !== null || myApp.location !== undefined ||
        myApp.location !== "") {
          myApp.queryData(myApp.timeRange, {"value":myApp.location, "type":myApp.locationType},
            function(data) {
              // Format the data
              var adsPerDay = {}, newData = [], i, time, month, inst, timeString, item;
              for (i = 0; i < data.length; ++i) {
                time = new Date(data[i]["time"]);
                month = (time.getMonth() + 1);
                if (month < 10) {
                  month = ("0" + month);
                }
                timeString = time.getFullYear() + "-"
                               + month + "-" +
                               + time.getDate();
                if (timeString in adsPerDay) {
                  adsPerDay[timeString] = adsPerDay[timeString] + 1;
                } else {
                  adsPerDay[timeString] = 1;
                }
              }

              for (item in adsPerDay) {
                inst = {};
                inst["date"] = item;
                inst["value"] = adsPerDay[item];
                newData.push(inst);
              }

              if (newData.length > 1) {
                myApp.displayStatistics(newData, false);
              }
          });
        }
      });

    myApp.map.draw();

    if (callback) {
      callback();
    }
  }

  // Query given a time duration
  //--------------------------------------------------------------------------
  myApp.queryData = function(timeRange, location, callback) {
    var url = "api/v1/data?limit=100000&duration=["+timeRange+"]";

    if (location !== undefined || location !== null || location !== '') {
      url += "&location="+location.value+"&location_type="+location.type;
    }

    $.ajax(url)
      .done(function(data) {

        $("#statistics").empty();

        if (callback !== undefined) {
          callback(data);
        }
      })
      .fail(function(err) {
        console.log(err);
      })
  }

  // Update extents and then render
  //--------------------------------------------------------------------------
  myApp.runQuery = function() {
    $("#spinner .gs-spinner-icon").show();
    $("#spinner .gs-spinner-text").hide();

    myApp.clearLastSearch();

    // Now run the query
    myApp.timeRange = $("#slider").slider("values");
    myApp.location = $("#search").val();
    myApp.queryData(myApp.timeRange,
      {"value":myApp.location, "type":myApp.locationType}, function(data) {
      // Clear out any previous information
      myApp.clearLastSearch();

      render(data, function() {
        $("#spinner .gs-spinner-icon").hide();
        $("#spinner .gs-spinner-text").show();
        $("#spinner .gs-spinner-text").text(data.length);

        // Update the UI
        myApp.updateView(null, myApp.timeRange);
      });
    });
  }

  // Update view
  //--------------------------------------------------------------------------
  myApp.updateView = function(timestamp, timeRange)  {
    $ ("#start").html((new Date(timeRange[0] * 1000)).toDateString());
    $ ("#end").html((new Date(timeRange[1] * 1000)).toDateString());
  }

  // Animate data
  //--------------------------------------------------------------------------
  myApp.animate = function(timestamp) {
    if (myApp.ready) {
      // First get the values from the slider
      var range = $( "#slider" ).slider( "values" ),
          min = $( "#slider" ).slider( "option", "min" ),
          max = $( "#slider" ).slider( "option", "max" ),
          delta = myApp.timeRange[1] - myApp.timeRange[0],
          newRange = null,
          elapsedTime = timestamp - myApp.prevTimestamp;

      if (elapsedTime * 0.001 > 1) {
        myApp.prevTimestamp = timestamp;

        //console.log('elapsedTime * 0.001 ', elapsedTime * 0.001);

        if (myApp.animationState == 3 || myApp.animationState == 1) {
          newRange = [ range[ 0 ] + delta, range[ 1 ] + delta ];
        } else if (myApp.animationState == 2) {
          newRange = [ range[ 0 ] - delta, range[ 1 ] - delta ];
        }

        if (range[1] === max) {
          newRange[0] = min;
          newRange[1] = newRange[0] + delta;
        }
        if (newRange[0] >= max) {
          newRange[0] = min;
        }
        if (newRange[0] <= min) {
          newRange[0] = min;
        }
        if (newRange[1] > max) {
          newRange[1] = max;
        }
        if (newRange[1] <= min) {
          newRange[1] = newRange[0] + delta;
        }

        // Set the slider value
        $( "#slider" ).slider( "option", "values", newRange );

        myApp.timeRange = newRange;
        myApp.location = $("#search-location").text();

        // Query the data and create vis again
        myApp.queryData(myApp.timeRange,
          {"value":myApp.location, "type":myApp.locationType}, function(data) {
          render(data, function() {
            myApp.updateView(null, newRange);

            if (myApp.animationState === 1) {
              window.requestAnimationFrame(myApp.animate);
            }
          });
        });

      } else {
        window.requestAnimationFrame(myApp.animate);
      }
    }
  }

  $.ajax( "api/v1/data" )
    .done(function(range) {
      var format = d3.time.format("%Y-%m-%d %H:%M:%S"),
          min = new Date(range.duration.start.time * 1000),
          max = new Date(range.duration.end.time * 1000);

      // Set the date range
      $( "#slider" ).slider({
        range: true,
        min: min.getTime()/1000,
        max: max.getTime()/1000,
        values: [ min.getTime()/1000, min.getTime()/1000 + 24 * 3600 * 180 ],
        stop: function( event, ui ) {
          myApp.runQuery();
        }
      });

      myApp.runQuery();
      myApp.ready = true;
    })
    .fail(function() {
      console.log('failed');
    });

  $( "#slider" ).slider();
});


// Event handlers
//--------------------------------------------------------------------------
myApp.buttonBackPress = function() {
  myApp.animationState = 2;
  window.requestAnimationFrame(myApp.animate);
}

myApp.buttonPlayPress = function() {
  myApp.animationState = 1;
  window.requestAnimationFrame(myApp.animate);
}

myApp.buttonStopPress = function() {
  myApp.animationState = 0;
  var min = $( "#slider" ).slider( "option", "min" ),
      max = $( "#slider" ).slider( "option", "max" ),
      range = $( "#slider" ).slider( "option", "values" );

  $( "#slider" ).slider( "option", "values",
    [ min, min + (range[1] - range[0]) ] );

  myApp.runQuery();
}

myApp.buttonForwardPress = function() {
  myApp.animationState = 3;
  window.requestAnimationFrame(myApp.animate);
}

// Clear out any information that is related to a particular time duration
//--------------------------------------------------------------------------
myApp.clearLastSearch = function(callback) {
  // Clear images
  $("#images").empty();
}

// Display search result
//--------------------------------------------------------------------------
myApp.displaySearchResults = function(data, clearPrev) {
  var div = $("#images"),
      newDiv = $(document.createElement('div')),
      i = null;

  if (clearPrev) {
    myApp.clearLastSearch();
  }

  div.append(newDiv);
  newDiv.addClass('row');
  if (data.images.length > 0) {
    for (i = 0; i < data.images.length; ++i) {
      var imageDiv = $(document.createElement('div'));
      imageDiv.addClass('col-xs-4');
      newDiv.append(imageDiv)
      var newAnchor = $(document.createElement('a'));
      var newImage = $(document.createElement('img'));
      newImage.addClass('img-responsive img-fluid img-blur');
      newImage.on('mouseover', function() {
        $(this).addClass('img-clear');
        $(this).removeClass('img-blur');
      });
      newImage.on('mouseout', function() {
        $(this).addClass('img-blur');
        $(this).removeClass('img-clear');
      });
      imageDiv.append(newAnchor);
      newAnchor.append(newImage);
      newImage.attr('src', data.images[i]);
    }
  }
}

// Create statistical plot
//--------------------------------------------------------------------------
myApp.displayStatistics = function(data, clearPrev) {
  // var data = [
  //   {"date": "2012-01-05",  "value": 28},
  //   {"date": "2012-01-10",  "value": 43}
  // ];
  var spec = {
    "width": $("#statistics").width() * 0.90,
    "height": $("#statistics").height(),
    "padding": {"top": 10, "left": 30, "bottom": 30, "right": 30},
    "data": [
      {
        "name": "table",
        "format": {"type":"json", "parse":{"date":"date", "value":"number"}},
        "values": data
      }
    ],
    "scales": [
      {
        "name": "x",
        "type": "time",
        "range": "width",
        "nice": true,
        "domain": {"data": "table", "field": "data.date"}
      },
      {
        "name": "y",
        "type": "linear",
        "range": "height",
        "nice": true,
        "domain": {"data": "table", "field": "data.value"}
      },
      {
        "name": "color",
        "type": "ordinal",
        "range": "category20"
      }
    ],
    "axes": [
      {"type": "x", "scale": "x"},
      {"type": "y", "scale": "y"}
    ],
    "marks": [
      {
        "type": "line",
        "from": {"data": "table"},
        "properties": {
          "enter": {
            "interpolate": {"value": "monotone"},
            "x": {"scale": "x", "field": "data.date"},
            "y": {"scale": "y", "field": "data.value"},
            "size": {"value": 50},
            "stroke": {"scale": "color", "field": "data.indexname"},
            "strokeWidth": {"value": 2}
          },
          "update": {
            "opacity": {"value": 1}
          },
          "hover": {
            "opacity": {"value": 0.5}
          }
        }
      }
    ]
  }

  if (clearPrev) {
    $("#statistics").empty();
  }

  try {
    vg.parse.spec(spec, function(chart) {
    var view = chart({el:"#statistics"})
      .update();
    });
  } catch(err) {
  }
}

//--------------------------------------------------------------------------
myApp.setLocationTypeToAddress = function() {
  myApp.locationType = "address";
}

//--------------------------------------------------------------------------
myApp.setLocationTypeToLatLon = function() {
  myApp.locationType = "4326";
}

