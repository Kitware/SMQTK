$(function () {
  'use strict';

  var map = geo.map({
    node: '#map',
    center: {
      x: -98.0,
      y: 39.5
    },
    zoom: 1
  });

  map.createLayer(
    'osm',
    {
      baseUrl: 'http://otile1.mqcdn.com/tiles/1.0.0/map/'
    }
  );

  $.ajax( "/api/v1/data?limit=50000" )
  .done(function(data) {
    map
      .createLayer('feature')
        .createFeature('point')
          .data(data)
          .position(function(d) {return {x:d.field7, y:d.field6}})
            .style('radius', 10)
            .style('fillColor', "red")
            .style('fillOpacity', 0.01)
            .style('strokeColor', {r: 0.3, g: 0.3, b: 0.3})
            .style('stroke', 0);
    map.draw();
  })
  .fail(function() {
    alert( "error" );
  });
});