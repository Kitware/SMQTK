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
      baseUrl: 'http://c.tile.stamen.com/terrain-labels/'
    }
  );

  $.ajax( "/api/v1/data?limit=10000" )
  .done(function(data) {
    map
      .createLayer('feature')
        .createFeature('point')
          .data(data)
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