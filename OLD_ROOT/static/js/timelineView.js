//
// LICENCE
// -------
// Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
// KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
// Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
//
var timelineView = function(elem, data, clipDuration) {
//        console.log('timeline view getting', clipDuration);
        // Variables
        var margin = {top: 20, right: 20, bottom: 40, left: 200},
            width =  $(elem).width() - margin.left - margin.right,
            height = $(elem).height() - margin.top - margin.bottom,
            barHeight = 6,
            barWidth = 2,
            padding = 10,
            x = d3.scale.linear().range([padding, width - padding * 0.5]),
            y = d3.scale.ordinal().rangeBands([0, height]),
            y2 = d3.scale.ordinal().rangeBands([0, height]),
            y2Domain = [],
            maxTimestamp = 0,
            maxDuration = 0,
            yDomain = [],
            that = this;

        // Compute maximum timestamp and duration
        // Minimum is 2 seconds as it is possible that data might not have it
        var duration = 2;
        data.forEach(function(row) {
          yDomain.push(replaceUnderscores(row.name));
          y2Domain.push(addUnderscores(row.name));
          row.top_timestamps.forEach(function(ts) {
            if (ts.timestamp > maxTimestamp) {
              maxTimestamp = ts.timestamp;
            }
            if (ts.duration) {
              duration = ts.duration;
            }
            if (duration > maxDuration) {
              maxDuration = duration;
            }
          });
        });

        // Set the domain (input range)
        x.domain([0, maxTimestamp + maxDuration]);
        y.domain(yDomain);
        y2.domain(y2Domain);

        // Now create new row for each timestamp (this code coudl be optimized later)
        var newRows = [];
        data.forEach(function(row) {
          if (row.top_timestamps[0].duration) {
            duration = row.top_timestamps[0].duration;
          }
          row.timestamp = row.top_timestamps[0].timestamp;
          row.duration = duration;

          row.top_timestamps.forEach(function(ts) {
            var newRow = {};
            newRow.group = row.group;
            newRow.timestamp = ts.timestamp;

            // Check if the timestamp is within duration
            if (newRow.timestamp < 0) {
//              console.log('timestamp is less than 0. Setting it to 0');
              newRows.timestamp = 0;
            }
            if (newRow.timestamp > clipDuration) {
              newRow.timestamp = clipDuration;
//              console.log('timestamp is past duration. Setting timestamp to clip duration');
            }

            if (ts.duration) {
              newRow.duration = ts.duration;
            } else {
              newRow.duration = duration;
            }

            // Check if timestamp + duration is within clip duration. If not
            // then the duration will be difference of timestamp and clipduration
            if ((newRow.timestamp + newRow.duration) > clipDuration) {
//              console.log('Clamping duration as timestamp + duration is longer than clip duration');
//              console.log('timestamp, duration, clip duration', newRow.timestamp, newRow.duration, clipDuration);
              newRow.duration = Math.max(0.0, clipDuration - newRow.timestamp);
            }

            newRow.name = addUnderscores(row.name);
            newRows.push(newRow);
          });
        });
        data = newRows;

        // Now since the data is ready, create the visualization
        var xAxis = d3.svg.axis()
            .scale(x)
            .orient("bottom");

        var yAxis = d3.svg.axis()
            .scale(y)
            .orient("left");

        var zoom = d3.behavior.zoom()
            .x(x)
            .y(y)
            .on("zoom", zoomed);

        var svg = d3.select(elem).append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
          .append("svg:g")
            .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

        svg.append("svg:rect")
            .attr("width", width)
            .attr("height", height)
            .attr("class", "plot");

        svg.append("g")
          .attr("class", "x axis")
          .attr("transform", "translate(0," + height + ")")
          .call(xAxis)
        .append("text")
          .attr("x", width * 0.5)
          .attr("dy", "3em")
          .style("text-anchor", "end")
          .text("Time Offset (seconds)");

        svg.append("g")
          .attr("class", "y axis")
          .call(yAxis)
        .append("text")
          .attr("transform", "rotate(-90)")
          .attr("y", 6)
          .attr("dy", "-3em")
          .style("text-anchor", "end")
          .text("");

        svg.selectAll(".bar")
          .data(data)
        .enter().append("rect")
          .attr("class", "bar")
          .attr("id", function(d) { return d.name; })
          .attr("timestamp", function(d) { return d.timestamp; })
          .attr("x", function(d) { return x(d.timestamp); })
          .attr("width", function(d) { return (x(d.timestamp + d.duration) - x(d.timestamp)); })
          .attr("y", function(d) { return (y2(d.name) + y2.rangeBand() * 0.5 - barHeight * 0.5); })
          .attr("height", function(d) { return barHeight; })
          .on('mousedown', function(d) {
            // This is required
            d3.event.stopPropagation();
            d3.event.preventDefault();

            clearAllSelections();
            d3.select(this).classed('selectedStandalone', true);

            // Fire an event
            $('body').trigger('selectedStandAlone', [d.name, parseInt(d.timestamp, 10),
              d3.event.clientX, d3.event.clientY]);

            //  Stop propagation for jquery as well
            $(this).click(function(event){
               event.stopPropagation();
            });
          });

        function zoomed() {
            svg.select(".x.axis").call(xAxis);
            svg.selectAll(".bar")
                .attr("x", function(d) { return x(d.timestamp) - barWidth * 0.5; })
        }

        function clearAllSelections() {
          svg.selectAll('rect.bar').classed('selected', false);
          svg.selectAll('rect.bar').classed('selectedStandalone', false);
          svg.selectAll('rect.bar').classed("defocus", false);
        }

        // Public API
        return {
            // Assuming that name is unique
            select : function(name) {
                var timestamps = [];

                clearAllSelections();
                name = addUnderscores(name);
                if (name === null) {
                    return;
                }
                svg.selectAll('rect.bar').classed("selected", false);
                svg.selectAll('rect.bar').classed("defocus", true);

                var that = this;
                var nodes = svg.selectAll('rect.bar').filter(function(d, i) {
                    if (name === addUnderscores(d.name))  {
                        timestamps.push(d.timestamp);
                        return d
                    } else {
                        return null;
                    }
                });

                if (nodes[0].length > 0) {
                    d3.selectAll(nodes[0]).classed("defocus", false);
                    d3.selectAll(nodes[0]).classed("selected", true);
                }
                return timestamps;
            },
            selectStandalone : function (d3elem) {
              clearAllSelections();
              if (d3elem) {
                svg.select(d3elem).classed('selectedStandalone');
              }
            }
        };
    };
