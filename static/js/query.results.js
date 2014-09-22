//
// LICENCE
// -------
// Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
// KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
// Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
//
// Containhiser class

var traverse = function (node)
    {
    var string = node.data.title;

    var children = node.getChildren();
    if(children === null)
        {
        return string;
        }
    else
        {
        for(var i=0; i < children.length; i ++)
            {
            string = string + " " + traverse(children[i])
            }
        }
    return string;
    }

function QueryResults(container, query)
    {
    // Initializee to defaults
    this.Init(container, query);
    }

QueryResults.prototype.UpdateQuery = function()
    {
    // Form the query from known information
    this.query = { "label" : this.label,
              "min" : this.min,
              "dataset" : this.dataset,
              "skip" : this.skip,
              }
    }

QueryResults.prototype.Update = function()
    {
    // Form the query from known information
    // Async make the query
    $(this.images).html("<div> <br/><center> <h4> Wait while loading .. </h4> <img src = '/img/loading.gif'/> </center></div>");

    this.UpdateQuery();
    var that = this;
    var request = $.ajax(
        {
        type: "GET",
        url: this.url,
        data : that.query,
        timeout:60000,
        error: function(x, t, m)
            {
            if(t==="timeout")
                {
                alert("got timeout");
                }
            else
                {
                alert(t);
                }
            },
        success: function (msg)
            {
            // Get imagelist and send to AddImag3s
            // TODO: Update variables
            // that.Init(msg)
            //$(that.info).html('<div class="btn-group"> <button class="btn" onclick="result.Navigate(-1)">Previous</button> <button class="btn" onclick="result.Navigate(1)">Next</button> </div>');
            $(that.info).html("<div class=span12>" + " Count: " + msg.count + ", <br/>Query : " + JSON.stringify(msg.query, null, ' ') + "</div>");
            that.AddClips(msg.clips);
            }
        });


    }

QueryResults.prototype.Init = function(container, query)
    {
    this.dataset = "algo";
    this.url = "/query_score";
    // Create a div
    this.div =document.createElement('div');
    // "width" : "200", "height" : "200",

    this.query = query || { root :  ' ', node: null}
    this.label = this.query.root;
    this.node = this.query.node;
    this.algo = this.query.algo

    $(this.div).css({"float" : "left",  "border" : "1px solid", "padding" : "4px", "margin" : "4px", "overflow" : "hidden"});
    $(this.div).css({"padding-right" : "10px","padding-bottom" : "10px", });
    $(this.div).addClass("ui-widget-content");
    $(this.div).html('<div class="ui-widget-header" style="overflow:hidden"> <span style="float:left;width:500px;">' + this.query.root  + ' with ' + this.query.algo + ' </span><a href="#" class="ui-icon ui-icon-close" style="float:right;" onclick="resultContainer.OnQueryClose($(this).parent())"></a> </div>');

    this.queryButton = document.createElement('button');
    $(this.queryButton).css({"float" : "left"});
    $(this.queryButton).addClass("btn");
    $(this.queryButton).html("update");
    var that = this;
    $(this.queryButton).click(function()
        {
            console.log("Updating");
            that.Update();
        });
    $(this.queryButton).appendTo($(this.div))
    //$(this.div).append('');
    $(this.div).resizable(
        {grid : 100,}
    );

    // Add two more divs
    this.controls = document.createElement("div");
    $(this.controls).appendTo($(this.div));

    this.info = document.createElement("div");
    $(this.info).appendTo($(this.div));

    this.images = document.createElement("div");
    $(this.images).appendTo($(this.div));

    $(this.div).appendTo($(container));


//    if(msg === undefined)
//        {
//
//        }
//    else
//        {
//        this.count = msg.count;
//        this.imagelist = msg.images
//        this.label = msg.query.label
//        }
    this.skip = 0;
    this.limit = 20;
    // this.Update();
   }

QueryResults.prototype.Close = function()
    {
    alert("Closing..");
    }

QueryResults.prototype.AddImages = function(imagelist)
    {
    // Clear existing images
    $(this.images).empty();

    // Add images
    for(var i = 0; i < imagelist.length; i++)
        {
        // Create an Image element
        // $('#images').append('<div style="width:100px;height:100px;float:left!important"> <img style="display:block;margin-top: auto;margin-bottom: auto;margin-left: auto; margin-right: auto"  src="/image?id=' + imagelist[i] + '"/> </div>');
        $(this.images).append('<div style="width:150px;height:150px;float:left!important;background-size: 95%!important;background:url(' + "/image?id=" + imagelist[i][0] + "&col=" + this.dataset + ') no-repeat center center;" title="Score: ' + imagelist[i][1] + ', Clip: ' + imagelist[i][2] + '"></div>');
        // img {}
        //    .image {min-height:50px}        }
        }
    }

QueryResults.prototype.AddClips = function(cliplist)
    {
    // Clear existing images
    $(this.images).empty();

    // Add images
    for(var i = 0; i < cliplist.length; i++)
        {
        // TODO: Create an image display element
        // Create an Image element
        // $('#images').append('<div style="width:100px;height:100px;float:left!important"> <img style="display:block;margin-top: auto;margin-bottom: auto;margin-left: auto; margin-right: auto"  src="/image?id=' + imagelist[i] + '"/> </div>');
        //$(this.images).append('<div style="width:150px;height:150px;float:left!important;background-size: 95%!important;background:url(' + "/image?id=" + cliplist[i].images[0][0] + "&col=" + this.dataset + ') no-repeat center center;" title="Score: ' + cliplist[i].images[0][1] + ', Clip: ' + cliplist[i].clip_id + ', Top Scoring Images : ' + cliplist[i].images.length  +  '"></div>');
        $(this.images).append('<div class="middle"> <img src = "/zero_shot/clip_middle?id=' + cliplist[i] + '" title="Score: "></div>');
        // img {}
        //    .image {min-height:50px}        }
        }
    }



//ImageResults.prototype.AddGraph = function(values)
//    {
//    // TODO: First clear graph
//    $("#chart_container").html('<div id="chart" onclick="result.Update();"></div>' )
//
//    this.data = [] ;
//
//    for(var i =0; i < values.length; i++)
//        {
//        this.data.push({'x' : i, 'y' : values[i]})
//        }
//    var that = this;
//    // instantiate our graph!
//    this.graph = new Rickshaw.Graph( {
//            element: document.getElementById("chart"),
//            min : "auto",
//            renderer: 'line',
//            height: 150,
//            width: 500,
//            series: [
//                {
//                    data: this.data,
//                    color: "#30c020"
//                }
//            ]
//        } );
//
//        //this.y_ticks = new Rickshaw.Graph.Axis.Y( {
//         //   graph: this.graph,
//          //  orientation: 'left',
//           // tickFormat: Rickshaw.Fixtures.Number.formatKMBT,
//          //  element: document.getElementById('y_axis'),
//        //} );
//
//        this.x_axis = new Rickshaw.Graph.Axis.X( { graph: this.graph } );
//
//        this.graph.render();
//
//        var hoverDetail = new Rickshaw.Graph.HoverDetail( {
//                graph: this.graph,
//                formatter: function(series, x, y) {
//                    var content = "percentile: " + x +", score: " + y.toFixed(2);
//                    // console.log(content);
//                    that.min = y.toFixed(2);
//                    return content;
//                }
//            } );
//
//
//
//    }
//
//ImageResults.prototype.UpdateImages = function()
//    {
//    // Either previous / next button is pressed or the threshold is changed
//    // Only request images
//
//    // Derive min and max values
//    var that = this;
//    $('#images').html("<center> <h2> Wait while loading .. </h2> <img src = '/img/loading.gif'/> </center>");
//
//    if(this.min === undefined)
//    {this.min = 0;}
//
//
//    var request = $.ajax(
//        {
//        type: "GET",
//        url: "/query_score?label=" +this.label + "&min=" + this.min + "&dataset=" + this.dataset + "&skip=" + this.skip,
//        timeout:25000,
//        error: function(x, t, m)
//            {
//            if(t==="timeout")
//                {
//                alert("got timeout");
//                }
//            else
//                {
//                alert(t);
//                }
//            },
//        success: function (msg)
//            {
//            // Get imagelist and send to AddImag3s
//            // TODO: Update variables
//            // that.Init(msg)
//            $("#query_info").html('<div class="span2 btn-group"> <button class="btn" onclick="result.Navigate(-1)">Previous</button> <button class="btn" onclick="result.Navigate(1)">Next</button> </div>');
//            $("#query_info").append("<div class=span10>" + " Count: " + msg.count + ", Query : " + JSON.stringify(msg.query) + "</div>");
//            that.AddImages(msg.images);
//            }
//        });
//
//    }
//
//ImageResults.prototype.Navigate = function(where)
//    {
//    if(where > 0)
//        {
//        this.skip = this.skip + this.limit;
//        }
//    else
//        {
//        if(this.skip === 0)
//            {
//            return;
//            }
//        this.skip = this.skip -  this.limit;
//        }
//    this.UpdateImages();
//    }
//

//ImageResults.prototype.Update = function()
//    {
//    // Updates and initiates the query by reading the values
//    this.node =$("#attributes").dynatree("getActiveNode");
//
//    if(this.node)
//        {
//        }
//    else
//    {
//    return;
//    }
//
//    console.log(this.node.data.title);
//
//    this.label = this.node.data.title;
//    console.log("Label : " + this.node.data.title);
//
//
//    this.dataset = $("#dataset").selectpicker().val();
//    console.log("Dataset : " + this.dataset);
//
//    this.dist = $("#dist").selectpicker().val();
//    console.log("Dist : " + this.dist);
//    var that = this;
//    var request = $.ajax(
//        {
//        type: "GET",
//        url: "/attribute_info?dist=" + this.dist + "&label=" +this.label,
//        success: function (msg)
//            {
//            that.OnData(msg);
//            that.UpdateImages();
//            //alert( "Data Saved: " + JSON.stringify(msg) );
//            }
//        });
//
//    }
//
//
//ImageResults.prototype.OnData = function(msg)
//    {
//    // Accept the incoming data
//    this.Init(msg);
//
//    // Update the state for results
//    $("#attrib_info").html(" <center> Label: " + this.label + ", Scores Index: " + msg.scores_index + "</center>");
//    // Update the state for images
//    // this.AddImages(this.imagelist.slice(this.firstImage, this.firstImage+this.numImages))
//    this.AddGraph(msg.dist)
//
//    }

