//
// LICENCE
// -------
// Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
// KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
// Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
//
// Inherits from query results
// Adds specifics for attribute node editing and results display

// Inherit from Queryresults

AttributeQueryResults.prototype = Object.create(QueryResults.prototype);
AttributeQueryResults.prototype.constructor = AttributeQueryResults;

function AttributeQueryResults(container, query) 
    {
    // Call base class init
    console.log("AttributesQueryResults")
    this.parent = QueryResults.call( this, container, query);

    // Now assume that basics are created Now add a slider for the threshold
    this.slider = document.createElement('div');

    this.sliderValue = document.createElement('div');
    $(this.sliderValue).css({"float" : "clear"});
    
    $(this.slider).css({ "width" : "100px", "margin": "5px", "float" : "left"});
    var that = this;
    $(this.slider).slider(
        {
        range: "max",
        min: -0.5,
        max: 1.,
        step:0.01,
        value: that.node.data.min,
        slide: function( event, ui ) 
            {
            $(that.sliderValue).html( ui.value );
            // Also update the value in the treenode
            that.node.data.min = ui.value;
            },
        });
        
    $(this.sliderValue).html($(this.slider).slider("value") );
    $(this.slider).appendTo($(this.controls));
    $(this.sliderValue).appendTo($(this.controls));
    

    // Assume that calibration exists for the attributes that are going to come
    // Fall back to the known percentile numbers in case calibration not existing
     
    // Initializee to defaults    
    }


AttributeQueryResults.prototype.UpdateQuery = function()
    {
    // Form the query from known information 
    // Async make the query
    this.url = "/group_score";
    this.query = { "label" : this.label,
              "min" : parseFloat($(this.slider).slider("value")),
              "dataset" : "algo",
              "skip" : this.skip,
              } 
    }

//    
//AttributeQueryResults.prototype.Update = function()
//    {
//    // Form the query from known information 
//    // Async make the query 
//    this.min = $(this.slider).slider( "value" );
//    this.dataset = "algo";
//    this.skip = 0;
//
//    var request = $.ajax(
//        {
//        type: "GET",
//        url: "/query_score?label=" +this.label + "&min=" + this.min + "&dataset=" + this.dataset + "&skip=" + this.skip,
//        timeout:60000,
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
//            //$("#query_info").html('<div class="span2 btn-group"> <button class="btn" onclick="result.Navigate(-1)">Previous</button> <button class="btn" onclick="result.Navigate(1)">Next</button> </div>');
//            //$("#query_info").append("<div class=span10>" + " Count: " + msg.count + ", Query : " + JSON.stringify(msg.query) + "</div>");
//            //that.AddImages(msg.images);
//            }
//        });
//    
//    
//    }
