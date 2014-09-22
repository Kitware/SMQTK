//
// LICENCE
// -------
// Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
// KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
// Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
//

// Displays clip

function Adjudication(config) {
    this.Init(config);
}

    // Accepts varioud configuration options
    //
    //    this.clip_id = clip_id;
    //    this.score = score;
    //    this.duration = null;
    //    this.container = container;
    //    this.ul = document.createElement('ul');
    //    this.div =document.createElement('div');
    //    this.p =document.createElement('p');
    //    $(this.ul).addClass("imageRow");
    //    $(this.div).addClass("filmstrip");
    //
    //    $(this.p).html("</br> <h4> Score: "+ score.toFixed(2) + "</h4>");
    //    // When the results come back load them here
    //    $(this.p).appendTo($(this.container))
    //    $(this.div).appendTo($(this.container));
    //    $(this.ul).appendTo($(this.div))
    //
    //    // Navigation
    //    this.skip = 0;
    //    this.limit = 20;
    //    // this.Update();

Adjudication.prototype.Highlight = function(whichclass, add) {
    // highlights the border around

    if (this.config.highlight === null){
        return;
    }

    if (add === undefined){
        add = true;
    }

    // Add start to the highlight element
    if(this.config.star_independent === true) {
        if(whichclass === "adjudication-star") {
            if (add) {
                this.stardiv = document.createElement("img");
                this.stardiv.src = "/img/star.png";
                $(this.stardiv).addClass("adjudication-star-image");
                $(this.stardiv).appendTo(this.config.highlight);
                $(this.stardiv).bind("click",function(event){
                    event.stopPropagation();
                    event.preventDefault();
                    //TODO: make sure this works as adjudication
                });
            }
            else {
                // Remove the star
                $(this.stardiv).remove();
            }
        }
        else{
            $(this.config.highlight).removeClass('adjudication-yes');
            $(this.config.highlight).removeClass('adjudication-no');

            if(add) {
                $(this.config.highlight).addClass(whichclass);
            }

        }
    }
    else {
        $(this.config.highlight).removeClass('adjudication-yes');
        $(this.config.highlight).removeClass('adjudication-no');
        $(this.config.highlight).removeClass('adjudication-star');

        if(add) {
            $(this.config.highlight).addClass(whichclass);
        }
    }
}

Adjudication.prototype.Init = function(config) {
    // Initializee to defaults

    this.config = {};

    $.extend(this.config, {
        "container" : null,
        "highlight" : null,
        "clip" : "unknown",
        "query" : "unknown",
        "display":false,
        "opacity" : 0.8,
        "star_independent" : false,
        "event_string" : "my_storage",
        "rank" : 0
    }, config );

    // Have different configuration strings for different configurations

    // console.log(this.config);
    // Create and add div to the container
    this.div = document.createElement('div');

    $(this.div).html('<img class="yes" height="24px"/> ' +
                     '<img class="no" height="24px"/> ' +
                     '<img class="star" height="24px"/>'
                     );

    $(this.div).addClass("clipAdjudication");
    $(this.div).css("opacity","" + this.config.opacity);


    if(this.config.display === false) {
        $(this.div).css("display","none");
    }

    $(this.div).appendTo(this.config.container);

    var that = this;

    if(that.config.display === false) {
        // Bind events
        $(this.config.container).mouseenter(function (event){
//          console.log("inside " + that.config.clip);
            $(that.div).css("display","inline-block");
        }).mouseleave(function (event){
//                console.log("leaving " + that.config.clip);
            $(that.div).css("display","none");
        });
    }

//    $(this.div).click(function (event) {
//        event.preventDefault();
//        event.stopPropagation();
//        console.log("clicked" + that.config.clip);
//    });
    //console.log(" Found " + $(this.div).find("img").length);

    $(this.div).find("img").click(function(event) {
        //console.log("status " + $(this).attr("class"));
        event.preventDefault();
        event.stopPropagation();
        // TODO: Send the ajax to api
        adj = that.GuessOperation(event.target);

        // Guess Adjudication

        var reqdata = {
            clip : that.config.clip,
            label : adj.label,
            query : that.config.query,
            op :adj.op,
            rank:that.config.rank
        };

        that.ApplyAdjudication(reqdata);
        // Change the local storage
        localStorage.setItem("adjudication", JSON.stringify(reqdata));
        // Make sure that this clip information is updated
        localStorage.setItem(reqdata.clip, JSON.stringify(reqdata));

        // Send out the ajax request to register this change at server-side
        $.getJSON("/adjudicate",reqdata, function(data){
            //console.log(data);
        });
    });

    // Bind the storage changed event
    $(window).on('storage', function (e) {
        // Listen to only adjudication storage
        if(e.originalEvent.key === "adjudication") {
            var reqdata = JSON.parse(e.originalEvent.newValue);

            if(reqdata.clip === that.config.clip) {
                // The change matters, adjust to the new state
                console.log("Value changed from another tab");
                that.ApplyAdjudication(reqdata);
            }
        }
    });

    console.log("Looking for " + this.config.clip);
//
//    for (var i = 0; i < localStorage.length; i++){
//        console.log(localStorage.key(i));
//        console.log(localStorage.getItem(localStorage.key(i)));
//    }

    // Check the adjudication state
    var statestr = localStorage.getItem(this.config.clip); //returns "Some Value"
    if(statestr !== null) {
        var reqdata = JSON.parse(statestr);
        if(reqdata.clip === that.config.clip) {
            // The change matters, adjust to the new state
            console.log("Value appears to be set already");
            that.ApplyAdjudication(reqdata);
        }
    }
};

Adjudication.prototype.GuessOperation = function(img) {
    // Based on the current image sources
    // Find out what operation was performed
    // todo: this can be done more elegantly with css and classes
    var toReturn = {};

    if ($(img).hasClass("yes")) {
       toReturn.label = "yes";
    }

    if ($(img).hasClass("adjudication")){
       toReturn.op = "remove";
    } else {
       toReturn.op = "add";
    }


    if ($(img).hasClass("no")) {
       toReturn.label = "no";
    }

    if ($(img).hasClass("star")) {
       toReturn.label = "star";
    }

//    if ($(img).hasClass("adjudication-yes")){
//           toReturn.op = "add";
//           toReturn.label = "yes";
//    } else if ($(img).hasClass("adjudication-yes-on")){
//           toReturn.op = "remove";
//           toReturn.label = "yes";
//    } else if ($(img).hasClass("adjudication-no-off")){
//           toReturn.op = "add";
//           toReturn.label = "no";
//    } else if ($(img).hasClass("adjudication-no-on")){
//           toReturn.op = "remove";
//           toReturn.label = "no";
//    } else if ($(img).hasClass("adjudication-star-off")){
//           toReturn.op = "add";
//           toReturn.label = "star";
//    } else if ($(img).hasClass("adjudication-star-on")){
//           toReturn.op = "remove";
//           toReturn.label = "star";
//    }

    return(toReturn);
};


Adjudication.prototype.ApplyAdjudication = function(reqdata) {
    // Do the ui changes to match the request

    $.event.trigger({
        type: this.config.event_string,
        message: reqdata,
        time: new Date()
    });

    // First determine what image we are dealing with
    // Then change its source to match what is required
    console.log("Applying Adjudication ..");
    console.log(reqdata);
    // Change the highlight
    this.ClearHighlight();
    var type = reqdata.label;

    if(reqdata.op === "add"){
        this.Highlight("adjudication-"+type, true);
        if(this.config.star_independent === true) {
            if(type !== "star") {
                $(this.div).find('img').removeClass("adjudication");
            }
        }
        else {
            $(this.div).find('img').removeClass("adjudication");
        }
        $(this.div).find('img.' + type).addClass("adjudication");
    }

    // Change the image src
    if(reqdata.op === "remove"){
        this.Highlight("adjudication-" + type, false);
        $(this.div).find('img.' + type).removeClass("adjudication");
    }

    // Add the class to image
    //  if

};

Adjudication.prototype.Close = function()
    {
    alert("Closing..");
    };

Adjudication.prototype.Update = function(clip_id)
    {
    // Updates and initiates the query by reading the values
    var that = this;
    var request = $.ajax(
        {
        type: "GET",
        //url: "/clip?dist=" + this.dist + "&label=" +this.label,
        url: "/frames?clip=" + this.clip_id,
        success: function (msg)
            {
            that.OnData(msg);
            //alert( "Data Saved: " + JSON.stringify(msg) );
            }
        });
    };

Adjudication.prototype.ClearHighlight = function()
    {
    // Remove all highlights
    $(this.ul).find('li').removeClass();
    $(this.div).scrollLeft(0);
    };


Adjudication.prototype.Clear = function()
    {
    // Remove all highlights
    $(this.div).find('img').removeClass("adjudication");
    this.Highlight("", false);
    };


Adjudication.prototype.ClearAdjudicationOn = function()
    {
    // Remove all highlights
    $(this.ul).find('li').removeClass();
    $(this.div).scrollLeft(0);
    };


Adjudication.prototype.HideAll = function(timestamp, classname)
    {
    // Hide all frames
    var elem = $(this.ul).find('li').addClass("hiddenframe")
    $(this.div).scrollLeft(0);
    };

Adjudication.prototype.SelectTimestamp = function(timestamp, classname)
    {
    // We want to select an li with data-duration=timestamp
    var elem = $(this.ul).find('li[data-duration="' + (timestamp) + '"]');
    //console.log($(elem).html());
    $(elem).removeClass();
    $(elem).addClass(classname);

    // find the position of
    if ($(elem).length > 0) {
        $(this.div).scrollLeft($(elem).offset().left- $(this.div).width() * 0.5);
    }

    // highlite
    };

Adjudication.prototype.OnData = function(msg)
    {
    // Update the state for results
    var strlist = "";
    $(this.ul).css("width",  '' + msg.frames.length * 170 + "px");

    for(var i = 0; i < msg.frames.length; i ++)
        {
        strlist = strlist + '<li data-duration="' + msg.frames[i].duration + '"> <img width="150px" src="/image?id=' + msg.frames[i]._id +  '&col=frames"/>' +  '</li>';
        }

    $(this.ul).html(strlist);
    // Update the state for images
    // this.AddImages(this.imagelist.slice(this.firstImage, this.firstImage+this.numImages))
    };

Adjudication.prototype.DownloadCSV = function() {
    // Compile the object
    var obj = {};
    obj["some"] = other;
};