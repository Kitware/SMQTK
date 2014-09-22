//
// LICENCE
// -------
// Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
// KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
// Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
//
/**
 * HoverClipView class displays a particular clip with other visual elements

 * @class
 */

/**
 * Constructor
 *
 */
function ClipView(config) {
    this.Init(config);
}

/**
 * Constructor
 *
 */

ClipView.prototype.Init = function(config) {

    this.config = {};

    $.extend(this.config, {
        "container" : null,
        "clipname" : "unknown",
        "score" : "unknown",
        "rank" : "unknown",
        "duration" : "unknown",
        "attribute_scores" : undefined,
        "query" : "unknown",
        "link_prefix" : "http://localhost/static/data/clips/"
    }, config );


    this.div =document.createElement('div');
    $(this.div).addClass("clip-view");

    this.strscores = "";
    if(this.config.attribute_scores) {
        for(var i =0; i < this.config.attribute_scores.labels.length; i++) {
            this.strscores = this.strscores + this.config.attribute_scores.labels[i] + " : " + this.config.attribute_scores.scores[i].toFixed(2) + "\n";
        }
    }
    var that = this;

    // When the results come back load them here
    this.scorediv = document.createElement("div");

    if(this.config.attribute_scores) {
        $(this.scorediv).html("<h4> #"+ this.config.rank + " : <button class='btn btn-mini'> "+ this.config.score.toFixed(2) + "</button> </h4> <br/>");
        $(this.scorediv).appendTo($(this.div)).find("button").bind("click", function(event) {
            event.stopPropagation();
            event.preventDefault();
            alert(that.strscores);
        });
    } else{
        $(this.scorediv).html("<h4> #"+ this.config.rank + " : " + this.config.clipname + " </h4> <br/>");
    }
    this.adiv = document.createElement('a');
    $(this.adiv).prop("href", this.config.link_prefix + this.config.clipname + ".q4.ogv");
    $(this.adiv).appendTo($(this.div));

    $(this.adiv).bind("click", function(event) {
        event.stopPropagation();
        event.preventDefault();
        window.open(this.href,"_blank");
    });

    this.data = [];


    this.clipdiv = document.createElement('div');
    $(this.clipdiv).addClass("clip-view-image");
    $(this.clipdiv).css("background", "transparent url(/clip?id=" + this.config.clipname + "&preview=middle_preview) 0px 0px no-repeat");
    $(this.clipdiv).appendTo($(this.adiv));

    this.adjudication = new Adjudication({container : this.clipdiv, clip : this.config.clipname, highlight : $(this.div), query : this.config.query });

    $(this.div).data("clipviewobj", this);
    $(this.div).attr("title", "" + this.config.duration + " sec");
    var that = this;
    $(this.clipdiv).ready(function() {
        //console.log("ready " + that.clipid);
        $(that.clipdiv).spriteOnHover({fps:4, rewind: "unanimate",repeat:true, loop : true});
    });
    $(this.div).appendTo($(this.config.container));
}



/*****************************************************************************
 *
 */
ClipView.prototype.Click = function() {
    console.log('clip view clicked');
}


