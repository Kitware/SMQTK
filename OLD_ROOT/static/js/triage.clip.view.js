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
function TriageClipView(container, kit_id, clip_id, score, kit_text, rank, duration) {
    this.Init(container, kit_id, clip_id, score, kit_text, rank, duration);
}


/**
 *
 */
function replaceUnderscores(str) {
    if (str) {
        return str.replace(/_/g, ' ');
    }

    return str;
}

/**
 *
 */
function addUnderscores(str) {
    if (str) {
        return str.replace(/ /g, '_');
    }

    return str;
}

/**
 * Init
 *
 */
TriageClipView.prototype.Init = function(container, kit_id, clip_id, score, kit_text, duration) {
    // Create a div
    this.kit_id = kit_id;
    this.duration = duration;
    this.rank = rank;
    this.clip_id = clip_id;
    this.score = score;
    this.kit_text = kit_text;
    this.container = container;
    this.div =document.createElement('div');
    $(this.div).addClass("clip-view");


    this.scorediv = document.createElement('div');
    this.clipdiv = document.createElement('div');
//    this.adiv = document.createElement('a');
//    $(this.adiv).prop("href","/triage_info_view?kit=" + this.kit_id + "&rank=" + this.rank);

    this.data = [];
    this.textView = null;
    this.timelineView = null;
    this.videoPlayer = null;
    this.currentClipId = null;

    // When the results come back load them here
    $(this.scorediv).html("<h4> #"+ rank +  ": "+ clip_id + "</h4> <br/>");
    $(this.scorediv).appendTo($(this.div));


    $(this.clipdiv).addClass("clip-view-image");
    $(this.clipdiv).css("background", "transparent url(/clip?id=" + clip_id + "&preview=middle_preview) 0px 0px no-repeat");

    $(this.clipdiv).appendTo($(this.div));

    this.adjudication = new Adjudication({container : this.div, clip : this.clip_id, highlight : $(this.div), event_string : "triage_storage", rank:this.rank});

//    console.log(this.adjudication);

    $(this.adiv).appendTo($(this.div));

    $(this.div).data("clipviewobj", this);
    $(this.div).attr("title", "" + this.duration + " sec");

    $(this.div).appendTo($(this.container));
    var that = this;
    $(this.clipdiv).ready(function() {
        //console.log("ready " + that.clipid);
        $(that.clipdiv).spriteOnHover({fps:4, rewind: "unanimate",repeat:true, loop : true});
    });

}


/*****************************************************************************
 *
 */
TriageClipView.prototype.Click = function() {
    console.log('hover clip view clicked');
}


