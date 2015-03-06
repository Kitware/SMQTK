//
// LICENCE
// -------
// Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
// KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
// Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
//
/**his.diide]
 * TriageViewStandalone class owns the div and displays all its elements
 * with other visual elements

 * @class
 */

/**
 * Constructor
 *
 */
function TriageViewStandalone(container, values) {
    // Container is unused (global hardcoded dialog is used)
    this.videoPlayer = null;
    this.container = container;

    this.div = document.createElement('div');
    $(this.div).html(' <h2> <span class="triageLabel"> </span> </h2> ' +
            '<br/> ' +
        '<div class="row-fluid"> ' +
            '<div class="span4"> ' +
                '<div class="row-fluid"> ' +
                    '<div class="triageClipEvidences" class="span4"> ' +
                    '</div> ' +
                '</div> ' +
                '<div class="row-fluid"> ' +
                    '<div class="triageAdjudicationHeading" class="span4"> ' +
                        '<h4> Adjudication </h4> <br/>' +

                        '<div class="triageAdjudication" class="span4"> </div>' +
                    '</div> ' +

                '</div> ' +
            '</div> ' +
            '<div class="triageVideo video-container span8"> ' +
           '</div> ' +
        '</div> ' +
        '<div class="row-fluid"> ' +
            '<div class="span12"> ' +
                '<div class="triageTimeLineView"></div> ' +
            '</div> ' +
        '</div> ' +
        '<div class="row-fluid"> ' +
            '<div class="span12"> ' +
                '<div class="triageFilmStripView"></div> ' +
           '</div> ' +
        '</div> '
        );

    $(this.div).appendTo($(this.container));
    this.Init(values);
    this.adjudication = new Adjudication({container : $(this.div).find(".triageAdjudication")[0], clip : this.data.clip, display:true, opacity : 0.5});
    $(this.adjudication.div).css("position","relative");

    var that = this;
    // TODO: This should be called after results
    $(this.div).find('.triageClipEvidences').on('click', 'div label', function(event) {
        //if (!that.textView || !that.timelineView) {
        //    return;
        //}

        event.stopPropagation();

        // Select current element
        that.textView.selectElement(this);
        var labelObj = $(this);

//        console.log('selecting......', labelObj.text());

        var timestamps = that.timelineView.select(labelObj.text());
        that.filmstripView.HideAll();
        for(var i=0; i < timestamps.length; i++)
            {
            that.filmstripView.SelectTimestamp(timestamps[i], "selectframe_text")
            }
        });

//    // Now initialize the timeline
//    $('#myModal').on('shown', function() {
//        if (that.currentClipId === null || that.currentClipId !== that.clip_id) {
//            return;
//        }
//        // Now show it
//        var clipEvidencesObj  = $('#clipEvidences');
//        var timelineViewObj = $('#timelineView');
//        var filmstripViewObj = $('#filmstripView');
//
//        // that.textView = new textView(clipEvidencesObj.get(0), that.data);
//        // that.timelineView = new timelineView(timelineViewObj.get(0), that.data);
//        // that.currentClipId = null;
//
//        // Get the data
//        //var algorithm = $("#algorithms-combo").find('option:selected').data('algo');


        //this.updateResults(that.kit_id, that.clip_id);

        $('body').on('selectedStandAlone', function(event, name, timestamp, x, y) {
            // that.filmstripView.SelectTime(event, timestamp);
            //alert('' + x+ y);
            that.filmstripView.ClearHighlight();
            that.filmstripView.SelectTimestamp(timestamp, "selectframe_timeline");
            that.selectStandalone(name, timestamp);
            });

//
//    $('#myModal').on('hidden', function () {
//        // Clear current rendering
//        that.currentClipId = null;
//        if (that.videoPlayer) {
//            that.videoPlayer.pause();
//            }
//        });
//
//    //$(this.div).click(function()
//

    // Respond to adjugation buttons

//   $(this.div).find(".triageAdjudication img").click(function(event) {
//
//       event.stopPropagation();
//        // TODO: Send the ajax to api
//
//        switch($(this).attr("class")) {
//           case "yes":
//               if ($(this).attr('src') === '/img/carbon-verified.png') {
//                   $(this).attr('src', '/img/carbon-verified_on.png')
//                   $(this).next().attr('src', '/img/carbon-rejected.png')
//                   $(this).next().next().attr('src', '/img/carbon-star2.png')
//               }  else {
//                   $(this).attr('src', '/img/carbon-verified.png')
//               }
//               break;
//           case "no":
//               if ($(this).attr('src') === '/img/carbon-rejected.png') {
//                   $(this).attr('src', '/img/carbon-rejected_on.png')
//                   $(this).next().attr('src', '/img/carbon-star2.png')
//                   $(this).prev().attr('src', '/img/carbon-verified.png')
//               }  else {
//                   $(this).attr('src', '/img/carbon-rejected.png')
//               }
//               break;
//           case "star":
//               if ($(this).attr('src') === '/img/carbon-star2.png') {
//                   $(this).attr('src', '/img/carbon-star_on.png')
//                   $(this).prev().attr('src', '/img/carbon-rejected.png')
//                   $(this).prev().prev().attr('src', '/img/carbon-verified.png')
//               }  else {
//                   $(this).attr('src', '/img/carbon-star2.png')
//               }
//               break;
//       }
//   });
//
//   var that = this;
//
//   $(this.div).find(".triageAdjudication div").click(function(event) {
//       event.stopPropagation();
//
//       // if ($(this).hasClass('btn')) {
//           // $(that.div).find(".triageAdjudication div").find('.myLabel').show();
//           // console.log($('#adjudication div').find('input').text());
//           // $(that.div).find(".triageAdjudication div").find('label').text($(that.div).find(".triageAdjudication div").find('input').val());
//           // $(that.div).find(".triageAdjudication div").find('input').addClass("disabled");
//       // }
//       // if ($(this).hasClass('myLabel')) {
//           // $(that.div).find(".triageAdjudication div").find('input').val($(this).text());
//           // $(this).hide();
//           // $(that.div).find(".triageAdjudication div").find('input').show();
//       // }
//       if ($(event.target).hasClass('input')) {
//           //console.log("enabled");
//           //$(event.target).removeAttr("disabled","");
//       }
//
//   });
//
//$(this.div).find('.triageAdjudication div').find('input').keypress(function(e) {
//        if(e.which == 13) {
//           $(that.div).find(".triageAdjudication div").find('.myLabel').show();
//           //console.log($('#adjudication div').find('input').text());
//           //$(that.div).find(".triageAdjudication div").find('label').text($(that.div).find(".triageAdjudication div").find('input').val());
//           //$(that.div).find(".triageAdjudication div").find('input').attr("disabled","");
//           $(that.div).find(".triageAdjudication div").find('input').blur();
//        }
//        e.stopPropagation();
//    });

}

TriageViewStandalone.prototype.Init = function(indata)
    {
    // Accept incoming data
    var defaults = {
        "algonames" : ["OB_max_avg_positive_hik_21", "sun_attribute_avg_positive_hik_27"],
        "url" : "/triage_info"
    }
    this.data = $.extend({}, defaults, indata);

    var that = this;
    // Register functions

    $(this.div).on('click', function(event) {
       if (!that.textView || !that.timelineView) {
           return;
       }

        // Register Deselection Handle deselection
        $(that.div).find('triageClipEvidences div label').removeClass('selected');
        that.timelineView.select(null);
        that.textView.selectStandalone(null);
        that.timelineView.selectStandalone(null);
        that.filmstripView.ClearHighlight();

        //console.log('that.currentClipId', that.currentClipId);
        //console.log('that.clipview.clip_id', that.clipview.clip_id);

       that.videoPlayer.play();

        //if (that.currentClipId !== null && that.currentClipId === that.clipview.clip_id) {
        //   console.log('Keeping video player');
        //   that.videoPlayer.play();
        //} else {
        //   console.log('Setting video player to null');
        //   that.videoPlayer = null;
        //}
   });

    // Call refresh with proper paramters
    this.Refresh()
    }

TriageViewStandalone.prototype.Refresh = function()
    {
    // this.data contains all relevant information

    // $(this.div).find(".triageLabel").html("Score: "+ this.data.score.toFixed(2));

    if(this.data === undefined)
        {
        // Do not refresh
        return;
        }

    this.clearVisuals();

    var that = this;

    // Display rank
    // console.log("Updating to " + that.data.rank);
    $(this.div).find(".triageRank").html("Rank: " + that.data.rank)

    //Create filmstripView
    $(this.div).find(".triageFilmStripView").empty();
    this.filmstripView = new FilmStripView($(this.div).find(".triageFilmStripView"), this.data.clip, this.data.score)
    this.filmstripView.Update(this.data.clip);


    // Create video player
    this.currentClipId = this.data.clip;

    //console.log('videoPlayer', this.videoPlayer);

    if (this.videoPlayer) {
        // Tricks to make sure the video is emptied before
        this.videoPlayer.pause();
        var source = $(this.div).find(".triageVideo").find('source');
        $(source).attr('src', '');
        //console.log('source', source);
    }

    // Use this line if using mp4
    // These codecs do not make sense codecs="dirac, speex"
    var time = new Date();

    $(this.div).find(".triageVideo").html(
        '<video height="250px" controls autoplay loop>' +
        '<source src="' + video_url_prefix + this.data.clip  + '.q4.ogv" type="video/ogg" preload="auto"></source>' +
        '</video>');

    // Load video
    this.videoPlayer = new videoPlayer($(this.div).find(".triageVideo").find("video").get()[0])

//    // Adjudication Resets
//    $(this.div).find(".triageAdjudication div").find('.myLabel').hide();
//    $(this.div).find(".triageAdjudication div").find('label').text();
//    $(this.div).find(".triageAdjudication div").find('input').show();
//    $(this.div).find(".triageAdjudication div").find('input').val('');
//    $(this.div).find('.yes').attr('src', '/img/carbon-verified.png');
//    $(this.div).find('.no').attr('src', '/img/carbon-rejected.png');

    // Now show it
    var clipEvidencesObj  = $('#clipEvidences');
    var timelineViewObj = $('#timelineView');
    var filmstripViewObj = $('#filmstripView');

    // that.textView = new textView(clipEvidencesObj.get(0), that.data);
    // that.timelineView = new timelineView(timelineViewObj.get(0), that.data);
     // Create adjudication

    // that.currentClipId = null;

    // Get the data
    //var algorithm = $("#algorithms-combo").find('option:selected').data('algo');

    this.updateResults(this.data.kit, this.data.clip, this.data.algonames );

    $('body').on('selectedStandAlone', function(event, name, timestamp, x, y) {
        if (!that.videoPlayer) {
            return console.log('Non existent video player');
        }

        //if (!that.videoPlayer.isValid(timestamp)) {
        //    console.log('Invalid video seek time');
        //    return;
        // }
        that.filmstripView.ClearHighlight();
        that.filmstripView.SelectTimestamp(timestamp, "selectframe_timeline");
        //that.selectStandalone(name, timestamp);
        });
    }


/**
 * Update results
 *
 */
TriageViewStandalone.prototype.updateResults = function(kitId, clipId, algonames)
    {
    // TODO: why this.currentClipId ? just use one from this.data

    this.evidences = [];

    for(var i =0 ; i < algonames.length ; i ++) {
        this.updateAlgorithmResults(kitId, clipId, this.evidences, algonames[i]);
    }

    // this.updateAlgorithmResults(kitId, clipId, this.evidences, 'sun_attribute_avg_positive_hik_27');

    // Now update the label. We delayed the label initialization as we didn't
    // have duration value when the clip was initialized.

    // Update the label
    $(this.div).find('.triageLabel').html("Clip " + this.data.clip + " ("
        + this.data.kit_text + ", "  + this.data.duration + " seconds)");


    document.title = "Triage Info :" + this.data.clip + " (" + this.data.kit_text + ")";

    //var clipEvidencesObj  = $(t)('#clipEvidences');
    //var timelineViewObj = $('#timelineView');

    this.textView = new textView($(this.div).find(".triageClipEvidences"), this.evidences);
    this.timelineView = new timelineView($(this.div).find(".triageTimeLineView").get()[0], this.evidences, this.data.duration);
    }

/**
 * Update algorithm specific results
 *
 */
TriageViewStandalone.prototype.updateAlgorithmResults = function(kitId, clipId, data, algorithm) {
    var that = this;

    // console.log('Updating results for', algorithm);
    var request = $.ajax({
        type: "GET",
        context: this,   // TODO: this line is not needed if var that = this is used
        async: false,
        url: that.data.url + "?kit=" + kitId + "&clip=" + clipId + "&algo=" + algorithm + "&strict=strict_v1",
        success: function (msg) {
            // Setup timer
            if("error" in msg) {
                console.log( "Submission not accepted" + JSON.stringify(msg));
            } else {
                // Update clip duration. Currenly this is the only time we get
                // clip duration value from database
                // if (!that.data.duration) {
                //     console.log('duration', msg.duration);
                //     that.data.duration = msg.duration;
                // } else if (that.data.duration !== msg.duration) {
                //     console.log('Mismatched clip duration from different algorithms');
                // }

                // TODO As per customer request hard-coding to
                // 5 evidences for now
                // for(i = 0; i < msg.evidences.length; ++i) {
                // Get the number of groups

                //var noOfEvidences = 5;
                for(i = 0; i < msg.evidences.length; ++i) {
                    data.push(msg.evidences[i]);
                }
            }
        }
    });
}


/*****************************************************************************
 *
 */
TriageViewStandalone.prototype.selectStandalone = function(name, timestamp) {
    // Text view should only have one entry
//    console.log("Selecting alone: " + name)
    this.textView.selectStandalone(name);


    // TODO For some reason webm clip does not advances manually
    // or programmatically in google-chrome 21.0.1180.79


    this.videoPlayer.seek(timestamp);
}


/**
 * Helper function to clear HTML content of text and timeline views
 *
 */
TriageViewStandalone.prototype.clearVisuals = function ()
    {
    $(this.div).find('.truageClipEvidences').html('');
    $(this.div).find('.truageTimeLineView').html('');
    }


