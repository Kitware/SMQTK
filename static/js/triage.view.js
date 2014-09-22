//
// LICENCE
// -------
// Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
// KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
// Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
//
/**
 * TriageView class owns the modal dialog and all its elements
 * with other visual elements

 * @class
 */

/**
 * Constructor
 *
 */
function TriageView(clipview) {
    // Container is unused (global hardcoded dialog is used)
    this.videoPlayer = null;
    this.Init(clipview);
    var that = this;

    $('#clipEvidences').on('click', 'div label', function(event)
        {
        if (!that.textView || !that.timelineView) {
            return;
        }

        event.stopPropagation();

        // Select current element
        that.textView.selectElement(this);
        var labelObj = $(this);

        console.log('selecting......', labelObj.text());

        var timestamps = that.timelineView.select(labelObj.text());
        that.filmstripView.HideAll();
        for(var i=0; i < timestamps.length; i++)
            {
            that.filmstripView.SelectTimestamp(timestamps[i], "selectframe_text")
            }
        });

    // Now initialize the timeline
    $('#myModal').on('shown', function() {
        if (that.currentClipId === null || that.currentClipId !== that.clip_id) {
            return;
        }
        // Now show it
        var clipEvidencesObj  = $('#clipEvidences');
        var timelineViewObj = $('#timelineView');
        var filmstripViewObj = $('#filmstripView');

        // that.textView = new textView(clipEvidencesObj.get(0), that.data);
        // that.timelineView = new timelineView(timelineViewObj.get(0), that.data);
        // that.currentClipId = null;

        // Get the data
        //var algorithm = $("#algorithms-combo").find('option:selected').data('algo');
        that.updateResults(that.kit_id, that.clip_id);

        $('body').on('selectedStandAlone', function(event, name, timestamp, x, y) {
            // that.filmstripView.SelectTime(event, timestamp);
            //alert('' + x+ y);
            that.filmstripView.ClearHighlight();
            that.filmstripView.SelectTimestamp(timestamp, "selectframe_timeline");
            that.selectStandalone(name, timestamp);
            });
        });

    $('#myModal').on('hidden', function () {
        // Clear current rendering
        that.currentClipId = null;
        if (that.videoPlayer) {
            that.videoPlayer.pause();
            }
        });

    //$(this.div).click(function()

    $('#adjudication img').click(function(event) {
        if (that.currentClipId === null || that.currentClipId !== that.clipview.clip_id) {
            return;
        }

        event.stopPropagation();

        // TOOD Optimized this code
        switch($('#adjudication img').index(this)) {
            case 0:
                if ($(this).attr('src') === '/img/carbon-verified.png') {
                    $(this).attr('src', '/img/carbon-verified_on.png')
                }  else {
                    $(this).attr('src', '/img/carbon-verified.png')
                }
                break;
            case 1:
                if ($(this).attr('src') === '/img/carbon-rejected.png') {
                    $(this).attr('src', '/img/carbon-rejected_on.png')
                }  else {
                    $(this).attr('src', '/img/carbon-rejected.png')
                }
                break;
        }
    });

    $('#adjudication div').click(function(event) {
        if (that.currentClipId === null || that.currentClipId !== that.clipview.clip_id) {
            return;
        }
        event.stopPropagation();

        if ($(this).hasClass('btn')) {
            $('#adjudication div').find('.myLabel').show();
            console.log($('#adjudication div').find('input').text());
            $('#adjudication div').find('label').text($('#adjudication div').find('input').val());
            $('#adjudication div').find('input').hide();
        }
        if ($(this).hasClass('myLabel')) {
            $('#adjudication div').find('input').val($(this).text());
            $(this).hide();
            $('#adjudication div').find('input').show();
        }
        });

    }

TriageView.prototype.Init = function(clipview)
    {
    var that = this;
    // Register functions
    $('.modal-body').on('click', function(event)
        {
        if (!that.textView || !that.timelineView)
            {
            return;
            }

        $('#clipEvidences div label').removeClass('selected');
        that.timelineView.select(null);
        that.textView.selectStandalone(null);
        that.timelineView.selectStandalone(null);
        that.filmstripView.ClearHighlight();

        console.log('that.currentClipId', that.currentClipId);
        console.log('that.clipview.clip_id', that.clipview.clip_id);

        if (that.currentClipId !== null && that.currentClipId === that.clipview.clip_id) {
            console.log('Keeping video player');
            that.videoPlayer.play();
        } else {
            console.log('Setting video player to null');
            that.videoPlayer = null;
        }

        });

    // Call refresh with proper paramters
    this.Refresh(clipview)
    }

TriageView.prototype.Refresh = function(clipview)
    {
    if(clipview === undefined)
        {
        // Do not refresh
        return;
        }
    this.clearVisuals();

    this.clipview = clipview;
    var that = this;

    console.log("Updating to " + that.clipview.rank);

    $("#myModalRank").html("Rank: " + that.clipview.rank)

    //Create filmstripView

    $('#filmstripView').empty();
    this.filmstripView = new FilmStripView($('#filmstripView'), this.clipview.clip_id, this.clipview.score)
    this.filmstripView.Update(this.clipview.clip_id);

    // Set information in mymodal
    // $('#myModalLabel').html("Event: " + this.clipview.kit_text  + ", Clip: " + this.clipview.clip_id)
    // TODO Use duration from query

    this.currentClipId = this.clipview.clip_id;

    console.log('videoPlayer', this.videoPlayer);
    if (this.videoPlayer) {
        this.videoPlayer.pause();
        var source = $("#myModalVid").find('source');
        $(source).attr('src', '');
        console.log('source', source);
    }

    var time = new Date();

    // Use this line if using mp4
    $("#myModalVid").html(
        '<video height="250px" controls autoplay loop>' +
//          '<source src="' + video_url_prefix + this.clipview.clip_id + '.q4.ogv?t='     + time.getTime() + '" type="video/ogg" preload="auto"></source>' +
            '<source src="' + video_url_prefix + this.clipview.clip_id + '.q4.ogv" type="video/ogg" preload="auto"> </source>' +
        '</video>');

    // Load video
    this.videoPlayer = new videoPlayer($('#myModalVid video').get(0))

    // Resets
    $('#adjudication div').find('.myLabel').hide();
    $('#adjudication div').find('label').text();
    $('#adjudication div').find('input').show();
    $('#adjudication div').find('input').val('');
    $('.yes').attr('src', '/img/carbon-verified.png');
    $('.no').attr('src', '/img/carbon-rejected.png');

    $('#myModal').modal("show");

    // Now show it
    var clipEvidencesObj  = $('#clipEvidences');
    var timelineViewObj = $('#timelineView');
    var filmstripViewObj = $('#filmstripView');

    // that.textView = new textView(clipEvidencesObj.get(0), that.data);
    // that.timelineView = new timelineView(timelineViewObj.get(0), that.data);
    // that.currentClipId = null;

    // Get the data
    //var algorithm = $("#algorithms-combo").find('option:selected').data('algo');

    this.updateResults(this.clipview.kit_id, this.clipview.clip_id);

    $('body').on('selectedStandAlone', function(event, name, timestamp, x, y) {
        if (!that.videoPlayer) {
            return console.log('Non existent video player');
        }

        if (!that.videoPlayer.isValid(timestamp)) {
            console.log('Invalid video seek time');
            return;
        }
        that.filmstripView.ClearHighlight();
        that.filmstripView.SelectTimestamp(timestamp, "selectframe_timeline");
        that.selectStandalone(name, timestamp);
        });
    }


/**
 * Update results
 *
 */
TriageView.prototype.updateResults = function(kitId, clipId)
    {

    if (this.currentClipId === null || this.currentClipId !== this.clipview.clip_id) {
        return;
    }

    this.data = [];

    this.updateAlgorithmResults(kitId, clipId, this.data, 'OB_max_avg_positive_hik_21');
    this.updateAlgorithmResults(kitId, clipId, this.data, 'sun_attribute_avg_positive_hik_27');

    // Now update the label. We delayed the label initialization as we didn't
    // have duration value when the clip was initialized.
    $('#myModalLabel').html("Clip " + this.clipview.clip_id + " ("
        + this.clipview.kit_text + ", "  + this.clipview.duration + " seconds)");

    var clipEvidencesObj  = $('#clipEvidences');
    var timelineViewObj = $('#timelineView');

    this.textView = new textView(clipEvidencesObj.get(0), this.data);
    this.timelineView = new timelineView(timelineViewObj.get(0), this.data, this.clipview.duration);
    }

/**
 * Update algorithm specific results
 *
 */
TriageView.prototype.updateAlgorithmResults = function(kitId, clipId, data, algorithm) {
    var that = this;

    if (that.currentClipId === null || that.currentClipId !== that.clipview.clip_id) {
        return;
    }

    console.log('Updating results for', algorithm);
    var request = $.ajax({
        type: "GET",
        context: this,
        async: false,
        url: "/triage_info?kit=" + that.clipview.kit_id + "&clip=" + that.clipview.clip_id + "&algo=" + algorithm + "&strict=strict_v1",
        success: function (msg) {
            // Setup timer
            if("error" in msg) {
                console.log( "Submission not accepted" + JSON.stringify(msg));
            } else {
                // Update clip duration. Currenly this is the only time we get
                // clip duration value from database
                if (!that.clipview.duration) {
                    console.log('duration', msg.duration);
                    that.clipview.duration = msg.duration;
                } else if (that.clipview.duration !== msg.duration) {
                    console.log('Mismatched clip duration from different algorithms');
                }

                // TODO As per customer request hard-coding to
                // 5 evidences for now
                // for(i = 0; i < msg.evidences.length; ++i) {
                var noOfEvidences = 5;
                for(i = 0; i < noOfEvidences; ++i) {
                    data.push(msg.evidences[i]);
                }
            }
        }
    });
}


/*****************************************************************************
 *
 */
TriageView.prototype.selectStandalone = function(name, timestamp) {
    // Text view should only have one entry
    console.log("Selecting alone: " + name)
    this.textView.selectStandalone(name);


    // TODO For some reason webm clip does not advances manually
    // or programmatically in google-chrome 21.0.1180.79
    this.videoPlayer.seek(timestamp);
}


/**
 * Helper function to clear HTML content of text and timeline views
 *
 */
TriageView.prototype.clearVisuals = function () {
    var clipEvidencesObj  = $('#clipEvidences');
    clipEvidencesObj.html('');

    var timelineViewObj = $('#timelineView');
    timelineViewObj.html('');
}


