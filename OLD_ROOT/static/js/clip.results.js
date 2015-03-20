//
// LICENCE
// -------
// Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
// KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
// Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
//
// Container class
// Base class for more query results classes.
// TODO: make it more configurable than Container used in zero-shot for carrying various results

function ClipResults(container) {
    // TODO: Create more divs
    // Get the div
    this.url = "/event_results"
    this.div = container
    $(this.div).empty();
    // Initializee to defaults

    // Get the div
    this.div = container
    $(this.div).empty();



    this.results = document.createElement("div");
    $(this.results).appendTo($(this.div));
    $(this.div).addClass("row-fluid");

    this.wheel = document.createElement("div");
    $(this.wheel).appendTo($(this.div));

    this.loadmore = document.createElement("div");
    $(this.loadmore).html("</center><button class='btn'> Load More </button> </center>");
    $(this.loadmore).appendTo($(this.div));
    $(this.loadmore).hide();

    var that = this;

    $(this.loadmore).bind('click', function(event) {
        that.LoadMore();
    });

    this.Init();
}

ClipResults.prototype.Init = function() {
    // Find out the size of grid that will fit in
    this.width = $(this.div).innerWidth();
    this.columns = parseInt((this.width -18) / 212);
    this.rows = 4;  // parseInt(this.height / 232);
    this.clips = [];
    this.skip = 0;
    this.limit = this.rows * this.columns;
    this.timerw = 0;
    this.taskid = 'some';
    //this.triageView = new TriageView()
}

ClipResults.prototype.ShowBusy = function(clear, message ) {

    if (message === undefined) {
        message = "";
    }
    if(clear == false){
        $(this.wheel).empty();
        $(this.loadmore).show();
    }else {
        $(this.wheel).html("<div><br/><center>" + message + "<br/> <img src = '/img/loading.gif' alt='Loading ...'/> </center></div>");
        $(this.loadmore).hide();
    }
}

ClipResults.prototype.AddClip = function(clip_id, score, kit_text, rank, duration) {
    var clip  = new TriageClipView($(this.results), this.eventkit_index, clip_id, score, kit_text, duration);
    //clip.Update();
    this.clips.push(clip);
    var that = this;

    $(clip.clipdiv).bind("click", function(evt) {

        window.open('triage_info_view?kit=' + that.eventkit_index + '&rank=' + rank,'_blank');

        //        var clipview = $(this).data("clipviewobj");
        //        // Ask the triageview to show itself with the new data
        //        that.triageView.Refresh(clipview);
        //        // console.log("Refreshing")

        evt.stopPropagation();
        return(false);
    });
}


ClipResults.prototype.Submit = function(eventkit_index, kit_text)
    {
    this.ShowBusy(true);
    this.eventkit_index = eventkit_index;
    this.kit_text= kit_text
    // Updates and initiates the query by reading the values
    var that = this;
    // TODO Remove this
    var values = [{'attribute1':10}, {'attribute2':20}, {'attribute3':30}];
    var clipId = null;
    var score = null;
    var request = $.ajax(
        {
        type: "GET",
        context: this,
        url: this.url + "?kit=" + this.eventkit_index + "&limit=" + this.limit + "&skip=" + this.skip,
        success: function (msg)
            {
            // Setup timer
            if("error" in msg)
                {
                this.ShowBusy(false);
                alert( "Submission not accepted" + JSON.stringify(msg));
                }
             else
                {
                //console.log("Submitted: " + this.eventkit_index);
                //console.log("Result: " + JSON.stringify(msg.clips))

                for(var i = 0; i < msg.clips.length; i ++) {
                    clipId = msg.clips[i].id
                    score = msg.clips[i].score;
                    rank = msg.clips[i].rank;
                    duration = msg.clips[i].duration;

                    this.AddClip(clipId, score, kit_text, rank, duration);
                }
                that.ShowBusy(false);
                }
            }
        });
    }

ClipResults.prototype.LoadMore = function()
    {
    this.skip = this.skip + this.limit;
    this.Submit(this.eventkit_index, this.kit_text);
    }


ClipResults.prototype.Poll = function()
    {
//    console.log("Polling")
    var request = $.ajax(
        {
        type: "GET",
        context: this,
        url: "/search?cmd=progress&what=" + this.taskid,
        success: function (msg)
            {
            // Setup timer
//            console.log(msg);
            if("error" in msg)
                {
                alert( "Progress not provided" + JSON.stringify(msg));
                }
            else
                {
                var that = this;
//                console.log("Success: " + JSON.stringify(msg));
                if(msg.progress < 1)
                    {

                     $("#progress").html("<br/><center> <h4> Processing .. " + parseFloat(msg.progress * 100).toFixed(1) + " % complete </h4></center>");
                    setTimeout(function() {that.Poll()} , 1000);
                    }
                else
                    {
                    that.GetResults()
//                    console.log("Done");
                    }
                }
            }
        });
    }

ClipResults.prototype.InitGrid = function() {
    this.width = $(this.div).innerWidth();
    this.columns = parseInt((this.width -18) / (212 + 14) );
    this.rows = 4;  // parseInt(this.height / 232);
    this.limit = this.rows * this.columns;
};

ClipResults.prototype.SubmitSuccess = function(result) {

};


ClipResults.prototype.GetResults = function()
    {
    $("#progress").html("<br/><center> <h4> Fetching results .. </h4> </center>");
    $("#wheel").empty();

    var request = $.ajax(
        {
        type: "GET",
        context: this,
        url: "/search?cmd=results&what=" + this.taskid,
        success: function (data)
            {
            $("#log").empty();
            $("#progress").empty();


            for(var i = 0; i < data.results.length; i ++)
                {
                this.AddClip(data.results[i][0], data.results[i][1]);
                }
                        //var someclips =  [ "290019", "326090","542672","683545","739212","758699","773557","804646", "806969","881622" ]

            //for(var i = 0; i < someclips.length; i ++)
              //  {
            //	this.AddClip(someclips[i], data.results[i][1]);
              //	}
            }
        });
    }
//ClipResults.prototype.Navigate = function(where)
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

//ClipResults.prototype.Clear() = function(imagelist)
//    {
//    // Clear existing images
//    $('#images').empty();
//    }


ClipResults.prototype.ResetAdjudication = function() {
    // Reload all the adjudications for all loaded clips
    var a = 0;

    // Clear all the appearance
    for(var i = 0; i < this.clips.length; i++ ){
        this.clips[i].adjudication.Clear()
    }
};
