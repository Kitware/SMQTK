//
// LICENCE
// -------
// Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
// KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
// Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
//
// Container
// Container class specific to event triage search list

TriageClipResults.prototype = Object.create(ClipResults.prototype);
TriageClipResults.prototype.constructor = ClipResults;


function TriageClipResults(container)
    {
    // TODO: Create more divs
    // Get the div
    ClipResults.call(this,container);
    this.url = "/event_results_from_training"
    }

TriageClipResults.prototype.Init = function() {
    // Find out the size of grid that will fit in
    this.width = $(this.div).innerWidth();
    this.columns = parseInt((this.width -18) / 212);
    this.rows = 4;  // parseInt(this.height / 232);
    this.clips = [];
    this.skip = 0;
    this.limit = this.rows * this.columns;
    this.timerw = 0;
    this.taskid = 'some';
    this.triageView = new TriageView()
};

TriageClipResults.prototype.AddClip = function(clip_id, score, kit_text, rank, duration) {
    var clip  = new TriageClipView($(this.results), this.eventkit_index, clip_id, score, kit_text, duration)
    //clip.Update();
    this.clips.push(clip);
    var that = this;

    $(clip.clipdiv).bind("click", function(evt) {

        window.open('triage_info_view_from_training?kit=' + that.eventkit_index + '&rank=' + rank + "&training=" + that.training ,'_blank');

        //        var clipview = $(this).data("clipviewobj");
        //        // Ask the triageview to show itself with the new data
        //        that.triageView.Refresh(clipview);
        //        // console.log("Refreshing")

        evt.stopPropagation();
        return(false);
    });
};

TriageClipResults.prototype.PrevTriage = function() {
    // Find if the triage
//    console.log(this.triageView.clipview.rank);
    var elem = $(this.triageView.clipview.div).prev();
    var clipview = $(elem).data("clipviewobj");
    if(clipview === null)
        {
        console.log("No next");
        return;
        }

    // Find next div
    this.triageView.Refresh(clipview)
};

TriageClipResults.prototype.NextTriage = function() {
    // Find if the triage
//    console.log(this.triageView.clipview.rank);
    this.url= "/event_results";
    var elem = $(this.triageView.clipview.div).next();
    var clipview = $(elem).data("clipviewobj");
    if(clipview === null)
        {
//        console.log("Loading more");
        this.LoadMore();
        return;
        }
    // Find next div
    this.triageView.Refresh(clipview)
};



TriageClipResults.prototype.Submit = function(eventkit_index, kit_text, training_dataset)
    {
    this.ShowBusy(true);
    this.InitGrid();
    this.eventkit_index = eventkit_index;
    this.kit_text= kit_text;
    this.training = training_dataset;
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
        url: this.url + "?kit=" + this.eventkit_index + "&limit=" + this.limit + "&skip=" + this.skip + "&training=" + training_dataset,
        success: function (msg)
            {
            // Setup timer
            this.ShowBusy(false);
            if("error" in msg)
                {
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
                this.ShowBusy(false);
                }
            }
        });
    }

TriageClipResults.prototype.LoadMore = function()
    {
    this.skip = this.skip + this.limit;
    this.Submit(this.eventkit_index, this.kit_text, this.training);
    }


TriageClipResults.prototype.Poll = function()
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

TriageClipResults.prototype.GetResults = function()
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
