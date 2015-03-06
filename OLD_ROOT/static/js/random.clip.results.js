//
// LICENCE
// -------
// Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
// KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
// Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
//
// Container class
// Container used in zero-shot for carrying various results


RandomClipResults.prototype = Object.create(ClipResults.prototype);
RandomClipResults.prototype.constructor = ClipResults;


function RandomClipResults(container) {
    // TODO: Create more divs
    ClipResults.call(this, container);
    this.url = "/random_results"
}

//
RandomClipResults.prototype.AddClip = function(clip_id, score, kit_text, rank, duration, attribute_scores)
    {
    var clip  = new ClipView( { container: $(this.results),
                                score: score,
                                rank : rank,
                                duration: duration,
                                clipname : clip_id,
                                attribute_scores: attribute_scores});

    //clip.Update();
    this.clips.push(clip);
    var that = this;

    $(clip.div).bind("click", function(evt) {
        var clipview = $(this).data("clipviewobj");

        // Ask the triageview to show itself with the new data
        that.triageView.Refresh(clipview);
        // console.log("Refreshing")
        evt.stopPropagation();
        return(false);
    });
}

//
//RandomClipResults.prototype.Prev = function()
//    {
//    // Find if the triage
//    console.log(this.triageView.clipview.rank);
//    var elem = $(this.triageView.clipview.div).prev()
//    var clipview = $(elem).data("clipviewobj")
//    if(clipview === null)
//        {
//        console.log("No next");
//        return;
//        }
//
//    // Find next div
//    this.triageView.Refresh(clipview)
//    }
//
//RandomClipResults.prototype.Next = function()
//    {
//    // Find if the triage
//    console.log(this.triageView.clipview.rank);
//    var elem = $(this.triageView.clipview.div).next()
//    var clipview = $(elem).data("clipviewobj")
//    if(clipview === null)
//        {
//        console.log("Loading more");
//        this.LoadMore();
//        return;
//        }
//    // Find next div
//    this.triageView.Refresh(clipview)
//    }
//
//
//
//RandomClipResults.prototype.Submit = function(query)
//    {
//    // Accepts the url which will give the results
//    $(this.div).empty();
//
//    $("#wheel").html("<div><br/><center><img src = '/img/loading.gif' alt='Loading ...'/> </center></div>");
//
//    this.query = query;
//
//    console.log(query);
//
//    var request = $.ajax(
//        {
//        type: "GET",
//        context: this,
//        url: this.url,
//        data: {
//            "limit" : this.limit,
//            "skip" : this.skip,
//            "query" : JSON.stringify(query),
//            "some" : "other"
//            },
//        success : function (msg)
//            {
//            // Setup timer
//            if("error" in msg)
//                {
//                alert( "Submission not accepted" + JSON.stringify(msg));
//                }
//             else
//                {
//                //console.log("Submitted: " + this.eventkitvcd.clip.results.jsindex);
//                //console.log("Result: " + JSON.stringify(msg.clips))
//
//
//                for(var i = 0; i < msg.clips.length; i ++) {
//                    clipId = msg.clips[i].id
//                    score = msg.clips[i].score;
//                    rank = msg.clips[i].rank;
//                    duration = msg.clips[i].duration;
//
//                    var scores = { "scores" : msg.clips[i].attribute_scores,
//                               "labels" : msg.sql.split(/[ ,]+/g).slice(2,2+msg.clips[i].attribute_scores.length)};
//
//
//                    this.AddClip(clipId, score, " ", rank, duration, scores);
//                }
//                window.setTimeout(function() {
//                // this will execute 1 second later
//                $("#wheel").empty();
//                }, 1000);
//                }
//            }
//        });
//    }
//
//RandomClipResults.prototype.LoadMore = function()
//    {
//    this.skip = this.skip + this.limit;
//    this.Submit(this.eventkit_index, this.kit_text);
//    }
//
//
