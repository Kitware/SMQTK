//
// LICENCE
// -------
// Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
// KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
// Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
//
// Container class
// Container used in zero-shot for carrying various results


VCDClipResults.prototype = Object.create(ClipResults.prototype);
VCDClipResults.prototype.constructor = ClipResults;


function VCDClipResults(container)
    {
    ClipResults.call(this, container);

    // TODO: Create more divs
    this.url = "/vcdi/search"
    this.dataset = "MEDTEST"
    }


VCDClipResults.prototype.AddClip = function(clip_id, score, kit_text, rank, duration, attribute_scores)
    {
    var clip  = new ClipView( { container: $(this.results),
                                clipname : clip_id,
                                score: score,
                                rank : rank,
                                duration: duration,
                                clip : clip_id,
                                attribute_scores: attribute_scores,
                                link_prefix: "/vcdi/info_view?clip=" + clip_id + "&rank=" + rank +"&ignore="
                            });

    //clip.Update();
    this.clips.push(clip);
    var that = this;
}

VCDClipResults.prototype.Prev = function()
    {
    // Find if the triage
    console.log(this.triageView.clipview.rank);
    var elem = $(this.triageView.clipview.div).prev();
    var clipview = $(elem).data("clipviewobj");
    if(clipview === null)
        {
        console.log("No next");
        return;
        }

    // Find next div
    this.triageView.Refresh(clipview)
    }

VCDClipResults.prototype.Next = function()
    {
    // Find if the triage
    console.log(this.triageView.clipview.rank);
    var elem = $(this.triageView.clipview.div).next();
    var clipview = $(elem).data("clipviewobj");
    if(clipview === null)
        {
        console.log("Loading more");
        this.LoadMore();
        return;
        }
    // Find next div
    this.triageView.Refresh(clipview)
    }



VCDClipResults.prototype.Submit = function(query)
    {
    // Accepts the url which will give the results
    this.ShowBusy(true, "Fetching results.. ");
    this.InitGrid();

    if(query !== undefined){
        $(this.results).empty();
        this.query = JSON.stringify(query);
        if(this.url === "/vcdi/search") {
            this.queryobj = query;
        }
        this.skip = 0;
    }

    console.log(query);

    var that2 = this;

    var request = $.ajax(
        {
        type: "GET",
        context: this,
        url: this.url,
        data: {
            "limit" : this.limit,
            "skip" : this.skip,
            "query" : this.query,
            "dataset" : this.dataset
            },
        error: function(msg) {
            this.ShowBusy(false);
        },
        success : function (msg)
            {
            this.ShowBusy(false);
            // Setup timer
            if("error" in msg)
                {
                alert( "Submission not accepted, error message: " + JSON.stringify(msg["error"]));
                that2.SubmitSuccess(false);
                }
            else
                {

                that2.SubmitSuccess(true);
                //console.log("Submitted: " + this.eventkitvcd.clip.results.jsindex);
                //console.log("Result: " + JSON.stringify(msg.clips))

                for(var i = 0; i < msg.clips.length; i ++) {
                    var clipId = msg.clips[i].id
                    var score = msg.clips[i].score;
                    var rank = msg.clips[i].rank;
                    var duration = msg.clips[i].duration;
                    var scores;

                    if(msg.clips[i].hasOwnProperty("attribute_scores")) {
                        scores = { "scores" : msg.clips[i].attribute_scores,
                               "labels" : msg.sql.split(/[ ,]+/g).slice(2,2+msg.clips[i].attribute_scores.length)};
                    }
                   else {
                        scores = { "scores" : [score],
                               "labels" : [clipId]
                        };
                    }

                    this.AddClip(clipId, score, " ", rank, duration, scores);
                }
                var that = this;
                window.setTimeout(function() {
                // this will execute 1 second later
                    $(that.wheel).empty();
                    }, 500);
                }
            }
        });
    }

VCDClipResults.prototype.LoadMore = function()
    {
    this.skip = this.skip + this.limit;
    this.Submit();
    }


