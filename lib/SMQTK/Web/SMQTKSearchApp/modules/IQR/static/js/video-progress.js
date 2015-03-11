


var VideoProgress = function(config) {
    this.config = {};
    this.videos = {};
    this.labels = ["File Upload",  "Process Preparation", "Frame Extraction",
                   "Sprite Creation", "Feature computation",
                   "Feature Aggregation"];
    $.extend(this.config, {
        "container" : null
    }, config );
    this.videos = [];
    this.videolist = [];
    this.timer = setInterval((function(self) {         //Self-executing func which takes 'this' as self
         return function() {   //Return a function in the context of 'self'
             self.Update(); //Thing you wanted to run as non-window 'this'
         }
     })(this), 1000);
};

VideoProgress.prototype.Init = function(){

};

VideoProgress.prototype.AddVideo = function(id) {
    // Create a div and insert it with all states greyed out
    this.videolist.push(id);
    console.log("Video Added: "+ id)
    var newvid = {}
    newvid.id = id
    newvid.div = document.createElement('div');
    $(newvid.div).html("id:" + id);
    $(newvid.div).append(this.CreateMeter());
    $(newvid.div).append('<br/>');
    $(newvid.div).appendTo($(this.config.container));
    var that = this;
    // Now send this to processing
    var res = $.ajax("/oneshot/process?id=" + newvid.id);
    res.success(function(data) {
        newvid.taskid = data.task;
        that.videos.push(newvid);
    });
};


VideoProgress.prototype.CreateMeter = function() {
    // Create a div and insert it with all states greyed out
    var ul=document.createElement('ul');
    $(ul).css("overflow","hidden");
    $(ul).css("list-style-type","none");
    $(ul).css("text-align","center");

    for(var i=0; i < this.labels.length; i ++) {
        li = document.createElement('li');
        $(li).addClass("progress-step");
        $(li).html(this.labels[i]);
        $(li).appendTo(ul);
    }
    return ul;
};


VideoProgress.prototype.Update = function() {
    // Loops through all the videos that are added
    for(var j =0; j < this.videos.length; j ++){
            // Polls for the status of each
        if(this.videos[j].done !== true) {
            // Ajax only if needed
            var that = this;
            var k = $.ajax("/task?id=" + that.videos[j].taskid + "&localid=" + j).done(function(data) {
                console.log(data);
                // Parse the data
                if(data.state === "SUCCESS") {
                    that.videos[data.localid].done = true;
                }
                console.log("State " + data.meta.state);
                var rank = parseInt(data.meta.state)
                $(that.videos[data.localid].div).find('li').slice(0, rank).addClass("video-done").removeClass("video-current");
                $(that.videos[data.localid].div).find('li').slice(rank,rank+1).addClass("video-current").removeClass("video-done");

                if (rank === 4) {
                    var percent = (data.current / data.total * 100.0);
                    $(that.videos[data.localid].div).find('li').slice(rank,rank+1).html("Feature computation " +
                      percent.toFixed(1) + "%");
                }

                if (rank > 4) {
                    $(that.videos[data.localid].div).find('li').slice(4,4+1).html("Features ready");
                }
            });
        }

            // Update the status if required
    }

    // var alldone = this.videos.length > 0;
    var alldone = true;
    for(var j =0; j < this.videos.length; j ++){
        if(this.videos[j].done !== true) {
            alldone = false;
        }
    }
    if (alldone === true) {
        // console.log("All done !")
        $("#beginiqr").removeAttr("disabled");
    }
    else {
        $("#beginiqr").attr("disabled","disabled");
        console.log("Still working ..")
    }
};

