//
// LICENCE
// -------
// Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
// KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
// Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
//
// Displays clip

function FilmStripView(container, clip_id, score)
    {
    // Initializee to defaults
    this.Init(container, clip_id, score);
    }

FilmStripView.prototype.Init = function(container, clip_id, score)
    {
    // Create a div
    this.clip_id = clip_id;
    this.score = score;
    this.duration = null;
    this.container = container;
    this.ul = document.createElement('ul');
    this.div =document.createElement('div');
    this.p =document.createElement('p');
    $(this.ul).addClass("imageRow");
    $(this.div).addClass("filmstrip");

    $(this.p).html("</br> <h4> Score: "+ score.toFixed(2) + "</h4>");
    // When the results come back load them here
    $(this.p).appendTo($(this.container))
    $(this.div).appendTo($(this.container));
    $(this.ul).appendTo($(this.div))

    // Navigation
    this.skip = 0;
    this.limit = 20;
    // this.Update();
   }

FilmStripView.prototype.Close = function()
    {
    alert("Closing..");
    }

FilmStripView.prototype.Update = function(clip_id)
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
    }

FilmStripView.prototype.ClearHighlight = function()
    {
    // Remove all highlights
    $(this.ul).find('li').removeClass();
    $(this.div).scrollLeft(0);
    }

FilmStripView.prototype.HideAll = function(timestamp, classname)
    {
    // Hide all frames
    var elem = $(this.ul).find('li').addClass("hiddenframe")
    $(this.div).scrollLeft(0);
    }

FilmStripView.prototype.SelectTimestamp = function(timestamp, classname)
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
    }

FilmStripView.prototype.OnData = function(msg)
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
    }

