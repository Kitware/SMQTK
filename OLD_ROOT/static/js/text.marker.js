//
// LICENCE
// -------
// Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
// KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
// Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
//
// p/optionaris
// 
// 
// 

var TextMarker = function()
    {
    this.name = $('#event-name');
    this.text = $('#event-text');
    rangy.init();
    this.css = rangy.createCssClassApplier("highlight", {normalize: true});
    this.ranges = {};
    this.Load();
    }

TextMarker.prototype.Load = function(kit) 
    {
    // Loads the possible names
    that = this;
    var request = $.ajax(
        {
        type: "GET",
        url: "/events",
        timeout:25000,
        error: function(x, t, m) 
            {
            if(t==="timeout") 
                {
                alert("got timeout");
                }
            else 
                {
                alert(t);
                }
            },
        success: function (msg)
            {
            
            var liststr;
            for(var i =0;  i < msg.list.length; i ++)
                {
                liststr = liststr + '<Option data-kit="' + msg.list[i].kit + '"> ' + msg.list[i].name + "</option>\n";
                }
            // Get the list of event names, and mae the title a selector
            //// TODO: Update variables 
            //// that.Init(msg)
            //$("#event-text").html("<p class='well'>" + msg.text + "</p>");
            $("#event-name").empty();
            this.sel = document.createElement("select");
            $(this.sel).html(liststr);
            $(this.sel).appendTo($("#event-name"));
            $(this.sel).selectpicker();
            $(this.sel).change(function ()
                {
                var selected = $(this).find('option:selected');
                var extra = selected.data('kit');
                that.Update(extra);
                console.log("Selected : " + extra);
                });
            that.Update("E001");
            //$("#event-text").mouseup( function (evt)
            //    {
            //    that.css.toggleSelection();
            //    var sel = rangy.getSelection();
            //    var seltext = sel.toString().trim();
            //    if(seltext.length > 0)
            //        {
            //        // console.log("Selected: " + sel);
            //        that.ranges[seltext] = sel.getRangeAt(0);
            //        $("#event-tags").tagit("createTag",  seltext);
            //        }
            //    // Check the selection here
            //    } );
                        
            }
        });
    }

TextMarker.prototype.Update = function(kit) 
    {
    that = this;
    var request = $.ajax(
        {
        type: "GET",
        url: "/events?kit=" +kit,
        timeout:25000,
        error: function(x, t, m) 
            {
            if(t==="timeout") 
                {
                alert("got timeout");
                }
            else 
                {
                alert(t);
                }
            },
        success: function (msg)
            {
            // Get imagelist and send to AddImag3s
            // TODO: Update variables 
            // that.Init(msg)
            $("#event-text").html("<p class='well'>" + msg.text + "</p>");
            //$("#event-name").html("<h2>" + msg.name + "</h2>");
            $("#event-text").mouseup( function (evt)
                {
                that.css.toggleSelection();
                var sel = rangy.getSelection();
                var seltext = sel.toString().trim();
                if(seltext.length > 0)
                    {
                    // console.log("Selected: " + sel);
                    that.ranges[seltext] = sel.getRangeAt(0);
                    $("#event-tags").tagit("createTag",  seltext);
                    }
                // Check the selection here
                } );
            }
        });
    }
    
TextMarker.prototype.Init = function() 
    {
    $("#event-tags").tagit(
        {
        // Options
        allowSpaces: true,
        allowDuplicates: false,
        beforeTagRemoved: function(event, ui) 
            {
            // do something special
            // console.log("Removing: " + ui.tagLabel);
            // Need to remove that.ranges[ui.tagLabel];
            var range = that.ranges[ui.tagLabel];
            that.css.undoToRange(range);
            // console.log(range);
            // console.log(that.ranges)
            console.log(that.ranges[ui.tagLabel])

            // for(var key in that.ranges) 
            // {
            //     if(key == ui.tagLabel)
            //          {
            //          console.log("Found " + key);
            //          }
            //    }
            },
        // readOnly:true,
        });
    }   