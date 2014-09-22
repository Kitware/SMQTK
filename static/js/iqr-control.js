//
// LICENCE
// -------
// Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
// KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
// Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
//
// Displays aggregated IQR control information
// Used internally only

function IQRState(config) {
    this.Init(config);
}

IQRState.prototype.Init = function(config) {
    // Initializee to defaults

    this.config = {};

    $.extend(this.config, {
        "positive" : [],
        "negative" : [],
        "idx" : 0
    }, config );

    // Have different configuration strings for different configurations

    // Create and add div to the container
    this.div = document.createElement('div');

    // Create basic structure
    $(this.div).html('<div class="pos" height="24px"/> ' +
                     '<img class="neg" height="24px"/> ' +
                     '<button class="reset" height="24px"/>'
                     );

    $(this.div).addClass("clipAdjudication");
    $(this.div).css("opacity","" + this.config.opacity);


    if(this.config.display === false) {
        $(this.div).css("display","none");
    }

    $(this.div).appendTo(this.config.container);

    var that = this;

    if(that.config.display === false) {
        // Bind events
        $(this.config.container).mouseenter(function (event){
//          console.log("inside " + that.config.clip);
            $(that.div).css("display","inline-block");
        }).mouseleave(function (event){
//                console.log("leaving " + that.config.clip);
            $(that.div).css("display","none");
        });
    }

//    $(this.div).click(function (event) {
//        event.preventDefault();
//        event.stopPropagation();
//        console.log("clicked" + that.config.clip);
//    });
    //console.log(" Found " + $(this.div).find("img").length);

    $(this.div).find("img").click(function(event) {
        //console.log("status " + $(this).attr("class"));
        event.preventDefault();
        event.stopPropagation();
        // TODO: Send the ajax to api
        adj = that.GuessOperation(event.target);

        // Guess Adjudication

        var reqdata = {
            clip : that.config.clip,
            label : adj.label,
            query : that.config.query,
            op :adj.op
        };

        that.ApplyAdjudication(reqdata);
        // Change the local storage
        localStorage.setItem("adjudication", JSON.stringify(reqdata));
        // Make sure that this clip information is updated
        localStorage.setItem(reqdata.clip, JSON.stringify(reqdata));

        // Send out the ajax request to register this change at server-side
        $.getJSON("/adjudicate",reqdata, function(data){
            //console.log(data);
        });
    });

    // Bind the storage changed event
    $(window).on('storage', function (e) {
        // Listen to only adjudication storage
        if(e.originalEvent.key === "adjudication") {
            var reqdata = JSON.parse(e.originalEvent.newValue);

            if(reqdata.clip === that.config.clip) {
                // The change matters, adjust to the new state
                console.log("Value changed from another tab");
                that.ApplyAdjudication(reqdata);
            }
        }
    });

    console.log("Looking for " + this.config.clip);
//
//    for (var i = 0; i < localStorage.length; i++){
//        console.log(localStorage.key(i));
//        console.log(localStorage.getItem(localStorage.key(i)));
//    }

    // Check the adjudication state
    var statestr = localStorage.getItem(this.config.clip); //returns "Some Value"
    if(statestr !== null) {
        var reqdata = JSON.parse(statestr);
        if(reqdata.clip === that.config.clip) {
            // The change matters, adjust to the new state
            console.log("Value appears to be set already");
            that.ApplyAdjudication(reqdata);
        }
    }
};

IQRState.prototype.Update = function(what, where) {
    // Updates the state arrays


}
