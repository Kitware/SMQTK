//
// LICENCE
// -------
// Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
// KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
// Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
//
// Container class
// Container used in zero-shot for carrying various results


IQRClipResults.prototype = Object.create(VCDClipResults.prototype);
IQRClipResults.prototype.constructor = VCDClipResults;


function IQRClipResults(container) {
    VCDClipResults.call(this, container);

    this.dataset = "EVENTKITS"
    // TODO: Create more divs

    this.statediv = document.createElement("div");
    $(this.statediv).prependTo($(this.div));
    $(this.statediv).addClass("row-fluid");

//    Create here good looking iqr state representation
//    this.iqrdiv = document.createElement("div");
//    $(this.iqrdiv).prependTo($(this.div));
//    $(this.iqrdiv).addClass("row-fluid");
//    $(this.iqrdiv).html('<button class="button refine" disabled="disabled"> Refine </button> <button class="button results" disabled="disabled"> Refresh Results </button> <br/>');
    this.statusdiv = document.createElement("div");
    $(this.statusdiv).prependTo($(this.div));
    $(this.statusdiv).addClass("row-fluid");

    this.wheel = document.createElement("div");
    $(this.wheel).prependTo($(this.div));
    $(this.wheel).addClass("row-fluid");

    this.buttondiv = document.createElement("div");
    $(this.buttondiv).html('<button type="button" class="btn refine" disabled="disabled" title="Please select at least one positive sample to enable refinement"> Refine </button> <button type="button" class="btn reset"> Reset </button> <button type="button" class="btn archive" disabled="disabled"> Search Large Archive  </a> <br/>');
    $(this.buttondiv).prependTo($(this.div));

    var that = this;

    $(this.buttondiv).find(".refine").click(function () {
        that.Refine();
    });

    $(this.buttondiv).find(".reset").click(function () {
        localStorage.clear();
        console.log("cleared adjudications");
        that.uid = undefined;
        that.yes = [];
        that.no = [];
        that.star = [];
        that.current_refinement = 0;
        that.refinement_states = [];

        that.Submit(that.queryobj);
        //that.ResetAdjudication();
        // TODO: reset the search session
    });


    $(this.buttondiv).find(".archive").click(function () {
        if(that.refinement_states.length < 1) {
            alert("Refinement model not yet ready to search archive with, please use refine button first");
            return;
        }
        var col = that.refinement_states.slice(-1)[0].collection;
        var uid = that.uid;
        window.open("/iqr/archive_search_view?state=" + col + "&session=" + uid,'_blank');
    });

    this.yes = [];
    this.no = [];
    this.star = [];
    this.current_refinement = 0;
    this.refinement_states = [];
    $(this.div).bind("click", function (event) {
        console.log("Div clicked .." + event)
    });

    // Bind the storage changed event
    $(document).on('my_storage', function (e) {
        //
        if(e.message.op == "add") {
            var idx = that[e.message.label].indexOf(e.message.clip);
            if(idx === -1) {
                // Does not exist
                that[e.message.label].push(e.message.clip);
            }

            // Remove the fellow from the other label
            if(e.message.label === "yes") {
                // Remove from no
                var idx = that.no.indexOf(e.message.clip);
                if(idx !== -1) {
                    // Does exist
                    that.no.splice(idx,1);
                }

            }

            if(e.message.label === "no") {
                var idx = that.yes.indexOf(e.message.clip);
                if(idx !== -1) {
                    // Does exist
                    that.yes.splice(idx,1);
                }
            }

        }
        if(e.message.op == "remove") {
            var idx = that[e.message.label].indexOf(e.message.clip);
            if(idx !== -1) {
                // Does exist
                that[e.message.label].splice(idx,1);
            }
        }
        that.RefreshState();
    });
}

IQRClipResults.prototype.Refine = function() {
    // Make sure that number of positives > 1
    if( this.yes.length === 0) {
        alert("Please add atleast one new positive before next refinement");
        return;
    }
    // Disable the button
    $(this.buttondiv).find(".refine").attr("disabled","disabled");
    //$(this.buttondiv).find(".refine").removeClass("btn-primary");

    this.ShowBusy(true, "Refining ");

    var that = this;
    var request = $.ajax({
      url: "/iqr/refine_search",
      type: "GET",
      data: {uid : that.uid,
             positive : JSON.stringify(that.yes),
             negative : JSON.stringify(that.no)
        },
      dataType: "json"
    });

    request.done(function( msg ) {
        that.RefreshState();
        // Create a new refinement state
        console.log(msg);
        var thisstate = {};
        thisstate.yes = that.yes.slice(0);
        thisstate.no = that.no.slice(0);
        thisstate.prevyes = thisstate.yes.length;
//        that.yes = [];
//        that.no = [];
        thisstate.collection = msg.collection;
        thisstate.idx = that.current_refinement;
        that.refinement_states.push(thisstate);

        that.current_refinement = that.current_refinement + 1;
        // Setup the polling
        that.Poll();

        that.RefreshState();
    });

    request.fail(function( jqXHR, textStatus ) {
      alert( "Request failed: " + textStatus );
      this.ShowBusy(false);
    });

    $(this.div).bind("click", function (event) {
//        console.log("Div clicked .." + event)
    });

    console.log("Ready to refine ..");
};



IQRClipResults.prototype.Results = function(query) {
    VCDClipResults.prototype.Submit.call(this, query);
};



IQRClipResults.prototype.RefreshState = function() {


    // append each state


    var statestr = JSON.stringify({ "iqr_session" : this.uid,
                                    "positive" : this.yes,
                                    "negative" : this.no,
                                    "states" : this.refinement_states
                               }, undefined, 2);

    // Enable the refine button if everything is all right

    var diff = 0;
    if(this.current_refinement > 0) {
        diff = this.yes.length - this.refinement_states[this.current_refinement-1].prevyes;
    } else {
        diff = this.yes.length;
    }

    if(diff > 0) {
        $(this.buttondiv).find(".refine").removeAttr("disabled");
    }
    else {
        $(this.buttondiv).find(".refine").attr("disabled", "disabled");
    }

    // $(this.statediv).html("<pre style='text-align:left'>" +statestr +  "</pre>");

};

IQRClipResults.prototype.Submit = function(query, feature) {
   if(this.url === "/iqr/search_results") {
       // Reset the search
       this.url = "/vcdi/search";
   }
    VCDClipResults.prototype.Submit.call(this, query);


    // Do the magic for initializing search session
    this.ShowBusy(true);
    $(this.buttondiv).find(".refine").addClass("btn-primary");
    var that = this;

    if(that.uid === undefined ||  that.url === "/vcdi/search") {
  
        var request = $.ajax({
            url: "/iqr/init_new_search_session",
            type: "GET",
            data: {
                query : JSON.stringify(query),
                feature: feature
            },
            dataType: "json"
        });

        request.done(function( msg ) {
            that.uid = msg.uid;
            that.RefreshState();
            that.ShowBusy(false);
//            $(that.buttondiv).find(".refine").removeAttr("disabled");
//            $(that.buttondiv).find(".refine").addClass("btn-primary");

    //        console.log(msg);
        });

        request.fail(function( jqXHR, textStatus ) {
            $("#btnsearch").removeClass("btn-primary");
            alert( "Request failed: " + textStatus );
        });
    }
};


IQRClipResults.prototype.Poll = function() {
    var that = this;
    $.ajax({
        type: "GET",
        context: this,
        url: "/iqr/search_state?uid=" + this.uid,
        success: function (msg) {
            // Add the count to state
            that.refinement_states[that.current_refinement-1].count = msg.count;
            that.RefreshState();
            if(msg.count > 20) {
                that.url = "/iqr/search_results"
                // Refresh the results
                that.Results(that.uid);
                // Set correct href
                $(that.buttondiv).find(".archive").removeAttr("disabled");

            }
            else {
                setTimeout(function() {that.Poll()} , 1000);
            }
        }
    });
};

IQRClipResults.prototype.LoadMore = function() {

    this.skip = this.skip + this.limit;
    if(this.url === "/iqr/search_results") {
        this.Results();
    }
    else {
        VCDClipResults.prototype.Submit.call(this);
    }
};

IQRClipResults.prototype.SubmitSuccess = function(result) {
    if(result === true){
        $(this.buttondiv).find(".refine").removeAttr("disabled");
        $(this.buttondiv).find(".refine").addClass("btn-primary");
        $(this.buttondiv).find(".reset").addClass("btn-danger");

        $("#btnsearch").removeClass("btn-primary");
    } else {
        $(this.buttondiv).find(".refine").attr("disabled","disabled");
        $(this.buttondiv).find(".refine").removeClass("btn-primary");
        $(this.buttondiv).find(".reset").removeClass("btn-danger");
        $("#btnsearch").addClass("btn-primary");
    }
    this.RefreshState();
}
