/**
 * Encapsulation of IQR Refinement queries and results viewing.
 *
 * @constructor
 */
function IqrRefineView(container) {
    Object.call(this);

    //
    // Members
    //
    this.show_more_step = 20;
    this.pre_to_display = 0;
    this.IQR_round = -1;
    this.session_id;

    // parallel arrays of ordered result UUIDs and relevancy scores.
    //this.displayed_search_results = [];
    this.refine_result_uuids = [];
    this.refine_result_score = [];
    // the UUIDs for pos selected and neg selected
    this.pos_selected_uuids = [];
    this.neg_selected_uuids = [];
    // number of refine results currently displayed from the above arrays.
    this.refine_results_displayed = 0;

    this.random_enabled = false;  // not displayed initially.
    this.random_ids = [];
    this.random_results_displayed = 0;

    //
    // Visual elements
    //

    // top-level container for layout
    this.iqr_view_container = $('<div id="iqr_view_instance"/>');

    // for DataView views
    this.refine_container = $('<div class="iqr-view-results-container"/>');
    this.refine_container.css('width', '100%');

    // for random result views, initially hidden
    this.random_container = $('<div class="iqr-view-results-container"/>');
    this.random_container.css('border', 'solid red 1px');

    // Shown at top of search view area (shouldn't cover random area)
    this.progress_bar_refine = new ActivityBar();
    this.progress_bar_random = new ActivityBar();

    this.button_container_refine_top = $('<div/>');
    this.button_container_refine_top.css('text-align', 'center');
    this.button_container_refine_bot = this.button_container_refine_top.clone();
    // this.button_container_random_top = this.button_container_refine_top.clone();
    // this.button_container_random_bot = this.button_container_refine_top.clone();

    this.button_refine_top = $('<button class="btn btn-primary" type="button"/>');
    this.button_refine_top.text("Refine Results");
    this.button_refine_bot = this.button_refine_top.clone();

    this.button_saliency_top = $('<button class="btn btn-success" type="button"/>');
    this.button_saliency_top.text("Saliency On");

    this.button_saliency_bot = this.button_saliency_top.clone();
    this.button_saliency_flag = true;

    // this.button_gt_top = $('<button class="btn btn-success" type="button"/>');
    // this.button_gt_top.text("GT label On");
    // this.button_gt_bot = this.button_gt_top.clone();
    // this.button_gt_flag = true;

    // To be put in top button container
    // this.button_toggle_random = $('<button class="btn" type="button"/>');
    // this.button_toggle_random.text('Toggle Random Results');

    // To be put in bottom button container in results panel
    // this.button_refine_showMore = $('<button class="btn" type="button"/>');
    // this.button_refine_showMore.text("Show More");

    // this.button_random_refresh = $('<button class="btn btn-danger" type="button"/>');
    // this.button_random_refresh.text("Refresh Random Results");

    // this.button_random_showMore = this.button_refine_showMore.clone();

    this.results_container_refine = $('<div/>');
    this.results_container_random = $('<div/>');

    //
    // Pos and neg container
    //
    this.pos_container = $('<div/>');
    this.neg_container = $('<div/>');
    this.pos_hr = $('<hr id="pos_hr">');
    this.neg_hr = $('<hr id="neg_hr">');

    this.pos_state = $('<label>');
    this.pos_state.text("Positive feedbacks:");

    this.neg_state = $('<label>');
    this.neg_state.text("Negative feedbacks:");


    this.status_inst;
    this.iqr_view_inst;

    this.saliency_flag = false;
    this.gt_flag = false;

    //
    // Construct layout
    //
    // Not constructing random pane here so that it can be added and removed via
    // functions.
    //
    $(container).append(this.iqr_view_container);
    this.construct_view(); // Everything lives underneath iqr_view_container

    return this;
}

/**
 * Global view construction
 */
IqrRefineView.prototype.construct_view = function () {
    this.construct_refine_pane();
    // construct random pane on request
};

/**
 * Global view update
 */
IqrRefineView.prototype.update_view = function () {
    this.update_refine_pane();
};

/**
 * Construct the IQR Search pane
 */
IqrRefineView.prototype.construct_refine_pane = function () {
    var inst = this;

    this.pos_container.remove();
    this.iqr_view_container.append(this.pos_state);
    this.iqr_view_container.append(this.pos_container);
    this.iqr_view_container.append(this.pos_hr);
    this.pos_state.hide();
    this.pos_hr.hide();

    this.neg_container.remove();
    this.iqr_view_container.append(this.neg_state);
    this.iqr_view_container.append(this.neg_container);
    this.iqr_view_container.append(this.neg_hr);
    this.neg_state.hide();
    this.neg_hr.hide();

    // remove the main container in case its already there
    this.refine_container.remove();
    this.iqr_view_container.append(this.refine_container);

    this.refine_container.append(
        this.progress_bar_refine.progress_div,
        this.button_container_refine_top,
        this.results_container_refine,
        this.button_container_refine_bot
    );

    this.button_container_refine_top.append(
        this.button_refine_top,
        this.button_saliency_top
        // this.button_gt_top,
        // this.button_toggle_random
    );

    this.button_container_refine_bot.append(
        this.button_refine_bot,
        this.button_saliency_bot
        // this.button_gt_bot,
        // this.button_refine_showMore
    );
    // initially hide bottom buttons
    this.button_container_refine_bot.hide();

    /**
     * Event handling
     */
    this.button_refine_top.click(function () {
        if (inst.IQR_round < 2) {
            inst.iqr_refine();
        } else {
            inst.finish_task();
        }
    });
    this.button_refine_bot.click(function () {
        if (inst.IQR_round < 2) {
            inst.iqr_refine();
        } else {
            inst.finish_task();
        }
    });
    // this.button_refine_showMore.click(function () {
    //     inst.show_more_refine_results();
    // });
    // this.button_toggle_random.click(function () {
    //     inst.toggle_random_pane();
    // });
    this.button_saliency_top.click(function () {
        inst.saliency_control();
    });
    this.button_saliency_bot.click(function () {
        inst.saliency_control();
    });
    // this.button_gt_top.click(function() {
    //     inst.gt_control();
    // });
    // this.button_gt_bot.click(function() {
    //     inst.gt_control();
    // });


    // sets element initial visible/hidden status
    this.update_refine_pane();
};

IqrRefineView.prototype.saliency_control = function () {
    if (this.button_saliency_flag) {
        this.saliency_flag = true;
        this.button_saliency_top.text("Saliency Off");
        this.button_saliency_bot.text("Saliency Off");
        this.button_saliency_flag = false;
        this.button_saliency_top.attr("class", "btn btn-danger");
        this.button_saliency_bot.attr("class", "btn btn-danger");
    } else {
        this.saliency_flag = false;
        this.button_saliency_top.text("Saliency On");
        this.button_saliency_bot.text("Saliency On");
        this.button_saliency_flag = true;
        this.button_saliency_top.attr("class", "btn btn-success");
        this.button_saliency_bot.attr("class", "btn btn-success");
    }
    this.pos_container.children().remove();
    this.neg_container.children().remove();
    this.results_container_refine.children().remove();
    this.show_more_refine_results(true);
};

IqrRefineView.prototype.gt_control = function () {
    if (this.button_gt_flag) {
        this.gt_flag = true;
        this.results_container_refine.children("div#iqr_res").children("div#gt_div").removeClass("iqr-gt-label-hide");
        this.results_container_refine.children("div#iqr_res").children("div#gt_div").addClass("iqr-gt-label-show");
        this.button_gt_flag = false;
        // this.button_gt_top.text("GT label Off");
        // this.button_gt_bot.text("GT label Off");
        // this.button_gt_top.attr("class", "btn btn-danger");
        // this.button_gt_bot.attr("class", "btn btn-danger");

    } else {
        this.gt_flag = false;
        this.results_container_refine.children("div#iqr_res").children("div#gt_div").removeClass("iqr-gt-label-show");
        this.results_container_refine.children("div#iqr_res").children("div#gt_div").addClass("iqr-gt-label-hide");
        this.button_gt_flag = true;
        // this.button_gt_top.text("GT label On");
        // this.button_gt_bot.text("GT label On");
        // this.button_gt_top.attr("class", "btn btn-success");
        // this.button_gt_bot.attr("class", "btn btn-success");
    }
};

IqrRefineView.prototype.finish_task = function () {
    var self = this;
    prompt("Please copy the following session ID and paste it back to the HIT webpage!",
        self.session_id);
    self.iqr_view_inst.reset_session();
};

/**
 * Refresh the view of the refine pane based on server-side model
 *
 */
IqrRefineView.prototype.update_refine_pane = function () {
    // clear children of results container
    // get ordered results information
    // display first X results
    this.pos_container.children().remove();
    this.neg_container.children().remove();
    this.results_container_refine.children().remove();
    if (this.saliency_flag) {
        this.button_saliency_top.attr("class", "btn btn-danger");
        this.button_saliency_bot.attr("class", "btn btn-danger");
        this.button_saliency_top.text("Saliency Off");
        this.button_saliency_bot.text("Saliency Off");
    } else {
        this.button_saliency_top.attr("class", "btn btn-success");
        this.button_saliency_bot.attr("class", "btn btn-success");
        this.button_saliency_top.text("Saliency On");
        this.button_saliency_bot.text("Saliency On");
    }

    this.refine_result_uuids = [];
    this.refine_result_score = [];

    this.pos_selected_uuids = [];
    this.neg_selected_uuids = [];
    this.refine_results_displayed = 0;

    var self = this;

    if (self.IQR_round === 2) {
        self.button_refine_top.text("Finish Task");
        self.button_refine_bot.text("Finish Task");
    }

    // Check initialization status of session
    // - When not initialized, disable buttons + don't try to show results
    //   (there aren't going to be any)
    // - When initialized, enable buttons + show ordered results
    $.ajax({
        url: "iqr_session_info",
        method: "GET",
        success: function (data) {
            self.session_id = data['uuid'];
            if (data["initialized"]) {
                // enable buttons
                self.button_container_refine_top.children().prop("disabled", false);
                self.button_saliency_bot.attr("disabled", true);
                self.button_saliency_top.attr("disabled", true);
                // Fetch ordered results + display
                $.ajax({
                    url: "iqr_ordered_results",
                    method: "GET",
                    success: function (data) {
                        // Update refinement result uuid/score arrays
                        for (var i=0; i < data["results"].length; i++) {
                            self.refine_result_uuids.push(data["results"][i][0]);
                            self.refine_result_score.push(data["results"][i][1]);
                        }

                        //calcualte accuracy and store the result into file
                        self.status_inst.cal_acc_store();


                        // create/show top N results
                        // - If no results, display text verification of no results
                        if (self.refine_result_uuids.length === 0) {
                            self.results_container_refine.append(
                                $("<span>No Results</span>")
                            );
                        }
                        else {
                            self.show_more_refine_results();
                        }
                    },
                    error: function (jqXHR, textStatus, errorThrown) {
                        alert_error("Error fetching refined results: " +
                            "("+errorThrown+") " + textStatus);
                    }
                });
            }
            else {
                // disable buttons + hide bottom button container
                self.button_container_refine_top.children().prop("disabled", true);
                self.button_container_refine_bot.hide();
            }

            if (data["positive_uids"].length > 0) {
                self.pos_state.show();
                self.pos_hr.show();
            } else {
                self.pos_state.hide();
                self.pos_hr.hide();
            }
            for (var i = 0; i < data["positive_uids"].length; i++) {
                self.pos_selected_uuids.push(data["positive_uids"][i]);
            }

            if (data["negative_uids"].length > 0) {
                self.neg_state.show();
                self.neg_hr.show();
            } else {
                self.neg_state.hide();
                self.neg_hr.hide();
            }
            for (var i = 0; i < data["negative_uids"].length; i++) {
                self.neg_selected_uuids.push(data["negative_uids"][i]);
            }
        }
    });
};

/**
 * Show more refinement results in the refinement view if we can.
 *
 * If we have shown all results possible, we hide the "Show More" button.
 */
IqrRefineView.prototype.show_more_refine_results = function (replay_flag) {
    var self = this;
    replay_flag = replay_flag || false;   //set default of replay_flag to false

    var to_display = 0;
    if (replay_flag) {
        to_display = this.pre_to_display;
        this.refine_results_displayed = 0;
    } else {
        to_display = Math.min(
        this.refine_result_uuids.length,
        this.refine_results_displayed + this.show_more_step);
    }
    this.pre_to_display = to_display;

    //show neg selected images on neg_container
    for (var i = 0; i < this.neg_selected_uuids.length; i++) {
        new DataView(this.neg_container,
                      i,  // show 1-indexed rank value
                      this.neg_selected_uuids[i],
                      0.0,
                      this.saliency_flag, false, false, false);
    }

    while (this.refine_results_displayed < to_display)
    {
        var cur_uid = this.refine_result_uuids[this.refine_results_displayed];
        //show pos selected images on pos_container
        if (this.pos_selected_uuids.includes(cur_uid)) {
            new DataView(this.pos_container,
                      this.refine_results_displayed,  // show 1-indexed rank value
                      this.refine_result_uuids[this.refine_results_displayed],
                      this.refine_result_score[this.refine_results_displayed],
                      this.saliency_flag, false, false, false);
            to_display++;

        } else if (this.neg_selected_uuids.includes(cur_uid)) {
            //show neg selected images on neg_container if the neg_selected will be
            //shown in the result_container (i.e., no other result will be shown,
            // very rare)
            new DataView(this.neg_container,
                      this.refine_results_displayed,  // show 1-indexed rank value
                      this.refine_result_uuids[this.refine_results_displayed],
                      this.refine_result_score[this.refine_results_displayed],
                      this.saliency_flag, false, false, false);
            to_display++;
        } else {

            new DataView(this.results_container_refine,
                this.refine_results_displayed,  // show 1-indexed rank value
                this.refine_result_uuids[this.refine_results_displayed],
                this.refine_result_score[this.refine_results_displayed],
                this.saliency_flag, false, false, true);
        }
        this.refine_results_displayed++;
    }

    if (this.gt_flag) {
        this.results_container_refine.children("div#iqr_res").children("div#gt_div").removeClass("iqr-gt-label-hide");
        this.results_container_refine.children("div#iqr_res").children("div#gt_div").addClass("iqr-gt-label-show");
        // this.button_gt_top.text("GT label Off");
        // this.button_gt_bot.text("GT label Off");
        // this.button_gt_top.attr("class", "btn btn-danger");
        // this.button_gt_bot.attr("class", "btn btn-danger");
    } else {
        this.results_container_refine.children("div#iqr_res").children("div#gt_div").removeClass("iqr-gt-label-show");
        this.results_container_refine.children("div#iqr_res").children("div#gt_div").addClass("iqr-gt-label-hide");
        // this.button_gt_top.text("GT label On");
        // this.button_gt_bot.text("GT label On");
        // this.button_gt_top.attr("class", "btn btn-success");
        // this.button_gt_bot.attr("class", "btn btn-success");
    }

    if (this.refine_results_displayed === 0) {
        this.button_container_refine_bot.hide();
    }
    else {
        this.button_container_refine_bot.show();

        // Conditionally show or hide the show-more button
        // if (this.refine_results_displayed === this.refine_result_uuids.length) {
        //     this.button_refine_showMore.hide();
        // }
        // else {
        //     this.button_refine_showMore.show();
        // }
    }
};

/**
 * Trigger an index refine action server-size, updating view when complete.
 * Requires that there be positive
 */
IqrRefineView.prototype.iqr_refine = function() {
    var self = this;

    if (typeof $('#liker_form input[name=likert]:checked').val() === 'undefined' && self.IQR_round !== -1) {
        //check whether the likert scale has been selected
        var info = "Please answer the question ";
        var question = "I have high confidence in my positive/negative label assignments.";
        alert(info+"\n" + "\t •" + question);
    } else {

        $.ajax({
            url: 'count_selection',
            method: 'POST',
            data:{
                iqr_round: self.IQR_round
            },
            success: function (iqr_count_flag){
                if (!iqr_count_flag['success']) {
                    cur_count = iqr_count_flag['count'];
                    var left_count = (self.IQR_round + 1) * 20 + 1 - cur_count;
                    alert("Please give feedbacks to all query images! only " + left_count + " lefted!");
                } else {
                    // helper methods for display stuff
                    function disable_buttons() {
                        self.button_container_refine_top.children().prop("disabled", true);
                        self.button_container_refine_bot.children().prop("disabled", true);
                    }

                    function enable_buttons() {
                        self.button_container_refine_top.children().prop("disabled", false);
                        self.button_container_refine_bot.children().prop("disabled", false);
                    }

                    function restore() {
                        self.progress_bar_refine.off();
                    }

                    disable_buttons();
                    self.progress_bar_refine.on("Refining");

                    // Refine and then display the first N results.
                    $.ajax({
                        url: 'iqr_refine',
                        method: "POST",
                        success: function (data) {
                            if (data['success']) {
                                enable_buttons();
                                self.button_saliency_top.attr("disabled", true);
                                self.button_saliency_bot.attr("disabled", true);

                                self.IQR_round = self.IQR_round + 1;
                                //reset AMT zone for next round IQR
                                $('#acc_stat').text("IQR Round " + self.IQR_round + "---Top-20 Accuracy: ");
                            }
                            else {
                                alert_error("IQR Refine error: " + data['message']);
                            }

                            self.update_refine_pane();

                            restore();
                        },
                        error: function (jqXHR, textStatus, errorThrown) {
                            alert_error("AJAX Error: " + errorThrown);
                            restore();
                        }
                    });
                }
            },
            error: function (jqXHR, textStatus, errorThrown) {
                alert_error("AJAX Error: count_selection: " + errorThrown);
                restore();
            }
        });
    }
};


//
// Random pane stuff
//
/**
 * Construct the random pane
 */
IqrRefineView.prototype.construct_random_pane = function () {
    // Remove the main container in case it's already there
    this.random_container.remove();

    this.iqr_view_container.append(this.random_container);

    this.random_container.append(this.progress_bar_random.progress_div);
    // this.random_container.append(this.button_container_random_top);
    this.random_container.append(this.results_container_random);
    // this.random_container.append(this.button_container_random_bot);

    // this.button_container_random_top.append(this.button_random_refresh);

    // this.button_container_random_bot.append(this.button_random_showMore);

    // Random pane events
    var inst = this;
    // this.button_random_refresh.click(function () {
    //     inst.refresh_random_ids();
    // });

    // this.button_random_showMore.click(function () {
    //     inst.iterate_more_random_results();
    // });

    // sets element initial visible/hidden status
    // this.show_random_functionals();
};

/**
 * Remove the random display pane.
 */
IqrRefineView.prototype.remove_random_pane = function () {
    this.random_container.remove();
};

/**
 * Show functional elements in the random container
 */
// IqrRefineView.prototype.show_random_functionals = function () {
//     // this.button_random_refresh.show();
//     // Only show the showMode button if there are any results displayed
//     if (this.random_results_displayed > 0) {
//         this.button_random_showMore.show();
//     }
//     else {
//         this.button_random_showMore.hide();
//     }
// };

/**
 * Hide functional elements in the random container
 */
// IqrRefineView.prototype.hide_random_functionals = function () {
//     // this.button_random_refresh.hide();
//     this.button_random_showMore.hide();
// };

/**
 * Clear currently displayed and stored random results.
 */
IqrRefineView.prototype.clear_random_results = function () {
    this.results_container_random.children().remove();
    this.random_results_displayed = 0;
};

/**
 * Show N more random results in the random results container based on our
 * currently stored random ID ordering. This will add nothing if we don't
 * currently have any stored random IDs.
 */
IqrRefineView.prototype.display_random_results_range = function (s, e) {
    var inst = this;

    // bound the start and end indices to prevent overflow
    s = Math.min(Math.max(0, s), this.random_ids.length);
    e = Math.min(e, this.random_ids.length);
    //alert_info("Displaying random range ("+s+", "+e+")");
    var r = null;
    for (; s < e; s++) {
        r = new DataView(this.results_container_random, s,
                         this.random_ids[s], 'rand', inst.saliency_flag);
        this.random_results_displayed++;
    }

    // update functionals shown.
    // this.show_random_functionals();

    // Hide "show more" button if there are no more to show
    // if (this.random_results_displayed === this.random_ids.length) {
    //     this.button_random_showMore.hide();
    // }
};

/**
 * Query from the server the list of data element IDs in a random order.
 */
IqrRefineView.prototype.refresh_random_ids = function () {
    // this.hide_random_functionals();
    this.clear_random_results();
    this.progress_bar_random.on("Refreshing random list");

    var inst = this;
    var restore = function () {
        inst.progress_bar_random.off();
        // inst.show_random_functionals();
    };

    $.ajax({
        url: "get_random_uids",
        success: function (data) {
            inst.random_ids = data['uids'];
            //alert_info("Random UIDs received: " + inst.random_ids);
            // If there are any IDs recorded, display an initial batch.
            inst.display_random_results_range(0, inst.show_more_step, inst.saliency_flag);
            restore();
        },
        error: function (jqXHR, textStatus, errorThrown) {
            alert_error("AJAX Error: " + errorThrown);
            restore();
        }
    });
};

/**
 * Display N more results in the random results container
 *
 * Where N is the configured ``show_more_step`` amount. If there are no more
 * results to show, this function does nothing (plus associated button is
 * hidden).
 */
IqrRefineView.prototype.iterate_more_random_results = function () {
    var N = this.show_more_step;
    var s = this.random_results_displayed;
    var e = s + N;
    //alert_info("Showing more randoms in range ("+s+", "+e+"]");
    this.display_random_results_range(s, e);
};

/**
 * Add or remove the random pane based on current enable state.
 *
 * When enabling, performs an initial query to display initial results.
 */
IqrRefineView.prototype.toggle_random_pane = function () {
    if (this.random_enabled)
    {
        // moving button first to avoid detaching handler attachments
        // this.button_container_refine_top.append(this.button_toggle_random);
        this.remove_random_pane();
        this.refine_container.css('width', "100%");
    }
    else
    {
        this.construct_random_pane();
        // this.button_container_random_top.append(this.button_toggle_random);
        this.refine_container.css('width', '49%');
        this.random_container.css('width', '49%');

        // Show initial random results
        this.refresh_random_ids();
    }

    // flip
    this.random_enabled = !this.random_enabled;
};
