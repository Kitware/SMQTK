/**
 * Encapsulation of IQR Search/Refinement results.
 *
 * This allows incremental loading of results, i.e.
 *
 * @constructor
 */
function IqrRefineView(container) {
    Object.call(this);

    //
    // Members
    //
    this.show_more_step = 20;

    // parallel arrays of ordered result UUIDs and relevancy scores.
    //this.displayed_search_results = [];
    this.refine_result_uuids = [];
    this.refine_result_score = [];
    // number of refine results currently displayed from the above arrays.
    this.refine_results_displayed = 0;

    this.random_enabled = false;  // not displayed initially.
    this.random_ids = [];
    this.displayed_random_results = [];

    //
    // Visual elements
    //

    // top-level container for layout
    this.iqr_view_container = $('<div id="iqr_view_instance"/>');

    // for IqrResult views
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
    this.button_container_random_top = this.button_container_refine_top.clone();
    this.button_container_random_bot = this.button_container_refine_top.clone();

    this.button_refine_top = $('<button class="btn btn-primary" type="button"/>');
    this.button_refine_top.text("Refine");
    this.button_refine_bot = this.button_refine_top.clone();

    // To be put in top button container
    this.button_toggle_random = $('<button class="btn" type="button"/>');
    this.button_toggle_random.text('Toggle Random Results');

    // To be put in bottom button container in results panel
    this.button_refine_showMore = $('<button class="btn" type="button"/>');
    this.button_refine_showMore.text("Show More");

    this.button_random_refresh = $('<button class="btn btn-danger" type="button"/>');
    this.button_random_refresh.text("Refresh Random Results");

    this.button_random_showMore = this.button_refine_showMore.clone();

    this.results_container_refine = $('<div/>');
    this.results_container_random = $('<div/>');

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
        this.button_toggle_random
    );

    this.button_container_refine_bot.append(
        this.button_refine_bot,
        this.button_refine_showMore
    );
    // initially hide bottom buttons
    this.button_container_refine_bot.hide();

    /**
     * Event handling
     */
    this.button_refine_top.click(function () {
        inst.iqr_refine();
    });
    this.button_refine_bot.click(function () {
        inst.iqr_refine();
    });
    this.button_refine_showMore.click(function () {
        inst.iterate_more_index_results();
    });
    this.button_toggle_random.click(function () {
        inst.toggle_random_pane();
    });

    // sets element initial visible/hidden status
    this.update_refine_pane();
};

/**
 * Refresh the view of the refine pane based on server-side model
 *
 */
IqrRefineView.prototype.update_refine_pane = function () {
    // clear children of results container
    // get ordered results information
    // display first X results
    this.results_container_refine.children().remove();
    this.refine_result_uuids = [];
    this.refine_result_score = [];
    this.refine_results_displayed = 0;

    var self = this;
    // Check initialization status of session
    // - When not initialized, disable buttons + don't try to show results
    //   (there aren't going to be any)
    // - When initialized, enable buttons + show ordered results
    $.ajax({
        url: "iqr_session_info",
        method: "GET",
        success: function (data) {
            if (data["initialized"]) {
                // enable buttons
                self.button_container_refine_top.children().prop("disabled", false);
                // Fetch ordered results + display
                $.ajax({
                    url: "iqr_ordered_results",
                    method: "GET",
                    success: function (data) {
                        // Update refinement result uuid/score arrays
                        for (var i=0; i < data["results"].length; i++) {
                            self.refine_result_uuids.push(data["results"][0]);
                            self.refine_result_score.push(data["results"][1]);
                        }

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
        }
    });
};

/**
 * Show more refinement results in the refinement view if we can.
 *
 * If we have shown all results possible, we hide the "Show More" button.
 */
IqrRefineView.prototype.show_more_refine_results = function () {
    var to_display = Math.min(
        this.refine_result_uuids.length,
        this.refine_results_displayed + this.show_more_step
    );
    while (this.refine_results_displayed < to_display)
    {
        new IqrResult(this.results_container_refine,
                      this.refine_results_displayed+1,  // show 1-indexed rank value
                      this.refine_result_uuids[this.refine_results_displayed],
                      this.refine_result_score[this.refine_results_displayed]);
        this.refine_results_displayed++;
    }

    if (this.refine_results_displayed === 0) {
        this.button_container_refine_bot.hide();
    }
    else {
        this.button_container_refine_bot.show();

        // Conditionally show or hide the show-more button
        if (this.refine_results_displayed === this.refine_result_uuids.length) {
            this.button_refine_showMore.hide();
        }
        else {
            this.button_refine_showMore.show();
        }
    }
};

/**
 * Trigger an index refine action server-size, updating view when complete.
 * Requires that there be positive
 */
IqrRefineView.prototype.iqr_refine = function() {
    this.hide_refine_functionals();
    this.clear_search_results();
    this.progress_bar_refine.on("... Searching ...");

    var inst = this;
    var restore = function () {
        inst.progress_bar_refine.off();
        inst.show_refine_functionals();
    };

    // Refine and then display the first N results.
    $.ajax({
        url: 'iqr_refine',
        method: "POST",
        success: function (data) {
            if (data['success']) {
                alert_info("Iqr refine complete");
            }
            else {
                alert_error("IQR Refine error: " + data['message']);
            }
            inst.update_refine_pane();
            restore();
        },
        error: function (jqXHR, textStatus, errorThrown) {
            alert_error("AJAX Error: " + errorThrown);
            restore();
        }
    });
};



/**
 * Construct the random pane
 */
IqrRefineView.prototype.construct_random_pane = function () {
    // Remove the main container in case it's already there
    this.random_container.remove();

    this.iqr_view_container.append(this.random_container);

    this.random_container.append(this.progress_bar_random.progress_div);
    this.random_container.append(this.button_container_random_top);
    this.random_container.append(this.results_container_random);
    this.random_container.append(this.button_container_random_bot);

    this.button_container_random_top.append(this.button_random_refresh);

    this.button_container_random_bot.append(this.button_random_showMore);

    // Random pane events
    var inst = this;
    this.button_random_refresh.click(function () {
        inst.refresh_random_ids();
    });

    this.button_random_showMore.click(function () {
        inst.iterate_more_random_results();
    });

    // sets element initial visible/hidden status
    this.show_random_functionals();
};

/**
 * Remove the random display pane.
 */
IqrRefineView.prototype.remove_random_pane = function () {
    this.random_container.remove();
};

/**
 * Show functional elements in the search container
 */
IqrRefineView.prototype.show_refine_functionals = function () {
    this.button_container_refine_top.show();
    // Only show bottom buttons if there are results being shown
    if (this.displayed_search_results.length > 0) {
        this.button_container_refine_bot.show();
        // explicitly show this one in case it was hidden due to display of
        // all results.
        this.button_refine_showMore.show();
    }
    else {
        this.button_container_refine_bot.hide();
    }
};

/**
 * Hide functional elements in the search container
 */
IqrRefineView.prototype.hide_refine_functionals = function () {
    this.button_container_refine_top.hide();
    this.button_container_refine_bot.hide();
};

/**
 * Show functional elements in the random container
 */
IqrRefineView.prototype.show_random_functionals = function () {
    this.button_random_refresh.show();
    // Only show the showMode button if there are any results displayed
    if (this.displayed_random_results.length > 0) {
        this.button_random_showMore.show();
    }
    else {
        this.button_random_showMore.hide();
    }
};

/**
 * Hide functional elements in the random container
 */
IqrRefineView.prototype.hide_random_functionals = function () {
    this.button_random_refresh.hide();
    this.button_random_showMore.hide();
};

/**
 * Clear currently displayed and stored search results
 */
IqrRefineView.prototype.clear_search_results = function () {
    this.results_container_refine.children().remove();
    this.displayed_search_results = [];
};

/**
 * Clear currently displayed and stored random results.
 */
IqrRefineView.prototype.clear_random_results = function () {
    this.results_container_random.children().remove();
    this.displayed_random_results = [];
};

/**
 * Add results from the current IQR session within the given ordered result
 * index range
 *
 * @param s Starting result index (inclusive)
 * @param e Ending result index (exclusive)
 */
IqrRefineView.prototype.display_search_results_range = function (s, e) {
    var inst = this;
    $.ajax({
        url: "iqr_ordered_results",
        data: {
            i: s,
            j: e
        },
        success: function (data) {
            // see IQRSearch.iqr_get_results_range for
            // results dictionary return format.
            var results = data['results'];
            var i = 0, r = null, c = null;
            for (; i < results.length; i++) {
                c = results[i];
                r = new IqrResult(inst.results_container_refine, s+i,
                                  c[0], c[1]);
                inst.displayed_search_results.push(r);
            }

            // Update functionals shown
            inst.show_refine_functionals();

            // Hide load more button if we received less results than what
            // we asked for, meaning that we just displayed the last of
            // available results.
            if (results.length != (e - s)) {
                inst.button_refine_showMore.hide();
            }
        },
        error: function (jqXHR, textStatus, errorThrown) {
            alert_error("AJAX Error: " + errorThrown);
        }
    });
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
        r = new IqrResult(this.results_container_random, s,
                          this.random_ids[s], 'rand', inst, true);
        this.displayed_random_results.push(r);
    }

    // update functionals shown.
    this.show_random_functionals();

    // Hide "show more" button if there are no more to show
    if (this.displayed_random_results.length == this.random_ids.length) {
        this.button_random_showMore.hide();
    }
};

/**
 * Query from the server the list of data element IDs in a random order.
 */
IqrRefineView.prototype.refresh_random_ids = function () {
    this.hide_random_functionals();
    this.clear_random_results();
    this.progress_bar_random.on("Refreshing random list");

    var inst = this;
    var restore = function () {
        inst.progress_bar_random.off();
        inst.show_random_functionals();
    };

    $.ajax({
        url: "get_random_uids",
        success: function (data) {
            inst.random_ids = data['uids'];
            //alert_info("Random UIDs received: " + inst.random_ids);
            // If there are any IDs recorded, display an initial batch.
            inst.display_random_results_range(0, inst.show_more_step);
            restore();
        },
        error: function (jqXHR, textStatus, errorThrown) {
            alert_error("AJAX Error: " + errorThrown);
            restore();
        }
    });
};

/**
 * Display N more results in the search results container
 *
 * Where N is the amount specified or our configured ``show_more_step``
 * amount. If there are no more results to show, we hide the button.
 */
IqrRefineView.prototype.iterate_more_index_results = function () {
    var N = this.show_more_step;
    var s = this.displayed_search_results.length;
    var e = s + N;
    //alert_info("Showing more from index range ("+s+", "+e+"]");
    this.display_search_results_range(s, e);
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
    var s = this.displayed_random_results.length;
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
        this.button_container_refine_top.append(this.button_toggle_random);
        this.remove_random_pane();
        this.refine_container.css('width', "100%");
    }
    else
    {
        this.construct_random_pane();
        this.button_container_random_top.append(this.button_toggle_random);
        this.refine_container.css('width', '49%');
        this.random_container.css('width', '49%');

        // Show initial random results
        this.refresh_random_ids();
    }

    // flip
    this.random_enabled = !this.random_enabled;
};
