/**
 * Encapsulation of IQR Search/Refinement results.
 *
 * This allows incremental loading of results, i.e.
 *
 * @constructor
 */
function IqrView(container) {
    Object.call(this);

    //
    // Members
    //
    this.show_more_step = 20;
    this.displayed_search_results = [];

    this.random_enabled = false;  // not displayed initially.
    this.random_ids = [];
    this.displayed_random_results = [];

    //
    // Visual elements
    //

    // top-level container for layout
    this.iqr_view_container = $('<div id="iqr_view_instance"/>');

    // for IqrResult views
    this.search_container = $('<div class="iqr-view-results-container"/>');
    this.search_container.css('width', '100%');

    // for random result views, initially hidden
    this.random_container = $('<div class="iqr-view-results-container"/>');
    this.random_container.css('border', 'solid red 1px');

    // Shown at top of search view area (shouldn't cover random area)
    this.progress_bar_search = new ActivityBar();
    this.progress_bar_random = new ActivityBar();

    this.button_container_search_top = $('<div/>');
    this.button_container_search_top.css('text-align', 'center');

    this.button_container_search_bot = this.button_container_search_top.clone();
    this.button_container_random_top = this.button_container_search_top.clone();
    this.button_container_random_bot = this.button_container_search_top.clone();

    this.button_search_top = $('<button class="btn btn-primary" type="button"/>');
    this.button_search_top.text("Search / Refine");

    this.button_search_bot = this.button_search_top.clone();

    this.button_reset_top = $('<button class="btn btn-danger" type="button"/>');
    this.button_reset_top.text("Reset IQR");

    this.button_reset_bot = this.button_reset_top.clone();

    // To be put in top button container
    this.button_toggle_random = $('<button class="btn" type="button"/>');
    this.button_toggle_random.text('Toggle Random Results');

    // To be put in bottom button container in results panel
    this.button_search_showMore = $('<button class="btn" type="button"/>');
    this.button_search_showMore.text("Show More");
    this.button_search_showMore.hide();

    this.button_random_refresh = $('<button class="btn btn-danger" type="button"/>');
    this.button_random_refresh.text("Refresh Random Results");

    this.button_random_showMore = this.button_search_showMore.clone();

    this.results_container_search = $('<div/>');
    this.results_container_search.css('text-align', 'center');

    this.results_container_random = this.results_container_search.clone();

    //
    // Construct layout
    //
    // Not constructing random pane here so that it can be added and removed via
    // functions.
    //

    // initially construct pane out-of-tree
    this.construct_search_pane();

    $(container).append(this.iqr_view_container);
    $(container).append($('<hr>'));

    // Update model to initial state
    var inst = this;
    $.ajax({
        url: "get_positive_uids",
        success: function (data) {
            var num_pos = data['uids'].length;
            inst.display_search_results_range(0, inst.show_more_step + num_pos);
        },
        error: function (jqXHR, textStatus, errorThrown) {
            alert_error("AJAX: " + textStatus + ": " + errorThrown);
        }
    });

    return this;
}

/**
 * Return the number of initial search results to display.
 *
 * This takes into account the number of results at the very top that are
 * already marked positive, and returns that number + the step amount. This
 * ensures that the user is given at least some non-redundant information
 * before having to click any buttons.
 *
 * @returns int Number of search results to initially display.
 */
//IqrView.prototype.get_initial_result_display_count= function() {
//    var pos_ids = 0;
//    $.ajax({
//        async: false,
//        url: "get_positive_uids",
//        success: function (data) {
//            pos_ids = data['uids'].length;
//        },
//        error: function (jqXHR, textStatus, errorThrown) {
//            alert_error("AJAX Error: " + errorThrown);
//        }
//    });
//    return this.show_more_step + pos_ids;
//};

/**
 * Construct the IQR Search pane
 */
IqrView.prototype.construct_search_pane = function () {
    var inst = this;

    // remove the main container in case its already there
    this.search_container.remove();

    this.iqr_view_container.append(this.search_container);

    this.search_container.append(this.progress_bar_search.progress_div);
    this.search_container.append(this.button_container_search_top);
    this.search_container.append(this.results_container_search);
    this.search_container.append(this.button_container_search_bot);

    this.button_container_search_top.append(this.button_search_top);
    this.button_container_search_top.append(this.button_reset_top);
    this.button_container_search_top.append(this.button_toggle_random);

    this.button_container_search_bot.append(this.button_search_bot);
    this.button_container_search_bot.append(this.button_reset_bot);
    this.button_container_search_bot.append(this.button_search_showMore);

    // Search pane events
    this.button_search_top.click(function () {
        inst.refine();
    });
    this.button_search_bot.click(function () {
        inst.refine();
    });

    this.button_reset_top.click(function () {
        inst.reset();
    });
    this.button_reset_bot.click(function () {
        inst.reset();
    });

    this.button_search_showMore.click(function () {
        inst.iterate_more_search_results();
    });

    this.button_toggle_random.click(function () {
        inst.toggle_random_pane();
    });

    // sets element initial visible/hidden status
    this.show_search_functionals();
};

/**
 * Construct the random pane
 */
IqrView.prototype.construct_random_pane = function () {
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
IqrView.prototype.remove_random_pane = function () {
    this.random_container.remove();
};

/**
 * Show functional elements in the search container
 */
IqrView.prototype.show_search_functionals = function () {
    this.button_container_search_top.show();
    // Only show bottom buttons if there are results being shown
    if (this.displayed_search_results.length > 0) {
        this.button_container_search_bot.show();
        // explicitly show this one in case it was hidden due to display of
        // all results.
        this.button_search_showMore.show();
    }
    else {
        this.button_container_search_bot.hide();
    }
};

/**
 * Hide functional elements in the search container
 */
IqrView.prototype.hide_search_functionals = function () {
    this.button_container_search_top.hide();
    this.button_container_search_bot.hide();
};

/**
 * Show functional elements in the random container
 */
IqrView.prototype.show_random_functionals = function () {
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
IqrView.prototype.hide_random_functionals = function () {
    this.button_random_refresh.hide();
    this.button_random_showMore.hide();
};

/**
 * Clear currently displayed and stored search results
 */
IqrView.prototype.clear_search_results = function () {
    this.results_container_search.children().remove();
    this.displayed_search_results = [];
};

/**
 * Clear currently displayed and stored random results.
 */
IqrView.prototype.clear_random_results = function () {
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
IqrView.prototype.display_search_results_range = function (s, e) {
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
                r = new IqrResult(inst.results_container_search, s+i,
                                  c[0], c[1]);
                inst.displayed_search_results.push(r);
            }

            // Update functionals shown
            inst.show_search_functionals();

            // Hide load more button if we received less results than what
            // we asked for, meaning that we just displayed the last of
            // available results.
            if (results.length != (e - s)) {
                inst.button_search_showMore.hide();
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
IqrView.prototype.display_random_results_range = function (s, e) {
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
IqrView.prototype.refresh_random_ids = function () {
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
IqrView.prototype.iterate_more_search_results = function () {
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
IqrView.prototype.iterate_more_random_results = function () {
    var N = this.show_more_step;
    var s = this.displayed_random_results.length;
    var e = s + N;
    //alert_info("Showing more randoms in range ("+s+", "+e+"]");
    this.display_random_results_range(s, e);
};

/**
 * Update the state of this view with the Iqr session model server-side
 */
IqrView.prototype.refine = function() {
    this.hide_search_functionals();
    this.clear_search_results();
    this.progress_bar_search.on("... Searching ...");

    var inst = this;
    var restore = function () {
        inst.progress_bar_search.off();
        inst.show_search_functionals();
    };

    // Refine and then display the first N results.
    $.ajax({
        url: 'iqr_refine',
        method: "POST",
        success: function (data) {
            if (data['success']) {
                //alert_info("Iqr refine complete");
                $.ajax({
                    url: "get_positive_uids",
                    success: function (data) {
                        var num_pos = data['uids'].length;
                        inst.display_search_results_range(
                            0, inst.show_more_step + num_pos
                        );
                        restore();
                    },
                    error: function (jqXHR, textStatus, errorThrown) {
                        alert_error("AJAX: " + textStatus + ": " + errorThrown);
                    }
                });
            }
            else {
                alert_error("IQR Refine error: " + data['message']);
                restore();
            }
        },
        error: function (jqXHR, textStatus, errorThrown) {
            alert_error("AJAX Error: " + errorThrown);
            restore();
        }
    });
};

/**
 * Reset current IQR session state.
 */
IqrView.prototype.reset = function () {
    this.hide_search_functionals();
    this.clear_search_results();
    this.clear_random_results();

    var inst = this;
    var restore = function () {
        inst.progress_bar_search.off();
        inst.show_search_functionals();

        // refresh random results if the pane is on
        if (inst.random_enabled)
        {
            inst.refresh_random_ids();
        }
    };

    // Call for IQR Session removal and initialization of a new session
    this.progress_bar_search.on("... Resetting current IQR Session ...");
    $.ajax({
        url: "reset_iqr_session",
        method: "POST",
        success: function (/*data*/) {
            //alert_info("[RESET]: " + data['message']);
            restore();
        },
        error: function (jqXHR, textStatus, errorThrown) {
            alert_error("AJAX Error: " + errorThrown);
            restore();
        }
    })
};

/**
 * Add or remove the random pane based on current enable state.
 *
 * When enabling, performs an initial query to display initial results.
 */
IqrView.prototype.toggle_random_pane = function () {
    if (this.random_enabled)
    {
        // moving button first to avoid detaching handler attachments
        this.button_container_search_top.append(this.button_toggle_random);
        this.remove_random_pane();
        this.search_container.css('width', "100%");
    }
    else
    {
        this.construct_random_pane();
        this.button_container_random_top.append(this.button_toggle_random);
        this.search_container.css('width', '49%');
        this.random_container.css('width', '49%');

        // Show initial random results
        this.refresh_random_ids();
    }

    // flip
    this.random_enabled = !this.random_enabled;
};
