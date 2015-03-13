/**
 * IQR Result encapsulation, exposing adjudication functions
 *
 * Image data should be loaded by an AJAX call that returns the literal data,
 * not a path.
 *
 * for image load progress status, see: http://usablica.github.io/progress.js/
 *
 * Explicit images are covered initially by default but may be uncovered
 * temporarily.
 *
 */
function IqrResult(container, rank, uid, probability) {
    Object.call(this);

    var inst = this;
    this.rank = rank;
    this.uid = uid;
    this.probability = probability;

    this.image_preview_data = null;
    this.image_loaded = false;
    this.is_explicit = false;
    this.img_long_side = 0;
    this.e_marked_servereide = false;

    this.is_positive = false;
    this.is_negative = false;

    // available default image stuff
    this.adj_pos_class = "result-positive";
    this.adj_pos_off_icon = "/static/img/carbon-verified.png";
    this.adj_pos_on_icon = "/static/img/carbon-verified_on.png";

    this.adj_neg_class = "result-negative";
    this.adj_neg_off_icon = "/static/img/carbon-rejected.png";
    this.adj_neg_on_icon = "/static/img/carbon-rejected_on.png";

    this.explicit_icon_on = "/static/img/explicit_marker_on.png";
    this.explicit_icon_off = "/static/img/explicit_marker_off.png";
    this.explicit_overlay_image = "/static/img/explicit_overlay.png";

    this.loading_gif = "static/img/loading.gif";

    //
    // Layout
    //

    // top-level container
    this.result = $('<div class="iqr-result"/>');
    this.result.appendTo($(container));

    // Header container textual identifiers
    this.header = $('<div/>');
    this.header.css('height', '24px');
    this.header.text("#" + (this.rank+1) + " | "
                     + (this.probability*100).toFixed(2) + "%");

    // adjudication icons / functionality
    this.adjudication_controls = $('<div class="adjudication-box"/>');
    this.adj_pos_icon = $('<img height="24px">');
    this.adj_pos_icon.css('padding-left', '12px');
    this.adj_pos_icon.attr('src', inst.adj_pos_off_icon);
    this.adj_neg_icon = $('<img height="24px">');
    this.adj_neg_icon.css('padding-left', '12px');
    this.adj_neg_icon.attr('src', inst.adj_neg_off_icon);
    this.explicit_icon = $('<img height="24px"/>');
    this.explicit_icon.addClass("pull-right");
    this.explicit_icon.attr('src', this.explicit_icon_off);

    this.adjudication_controls.append(this.adj_pos_icon);
    this.adjudication_controls.append(this.adj_neg_icon);
    this.adjudication_controls.append(this.explicit_icon);

    // image container image data and adjudication buttons
    this.image_container = $('<div class="iqr-result-img-container"/>');
    this.image_data_view = $('<img>');
    this.image_data_view.attr('src', this.loading_gif);

    // Assemble result box
    this.result.append(this.header);
    this.result.append(this.adjudication_controls);
    this.result.append(this.image_container);

    this.image_container.append(this.image_data_view);

    //
    // Control
    //

    // react to clicking positive adjudication marker
    this.adj_pos_icon.click(function () {
        inst.set_positive();
    });

    // react to clicking negative adjudication marker
    this.adj_neg_icon.click(function () {
        inst.set_negative();
    });

    // react to clicking explicit icon
    this.explicit_icon.click(function () {
        inst.toggle_explicit();
    });

    // link click controls
    this.image_data_view.click(function () {
        inst.display_full_image();
    });

    // Update to initial view
    this.update_view(true);

    return this;
}

/**
 * Return boolean value indicating if this result has or has not been
 * adjudicated yet.
 *
 * @returns boolean True of this result has been adjudicated and false if
 *      not.
 */
IqrResult.prototype.is_adjudicated = function () {
    return (this.is_negative || this.is_positive);
};

/**
 * Update the view of this element based on current state.
 */
IqrResult.prototype.update_view = function (server_update) {
    var inst = this;
    server_update = server_update || false;

    // Fetch/Update adjudication status
    function update_adj_button_view() {
        if (inst.is_positive) {
            inst.adj_pos_icon.attr('src', inst.adj_pos_on_icon);
            inst.adj_neg_icon.attr('src', inst.adj_neg_off_icon);
            inst.result.addClass(inst.adj_pos_class);
            inst.result.removeClass(inst.adj_neg_class);
        }
        else if (inst.is_negative) {
            inst.adj_pos_icon.attr('src', inst.adj_pos_off_icon);
            inst.adj_neg_icon.attr('src', inst.adj_neg_on_icon);
            inst.result.removeClass(inst.adj_pos_class);
            inst.result.addClass(inst.adj_neg_class);
        }
        else {
            inst.adj_pos_icon.attr('src', inst.adj_pos_off_icon);
            inst.adj_neg_icon.attr('src', inst.adj_neg_off_icon);
            inst.result.removeClass(inst.adj_pos_class);
            inst.result.removeClass(inst.adj_neg_class);
        }
    }
    if (server_update)
    {
        $.ajax({
            url: "get_item_adjudication?uid="+this.uid,
            success: function (data)
            {
                inst.is_positive = data['is_pos'];
                inst.is_negative = data['is_neg'];

                update_adj_button_view();
            },
            error: function (jqXHR, textStatus, errorThrown)
            {
                alert_error("AJAX Error: " + errorThrown);
            }
        });
    }
    else
    {
        update_adj_button_view();
    }

    // display based on explicit settings
    function update_image()
    {
        if (inst.is_explicit)
        {
            inst.explicit_icon.attr('src', inst.explicit_icon_on);
            inst.image_data_view.attr('src', inst.explicit_overlay_image);
        }
        else
        {
            inst.explicit_icon.attr('src', inst.explicit_icon_off);
            inst.image_data_view.attr('src', inst.image_preview_data);
            // balance side scaling.
            if (inst.img_long_side)
            {
                inst.image_data_view.css('height', '192px');
            }
            else
            {
                inst.image_data_view.css('width', '192px');
            }
        }
    }
    if (this.image_loaded)
    {
        update_image();
    }
    else
    {
        // Get the preview image information from the server
        $.ajax({
            url: "get_ingest_image_preview_data?uid=" + this.uid,
            success: function (data) {
                // Check for failures
                if  (!data.success) {
                    alert_error("Image fetch error: " + data.message);
                    inst.image_preview_data = 'url("'+inst.adj_neg_on_icon+'")';
                }
                else {
                    inst.image_preview_data =
                        'data:image/'+data['ext']+';base64,'+ data['data'];
                    inst.image_loaded = true;
                    inst.is_explicit = inst.e_marked_servereide =
                        data['is_explicit'];
                    inst.img_long_side = parseInt(data['long_side'], 10);
                }
                update_image();
            },
            error: function (jqXHR, textStatus, errorThrown) {
                alert_error("Failed to load image for ID " + inst.uid + ": " +
                        errorThrown);
                inst.image.css('background-image',
                    'url("'+inst.adj_neg_on_icon+'")');
            }
        });
    }
};

/**
 * Set display to positive indication and let the server know of change.
 */
IqrResult.prototype.set_positive = function () {
    var ajax_param = '';
    //var adj_type = '';  // for alert_info below
    if (!this.is_positive) {
        // was negative or neutral, setting to positive
        this.is_positive = true;
        this.is_negative = false;

        ajax_param = "add_pos=["+this.uid+"]";
        //adj_type = "positive";  // for alert_info below
    }
    else {
        // was positive, reset to neutral
        this.is_positive = false;

        ajax_param = "remove_pos=["+this.uid+"]";
        //adj_type = 'neutral';  // for alert_info below
    }

    $.ajax({
        url: "adjudicate?" + ajax_param,
        dataType: 'json',
        success: function (data) {
            //alert_info("Marked ID "+inst.uid+" as "+adj_type);
        },
        error: function (jqXHR, textStatus, errorThrown) {
            alert_error("AJAX Error: " + errorThrown);
        }
    });

    this.update_view();
};

/**
 * Set display of negative indication and let the server know.
 */
IqrResult.prototype.set_negative = function () {
    var ajax_param = '';
    //var adj_type = '';  // for alert_info below
    if (!this.is_negative) {
        // was negative or neutral, setting to positive
        this.is_positive = false;
        this.is_negative = true;

        ajax_param = "add_neg=["+this.uid+"]";
        // adj_type = "negative";  // for alert_info below
    }
    else {
        // was positive, reset to neutral
        this.is_negative = false;

        ajax_param = "remove_neg=["+this.uid+"]";
        // adj_type = 'neutral';  // for alert_info below
    }

    var inst = this;
    $.ajax({
        url: "adjudicate?" + ajax_param,
        dataType: 'json',
        success: function (data) {
            //alert_info("Marked ID "+inst.uid+" as "+adj_type);
        },
        error: function (jqXHR, textStatus, errorThrown) {
            alert_error("AJAX Error: " + errorThrown);
        }
    });

    this.update_view();
};

/**
 * Locally turn off image explicit label and overlay.
 *
 * Alerts with an acknowledgment confirmation.
 */
IqrResult.prototype.turn_off_explicit = function () {
    var c = confirm("This image has been marked as containing explicit " +
        "content.\n\nAre you sure you would like to un-mark this image " +
        "as explicit?");
    if (c) {
        this.is_explicit = false;
        this.update_view();
    }
};

/**
 * Turn on explicit label + overlay.
 *
 * Notify server of an image being marked as explicit.
 */
IqrResult.prototype.turn_on_explicit = function () {
    this.is_explicit = true;
    this.update_view();

    if (!this.e_marked_servereide && confirm("Permanently mark this image as explicit?")) {
        // notifying server that this image has been marked as explicit so the
        // decision is reflected in future views of this image.
        var inst = this;
        $.ajax({
            url: "_mark_img_explicit?id=" + this.uid,
            success: function (data) {
                if (data['success']) {
                    inst.e_marked_servereide = true;
                }
                else {
                    alert_error("Image ID "+inst.uid+" not marked " +
                        "server-side due to error. (See log)");
                }
            },
            error: function (jqXHR, textStatus, errorThrown) {
                alert_error("AJAX Error: " + errorThrown);
            }
        });
    }
};

/**
 * Toggle the explicit state of this image.
 */
IqrResult.prototype.toggle_explicit = function () {
    if (this.is_explicit) {
        this.turn_off_explicit();
    }
    else {
        this.turn_on_explicit();
    }
};

/**
 * Open this image up in a new window
 *
 * If this image is explicit, query the user if they're sure they still want
 * to view it.
 */
IqrResult.prototype.display_full_image = function () {
    if (this.image_loaded) {
        if (this.is_explicit) {
            if (!confirm("This image has been marked as explicit\n\nAre " +
                "you sure you would still like to view the full " +
                "image?")) {
                return;
            }
        }
        // TODO: Should make this better...
        //       like open a webpage instead of just opening image data...
        window.open(this.image_preview_data);
    }
};





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
    var inst = this;
    this.show_more_step = 20;
    this.displayed_search_results = [];

    this.random_enabled = false;  // not displayed initially.
    this.random_ids = [];
    this.displayed_random_results = [];

    // TODO: Parametrize / wrap in AJAX call asking server for initial center
    //       Should be based on Server's currently represented dataset (configuration)
    // Center of Ukraine
    //this.results_map = new IqrResultMap(null, [48.5, 32.1], 6);
    // Center of Europe
    this.results_map = new IqrResultMap(null, [49.237810, 16.197492], 4);

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
    this.progress_bar_search = new SpinnyWait();
    this.progress_bar_random = new SpinnyWait();

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
    // Functions
    //

    // ### PRIVATE ###

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
    function get_initial_result_display_count() {
        var top_cluster_size = 0;
        var geofilter = '';
        if (inst.results_map.filter_bounds)
        {
            geofilter = "?geofilter="+inst.results_map.filter_bounds.toBBoxString();
        }
        $.ajax({
            async: false,
            url: "_iqr_get_positive_id_groups"+geofilter,
            success: function (data) {
                var clusters = data['clusters'];
                // looking at the ID of the first result in the first cluster
                if (clusters.length && clusters[0][0][0] == 0)
                {
                    top_cluster_size = clusters[0].length;
                }
            },
            error: function (jqXHR, textStatus, errorThrown) {
                alert_error("AJAX Error: " + errorThrown);
            }
        });
        return inst.show_more_step + top_cluster_size;
    }


    // ### PUBLIC ###

    /**
     * Construct the IQR Search pane
     */
    this.construct_search_pane = function () {
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
    this.construct_random_pane = function () {
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
        this.show_random_functionls();
    };

    /**
     * Remove the random display pane.
     */
    this.remove_random_pane = function () {
        this.random_container.remove();
    };

    /**
     * Show functional elements in the search container
     */
    this.show_search_functionals = function () {
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
    this.hide_search_functionals = function () {
        this.button_container_search_top.hide();
        this.button_container_search_bot.hide();
    };

    /**
     * Show functional elements in the random container
     */
    this.show_random_functionls = function () {
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
    this.hide_random_functionals = function () {
        this.button_random_refresh.hide();
        this.button_random_showMore.hide();
    };

    /**
     * Clear currently displayed and stored search results
     */
    this.clear_search_results = function () {
        this.results_container_search.children().remove();
        this.displayed_search_results = [];
    };

    /**
     * Clear currently displayed and stored random results.
     */
    this.clear_random_results = function () {
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
    this.display_search_results_range = function (s, e) {
        var inst = this;
        var geofilter = '';
        if (this.results_map.filter_bounds)
        {
            geofilter = "&geofilter="+this.results_map.filter_bounds.toBBoxString();
        }
        $.ajax({
            url: "_iqr_get_results_range?s="+s+"&e="+e+geofilter,
            dataType: 'json',
            success: function (data) {
                // see masir.web.mods.SearchMod.init.iqr_get_results_range for
                // results dictionary return format.
                var results = data['results'];
                var i = 0, r = null, c = null;
                for (; i < results.length; i++) {
                    c = results[i];
                    r = new IqrResult(inst.results_container_search, s+i,
                                      c[0], c[1], inst);
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
    this.display_random_results_range = function (s, e) {
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
        this.show_random_functionls();

        // Hide "show more" button if there are no more to show
        if (this.displayed_random_results.length === this.random_ids.length) {
            this.button_random_showMore.hide();
        }
    };

    /**
     * Query from the server the list of data element IDs in a random order.
     */
    this.refresh_random_ids = function () {
        this.hide_random_functionals();
        this.clear_random_results();
        this.progress_bar_random.on("Refreshing random list");

        var inst = this;
        var restore = function () {
            inst.progress_bar_random.off();
            inst.show_random_functionls();
        };

        var geofilter = '';
        if (this.results_map.filter_bounds)
        {
            geofilter = "?geofilter="+this.results_map.filter_bounds.toBBoxString();
        }

        $.ajax({
            url: "_iqr_get_random_id_order"+geofilter,
            success: function (data) {
                inst.random_ids = data['rand_ids'];
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
    this.iterate_more_search_results = function () {
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
    this.iterate_more_random_results = function () {
        var N = this.show_more_step;
        var s = this.displayed_random_results.length;
        var e = s + N;
        //alert_info("Showing more randoms in range ("+s+", "+e+"]");
        this.display_random_results_range(s, e);
    };

    /**
     * Update the state of this view with the Iqr session model server-side
     */
    this.refine = function() {
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
            url: '_iqr_refine',
            success: function (data) {
                if (data['success']) {
                    //alert_info("Iqr refine complete");
                    inst.display_search_results_range(
                        0, get_initial_result_display_count()
                    );
                    restore();
                }
                else {
                    alert_error("IQR Refine error: " + data['message']);
                    restore();
                }

                // Map out the top N results
                inst.results_map.mark_top_N();
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
    this.reset = function () {
        this.hide_search_functionals();
        this.clear_search_results();
        this.clear_random_results();
        this.results_map.clear_markers();

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
        this.progress_bar_search.on("... Removing current IQR Session ...");
        $.ajax({
            url: "_iqr_session_remove",
            success: function (data) {
                //alert_info("[RESET]: " + data['message']);
                inst.progress_bar_search.on("... initializing more IQR Session ...");
                $.ajax({
                    url: "_initialize_new_iqr_session",
                    success: function (data) {
                        //alert_info("[RESET]: " + data['message']);
                        restore();
                    },
                    error: function (jqXHR, textStatus, errorThrown) {
                        alert_error("AJAX Error: " + errorThrown);
                        restore();
                    }
                })
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
    this.toggle_random_pane = function () {
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

    // ### End functions #######################################################

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
    $(container).append(this.results_map.map_container);

    // now that the element path to the map is fully constructed, re-initialize
    this.results_map.initialize();


    // Update model to initial state
    this.display_search_results_range(0, get_initial_result_display_count());
    this.results_map.mark_top_N();

    return this;
}
