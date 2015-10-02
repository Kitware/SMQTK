/**
 * IQR Result encapsulation, exposing adjudication functions
 *
 * Image data should be loaded by an AJAX call that returns the literal data,
 * not a path.
 *
 * for image load progress status, see: http://usablica.github.io/progress.js/
 *
 * @param is_example: Boolean flag for whether or not this is a view of example
 *                    data, in contrast to data in the IQR working index.
 *
 */
function DataView(container, rank, uid, probability, is_example) {
    Object.call(this);

    var inst = this;
    this.rank = rank;
    this.uid = uid;
    this.probability = probability;
    this.is_example = is_example === undefined ? false : is_example;

    // image ``src`` reference to use for display in an <img>.
    this.image_preview_data = null;
    // link to statically hosted file for full view
    this.static_view_link = null;
    // If we have received image preview data yet
    this.image_loaded = false;
    // Used for image view size clamping
    // -> this is 1 if height > width, 0 if otherwise.
    this.img_long_side = 0;

    // adjudication status
    this.is_positive = false;
    this.is_negative = false;

    // available default image stuff
    this.adj_pos_class = "result-positive";
    this.adj_pos_off_icon = "static/img/carbon-verified.png";
    this.adj_pos_on_icon = "static/img/carbon-verified_on.png";

    this.adj_neg_class = "result-negative";
    this.adj_neg_off_icon = "static/img/carbon-rejected.png";
    this.adj_neg_on_icon = "static/img/carbon-rejected_on.png";

    this.loading_gif = "static/img/loading.gif";

    //
    // View Layout
    //

    // top-level container
    this.result = $('<div class="iqr-result"/>');
    this.result.appendTo($(container));

    // Header container textual identifiers
    this.header = $('<div/>');
    this.header.css('height', '24px');
    this.header.text("#" + (this.rank+1) + " | "
                     //+ "UID: " + this.uid + " | "
                     + (this.probability*100).toFixed(2) + "%");

    // adjudication icons / functionality
    this.adjudication_controls = $('<div class="adjudication-box"/>');
    this.adj_pos_icon = $('<img height="24px">');
    this.adj_pos_icon.css('padding-left', '12px');
    this.adj_pos_icon.attr('src', inst.adj_pos_off_icon);
    this.adj_neg_icon = $('<img height="24px">');
    this.adj_neg_icon.css('padding-left', '12px');
    this.adj_neg_icon.attr('src', inst.adj_neg_off_icon);

    this.adjudication_controls.append(this.adj_pos_icon);
    this.adjudication_controls.append(this.adj_neg_icon);

    // image container image data and adjudication buttons
    this.image_container = $('<div class="iqr-result-img-container"/>');
    this.image_data_view = $('<img>');
    // Showing loading GIF by default until image preview actually loaded via
    // ajax call.
    this.image_data_view.attr('src', this.loading_gif);

    // Assemble result box
    if (! this.is_example) {
        this.result.append(this.header);
        this.result.append(this.adjudication_controls);
    }
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

    // link click controls
    this.image_data_view.click(function () {
        inst.display_full_image();
    });

    // Update to initial view from the server's state
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
DataView.prototype.is_adjudicated = function () {
    return (this.is_negative || this.is_positive);
};

/**
 * Update the view of this element based on current state.
 */
DataView.prototype.update_view = function (server_update) {
    var inst = this;
    server_update = server_update || false;

    // Fetch/Update adjudication status
    function update_adj_button_view()
    {
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

    // helper function for display based on explicit settings
    function update_image()
    {
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

    if (server_update)
    {
        var query_url = null;
        if (this.is_example) {
            query_url = "get_example_adjudication";
        }
        else {
            query_url = "get_index_adjudication";
        }

        $.ajax({
            url: query_url+"?uid="+this.uid,
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

    if (this.image_loaded)
    {
        update_image();
    }
    else
    {
        // Get the preview image information from the server
        $.ajax({
            url: "get_data_preview_image?uid=" + this.uid,
            success: function (data) {
                // Check for failures
                if  (!data.success) {
                    alert_error("Image fetch error: " + data.message);
                    inst.image_preview_data = inst.adj_neg_on_icon;
                }
                else {
                    inst.image_preview_data = data.static_preview_link;
                    inst.static_view_link = data.static_file_link;
                    inst.image_loaded = true;

                    inst.img_long_side =
                        parseInt(data.shape[1]) > parseInt(data.shape[0]);
                }
                update_image();
            },
            error: function (jqXHR, textStatus, errorThrown) {
                alert_error("Failed to load preview image for ID " + inst.uid
                        + ": " + errorThrown);
                inst.image_preview_data = "broken-image-src";
                update_image();
            }
        });
    }
};

/**
 * Set display to positive indication and let the server know of change.
 */
DataView.prototype.set_positive = function () {
    var post_data = {};
    //var adj_type = '';  // for alert_info below
    if (!this.is_positive) {
        // was negative or neutral, setting to positive
        this.is_positive = true;
        this.is_negative = false;

        post_data.add_pos = "[\""+this.uid+"\"]";
        //adj_type = "positive";  // for alert_info below
    }
    else {
        // was positive, reset to neutral
        this.is_positive = false;

        post_data.remove_pos = "[\""+this.uid+"\"]";
        //adj_type = 'neutral';  // for alert_info below
    }

    $.ajax({
        url: "adjudicate",
        data: post_data,
        method: "POST",
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
DataView.prototype.set_negative = function () {
    var post_data = {};
    //var adj_type = '';  // for alert_info below
    if (!this.is_negative) {
        // was negative or neutral, setting to positive
        this.is_positive = false;
        this.is_negative = true;

        post_data.add_neg = '[\"'+this.uid+'\"]';
        // adj_type = "negative";  // for alert_info below
    }
    else {
        // was positive, reset to neutral
        this.is_negative = false;

        post_data.remove_neg = '[\"'+this.uid+'\"]';
        // adj_type = 'neutral';  // for alert_info below
    }

    //var inst = this;
    $.ajax({
        url: "adjudicate",
        data: post_data,
        method: "POST",
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
 * Open this image up in a new window
 */
DataView.prototype.display_full_image = function () {
    if (this.image_loaded) {
        // TODO: Should make this better...
        //       like open a webpage instead of just opening static data...
        window.open(this.static_view_link);
    }
};
