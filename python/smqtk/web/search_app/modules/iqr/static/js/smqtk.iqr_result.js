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
    // this is 1 if height > width, 0 if otherwise
    this.img_long_side = 0;
    this.e_marked_servereide = false;

    this.is_positive = false;
    this.is_negative = false;

    // available default image stuff
    this.adj_pos_class = "result-positive";
    this.adj_pos_off_icon = "static/img/carbon-verified.png";
    this.adj_pos_on_icon = "static/img/carbon-verified_on.png";

    this.adj_neg_class = "result-negative";
    this.adj_neg_off_icon = "static/img/carbon-rejected.png";
    this.adj_neg_on_icon = "static/img/carbon-rejected_on.png";

    this.explicit_icon_on = "static/img/explicit_marker_on.png";
    this.explicit_icon_off = "static/img/explicit_marker_off.png";
    this.explicit_overlay_image = "static/img/explicit_overlay.png";

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

        if (inst.is_explicit)
        {
            inst.explicit_icon.attr('src', inst.explicit_icon_on);
            inst.image_data_view.addClass("explicit-image");
        }
        else
        {
            inst.explicit_icon.attr('src', inst.explicit_icon_off);
            inst.image_data_view.removeClass("explicit-image")
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
                        'data:image/'+data.ext+';base64,'+ data.data;
                    inst.image_loaded = true;
                    inst.is_explicit = inst.e_marked_servereide =
                        data.is_explicit;

                    inst.img_long_side =
                        parseInt(data.shape[1]) > parseInt(data.shape[0]);
                }
                update_image();
            },
            error: function (jqXHR, textStatus, errorThrown) {
                alert_error("Failed to load preview image for ID " + inst.uid
                        + ": " + errorThrown);
                //inst.image_preview_data = inst.adj_neg_on_icon;
                inst.image_preview_data = "broken-image";
                update_image();
            }
        });
    }
};

/**
 * Set display to positive indication and let the server know of change.
 */
IqrResult.prototype.set_positive = function () {
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
IqrResult.prototype.set_negative = function () {
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
            url: "mark_uid_explicit",
            method: "POST",
            data: {
                uid: this.uid
            },
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
