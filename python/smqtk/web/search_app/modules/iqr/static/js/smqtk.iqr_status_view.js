/**
 * View of IQR session stat/status, including uploaded example data and views of
 * adjudicated data.
 */
function IqrStatusView (container) {
    //
    // Members
    //

    // View components
    this.example_pos_data_zone = $('<div>');
    this.example_neg_data_zone = $('<div>');
    this.buttons_bottom = $('<div>');

    this.button_view_pos = $('<button class="btn btn-primary" type="button"/>');
    this.button_view_neg = $('<button class="btn btn-primary" type="button"/>');

    //
    // Setup
    //
    this.construct_view(container);

    return this;
}

/**
 * Build up the view.
 *
 * @param container: Parent container element to append view elements to.
 */
IqrStatusView.prototype.construct_view = function (container) {
    container.append(
        $("<span><h3>Positive Examples:</h3></span>")
        ,this.example_pos_data_zone
        //,$("<span><h3>Negative Examples:</h3></span>")
        //,this.example_neg_data_zone
        //,$("<span><h4>View Index Adjudications:</h4></span>")
        //,this.buttons_bottom
    );

    this.buttons_bottom.append(
        this.button_view_pos,
        this.button_view_neg
    );

    this.example_pos_data_zone.attr("id", "pos_example_zone");
    this.example_neg_data_zone.attr("id", "neg_example_zone");

    this.button_view_pos.text("Positive");
    this.button_view_neg.text("Negative");

    //
    // Control
    //
    // TODO
    this.button_view_pos.click(function () {
        alert("TODO: toggle on pos adj sub-window");
    });
    this.button_view_neg.click(function () {
        alert("TODO: toggle on neg adj sub-window");
    });

    // of course, update what we just constructed
    this.update_view();
};

/**
 * Update content shown in positive example view based on server state
 *
 * @param iqr_sess_state: Object encapsulating IQR session state information.
 *                        This should be the JSON object returned from a call to
 *                        "/iqr_session_info".
 */
IqrStatusView.prototype.update_pos_zone_content = function (iqr_sess_state) {
    // clear current window, reconstruct views for received UUIDs
    this.example_pos_data_zone.children().remove();
    for (var i=0; i < iqr_sess_state["ex_pos"].length; i++) {
        new DataView(this.example_pos_data_zone, 0, iqr_sess_state["ex_pos"][i],
                     0, true);
    }
};

/**
 * Update content shown in negative example view based on server state
 *
 * @param iqr_sess_state: Object encapsulating IQR session state information.
 *                        This should be the JSON object returned from a call to
 *                        "/iqr_session_info".
 */
IqrStatusView.prototype.update_neg_zone_content = function (iqr_sess_state) {
    // clear current window, reconstruct views for received UUIDs
    this.example_neg_data_zone.children().remove();
    for (var i=0; i < iqr_sess_state["ex_neg"].length; i++) {
        new DataView(this.example_neg_data_zone, 0, iqr_sess_state["ex_neg"][i],
                     0, true);
    }
};

/**
 * Update State view based on IQR session state
 */
IqrStatusView.prototype.update_view = function () {
    var self = this;
    $.ajax({
        url: "iqr_session_info",
        method: "GET",
        dataType: "json",
        success: function (data) {
            self.update_pos_zone_content(data);
            self.update_neg_zone_content(data);
        },
        error: function (jqXHR, textStatus, errorThrown) {
            alert_error("Error fetching IQR session status information: " +
                "("+errorThrown+") " + textStatus);
        }
    });
};
