/**
 * View of IQR session stat/status, including uploaded example data and views of
 * adjudicated data.
 */
function IqrStatusView (container) {
    //
    // Members
    //

    // View components
    this.pos_AMT_zone = $('<div>');
    this.example_pos_data_zone = $('<div style="float:left;width:50%;height:300px;">');
    this.AMT_zone = $('<div style="float:right;width:50%;height:300px;">');
    this.example_neg_data_zone = $('<div>');

    this.query_target_statement = "<label class=\"likert_statement\">Query Target:</label>";
    this.query_target_dropdown = $('<select>');
    this.query_target_dropdown.attr('id', "target_list");
    this.query_target_dropdown.append( new Option("--select--", "None"));

    this.acc_statement = "<label class=\"likert_statement\">IQR Accuracy:</label>";
    this.acc_input = $('<input type="text">');

    this.likert_statement = "<label class=\"likert_statement\">" +
        "I give feedback with 100% confidence.</label>";
    this.likert_scale = "<ul class='likert'>\n" +
        "      <li>\n" +
        "        <input type=\"radio\" name=\"likert\" value=5>\n" +
        "        <label>Strongly agree</label>\n" +
        "      </li>\n" +
        "      <li>\n" +
        "        <input type=\"radio\" name=\"likert\" value=4>\n" +
        "        <label>Agree</label>\n" +
        "      </li>\n" +
        "      <li>\n" +
        "        <input type=\"radio\" name=\"likert\" value=3>\n" +
        "        <label>Neutral</label>\n" +
        "      </li>\n" +
        "      <li>\n" +
        "        <input type=\"radio\" name=\"likert\" value=2>\n" +
        "        <label>Disagree</label>\n" +
        "      </li>\n" +
        "      <li>\n" +
        "        <input type=\"radio\" name=\"likert\" value=1>\n" +
        "        <label>Strongly disagree</label>\n" +
        "      </li>\n" +
        "    </ul>";

    this.cal_acc_button = $('<button class="btn" type="button"/>');


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
    var self = this;

    this.AMT_zone.append(
        $("<span><h3>AMT Control:</h3></span>"),
        this.query_target_statement,
        this.query_target_dropdown,
        this.cal_acc_button,
        $('<br/>'),
        this.acc_statement,
        this.acc_input,
        $('<br/>'),
        this.likert_statement,
        this.likert_scale
    );

    this.cal_acc_button.text("Calculate Accuracy");
    this.cal_acc_button.click(function () {
        self.cal_acc();
    });

    this.acc_input.val('98.76%');
    this.acc_input.attr('readonly',true);

    this.pos_AMT_zone.append(
        this.example_pos_data_zone,
        this.AMT_zone
    );

    container.append(
        this.pos_AMT_zone
    );

    this.pos_AMT_zone.attr("id", "pos_AMT_zone");
    this.AMT_zone.attr("id", "AMT_zone");
    this.example_pos_data_zone.attr("id", "pos_example_zone");
    this.example_neg_data_zone.attr("id", "neg_example_zone");

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
    this.example_pos_data_zone.append(
        $("<span><h3>Query Image:</h3></span>")
    );
    for (var i=0; i < iqr_sess_state["ex_pos"].length; i++) {
        new DataView(this.example_pos_data_zone, 0, iqr_sess_state["ex_pos"][i],
                     0, false, true);
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
                     0, false, true);
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


IqrStatusView.prototype.reset_target_list = function (list) {
    var self = this;
    $("#target_list option").remove();
    self.query_target_dropdown.append( new Option("--select--", "None"));
};

IqrStatusView.prototype.update_target_list = function (list) {
    var self = this;
    self.reset_target_list();
    for (var i=0; i < list.length; i++) {
        self.query_target_dropdown.append(new Option(list[i], list[i]));
    }
};

IqrStatusView.prototype.cal_acc = function () {
    var target = $("#target_list :selected").text();

    //call cal_acc
};