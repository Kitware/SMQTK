/**
 * View of IQR session stat/status, including uploaded example data and views of
 * adjudicated data.
 */
function IqrStatusView (container, result_inst) {
    //
    // Members
    //

    this.result_inst = result_inst;
    // View components
    this.pos_AMT_zone = $('<div>');
    this.example_pos_data_zone = $('<div style="float:left;width:50%;height:350px;">');
    this.AMT_zone = $('<div style="float:right;width:50%;height:350px;">');
    this.example_neg_data_zone = $('<div>');

    this.query_target_statement = "<label class=\"likert_statement\">Query Target:</label>";
    this.query_target_dropdown = $('<select>');
    this.query_target_dropdown.attr('id', "target_list");
    this.query_target_dropdown.append( new Option("--select--", "None"));

    this.acc_statement = $('<lable>');
    this.acc_statement.attr('class', 'likert_statement');
    this.acc_statement.attr('id', 'acc_stat');
    this.acc_statement.text("IQR Round 0---Top-20 Accuracy: ");

    this.acc_input = $('<input type="text">');
    this.acc_input.attr('id', 'acc_input');

    this.likert_statement = "<label class=\"likert_statement\">" +
        "I give feedback with 100% confidence.</label>";
    this.likert_form = $('<form>');
    this.likert_form.attr('id', 'liker_form');
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

    this.re_acc;


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

    this.likert_form.append(this.likert_scale);
    this.AMT_zone.append(
        $("<span><h4>AMT Control:</h4></span>"),
        // this.query_target_statement,
        // this.query_target_dropdown,
        this.acc_statement,
        this.acc_input,
        $('<br/>'),
        this.likert_statement,
        this.likert_form
    );

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
        $("<span><h4>Query Image:</h4></span>"),
        this.query_target_statement,
        this.query_target_dropdown,
        $('<br/>')
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
    this.acc_statement.text("IQR Round 0---Top-20 Accuracy: ");
    this.acc_input.val('');
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


IqrStatusView.prototype.reset_target_list = function () {
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

IqrStatusView.prototype.cal_acc_store = function (pos_num, neg_num) {
    var self = this;
    var target = $("#target_list :selected").text();
    self.query_target_dropdown.attr('disabled', 'disabled');

    $.ajax({
        url: "cal_retrival_acc?target="+target,

        success: function (data)
        {
            self.re_acc = data['acc'];
            self.acc_input.val(self.re_acc + "%");
            self.store_input(pos_num, neg_num);


        },
        error: function (jqXHR, textStatus, errorThrown)
        {
            alert_error("AJAX Error: " + errorThrown);
        }
    });

};

IqrStatusView.prototype.store_input = function (pos_num, neg_num) {
    var self = this;

    //retrival target
    var target = $("#target_list :selected").text();
    //iqr round number
    var iqr_round = self.result_inst.IQR_round;

    //confidence
    var likert_score = $('#liker_form input[name=likert]:checked').val();


    alert_info("Iqr round: " + iqr_round + "target: " + target + " acc: " + self.re_acc +
        "pos_num: "+ pos_num + "----neg_num: " + neg_num + "likert_score: " + likert_score);

    // $.ajax({
    //     url: "record_AMT_input",
    //     type: 'GET',
    //     data: {
    //         target: target,
    //         iqr_round: iqr_round,
    //         acc: self.re_acc,
    //         likert_score: likert_score
    //     },
    //     success: function (data)
    //     {
    //         acc = data['acc'];
    //         self.acc_input.val(acc + "%");
    //
    //     },
    //     error: function (jqXHR, textStatus, errorThrown)
    //     {
    //         alert_error("AJAX Error: " + errorThrown);
    //     }
    // });
    //call cal_acc_store

};