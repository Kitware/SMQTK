/**
 * High level IQR interview view controller.
 *
 * @param container: JQuery reference to the element that will contain this view
 * @param upload_post_url: URL to POST file uploads to via Flow.js
 *
 * @constructor
 */
function IqrView(container, upload_post_url) {
    //
    // Members
    //
    this.container = container;
    this.upload_post_url = upload_post_url;

    // Instance of FlowUploadZone being used.
    this.flow_inst = null;
    // Instance of IqrStatusView being used.
    this.status_inst = null;
    // Instance of IqrResultsView being used.
    this.results_view_inst = null;

    this.query_uuids = [];

    // View elements
    // - contains flow.js controls
    this.flow_zone = $('<div>');

    this.AMT_label = "<label>AMT ID</label>";
    this.AMT_id = $('<input type="text" class="form-control" placeholder="Please paste your AMT ID">');
    this.AMT_id.attr('id', "AMT_ID");
    this.AMT_id.attr('size', 60);
    this.AMT_ins = "<small class=\"form-text text-muted\">" +
        "<a href=\"https://blog.mturk.com/get-to-know-the-new-worker-site-4a69967d90c3\">" +
        "where to find AMT ID?</a></small>";

    this.uuid_label = "<label>HIT ID</label>";
    this.uuid_input = $('<input type="text" class="form-control" placeholder="Please paste the assigned HIT ID">');
    this.uuid_input.attr('id', 'uuid_input');
    this.uuid_input.attr('size', 60);
    this.button_uuid_fetch = $('<button class="btn btn-primary" type="button"/>');

    // - contains temporary ingest progress bars (descriptor computation)
    this.ingest_progress_zone = $('<div>');
    this.ingest_progress_zone.attr('id', 'ingest_progress_zone');
    // - contains session status view + controls
    this.status_zone = $('<div>');
    this.status_zone.attr('id', 'status_zone');
    // -- contains IQR initialization/reset controls
    this.control_zone = $('<div>');
    this.control_zone.attr('id', 'control_zone');
    // -- contains IQR refinement results + controls
    this.results_zone = $('<div>');
    this.results_zone.attr('id', 'result_zone');

    this.button_index_initialize = $('<button class="btn btn-primary" type="button"/>');
    this.button_reset_session = $('<button class="btn btn-danger" type="button"/>');
    // this.button_state_save = $('<button class="btn" type="button"/>');

    this.hr = $('<hr>');

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
IqrView.prototype.construct_view = function (container) {
    var self = this;

    this.flow_zone.append(
        this.AMT_label,
        this.AMT_id,
        this.AMT_ins,
        $('<br/>'),
        $('<br/>'),
        this.uuid_label,
        this.uuid_input,
        $('<br/>'),
        this.button_uuid_fetch
    );

    this.status_zone.append(this.hr);
    this.control_zone.append(this.hr);
    container.append(
        this.flow_zone,
        this.ingest_progress_zone,
        this.status_zone,
        this.control_zone,
        this.results_zone
    );

    this.control_zone.append(
        this.button_index_initialize,
        this.button_reset_session
        // this.button_state_save
    );

    // this.flow_inst = new FlowUploadZone(this.flow_zone, this.upload_post_url);
    self.status_zone.addClass("sticky");
    this.results_view_inst = new IqrRefineView(this.results_zone);
    this.status_inst = new IqrStatusView(this.status_zone, this.results_view_inst);
    this.results_view_inst.status_inst = this.status_inst;
    this.results_view_inst.iqr_view_inst = this;

    this.button_index_initialize.text("Run Query");
    this.button_reset_session.text("Choose a New Query");
    // this.button_state_save.text("Save IQR state");
    // this.button_state_save.addClass('pull-right');
    this.button_uuid_fetch.text("Fetch the query task");

    //
    // Control
    //
    // self.show_query_image();

    this.button_index_initialize.click(function () {
        self.initialize_index();
    });

    this.button_reset_session.click(function () {
        self.reset_session();
    });

    // this.button_state_save.attr({
    //     onclick: "location.href='get_iqr_state'"
    // });

    this.button_uuid_fetch.click(function() {
        var AMT_id = self.AMT_id.val().trim();
        var input_str = self.uuid_input.val().trim();
        self.uuid_input.val('');
        self.show_query_image(AMT_id, input_str);
    });

    this.status_zone.hide();
    this.control_zone.hide();
    this.results_zone.hide();
};


IqrView.prototype.show_query_image = function (AMT_id, input_str) {
    // clear children of results container
    // get ordered results information
    // display first X results
    var self = this;

    self.flow_zone.children().remove();
    this.flow_zone.append($("<span><h3>Select one of the query images by click it</h3></span>"));

    query_uuids = [];
    query_catNMs = [];

    // Fetch query image + display
    $.ajax({
        // url: "iqr_ordered_results",
        url: "fetch_query_imgs",
        method: "GET",
        data: {
            AMT_id: AMT_id,
            input_str: input_str
        },
        success: function (data) {
            // Update refinement result uuid/score arrays
            for (var i=0; i < data["results"].length; i++) {
                query_uuids.push(data["results"][i][0]);
                query_catNMs.push(data["results"][i][1]);
            }

            self.results_view_inst.button_saliency_flag=data["sa_flag"];
            self.results_view_inst.saliency_control();

            // create/show top N results
            // - If no results, display text verification of no results
            if (query_uuids.length === 0) {
                self.flow_zone.append(
                    $("<span>No Query Image</span>")
                );
            }
            else {
                var displayed = 0;

                while (displayed < query_uuids.length)
                {
                    new QueryView(self.flow_zone,
                                  self.ingest_progress_zone,
                                  self.status_inst,
                                  query_uuids[displayed],
                                  query_catNMs[displayed]);
                    displayed++;
                }
            }
        },
        error: function (jqXHR, textStatus, errorThrown) {
            alert_error("Error fetching query images: " +
                "("+errorThrown+") " + textStatus);
        }
    });
};

/**
 * Reset IQR Session
 *
 * This clears uploaded examples as well as any initialized working index.
 */
IqrView.prototype.reset_session = function() {
    var self = this;
    $.ajax({
        url: "reset_iqr_session",
        success: function (data) {
            if ( data.success ) {
                self.status_inst.update_view();
                // self.status_inst.acc_statement.text("IQR Round 0---Top-20 Accuracy: ");
                // self.status_inst.acc_input.val('');
                self.results_view_inst.saliency_flag = false;

                $('#liker_form input[name=likert]').prop('checked', false);

                self.results_view_inst.button_gt_flag = false;
                self.results_view_inst.gt_control();

                self.results_view_inst.update_refine_pane();
                self.results_view_inst.IQR_round = -1;
                if (self.results_view_inst.random_enabled) {
                    self.results_view_inst.toggle_random_pane();
                }

                // self.container.children().remove();
                self.flow_zone.children().remove();
                self.status_zone.children().remove();
                self.control_zone.children().remove();
                self.results_zone.children().remove();
                self.construct_view(self.container);

                self.flow_zone.slideDown();

                self.status_zone.hide();
                self.control_zone.hide();
                self.results_zone.hide();

                self.status_inst.reset_target_list();
                alert_success("IQR Session Reset");
            }
        }
    });
};

/**
 * Initialize new IQR index.
 *
 * - clears existing results view
 * - query server for initialization
 * - call IqrRefineView.refine for initial view
 *
 */
IqrView.prototype.initialize_index = function () {

    if ($("#target_list :selected").val() === 'None') {
        alert("Please select retrival target!");
    } else {
        var prog_bar = new ActivityBar(this.control_zone, "Initializing IQR Index");
        prog_bar.on();
        $('#result_zone').slideDown();

        function remove_prog_bar(message, color_class) {
            prog_bar.on(message);
            prog_bar.stop_active(color_class);
            prog_bar.progress_div.fadeOut('slow', function () {
                prog_bar.remove();
            });
        }

        var self = this;
        $.ajax({
            url: "iqr_initialize",
            method: "POST",
            success: function (data) {
                if (data.success) {
                    remove_prog_bar("Initialization Complete", "success");
                    self.results_view_inst.iqr_refine(false);
                }
                else {
                    remove_prog_bar("Initialization Failure", "danger");
                    alert_error("Error occurred initializing new index: " + data.message);
                }
            }
        })
    }
};
