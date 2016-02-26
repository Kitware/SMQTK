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

    this.upload_post_url = upload_post_url;

    // Instance of FlowUploadZone being used.
    this.flow_inst = null;
    // Instance of IqrStatusView being used.
    this.status_inst = null;
    // Instance of IqrResultsView being used.
    this.results_view_inst = null;

    // View elements
    // - contains flow.js controls
    this.flow_zone = $('<div>');
    // - contains temporary ingest progress bars (descriptor computation)
    this.ingest_progress_zone = $('<div>');
    // - contains session status view + controls
    this.status_zone = $('<div>');
    // -- contains IQR initialization/reset controls
    this.control_zone = $('<div>');
    // -- contains IQR refinement results + controls
    this.results_zone = $('<div>');

    this.button_index_initialize = $('<button class="btn btn-primary" type="button"/>');
    this.button_reset_session = $('<button class="btn btn-danger" type="button"/>');
    this.button_state_save = $('<button class="btn" type="button"/>');

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

    container.append(
        this.flow_zone,
        this.ingest_progress_zone,
        $("<hr>"),
        this.status_zone,
        $("<hr>"),
        this.control_zone,
        this.results_zone
    );

    this.control_zone.append(
        this.button_index_initialize,
        this.button_reset_session,
        this.button_state_save
    );

    this.flow_inst = new FlowUploadZone(this.flow_zone, this.upload_post_url);
    this.status_inst = new IqrStatusView(this.status_zone);
    this.results_view_inst = new IqrRefineView(this.results_zone);

    this.button_index_initialize.text("Initialize Index");
    this.button_reset_session.text("Reset IQR Session");
    this.button_state_save.text("Save IQR state");
    this.button_state_save.addClass('pull-right');

    //
    // Control
    //
    // Add uploaded content to current session online ingest
    this.flow_inst.onFileSuccess(function(file) {
        var fname = file.name;
        var fid = file.uniqueIdentifier;

        var message_prefix = "Ingesting file ["+fname+"]: ";
        var bar = new ActivityBar(self.ingest_progress_zone,
                                  message_prefix+"Ingesting...");
        bar.on();

        $.ajax({
            url: "iqr_ingest_file",
            type: 'POST',
            data: {
                fid: fid
            },
            success: function(data) {
                bar.on(message_prefix+"Complete");
                bar.stop_active("success");
                bar.progress_div.fadeOut('slow', function () {
                    bar.remove();
                });
                self.status_inst.update_view();
            },
            error: function(jqXHR, textStatus, errorThrown) {
                bar.on("ERROR: "+errorThrown);
                bar.stop_active("danger");
                alert_error("Error during file upload: "
                            + jqXHR.responseText);
            }
        });
    });

    this.button_index_initialize.click(function () {
        self.initialize_index();
    });

    this.button_reset_session.click(function () {
        self.reset_session();
    });

    this.button_state_save.attr({
        onclick: "location.href='get_iqr_state'"
    })
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
                self.results_view_inst.update_refine_pane();
                if (self.results_view_inst.random_enabled) {
                    self.results_view_inst.toggle_random_pane();
                }
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
    var prog_bar = new ActivityBar(this.control_zone, "Initializing IQR Index");
    prog_bar.on();

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
                self.results_view_inst.iqr_refine();
            }
            else {
                remove_prog_bar("Initialization Failure", "danger");
                alert_error("Error occurred initializing new index: "+data.message);
            }
        }
    })
};
