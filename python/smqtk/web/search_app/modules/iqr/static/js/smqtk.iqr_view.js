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
    this.flow_zone = $('<div>');
    this.ingest_progress_zone = $('<div>');
    this.status_zone = $('<div>');
    this.results_zone = $('<div>');

    this.button_reset_session = $('<button class="btn btn-danger" type="button"/>');

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
    container.append(
        this.flow_zone,
        this.ingest_progress_zone,
        $("<hr>"),
        this.status_zone,
        $("<hr>"),
        this.button_reset_session,
        this.results_zone
    );

    this.flow_inst = new FlowUploadZone(this.flow_zone, this.upload_post_url);
    this.status_inst = new IqrStatusView(this.status_zone);
    this.results_view_inst = null; //new IqrRefineView(this.results_zone);

    this.button_reset_session.text("Reset IQR Session");

    //
    // Control
    //
    var self = this;

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
                bar.remove();
                bar.progress_div.fadeOut('slow', function () {
                    bar.remove();
                });
                self.status_inst.update_view();
            },
            error: function(jqXHR, textStatus, errorThrown) {
                bar.on("ERROR: "+errorThrown);
                bar.stop_active("danger");
                alert_error("Error during FID upload POST: "
                            + textStatus);
            }
        });
    });

    this.button_reset_session.click(function () {
        $.ajax({
            url: "reset_iqr_session",
            success: function (data) {
                if ( data.success ) {
                    self.status_inst.update_view();
                    alert_success("IQR Session Reset");
                }
            }
        })
    });
};
