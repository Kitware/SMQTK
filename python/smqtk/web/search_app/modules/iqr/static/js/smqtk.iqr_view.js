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
    // Instance of IqrResultsView being used
    this.results_view_inst = null;

    // View elements
    this.flow_zone = $('<div>');
    this.status_zone = $('<div>');
    this.results_zone = $('<div>');

    //
    // Setup
    //
    this.construct_view(container);

    return this;
}

/**
 * Build up the view.
 */
IqrView.prototype.construct_view = function (container) {
    container.append(
        this.flow_zone,
        $("<hr>"),
        this.status_zone,
        $("<hr>"),
        this.results_zone
    );

    this.flow_inst = new FlowUploadZone(this.flow_zone, this.upload_post_url);
    this.results_view_inst = null; //new IqrRefineView(this.results_zone);

    //
    // Control
    //

    // Add uploaded content to current session online ingest
    this.flow_inst.onFileSuccess(function(file) {
        var fname = file.name;
        var fid = file.uniqueIdentifier;

        var message_prefix = "Ingesting file ["+fname+"]: ";
        var bar = new ActivityBar($("#ingest_progress"),
                                  message_prefix+"Ingesting...");
        bar.on();

        $.ajax({
            url: "iqr_ingest_file",
            type: 'POST',
            data: {
                fid: fid
            },
            success: function(data) {
                bar.on(message_prefix+data);
                bar.stop_active("success");
                bar.remove();
                // TODO: Call status zone method to add data UUID reference
            },
            error: function(jqXHR, textStatus, errorThrown) {
                bar.on("ERROR: "+errorThrown);
                bar.stop_active("danger");
                alert_error("Error during FID upload POST: "
                            + textStatus);
            }
        });
    });
};
