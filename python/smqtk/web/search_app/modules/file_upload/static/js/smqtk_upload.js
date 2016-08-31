/**
 * Container for file progress information
 *
 * @param container The container this progress indicator shall be nested under
 * @param flow_file FlowFile object this indicator is attached to
 *
 * @constructor
 */
function FileProgressIndicator(container, flow_file) {
    // The element that's containing our indicator (probably a div)
    this.c = $(container);
    // associated FlowFile object
    this.flow_file = flow_file;
    var self = this;

    this.progress_c = $('<div id="file-' + flow_file.uniqueIdentifier + '"/>');
    this.progress_c.appendTo(this.c);

    // content before the loading bar
    this.header = $('<span>'+self.flow_file.name+'</span>');
    this.header.appendTo(this.progress_c);

    this.remove_icon = $('<img width=12px height=12px ' +
                          'src="/static/img/cancel.png">');
    this.remove_icon.click(function(event) {
        self.remove();
    });
    this.remove_icon.prependTo(this.header);
    this.remove_icon.hide();

    this.progress_bar = $('<div class="progress-bar progress-bar-success active" \
                                role="progressbar" \
                                aria-valuenow="0" \
                                aria-valuemin="0" aria-valuemax="100">\
                               0%\
                           </div>');
    this.progress_c.append($('<div class="progress"/>')
                   .append(this.progress_bar));

    return this;
}

FileProgressIndicator.prototype = {
    update: function() {
        var p_done = Math.floor(this.flow_file.progress() * 100.0);
        this.progress_bar.css("width", p_done + "%");
        this.progress_bar.attr("aria-valuenow", p_done);
        this.progress_bar.text(p_done + "%");

        if (this.flow_file.isComplete()) {
            //this.progress_bar.progressbar("value", 100);
            this.remove_icon.show();
        }
    },

    remove: function() {
        // cancel upload if still in progress
        if (!this.flow_file.isComplete()) {
            this.flow_file.cancel();
            alert_error("Upload of file '" + this.flow_file.name
                + "' was canceled.");
        }

        this.progress_c.remove();
    }
};

/**
 * Encapsulation of the Flow.js drop-zone for file upload to a URL target (POST)
 *
 * @param container The container this encapsulation should nest its element
 *      structure under.
 * @param upload_url The URL to target for uploads. Take into consideration the
 *      FileUploadMod blueprint prefix and target function string.
 *
 * @constructor
 */
function FlowUploadZone(container, upload_url) {
    var flow = new Flow({
        target: upload_url,
        // 1 for serial debugging, may be higher for production
        simultaneousUploads: 4
    });

    // Notify if Flow is not supported
    if (!flow.support)
    {
        alert_error("Flow is not supported in current browser");
        return;
    }

    // Association from FlowFile ID to FileProgressIndicator for that
    // object. -- {fid -> fpi}
    var ffid_fpi_map = {};

    // Hook placeholders for upload stages
    var hook_fileSuccess = null;

    // GUI Components
    this.flow_drop = $("<div/>");
    this.flow_drop.addClass('flow-drop');
    this.flow_drop.attr({
        ondragenter: "jQuery(this).addClass('flow-dragover');",
        ondragend:   "jQuery(this).removeClass('flow-dragover');",
        ondrop:      "jQuery(this).removeClass('flow-dragover');"
    });
    this.flow_drop.text("To see the search, drag & drop files here or ");

    this.flow_browse = $("<a/>");
    this.flow_browse.addClass('flow-browse');
    this.flow_browse.text("select one or more files");

    this.flow_progress = $("<div/>");
    this.flow_progress.addClass("flow-progress");

    $(container).append(this.flow_drop);
    this.flow_drop.append(this.flow_browse);
    $(container).append(this.flow_progress);

    // Flow hookups to GUI components
    // Assign flow drop and browse locations
    flow.assignDrop(this.flow_drop[0]);
    ////noinspection JSCheckFunctionSignatures
    //flow.assignBrowse($(".flow-browse-folder")[0], false, false);
    //noinspection JSCheckFunctionSignatures
    flow.assignBrowse(this.flow_browse[0], false, false);

    // Error message forwarding
    flow.on("error", function(message, file) {
        alert_error("[Flow] ERROR: (" + file.name + ") " + message);
        return true;
    });

    flow.on("fileAdded", function(file, event){
        // TODO: add rejection rules via hook functions?
        ffid_fpi_map[file.uniqueIdentifier] =
                new FileProgressIndicator($('.flow-progress'), file);
    });

    flow.on("filesSubmitted", function(array, event) {
        flow.upload();
    });

    flow.on("fileProgress", function (file) {
        // Update file's associated progress indicator element
        ffid_fpi_map[file.uniqueIdentifier].update();
    });

    flow.on("fileSuccess", function (file, message) {
        alert_success("[Flow] Success uploading file '" + file.name + ": "
                      + message);
        hook_fileSuccess && hook_fileSuccess(file);

        // Fade out the bar when done.
        var fpi = ffid_fpi_map[file.uniqueIdentifier];
        fpi.progress_c.fadeOut('slow', function () {
            fpi.progress_c.remove();
        });

        delete ffid_fpi_map[file.uniqueIdentifier];
    });

    flow.on("fileError", function(file, message) {
        alert_error("[Flow] Error uploading file '" + file.name + ": "
                + message);
        delete ffid_fpi_map[file.uniqueIdentifier];
    });

    //
    // Functions
    //

    /**
     * Add a hook function for when a file is successfully uploaded
     *
     * This is usually set to a function that contains an ajax call that tells
     * the server that a particular file ID has completed upload and should be
     * fetched from the uploader module on the python side.
     *
     * @param func Function taking one argument that is the FlowFile instance.
     */
    this.onFileSuccess = function( func ) {
        hook_fileSuccess = func;
    }
}
