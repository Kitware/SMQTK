/**
 * Alert message functions
 */

function alert_error(message) {
    var c = $('#alert_placeholder');
    var alert = $(
       '<div id="alert_banner_error" \
             class="ui-widget ui-state-error ui-corner-all" \
             style="padding: 0 .7em;"> \
            <span class="ui-icon ui-icon-alert" \
                  style="float: left; margin-right: .3em;"></span> \
            <strong>Alert:</strong> \
                <span style="white-space: pre-wrap">' + message + '</span>\
            <a class="close" data-dismiss="alert">x</a>\
        </div>'
    );
    alert.hide();
    c.append(alert);
    alert.fadeIn('fast');

    //// message timeout
    //// TODO: add a little "loading bar" to show timeout
    //setTimeout(function() {
    //    alert.fadeOut();
    //}, 3000)
}


function alert_info(message) {
    var c = $('#alert_placeholder');
    var alert = $(
       '<div id="alert_banner_info" \
             class="ui-widget ui-state-highlight ui-corner-all"\
             style="padding: 0 .7em;"> \
                <span class="ui-icon ui-icon-info" \
                      style="float: left; margin-right: .3em;"></span>\
                <strong>Info:</strong> \
                <span style="white-space: pre-wrap">' + message + '</span>\
                <a class="close" data-dismiss="alert">x</a>\
        </div>'
    );
    alert.hide();
    c.append(alert);
    alert.fadeIn('fast');

    //// message timeout
    //setTimeout(function() {
    //    alert.fadeOut();
    //}, 3000)
}


function alert_success(message) {
    var c = $('#alert_placeholder');
    var alert = $(
       '<div id="alert_banner_success" \
             class="ui-widget alert-success ui-corner-all"\
             style="padding: 0 .7em; border: 1px solid #3c763d"> \
                <span class="ui-icon ui-icon-check" \
                      style="float: left; margin-right: .3em;"></span>\
                <strong>Success:</strong> \
                <span style="white-space: pre-wrap">' + message + '</span>\
                <a class="close" data-dismiss="alert">x</a>\
        </div>'
    );
    alert.hide();
    c.append(alert);
    alert.fadeIn('fast');

    // message timeout
    setTimeout(function() {
        alert.fadeOut();
    }, 3000)
}