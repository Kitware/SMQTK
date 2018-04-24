/**
 * IQR Result encapsulation, exposing adjudication functions
 *
 * Image data should be loaded by an AJAX call that returns the literal data,
 * not a path.
 *
 * for image load progress status, see: http://usablica.github.io/progress.js/
 *
 * @param is_example: Boolean flag for whether or not this is a view of example
 *                    data, in contrast to data in the IQR working index.
 *
 */
function DataView(container, rank, uid, probability, saliency_flag, is_example) {
    Object.call(this);

    var inst = this;
    this.rank = rank;
    this.uid = uid;
    this.probability = probability;
    this.saliency_flag = saliency_flag;
    this.is_example = is_example === undefined ? false : is_example;

    // image ``src`` reference to use for display in an <img>.
    this.image_preview_data = null;
    // saliency map ``src`` reference to use for display in an <img>.
    this.sm_preview_data = null;
    // link to statically hosted file for full view
    this.static_view_link = null;
    // link to statically hosted file for full saliency map
    this.sm_static_view_link = null;
    // If we have received image preview data yet
    this.image_loaded = false;
    // Used for image view size clamping
    // -> this is 1 if height > width, 0 if otherwise.
    this.img_long_side = 0;

    // adjudication status
    this.is_positive = false;
    this.is_negative = false;

    // available default image stuff
    this.adj_pos_class = "result-positive";
    this.adj_pos_off_icon = "static/img/carbon-verified.png";
    this.adj_pos_on_icon = "static/img/carbon-verified_on.png";

    this.adj_neg_class = "result-negative";
    this.adj_neg_off_icon = "static/img/carbon-rejected.png";
    this.adj_neg_on_icon = "static/img/carbon-rejected_on.png";

    this.loading_gif = "static/img/loading.gif";

    //
    // View Layout
    //

    // float view container
    this.float_div = $('<div id="float_div_'+this.rank+'" style="display:none; vertical-align: top"/>');
    this.float_div.appendTo($(container));

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

    this.adjudication_controls.append(this.adj_pos_icon);
    this.adjudication_controls.append(this.adj_neg_icon);

    // image container image data and adjudication buttons
    this.image_container = $('<div class="iqr-result-img-container"/>');
    this.image_data_view = $('<img>');
    // Showing loading GIF by default until image preview actually loaded via
    // ajax call.
    this.image_data_view.attr('src', this.loading_gif);

    //display saliency map
    this.saliency_data_view = $('<img>');
    this.saliency_data_view.attr('src', this.loading_gif);


    // Assemble result box
    if (! this.is_example) {
        this.result.append(this.header);
        this.result.append(this.adjudication_controls);
    }
    this.result.append(this.image_container);

    this.image_container.append(this.image_data_view);
    if (this.saliency_flag) {
        this.image_container.append(this.saliency_data_view);
    }

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

    // link click controls
    this.image_data_view.click(function () {
        inst.display_full_image();
    });

    // link click controls
    this.saliency_data_view.click(function () {
        inst.display_full_saliencyMap();
    });



    // Update to initial view from the server's state
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
DataView.prototype.is_adjudicated = function () {
    return (this.is_negative || this.is_positive);
};

/**
 * Update the view of this element based on current state.
 */
DataView.prototype.update_view = function (server_update) {
    var inst = this;
    server_update = server_update || false;

    // Fetch/Update adjudication status
    function update_adj_button_view()
    {
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

    // helper function for display based on explicit settings
    function update_image()
    {
        inst.image_data_view.attr('src', inst.image_preview_data);
        inst.saliency_data_view.attr('src', inst.sm_preview_data);

        inst.image_data_view.mouseenter(function () {
            console.log('input image: ' + inst.sm_static_view_link);
            inst.showtrail(inst.static_view_link, 192, 192);
        });

        inst.image_data_view.mouseleave(function () {
            inst.hidetrail();
        });


        inst.saliency_data_view.mouseenter(function () {
            console.log('input image: ' + inst.sm_static_view_link);
            inst.showtrail(inst.sm_static_view_link, 192, 192);
        });

        inst.saliency_data_view.mouseleave(function () {
            inst.hidetrail();
        });


        var data_veiw_len = 192;
        if (inst.saliency_flag) {
            data_veiw_len = Math.floor(data_veiw_len / 2);
        }
        var data_veiw_len_str = data_veiw_len.toString() + 'px';

        // balance side scaling.
        if (inst.img_long_side) {
            inst.image_data_view.css('height', data_veiw_len_str);
            if (inst.saliency_flag) {
                inst.saliency_data_view.css('height', data_veiw_len_str);
            }
        }
        else
        {
            inst.image_data_view.css('width', data_veiw_len_str);
            if (inst.saliency_flag) {
                inst.saliency_data_view.css('width', data_veiw_len_str);
            }
        }
    }

    if (server_update)
    {
        var query_url = null;
        if (this.is_example) {
            query_url = "get_example_adjudication";
        }
        else {
            query_url = "get_index_adjudication";
        }

        $.ajax({
            url: query_url+"?uid="+this.uid,
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

    if (this.image_loaded)
    {
        update_image();
    }
    else
    {
        // Get the preview image information from the server
        $.ajax({
            url: "get_data_preview_image?uid=" + this.uid,
            success: function (data) {
                // Check for failures
                if  (!data.success) {
                    alert_error("Image fetch error: " + data.message);
                    inst.image_preview_data = inst.adj_neg_on_icon;
                }
                else {
                    inst.image_preview_data = data.static_preview_link;
                    inst.sm_preview_data = data.smap_preview_link;
                    inst.static_view_link = data.static_file_link;
                    inst.sm_static_view_link = data.smap_static_file_link;
                    inst.image_loaded = true;

                    inst.img_long_side =
                        parseInt(data.shape[1]) > parseInt(data.shape[0]);
                }
                update_image();
            },
            error: function (jqXHR, textStatus, errorThrown) {
                alert_error("Failed to load preview image for ID " + inst.uid
                        + ": " + errorThrown);
                inst.image_preview_data = "broken-image-src";
                update_image();
            }
        });
    }
};

/**
 * Set display to positive indication and let the server know of change.
 */
DataView.prototype.set_positive = function () {
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
DataView.prototype.set_negative = function () {
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
 * Open this image up in a new window
 */
DataView.prototype.display_full_image = function () {
    if (this.image_loaded) {
        // TODO: Should make this better...
        //       like open a webpage instead of just opening static data...
        window.open(this.static_view_link);
    }
};

/**
 * Open this image up in a new window
 */
DataView.prototype.display_full_saliencyMap = function () {
    if (this.image_loaded) {
        // TODO: Should make this better...
        //       like open a webpage instead of just opening static data...
        window.open(this.sm_static_view_link);
    }
};


var timer;

DataView.prototype.hidetrail = function (){
    var inst = this;
    inst.float_div.html("");
    $(document).unbind('mousemove');
    clearTimeout(timer);
};

DataView.prototype.showtrail = function (imagename, width, height) {
    var inst = this;
    var offsetfrommouse=[15, 15];    //image x,y offsets from cursor position in pixels. Enter 0,0 for no offset
    var defaultimageheight = 40;	// maximum image size.
    var defaultimagewidth = 40;	    // maximum image size.

    function truebody(){
        return (!window.opera && document.compatMode && document.compatMode != "BackCompat")? document.documentElement : document.body
    }

    function followmouse(e) {
        var xcoord=offsetfrommouse[0];
        var ycoord=offsetfrommouse[1];

        var docwidth=document.all? truebody().scrollLeft+ truebody().clientWidth : pageXOffset+window.innerWidth-15;
        var docheight=document.all? Math.min(truebody().scrollHeight, truebody().clientHeight) : Math.min(window.innerHeight);

        if (typeof e != "undefined"){
            if (docwidth - e.pageX < defaultimagewidth + 2*offsetfrommouse[0]){
                xcoord = e.pageX - xcoord - defaultimagewidth; // Move to the left side of the cursor
            } else {
                xcoord += e.pageX;
            }
            if (docheight - e.pageY < defaultimageheight + 2*offsetfrommouse[1]){
                ycoord += e.pageY - Math.max(0,(2*offsetfrommouse[1] + defaultimageheight + e.pageY - docheight - truebody().scrollTop));
            } else {
                ycoord += e.pageY;
            }

        } else if (typeof window.event != "undefined"){
            if (docwidth - event.clientX < defaultimagewidth + 2*offsetfrommouse[0]){
                xcoord = event.clientX + truebody().scrollLeft - xcoord - defaultimagewidth; // Move to the left side of the cursor
            } else {
                xcoord += truebody().scrollLeft+event.clientX
            }
            if (docheight - event.clientY < (defaultimageheight + 2*offsetfrommouse[1])){
                ycoord += event.clientY + truebody().scrollTop - Math.max(0,(2*offsetfrommouse[1] + defaultimageheight + event.clientY - docheight));
            } else {
                ycoord += truebody().scrollTop + event.clientY;
            }
        }
        inst.float_div.css('left', xcoord+"px");
        inst.float_div.css('top', ycoord+"px");
    }

    function show(imagename, width, height) {
        console.log('[inside show] imagename: ' + imagename);
        var docwidth=document.all? truebody().scrollLeft+truebody().clientWidth : pageXOffset+window.innerWidth - offsetfrommouse[0]
        var docheight=document.all? Math.min(truebody().scrollHeight, truebody().clientHeight) : Math.min(window.innerHeight)

        if( (navigator.userAgent.indexOf("Konqueror")==-1  || navigator.userAgent.indexOf("Firefox")!=-1 || (navigator.userAgent.indexOf("Opera")==-1 && navigator.appVersion.indexOf("MSIE")!=-1)) && (docwidth>650 && docheight>500)) {
            ( width == 0 ) ? width = defaultimagewidth: '';
            ( height == 0 ) ? height = defaultimageheight: '';

            width+=30;
            height+=55;
            defaultimageheight = height;
            defaultimagewidth = width;

            $(document).mousemove(function(e) {
                followmouse(e);
            });

            newHTML = '<div class="border_preview">';
            console.log('inst.img_long_side' + inst.img_long_side);
            if(inst.img_long_side) {
                console.log('height: ' + height);
                newHTML = newHTML + '<div class="preview_temp_load">' +
                    '<img style="height:' + height + 'px" src="' + imagename + '" border="0"></div>';
            } else {
                console.log('width: ' + width);
                newHTML = newHTML + '<div class="preview_temp_load">' +
                    '<img style="width:' + width + 'px" src="' + imagename + '" border="0"></div>';
            }
            newHTML = newHTML + '</div>';

            if(navigator.userAgent.indexOf("MSIE") != -1 && navigator.userAgent.indexOf("Opera") == -1 ){
                newHTML = newHTML+ $('<iframe src="about:blank" scrolling="no" frameborder="0" width="'+width+'" height="'+height+'"></iframe>');
            }

            inst.float_div.html(newHTML);
            inst.float_div.attr('style', "display: block");
            inst.float_div.css('z-index', 110);
            inst.float_div.css('position', "absolute");
            inst.float_div.css('vertical-align', top);

        }
    }

    console.log('image_name: ' + imagename);
    i = imagename;
    w = width;
    h = height;
    timer = setTimeout(show(i, w, h), 2000);
};
