/**
 * General purpose progress bar encapsulation
 */
function GeneralProgressBar(container, min, max) {
    Object.call(this);

    this.contained_by = $(container);
    this.min = min || 0;
    this.max = max || 100;
    this.value = 0;
    this.message = '';

    this.progress_div = $('<div class="progress"/>');
    this.progress_div.css('text-align', 'center');
    this.bar = $('<div class="progress-bar" role="progressbar"/>');
    this.bar.css({width: "100%"});
    this.bar.appendTo(this.progress_div);
    this.progress_div.appendTo(this.contained_by);

    // Initial setup
    this.update(this.value);

    return this;
}

/**
 * Hide the progress bar.
 */
GeneralProgressBar.prototype.hide = function() {
    this.progress_div.hide();
};

/**
 * show the progress bar
 */
GeneralProgressBar.prototype.show = function() {
    this.progress_div.show();
};

/**
 * Update the display of the progress bar, reflecting the current value in
 * relation to the set min and max.
 *
 * @param new_val Optional new value to the current object's value.
 */
GeneralProgressBar.prototype.update = function (new_val) {
    this.value = (typeof new_val === 'undefined') ? this.value : new_val;

    //var w = Math.floor((this.value / this.max) * 100);
    this.bar.attr('aria-valuenow', this.value);
    this.bar.attr('aria-valuemin', this.min);
    this.bar.attr('aria-valuemax', this.max);
    //this.bar.css('width', w + '%');
    this.bar.text(this.message);
};


/**
 * Animated bar that represents indeterminately ongoing progress.
 *
 * @param container Containing element
 * @param message Optional Initial message to show
 * @constructor
 */
function ActivityBar(container, message) {
    Object.call(this);

    this.message = (typeof message === 'undefined') ? "" : message;

    // jQuery div container
    this.c = $(container);
    this.color_class = null;

    this.progress_div = $('<div class="progress" style="text-align: center;"/>');
    this.progress_div.appendTo(this.c);
    this.progress_div.hide();

    this.loading_bar = $('<div class="progress-bar progress-bar-striped active" \
                               role="progressbar" \
                               aria-valuenow="100" style="width: 100%" \
                               aria-valuemin="0" aria-valuemax="100">\
                          </div>');
    this.loading_bar.text(this.message);
    this.loading_bar.appendTo(this.progress_div);

    return this;
}

/**
 * Turn on as well as update the message shown.
 * @param message Message to show.
 */
ActivityBar.prototype.on = function(message) {
    this.loading_bar.text(message || this.message);
    this.progress_div.show();
};

/**
 * Remove animation and striping from bar
 *
 * For valid color class strings, see:
 *     http://www.w3schools.com/bootstrap/bootstrap_progressbars.asp
 *
 */
ActivityBar.prototype.stop_active = function(color_class) {
    this.loading_bar.removeClass("progress-bar-striped");
    this.loading_bar.removeClass("active");
    if( color_class ) {
        this.color_class = "progress-bar-"+color_class;
        this.loading_bar.addClass(this.color_class);
    }
    else if (this.color_class) {
        this.loading_bar.removeClass(this.color_class);
    }
};

ActivityBar.prototype.off = function() {
    this.progress_div.hide();
};

ActivityBar.prototype.remove = function() {
    this.progress_div.remove();
};
