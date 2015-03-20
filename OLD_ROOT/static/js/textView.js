//
// LICENCE
// -------
// Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
// KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
// Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
//
var textView = function(elem, data) {
    $(elem).html('');

    var element = elem,
        bins = {},
        i = 0;

    // Create bins based on the group
    data.forEach(function(row) {
        if (typeof row['group'] === 'undefined') {
            row['group'] = 'anonymous';
        }

        if (typeof bins[row['group']] === 'undefined') {
            bins[row['group']] = [];
        }

        // A evidence can have multiple timestamps
        var timestamps = [];
        row['top_timestamps'].forEach(function(instance) {
            timestamps.push(instance.timestamp);
        });

        bins[row['group']].push({
            'name':row['name'],
            'timestamp': timestamps[0],
            'timestamps':timestamps,
            'duration':row['duration'],
        });
    });

    // Now append each group attributes to the HTML
    for (group in bins) {
        if (bins.hasOwnProperty(group)) {
            var newDiv = $(document.createElement('div'));
            // newDiv.addClass('clipAttributes');
            newDiv.addClass('row-fluid evidence-group');
            $(elem).append(newDiv);
            var h4 = $(document.createElement('h4'));
            h4.text('Top Evidence (' + replaceUnderscores(group) + ')');
            newDiv.append(h4);
            var ul = $(document.createElement('ul'));
            newDiv.append(ul);
            for (i = 0; i < bins[group].length; ++i) {
                var label = $(document.createElement('label'));
                label.addClass('clipEvidence');
                label.text(replaceUnderscores(bins[group][i].name));
                label.attr('timestamp', bins[group][i].timestamp);
                label.attr('timestamps', bins[group][i].timestamps.toString());
                label.attr('duration', bins[group][i].duration);
                var li = $(document.createElement('li'));
                li.append(label);
                ul.append(li);
            } // endfor
        } // endif
    } // endfor

    function clearAllSelections() {
        jQuery(element).find('label').each(function() {
            $(this).removeClass('selected');
            $(this).removeClass('selectedStandalone');
        });
    }

    // Public API
    return {
        selectElement : function(elem) {
            clearAllSelections();

            if (!elem) {
                return;
            }

            var labelObj = $(elem);
            labelObj.addClass('selected');

        },
        selectStandalone : function(name) {
            clearAllSelections();
            if (!name) {
                return;
            }

            jQuery(element).find('label').each(function() {
                if (addUnderscores($(this).text()) === name) {
                    $(this).addClass('selectedStandalone');
                    return false;
                }
            });
        }
    };
};
