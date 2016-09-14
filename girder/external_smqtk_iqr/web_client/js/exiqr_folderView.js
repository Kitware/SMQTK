

girder.views.exiqrFolderWidget = girder.View.extend({

    initialize: function (settings) {
        this.folderModel = settings.folderModel;
        this.folderModel.on('change:meta', function () {
            this.render();
        }, this);
        this.render();
    },

    render: function () {
        var folder_meta = this.folderModel.get('meta');

        // Must have both metadata fields to be considered an IQR-able folder
        if( folder_meta['smqtk_iqr'] !== undefined &&
            folder_meta['smqtk_iqr_root'] !== undefined )
        {
            // TODO: Check if IQR root URL is accessible (try GET on root)
            // TODO: Check if IQR instance for this folder already exists

            this.$el.html(girder.templates.exiqr_folderView());
        }
        else
        {
            // clear the GUI content
            this.$el.html("");
        }

        return this;
    },

    events: {
        "click .g-exiqrFolderView-header a.g-exiqr-link": function (event) {
            // Open SMQTK IQR-lite in a new window/tab with the nested
            // configuration.  Should only get to this point the parent folder
            // has a "configuration", but it may not be valid.  IQR site should
            // do validation?
            var folder_meta = this.folderModel.get('meta'),
                iqr_config = folder_meta['smqtk_iqr'],
                iqr_root = folder_meta['smqtk_iqr_root'];

            // Call to initialize IQR config + open in new window
            $.ajax({
                url: iqr_root+"/",
                method: "POST",
                data: {
                    prefix: this.folderModel.id,
                    config: JSON.stringify(iqr_config),

                    girder_token: girder.currentToken,
                    girder_origin: window.location.origin,
                    girder_apiRoot: girder.apiRoot
                },
                dataType: 'json',
                success: function (data, textStatus, jqXHR) {

                    window.open(data['url']
                        + "?girder_token="+girder.currentToken
                        + "&girder_origin="+window.location.origin
                        + "&girder_apiRoot="+girder.apiRoot
                    );
                },
                error: function (jqXHR, textStatus, errorThrown) {
                    girder.events.trigger('g:alert', {
                        text: 'Error initializing IQR state for current folder'
                            + " :: " + errorThrown
                            + " :: " + textStatus,
                        type: 'warning',
                        icon: 'info'
                    });
                }
            });
            girder.events.trigger('g:alert', {
                text: 'Initializing IQR state for this folder...',
                type: 'info',
                icon: 'info'
            });
        }
    }

});


girder.wrap(girder.views.HierarchyWidget, 'render', function (render) {
    render.call(this);

    // Only on folder views:
    if (this.parentModel.resourceName === 'folder')
    {
        // Add the item-previews-container.
        var container_el = $('<div class="g-exiqr-container">');
        this.$el.prepend(container_el);

        this.exiqrFolderView = new girder.views.exiqrFolderWidget({
            folderModel: this.parentModel,
            parentView: this,
            el: container_el
        });
    }

    return this;
});
