

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

        if( folder_meta['smqtk_iqr'] !== undefined )
        {
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
            var iqr_config = this.folderModel.get('meta')['smqtk_iqr'];
            window.open("http://localhost:5050/?config="+JSON.stringify(iqr_config));
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
