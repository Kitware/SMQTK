import HierarchyWidget from 'girder/views/widgets/HierarchyWidget';
import { restRequest } from 'girder/rest';
import router from 'girder/router';
import { wrap } from 'girder/utilities/PluginUtils';
import HierarchyWidgetMenuTemplate from '../templates/hierarchyWidgetMenu.pug';
import events from 'girder/events';

HierarchyWidget.prototype.events['click .smqtk-process-images'] = function (e) {
    restRequest({
        path: 'smqtk/process_images',
        type: 'POST',
        data: {
            id: this.parentModel.id
        }
    }).error(console.error);
};

HierarchyWidget.prototype.events['click .smqtk-view-as-gallery'] = function (e) {
    if (this.parentModel.resourceName === 'folder') {
        // Fetch the .smqtk folder id (effectively the descriptor index id)
        // This is necessary to store in the URL bar for state since every view
        // deriving from the gallery (NN and IQR) require a specific descriptor index
        // for access control.
        restRequest({
            path: 'folder',
            type: 'GET',
            data: {
                parentType: 'folder',
                parentId: this.parentModel.id,
                name: '.smqtk',
                limit: 1
            }
        }).then(resp => {
            if (_.size(resp) == 1) {
                router.navigate(`gallery/${resp[0]._id}/${this.parentModel.id}`, {trigger: true});
            } else {
                events.trigger('g:alert', {
                    text: 'Can\'t view as gallery until images have been processed.',
                    type: 'warning'
                });
            }
        });
    }
};

wrap(HierarchyWidget, 'render', function (render) {
    render.call(this);

    if (this.parentModel.get('_modelType') === 'folder') {
        this.$el.find('ul.g-folder-actions-menu.dropdown-menu').append(HierarchyWidgetMenuTemplate());
    }
});
