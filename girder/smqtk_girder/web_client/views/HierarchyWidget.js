import HierarchyWidget from 'girder/views/widgets/HierarchyWidget';
import { restRequest } from 'girder/rest';
import router from 'girder/router';
import { wrap } from 'girder/utilities/PluginUtils';
import HierarchyWidgetMenuTemplate from '../templates/hierarchyWidgetMenu.pug';

wrap(HierarchyWidget, 'initialize', function (initialize, settings) {
    this.events = _.extend(this.events, {
        'click .smqtk-process-images': function (e) {
            restRequest({
                path: 'smqtk/process_images',
                type: 'POST',
                data: {
                    id: this.parentModel.id
                }
            }).error(console.error);
        },

        'click .smqtk-view-as-gallery': function (e) {
            if (this.parentModel.resourceName === 'folder') {
                router.navigate('gallery/' + this.parentModel.id, {trigger: true});
            }
        }
    });

    initialize.call(this, settings);

    return this;
});

wrap(HierarchyWidget, 'render', function (render) {
    render.call(this);

    if (this.parentModel.get('_modelType') === 'folder') {
        this.$el.find('ul.g-folder-actions-menu.dropdown-menu').append(HierarchyWidgetMenuTemplate());
    }
});
