import _ from 'underscore';

import PluginConfigBreadcrumbWidget from 'girder/views/widgets/PluginConfigBreadcrumbWidget';
import View from 'girder/views/View';
import events from 'girder/events';
import { restRequest } from 'girder/rest';

import ConfigViewTemplate from '../templates/configView.pug';

var ConfigView = View.extend({
    _settingKeyToSelector: {
        'smqtk_girder.db_host': '#smqtk-girder-db-host',
        'smqtk_girder.db_name': '#smqtk-girder-db-name',
        'smqtk_girder.db_user': '#smqtk-girder-db-user',
        'smqtk_girder.db_pass': '#smqtk-girder-db-pass',
        'smqtk_girder.db_descriptors_table': '#smqtk-girder-db-descriptors-table',
        'smqtk_girder.image_batch_size': '#smqtk-girder-image-batch-size',
        'smqtk_girder.caffe_network_model': '#smqtk-girder-caffe-network-model',
        'smqtk_girder.caffe_network_prototxt': '#smqtk-girder-caffe-network-prototxt',
        'smqtk_girder.caffe_image_mean': '#smqtk-girder-caffe-image-mean'
    },

    events: {
        'submit #smqtk-girder-settings-form': function (event) {
            event.preventDefault();
            this.$('#smqtk-girder-settings-error-message').empty();
            var settings = [];

            _.each(this._settingKeyToSelector, function (selector, settingKey) {
                if (settingKey !== 'smqtk_girder.db_pass') {
                    settings.push({ key: settingKey,
                                    value: this.$(selector).val().trim() });
                } else {
                    settings.push({ key: settingKey,
                                    value: this.$(selector).val() });
                }
            }, this);

            this._saveSettings(settings);
        }
    },

    initialize: function () {
        restRequest({
            type: 'GET',
            path: 'system/setting',
            data: {
                list: JSON.stringify(_.keys(this._settingKeyToSelector))
            }
        }).done(_.bind(function (resp) {
            this.render();

            _.each(resp, _.bind(function (settingValue, settingKey) {
                this.$(this._settingKeyToSelector[settingKey]).val(settingValue);
            }, this));
        }, this));
    },

    render: function () {
        this.$el.html(ConfigViewTemplate());

        if (!this.breadcrumb) {
            this.breadcrumb = new PluginConfigBreadcrumbWidget({
                pluginName: 'SMQTK Girder',
                el: this.$('.g-config-breadcrumb-container'),
                parentView: this
            });
        }

        this.breadcrumb.render();

        return this;
    },

    _saveSettings: function (settings) {
        restRequest({
            type: 'PUT',
            path: 'system/setting',
            data: {
                list: JSON.stringify(settings)
            },
            error: null
        }).done(_.bind(function (resp) {
            events.trigger('g:alert', {
                icon: 'ok',
                text: 'Settings saved.',
                type: 'success',
                timeout: 4000
            });
        }, this)).error(_.bind(function (resp) {
            this.$('#smqtk-girder-settings-error-message').text(
                resp.responseJSON.message);
        }, this));
    }
});

export default ConfigView;
