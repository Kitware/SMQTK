import _ from 'underscore';

import { restRequest } from 'girder/rest';

import '../stylesheets/iqrItemView.styl';

import GalleryItemView from './GalleryItemView';
import IqrItemViewTemplate from '../templates/iqrItemView.pug';

var IqrItemView = GalleryItemView.extend({
    events: {
        'click .smqtk-iqr-annotate': function (e) {
            var button = $(e.currentTarget),
                done = _.bind(this.render, this);

            if (button.hasClass('smqtk-iqr-positive')) {
                if (this.isIqrPositive()) {
                    this.iqrSession.removePositiveUuid(this.item.get('meta').smqtk_uuid, done);
                } else {
                    this.iqrSession.addPositiveUuid(this.item.get('meta').smqtk_uuid, done);
                }
            } else if (button.hasClass('smqtk-iqr-negative')) {
                if (this.isIqrNegative()) {
                    this.iqrSession.removeNegativeUuid(this.item.get('meta').smqtk_uuid, done);
                } else {
                    this.iqrSession.addNegativeUuid(this.item.get('meta').smqtk_uuid, done);
                }
            }
        }
    },

    initialize: function (settings) {
        this.item = settings.item;
        this.iqrSession = this.parentView.iqrSession;

        _.extend(this.events, GalleryItemView.prototype.events);
    },

    isIqrPositive: function () {
        return (this.iqrSession.has('meta') &&
                _.contains(this.iqrSession.get('meta').pos_uuids, this.item.get('meta').smqtk_uuid));
    },

    isIqrNegative: function () {
        return (this.iqrSession.has('meta') &&
                _.contains(this.iqrSession.get('meta').neg_uuids, this.item.get('meta').smqtk_uuid));
    },

    render: function () {
        this.$el.html(IqrItemViewTemplate({
            item: this.item,
            positive: this.isIqrPositive(),
            negative: this.isIqrNegative()
        }));

        return this;
    }
});

export default IqrItemView;
