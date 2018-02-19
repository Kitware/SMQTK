import View from 'girder/views/View';

import { cancelRestRequests } from 'girder/rest';
import { restRequest } from 'girder/rest';

import GalleryViewTemplate from '../templates/galleryView.pug';

import IqrItemView from './IqrItemView';

import IqrSessionModel from '../models/IqrSessionModel';

import IqrResultsCollection from '../collections/IqrResultsCollection';

import _ from 'underscore';
import Backbone from 'backbone';

var IqrView = View.extend({
    events: {
        'click a.smqtk-iqr-refine': function (e) {
            if (_.size(this.iqrSession.get('meta').pos_uuids) === 0) {
                alert('Refinement requires at least 1 positive example.');
                return;
            }

            restRequest({
                path: `smqtk_iqr/refine/${this.iqrSession.get('name')}`,
                type: 'POST',
                data: {
                    pos_uuids: JSON.stringify(this.iqrSession.get('meta').pos_uuids),
                    neg_uuids: JSON.stringify(this.iqrSession.get('meta').neg_uuids)
                }
            }).done(_.bind(function () {
                // See documentation when starting an IQR session
                window.onbeforeunload = null;

                // either this is already an iqr results collection and we can just call .fetch
                if (this.collection instanceof IqrResultsCollection) {
                    this.collection.fetch(this.collection.params || {}, true);
                } else {
                    // or it's a seedCollection and we need to supplant our collection with the new iqr one
                    this.collection = new IqrResultsCollection();
                    this.collection.params = this.collection.params || {};
                    this.collection.params.sid = this.iqrSession.get('name');

                    this.collection.on('g:changed', this.render, this).fetch(this.collection.params || {}, true);
                }
            }, this)).error(console.error);
        }
    },

    initialize: function (settings) {
        cancelRestRequests('fetch');
        this.$el = settings.el;
        this.seedUrl = settings.seedUrl;
        this.indexId = settings.indexId;

        if (_.has(settings, 'seedCollection') &&
            _.has(settings, 'iqrSession')) {
            this.collection = settings.seedCollection;
            this.iqrSession = settings.iqrSession;

            this.collection.on('g:changed', this.render, this).fetch(this.collection.params || {});
        } else {
            this.iqrSession = new IqrSessionModel({ _id: Backbone.history.fragment.split('/')[3],
                                                    smqtkFolder: this.indexId});

            this.iqrSession.once('g:fetched', _.bind(function () {
                this.collection = new IqrResultsCollection();
                this.collection.params = this.collection.params || {};
                this.collection.params.sid = this.iqrSession.get('name');

                this.collection.on('g:changed', this.render, this).fetch(this.collection.params || {});
            }, this)).fetch();
        }
    },

    render: function () {
        this.$el.html(GalleryViewTemplate({
            iqrEnabled: true
        }));

        this.collection.each(function (item) {
            var iqrItemView = new IqrItemView({
                item: item,
                parentView: this
            });

            this.$el.find('#gallery-images').append(iqrItemView.render().$el);
        }, this);

        return this;
    }
});

export default IqrView;
