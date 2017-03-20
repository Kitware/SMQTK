import ItemCollection from 'girder/collections/ItemCollection';
import FolderModel from 'girder/models/FolderModel';
import GalleryItemView from './GalleryItemView';
import IqrView from './IqrView';
import View from 'girder/views/View';
import { cancelRestRequests } from 'girder/rest';
import events from 'girder/events';
import GalleryViewTemplate from '../templates/galleryView.pug';
import ResultsCollection from '../collections/ResultsCollection';

import IqrSessionModel from '../models/IqrSessionModel';

import _ from 'underscore';
import Backbone from 'backbone';

import router from 'girder/router';

import IqrResultsCollection from '../collections/IqrResultsCollection';

/**
 * This view shows a single gallery as a hierarchy widget.
 */
var GalleryView = View.extend({
    events: {
        'click a.show-more-items': function (e) {
            this.collection.fetchNextPage();
        },

        // @todo check user is logged in
        'click a.smqtk-start-iqr': function (e) {
            var iqrSession = new IqrSessionModel();

            iqrSession.once('g:saved', _.bind(function () {
                events.trigger('g:navigateTo', IqrView, {
                    seedCollection: this.collection,
                    seedUrl: Backbone.history.fragment,
                    iqrSession: iqrSession
                });

                window.onbeforeunload = function () {
                    return false;
                };

                router.navigate('/gallery/iqr/' + iqrSession.id, { trigger: false });
            }, this)).save();
        }
    },

    initialize: function (settings) {
        cancelRestRequests('fetch');
        this.folder = settings.folder;
        this.$el = settings.el;

        if (_.isUndefined(settings.collection)) {
            this.collection = new ItemCollection();
            this.collection.pageLimit = 24;
            this.collection.append = true;
            this.collection.params = _.extend(this.collection.params || {}, {
                folderId: this.folder.id
            });
        }

        this.collection.on('g:changed', this.render, this).fetch(this.collection.params);
    },

    render: function () {
        this.$el.html(GalleryViewTemplate({}));

        this.collection.each(function (item) {
            var galleryItemView = new GalleryItemView({
                item: item,
                parentView: this
            });

            this.$el.find('#gallery-images').append(galleryItemView.render().$el);
        }, this);

        return this;
    }
}, {
    /**
     * Helper function for fetching the folder by id, then render the view.
     */
    fetchAndInit: function (id, params) {
        var folder = new FolderModel();
        folder.set({ _id: id }).on('g:fetched', function () {
            events.trigger('g:navigateTo', GalleryView, _.extend({
                folder: folder
            }, params || {}));
        }, this).fetch();
    },

    fetchAndInitNns: function (id, params) {
        var collection = new ResultsCollection();
        collection.params = _.extend(collection.params || {}, { itemId: id });

        events.trigger('g:navigateTo', GalleryView, _.extend({
            collection: collection
        }));
    },

    fetchAndInitIqr: function (sessionId, params) {
        events.trigger('g:navigateTo', IqrView, {});
    }

});

export default GalleryView;
