import ItemCollection from 'girder/collections/ItemCollection';
import FolderModel from 'girder/models/FolderModel';
import GalleryItemView from './GalleryItemView';
import IqrView from './IqrView';
import View from 'girder/views/View';
import { restRequest, cancelRestRequests } from 'girder/rest';
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
            var iqrSession = new IqrSessionModel({
                smqtkFolder: this.indexId
            });

            iqrSession.once('g:saved', _.bind(function () {
                events.trigger('g:navigateTo', IqrView, {
                    seedCollection: this.collection,
                    seedUrl: Backbone.history.fragment,
                    iqrSession: iqrSession,
                    indexId: this.indexId
                });

                /*
                 * Once the user has started an IQR session, returning to it before
                 * their first refine will result in an empty page. Setting this
                 * function to return false will give the user the "Are you sure
                 * you want to leave this page, changes may be lost?" warning.
                 * On the first refine, we unset this back to null so the warning
                 * won't occur.
                 **/
                window.onbeforeunload = function () {
                    return false;
                };

                router.navigate(`/gallery/iqr/${this.indexId}/${iqrSession.id}`, { trigger: false });
            }, this)).save();
        }
    },

    initialize: function (settings) {
        cancelRestRequests('fetch');
        this.folder = settings.folder;
        this.$el = settings.el;
        this.indexId = settings.indexId;

        // Accept either a collection, or default to assume we're operating on a folder level
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
    fetchAndInit: function (indexId, id, params) {
        var folder = new FolderModel();

        return folder.set({ _id: id }).fetch().then(() => {
            events.trigger('g:navigateTo', GalleryView, _.extend({
                folder: folder,
                indexId: indexId
            }, params || {}));
        });
    },

    fetchAndInitNns: function (indexId, id, params) {
        var collection = new ResultsCollection();
        collection.params = _.extend(collection.params || {}, { itemId: id });

        events.trigger('g:navigateTo', GalleryView, _.extend({
            collection: collection,
            indexId: indexId
        }));
    },

    fetchAndInitIqr: function (sessionId, params) {
        events.trigger('g:navigateTo', IqrView, {});
    }

});

export default GalleryView;
