import $ from 'jquery';
import _ from 'underscore';

import FolderCollection from 'girder/collections/FolderCollection';
import LoadingAnimation from 'girder/views/widgets/LoadingAnimation';
import View from 'girder/views/View';

import GalleryDetailWidget from './GalleryDetailWidget';
import GalleryItemViewTemplate from '../templates/galleryItemView.pug';

import '../stylesheets/galleryItemView.styl';

var GalleryItemView = View.extend({
    events: {
        'click .im-details': function (event) {
            this.galleryDetailWidget = new GalleryDetailWidget({
                el: $('#g-dialog-container'),
                item: this.item,
                parentView: this
            });
            this.galleryDetailWidget.render();
        }
    },

    initialize: function (settings) {
        this.item = settings.item;
    },

    render: function () {
        this.$el.html(GalleryItemViewTemplate({
            item: this.item
        }));

        return this;
    }
});

export default GalleryItemView;
