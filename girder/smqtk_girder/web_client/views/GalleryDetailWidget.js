import router from 'girder/router';
import View from 'girder/views/View';
import GalleryView from './GalleryView';
import GalleryDetailWidgetTemplate from '../templates/galleryDetailWidget.pug';

var GalleryDetailWidget = View.extend({
    events: {
        'click .prev-image': function (event) {
            if (this.canScroll) {
                this.parentView.$el.prev().find('.im-details:first').click();
            }
        },

        'click .next-image': function (event) {
            if (this.canScroll) {
                this.parentView.$el.next().find('.im-details:first').click();
            }
        },

        'click .smqtk-nn-search': function (event) {
            router.navigate('gallery/nearest-neighbors/' + this.item.id, {trigger: true});
        }
    },

    initialize: function (settings) {
        this.item = settings.item || null;

        // If it's being displayed in a grid/list as part of results, let the user
        // scroll through modals
        this.canScroll = _.has(this, 'parentView') && _.has(this.parentView, 'parentView') &&
            this.parentView.parentView instanceof GalleryView;
    },

    render: function () {
        var modal = this.$el.html(GalleryDetailWidgetTemplate({
            item: this.item,
            hasPrev: this.canScroll && this.parentView.$el.prev().length,
            hasNext: this.canScroll && this.parentView.$el.next().length
        })).girderModal(this);

        $('.modal-body').css('height', $(window).height() * 0.6);
        $('.modal-body').css('overflow', 'auto');

        // Bizzare bug in FF causes the scrollbar to remember its position
        // https://bugzilla.mozilla.org/show_bug.cgi?id=706792
        $('.modal-body').scrollTop(0);

        modal.trigger($.Event('ready.girder.modal', {relatedTarget: modal}));

        return this;
    }
});

export default GalleryDetailWidget;
