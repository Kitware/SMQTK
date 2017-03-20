import router from 'girder/router';
import GalleryView from './views/GalleryView';

router.route('gallery/:id', 'gallery', function (id, params) {
    GalleryView.fetchAndInit(id, params);
});

router.route('gallery/nearest-neighbors/:id', 'gallery-nearest-neighbors', function (id, params) {
    GalleryView.fetchAndInitNns(id, params);
});

// this :id is the IQR Session ID
router.route('gallery/iqr/:id', 'gallery-iqr', function (id, params) {
    GalleryView.fetchAndInitIqr(id, params);
});
