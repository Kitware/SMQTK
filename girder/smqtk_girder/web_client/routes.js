import router from 'girder/router';
import GalleryView from './views/GalleryView';
import ConfigView from './views/ConfigView';
import events from 'girder/events';
import { exposePluginConfig } from 'girder/utilities/PluginUtils';

exposePluginConfig('smqtk_girder', 'plugins/smqtk_girder/config');

router.route('plugins/smqtk_girder/config', 'smqtkGirderConfig', function () {
    events.trigger('g:navigateTo', ConfigView);
});

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
