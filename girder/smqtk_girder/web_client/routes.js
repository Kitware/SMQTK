import router from 'girder/router';
import GalleryView from './views/GalleryView';
import ConfigView from './views/ConfigView';
import events from 'girder/events';
import { exposePluginConfig } from 'girder/utilities/PluginUtils';

exposePluginConfig('smqtk_girder', 'plugins/smqtk_girder/config');

router.route('plugins/smqtk_girder/config', 'smqtkGirderConfig', function () {
    events.trigger('g:navigateTo', ConfigView);
});

router.route('gallery/:indexId/:id', 'gallery', function (indexId, id, params) {
    GalleryView.fetchAndInit(indexId, id, params);
});

router.route('gallery/nearest-neighbors/:indexId/:id', 'gallery-nearest-neighbors', function (indexId, id, params) {
    GalleryView.fetchAndInitNns(indexId, id, params);
});

// this :id is the IQR Session ID
router.route('gallery/iqr/:indexId/:id', 'gallery-iqr', function (indexId, id, params) {
    GalleryView.fetchAndInitIqr(indexId, id, params);
});
