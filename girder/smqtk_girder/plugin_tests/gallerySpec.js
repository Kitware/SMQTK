girderTest.addCoveredScripts([
    '/plugins/smqtk_girder/plugin_tests/testUtils.js',
    '/clients/web/static/built/plugins/smqtk_girder/plugin.min.js'
]);

girderTest.importStylesheet('/static/built/plugins/smqtk_girder/plugin.min.css');

girderTest.startApp();

$(function () {
    describe('Test viewing a folder as a gallery', function () {
        it('Tries to view an image', function () {
            smqtkGirderTest.viewAsGallery()();

            runs(function () {
                $('img.im-image-thumbnail:first').click();

                waitsFor(function () {
                    return $('.modal a.btn:contains(Close):visible').length === 1;
                }, 'the dialog close button to appear');

                runs(function () {
                    $('.modal a.btn:contains(Close)').click();
                });
                girderTest.waitForLoad();
            });
        });
    });

    // Test sort order for distance in NNS
});
