window.smqtkGirderTest = {};

smqtkGirderTest.viewAsGallery = function () {
    return function () {
        girderTest.login('girder', 'girder', 'girder', 'girder')();

        runs(function () {
            expect($('#g-user-action-menu.open').length).toBe(0);
            $('.g-user-text>a:first').click();
        });
        girderTest.waitForLoad();

        runs(function () {
            expect($('#g-user-action-menu.open').length).toBe(1);
            $('a.g-my-folders').click();
        });
        girderTest.waitForLoad();

        runs(function () {
            $('a.g-folder-list-link:last').click();
        });
        girderTest.waitForLoad();

        runs(function () {
            $('.g-folder-actions-button').click();
        });

        waitsFor(function () {
            return $('.smqtk-view-as-gallery:visible').length === 1;
        }, 'the view as gallery action to appear');


        runs(function () {
            $('.smqtk-view-as-gallery').click();
        });

        waitsFor(function () {
            return $('#gallery-images').length === 1 &&
                $('.im-thumbnail').length === 14;
        }, 'items to appear as thumbnails');
    };
};
