import ItemCollection from 'girder/collections/ItemCollection';
import ItemModel from 'girder/models/ItemModel';

var IqrResultsCollection = ItemCollection.extend({
    resourceName: 'item',
    model: ItemModel,
    altUrl: 'smqtk_iqr/results',
    pageLimit: 100,
    comparator: function (item) {
        return -item.get('meta').smqtk_iqr_confidence;
    }
});

export default IqrResultsCollection;
