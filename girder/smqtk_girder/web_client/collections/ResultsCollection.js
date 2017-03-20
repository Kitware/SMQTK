import ItemCollection from 'girder/collections/ItemCollection';
import ItemModel from 'girder/models/ItemModel';

var ResultsCollection = ItemCollection.extend({
    resourceName: 'item',
    model: ItemModel,
    altUrl: 'smqtk_nearest_neighbors/nn',
    pageLimit: 100,
    comparator: 'smqtk_distance'
});

export default ResultsCollection;
