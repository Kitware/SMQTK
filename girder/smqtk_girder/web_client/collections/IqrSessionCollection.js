import ItemCollection from 'girder/collections/ItemCollection';
import IqrSessionModel from '../models/IqrSessionModel';

var IqrSessionCollection = ItemCollection.extend({
    model: IqrSessionModel,
    altUrl: 'smqtk_iqr/session'
});

export default IqrSessionCollection;
