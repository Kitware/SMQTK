import ItemCollection from 'girder/collections/ItemCollection';
import IqrSessionModel from '../models/IqrSessionModel';

var IqrSessionCollection = ItemCollection.extend({
    model: IqrSessionModel,
    altUrl: IqrSessionModel.altUrl
});

export default IqrSessionCollection;
