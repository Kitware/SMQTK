#!/usr/bin/env python

import json
import requests

girder_url = 'http://localhost:8080/api/v1'
user = 'aashish24'
password = '2pw4kw'


class Client(object):

    def __init__(self, girder_url, user, password):

        # store for later
        self._root = girder_url
        self._user = user
        self._password = password
        self._token = None

        # log in
        r = requests.get(girder_url + '/user/authentication', auth=(user, password))

        if r.ok:
            self._token = r.json()['authToken']['token']
        else:
            print "Could not log in"

    def request(self, method, endpoint, params={}, data=None):

        # add the login token to the parameters
        p = dict(params)
        p['token'] = self._token

        # make the request
        r = method(
            url=self._root + endpoint,
            data=data,
            params=p
        )

        if r.ok:
            return r.json()
        else:
            print r.content


    def get(self, endpoint, **kw):
        return self.request(requests.get, endpoint, **kw)

    def put(self, endpoint, **kw):
        return self.request(requests.put, endpoint, **kw)

    def post(self, endpoint, **kw):
        return self.request(requests.post, endpoint, **kw)

    def delete(self, endpoint, **kw):
        return self.request(requests.delete, endpoint, **kw)


if __name__ == '__main__':
    import json
    data_file = open('foo.json', 'r')
    data = json.load(data_file)

    girder = Client(girder_url, user, password)

    # First get all of the collections
    collections = girder.get('/collection')

    for collection in collections:
        if collection['name'] == "mycollection":
            print girder.delete('/collection/' + collection['_id'])

    # create a collection
    collection = girder.post('/collection', params={
        'name': 'mycollection',
        'description': 'This is an optional description'
    })

    # create a folder
    folder = girder.post('/folder', params={
        'parentType': 'collection',
        'parentId': collection['_id'],
        'name': 'afolder'
    })

    # create an item
    print data['collection']
    for collection in data['collection']:
        for data_item in collection['items']:
            print data_item
            item = girder.post('/item', params={
                'folderId': folder['_id'],
                'name': 'data_item'
            })

            # add metadata to the item
            meta = {
                'item': data_item['item']
            }
            girder.put(
                '/item/' + item['_id'] + '/metadata',
                data=json.dumps(meta)
            )
