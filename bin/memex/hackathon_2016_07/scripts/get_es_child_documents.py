import certifi
import json
import os
import sys

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q


def search(obj_parents_l, es):
    return Search() \
        .using(es) \
        .filter(Q('match',
                  obj_parent=json.dumps(obj_parents_l)))\
        .scan()

if __name__ == '__main__':
    """
    Returns the Elasticsearch documents whose parents are in sys.argv.

    This is useful for retrieving image documents given a series of
    ad document ids.
    """
    config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               'es_config.json')
    if not os.path.isfile(config_file):
        print 'Configuration file not found.'
        sys.exit(1)
    elif len(sys.argv) < 2:
        print 'Usage: get_es_child_documents.py ad_cdr_id...'
        sys.exit(1)
    else:
        with open(config_file, 'rb') as infile:
            config = json.load(infile)

        es = Elasticsearch(config['url'],
                           http_auth=(config['user'], config['password']),
                           use_ssl=True,
                           verify_certs=True,
                           ca_certs=certifi.where())

        cdr_ids = sys.argv[1:]

        try:
            results = search(json.dumps(cdr_ids), es)

            for result in results:
                print(json.dumps(result.to_dict()))
        except Exception as e:
            print str(e)
            sys.exit(5)
