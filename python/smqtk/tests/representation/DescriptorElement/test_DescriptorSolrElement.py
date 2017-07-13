import numpy
import requests
import unittest

from smqtk.representation.descriptor_element.solr_element import SolrDescriptorElement


SOLR_URL = 'http://localhost:8983/solr'  # is also a web-page


# Conduct test only if we have the solr module and if  there is a default solr
# instance on localhost
try:
    requests.get(SOLR_URL)
    solr_accessible = True
except requests.ConnectionError:
    solr_accessible = False


if solr_accessible:

    class TestDescriptorSolrElement (unittest.TestCase):

        def test_configuration(self):
            default_config = SolrDescriptorElement.get_default_config()
            self.assertEqual(default_config, {
                "solr_conn_addr": None,
                "type_field": None,
                "uuid_field": None,
                "vector_field": None,
                "timestamp_field": None,
                "timeout": 10,
                "persistent_connection": False,
                "commit_on_set": True,
            })

            default_config.update({
                'solr_conn_addr': SOLR_URL,
                "type_field": "type_s",
                "uuid_field": "uuid_s",
                "vector_field": "descriptor_fs",
                "timestamp_field": "timestamp_f",
            })
            inst1 = SolrDescriptorElement.from_config(default_config,
                                                      'test', 'a')
            inst1.set_vector(numpy.array([1, 2, 3]))

            self.assertEqual(default_config, inst1.get_config())
            self.assertEqual(inst1.solr.url, SOLR_URL)
            self.assertEqual(inst1.solr.timeout, 10)
            self.assertEqual(inst1.solr.persistent, False)
            self.assertEqual(inst1.type_field, 'type_s')
            self.assertEqual(inst1.uuid_field, 'uuid_s')
            self.assertEqual(inst1.vector_field, 'descriptor_fs')
            self.assertEqual(inst1.timestamp_field, 'timestamp_f')
            self.assertEqual(inst1.commit_on_set, True)

            # vector-based equality
            inst2 = SolrDescriptorElement.from_config(inst1.get_config(),
                                                      'test', 'a')
            self.assertEqual(inst1, inst2)

            inst1.solr.delete_query('id:%s' % inst1._base_doc()['id'])
            inst1.solr.commit()
