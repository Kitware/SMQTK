import nose.tools as ntools
import numpy
import requests
import unittest

from smqtk.representation.descriptor_element.solr_element import SolrDescriptorElement


__author__ = "paul.tunison@kitware.com"


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
            ntools.assert_equal(default_config, {
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

            ntools.assert_equal(default_config, inst1.get_config())
            ntools.assert_equal(inst1.solr.url, SOLR_URL)
            ntools.assert_equal(inst1.solr.timeout, 10)
            ntools.assert_equal(inst1.solr.persistent, False)
            ntools.assert_equal(inst1.type_field, 'type_s')
            ntools.assert_equal(inst1.uuid_field, 'uuid_s')
            ntools.assert_equal(inst1.vector_field, 'descriptor_fs')
            ntools.assert_equal(inst1.timestamp_field, 'timestamp_f')
            ntools.assert_equal(inst1.commit_on_set, True)

            # vector-based equality
            inst2 = SolrDescriptorElement.from_config(inst1.get_config(),
                                                      'test', 'a')
            ntools.assert_equal(inst1, inst2)

            inst1.solr.delete_query('id:%s' % inst1._base_doc()['id'])
            inst1.solr.commit()
