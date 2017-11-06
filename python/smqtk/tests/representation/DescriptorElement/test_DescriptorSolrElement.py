import unittest

import mock

from smqtk.representation.descriptor_element.solr_element import \
    SolrDescriptorElement


if SolrDescriptorElement.is_usable():

    class TestDescriptorSolrElement (unittest.TestCase):

        TEST_URL = 'http://localhost:8983/solr'

        @mock.patch("solr.Solr")
        def test_configuration(self, _mock_Solr):
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
                'solr_conn_addr': self.TEST_URL,
                "type_field": "type_s",
                "uuid_field": "uuid_s",
                "vector_field": "descriptor_fs",
                "timestamp_field": "timestamp_f",
                "timeout": 101,
                "persistent_connection": True,
                "commit_on_set": False,
            })

            # Result instance should have parameters matching those in config.
            #: :type: SolrDescriptorElement
            inst1 = SolrDescriptorElement.from_config(default_config,
                                                      'test', 'a')
            self.assertEqual(default_config, inst1.get_config())
            self.assertEqual(inst1.solr_conn_addr, self.TEST_URL)
            self.assertEqual(inst1.solr_timeout, 101)
            self.assertEqual(inst1.solr_persistent_connection, True)
            self.assertEqual(inst1.type_field, 'type_s')
            self.assertEqual(inst1.uuid_field, 'uuid_s')
            self.assertEqual(inst1.vector_field, 'descriptor_fs')
            self.assertEqual(inst1.timestamp_field, 'timestamp_f')
            self.assertEqual(inst1.solr_commit_on_set, False)

            # State parameters should be equal.
            #: :type: SolrDescriptorElement
            inst2 = SolrDescriptorElement.from_config(inst1.get_config(),
                                                      'test', 'a')
            self.assertEqual(inst1.type_field, inst2.type_field)
            self.assertEqual(inst1.uuid_field, inst2.uuid_field)
            self.assertEqual(inst1.vector_field, inst2.vector_field)
            self.assertEqual(inst1.timestamp_field, inst2.timestamp_field)
            self.assertEqual(inst1.solr_conn_addr, inst2.solr_conn_addr)
            self.assertEqual(inst1.solr_timeout, inst2.solr_timeout)
            self.assertEqual(inst1.solr_persistent_connection,
                             inst2.solr_persistent_connection)
            self.assertEqual(inst1.solr_commit_on_set, inst2.solr_commit_on_set)
