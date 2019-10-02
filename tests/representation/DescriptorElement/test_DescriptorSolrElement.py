import unittest

import mock
import pytest

from smqtk.representation.descriptor_element.solr_element import \
    SolrDescriptorElement
from smqtk.utils.configuration import configuration_test_helper


@pytest.mark.skipif(not SolrDescriptorElement.is_usable(),
                    reason='SolrDescriptorElement reports as not usable.')
class TestDescriptorSolrElement (unittest.TestCase):

    TEST_URL = 'http://localhost:8983/solr'

    @mock.patch("solr.Solr")
    def test_configuration(self, _mock_Solr):
        inst = SolrDescriptorElement(
            'test', 'a',
            solr_conn_addr=self.TEST_URL,
            type_field='type_s', uuid_field='uuid_s', vector_field='vector_fs',
            timestamp_field='timestamp_f', timeout=101,
            persistent_connection=True, commit_on_set=False,
        )
        for i in configuration_test_helper(inst, {'type_str', 'uuid'},
                                           ('test', 'abcd')):  # type: SolrDescriptorElement
            assert i.solr_conn_addr == self.TEST_URL
            assert i.type_field == 'type_s'
            assert i.uuid_field == 'uuid_s'
            assert i.vector_field == 'vector_fs'
            assert i.timestamp_field == 'timestamp_f'
            assert i.solr_timeout == 101
            assert i.solr_persistent_connection is True
            assert i.solr_commit_on_set is False
