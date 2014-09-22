"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import nose.tools
import numpy as np
import os.path as osp
import unittest

from SMQTK_Backend.FeatureMemory import \
    initFeatureManagerConnection,\
    getFeatureManager
from SMQTK_Backend.RefinableSearchState import RefinableSearchState
from SMQTK_Backend.SharedAttribute import register_mdb_loc
from SMQTK_Backend.utils.DatabaseInfo import DatabaseInfo
from SMQTK_Backend.utils.jsmin import jsmin


#noinspection PyMethodMayBeStatic,PyAttributeOutsideInit,PySetFunctionToLiteral
# noinspection PyPep8Naming
class test_RefinableSearchState (unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # local FeatureManager server may have already been initialized by
        # another unittest
        try:
            initFeatureManagerConnection()
        except ValueError:
            pass

        data_dir = osp.abspath(osp.join(osp.dirname(__file__), 'data'))
        classifier_config_file = osp.join(data_dir, 'classifier_config.json')
        cls.classifier_config = jsmin(open(classifier_config_file).read())

        symmetric_clipids = np.loadtxt(osp.join(data_dir,
                                                'symmetric_clipids.txt'))
        symmetric_bgflags = np.loadtxt(osp.join(data_dir,
                                                'symmetric_bgflags.txt'))
        # noinspection PyCallingNonCallable
        symmetric_dk_mat = \
            np.matrix(np.load(osp.join(data_dir,
                                       'symmetric_distance_kernel.npy')))
        symmetric_bgclipids = np.array([symmetric_clipids[i]
                                        for i, e in enumerate(symmetric_bgflags)
                                        if e])
        # noinspection PyUnresolvedReferences
        #: :type: DistanceKernel
        cls.dk_iqr = getFeatureManager().DistanceKernel(symmetric_clipids,
                                                        symmetric_clipids,
                                                        symmetric_dk_mat,
                                                        symmetric_bgclipids)

        asymmetric_row_ids = np.loadtxt(osp.join(data_dir,
                                                 'asymmetric_clipids_rows.txt'))
        asymmetric_col_ids = np.loadtxt(osp.join(data_dir,
                                                 'asymmetric_clipids_cols.txt'))
        # noinspection PyCallingNonCallable
        asymmetric_dk_mat = \
            np.matrix(np.load(osp.join(data_dir,
                                       'asymmetric_distance_kernel.npy')))
        # noinspection PyUnresolvedReferences
        #: :type: DistanceKernel
        cls.dk_arc = getFeatureManager().DistanceKernel(asymmetric_row_ids,
                                                        asymmetric_col_ids,
                                                        asymmetric_dk_mat)

        cls.mdb_info = DatabaseInfo('localhost', 27017, 'RSS_Test')

        register_mdb_loc(cls.mdb_info.host, cls.mdb_info.port)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    @nose.tools.raises(ValueError)
    def test_first_init_fail1(self):
        RefinableSearchState(1)

    @nose.tools.raises(ValueError)
    def test_first_init_fail2(self):
        RefinableSearchState(1, 'foo bar')

    @nose.tools.raises(ValueError)
    def test_first_init_fail3(self):
        RefinableSearchState(1, 'foo bar', self.dk_iqr)

    @nose.tools.raises(ValueError)
    def test_first_init_fail4(self):
        RefinableSearchState(1, 'foo bar', self.dk_iqr,
                             self.classifier_config)

    @nose.tools.raises(ValueError)
    def test_first_init_fail4(self):
        RefinableSearchState(1, 'foo bar', self.dk_iqr,
                             self.classifier_config, self.mdb_info)

    def test_first_init(self):
        eID = None
        query = 'foo and bar'
        rMID = 'results'
        rss = RefinableSearchState(eID, query, self.dk_iqr,
                                   self.classifier_config, self.mdb_info, rMID)
        assert rss.search_event_type == eID
        assert rss.search_query == query
        assert rss.distance_kernel == self.dk_iqr
        assert rss.classifier_config == self.classifier_config
        assert rss.negatives == set()
        assert rss.positives == set()
        assert rss.parent_state is None
        assert rss.child_state is None
        assert rss.num_parents == 0
        assert rss.num_children == 0
        assert rss.result_mID == rMID

    def test_equalty(self):
        eID = None
        query = 'foo and bar'
        rMID = 'results'
        rss1 = RefinableSearchState(eID, query, self.dk_iqr,
                                    self.classifier_config, self.mdb_info, rMID)
        rss2 = RefinableSearchState(eID, query, self.dk_iqr,
                                    self.classifier_config, self.mdb_info, rMID)
        assert rss1.uuid != rss2.uuid
        assert rss1 != rss2
        assert rss1 == rss1
        assert rss2 == rss2
        assert hash(rss1) != hash(rss2)
        assert hash(rss1) == hash(rss1)
        assert hash(rss2) == hash(rss2)

    def test_state_inherit_construction(self):
        eID = 1
        query = 'foo and bar'
        rMID = 'results'
        rss1 = RefinableSearchState(eID, query, self.dk_iqr,
                                    self.classifier_config, self.mdb_info, rMID)
        rss2 = RefinableSearchState(rss1)
        assert rss1.uuid != rss2.uuid
        assert rss1.num_children == 1
        assert rss1.child_state is rss2
        assert rss2.parent_state is rss1
        assert rss2.num_parents == 1
        assert rss2.search_uuid == rss1.search_uuid

        rss3 = RefinableSearchState(eID, query, self.dk_iqr,
                                    self.classifier_config, self.mdb_info, rMID)
        assert rss1.search_uuid != rss3.search_uuid

    def test_state_inherit_ignores_construction_params(self):
        eID = 2L
        query = 'foo and bar'
        rMID = 'results'
        rss1 = RefinableSearchState(eID, query, self.dk_iqr,
                                    self.classifier_config, self.mdb_info, rMID)
        rss2 = RefinableSearchState(rss1, 'other_query', self.dk_arc, {},
                                    self.mdb_info, 'somewhereelse')
        assert rss1.uuid != rss2.uuid
        assert rss1.search_event_type == rss2.search_event_type
        assert rss1.search_query == rss2.search_query
        assert rss1.distance_kernel == rss2.distance_kernel
        assert rss1.classifier_config == rss2.classifier_config
        assert rss1.result_mID == rss2.result_mID

    def test_config_setter(self):
        eID = None
        query = 'foo and bar'
        rss1 = RefinableSearchState(eID, query, self.dk_iqr,
                                    self.classifier_config, self.mdb_info,
                                    'results')
        config2 = {'foo': 'bar'}
        assert rss1.classifier_config != config2
        rss1.classifier_config = config2
        assert rss1.classifier_config == config2
        assert rss1.classifier_config != self.classifier_config

    def test_setting_positives(self):
        eID = None
        query = 'foo and bar'
        rss = RefinableSearchState(eID, query, self.dk_iqr,
                                   self.classifier_config, self.mdb_info,
                                   'results')
        assert rss.positives == set()
        rss.register_positive_feedback(1)
        assert rss.positives == set((1,))

    def test_setting_negatives(self):
        eID = None
        query = 'foo and bar'
        rss = RefinableSearchState(eID, query, self.dk_iqr,
                                   self.classifier_config, self.mdb_info,
                                   'results')
        assert rss.negatives == set()
        rss.register_negative_feedback(1)
        assert rss.negatives == set((1,))

    def test_positive_removal(self):
        eID = None
        query = 'foo and bar'
        rss = RefinableSearchState(eID, query, self.dk_iqr,
                                   self.classifier_config, self.mdb_info,
                                   'results')
        rss.register_positive_feedback(1)
        rss.remove_positive_feedback(1)
        assert rss.positives == set()

    def test_negative_removal(self):
        eID = None
        query = 'foo and bar'
        rss = RefinableSearchState(eID, query, self.dk_iqr,
                                   self.classifier_config, self.mdb_info,
                                   'results')
        rss.register_negative_feedback(1)
        rss.remove_negative_feedback(1)
        self.assertSetEqual(rss.negatives, set())

    def test_refinement_inheritance(self):
        eID = None
        query = 'foo and bar'
        pos_set = set((23, 25, 27, 60, 123, 227, 457, 967))
        neg_set = set((794, 986, 5967, 7254, 23456, 82456))

        rss = RefinableSearchState(eID, query, self.dk_iqr,
                                   self.classifier_config, self.mdb_info,
                                   'results')
        rss.register_positive_feedback(pos_set)
        rss.register_negative_feedback(neg_set)

        rss2 = RefinableSearchState(rss)

        self.assertSetEqual(rss.positives, rss2.positives)
        self.assertSetEqual(rss2.positives, pos_set)
        self.assertSetEqual(rss.negatives, rss2.negatives)
        self.assertSetEqual(rss2.negatives, neg_set)

        rss.register_positive_feedback(0)
        rss2.register_positive_feedback(1)
        self.assertNotIn(0, rss2.positives)
        self.assertNotIn(1, rss.positives)

        rss.register_negative_feedback(2)
        rss2.register_negative_feedback(3)
        self.assertNotIn(2, rss2.negatives)
        self.assertNotIn(3, rss.negatives)

    def test_refinement_side_effect(self):
        eID = None
        query = 'foo and bar'
        pos_set = set(range(0, 10))
        neg_set = set(range(10, 20))

        rss = RefinableSearchState(eID, query, self.dk_iqr,
                                   self.classifier_config, self.mdb_info,
                                   'results')
        rss.register_positive_feedback(pos_set)
        rss.register_negative_feedback(neg_set)

        rss.register_positive_feedback(10)
        self.assertNotIn(10, rss.negatives)
        self.assertIn(10, rss.positives)

        rss.register_negative_feedback(0)
        self.assertNotIn(0, rss.positives)
        self.assertIn(0, rss.negatives)
