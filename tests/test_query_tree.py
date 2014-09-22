"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""
# implement a basic test under somepackage.tests
import unittest
import sys
import os
sys.path.append(os.path.abspath(".."))
from WebUI.querytree import QueryTree, RankFusionQueryTree, AvgQueryTree, ScoreFusionQueryTree, BooleanQueryTree
import json



class TestQueryTree(unittest.TestCase):
    def test_single_node(self):
        self.assertEqual(True, True)

    def test_heirarchy_and(self):
        treestr = '{"$and": [{"reading": {"$gte": 0}}, {"teaching_or_training": {"$gte": 0}}]}'
        tree = json.loads(treestr)
        qt = QueryTree(tree)
        print qt.sql()

    def test_heirarchy_two_levels(self):
        treestr = '{"$and": [{"reading": {"$gte": 0}}, {"teaching_or_training": {"$gte": 0}}, {"$or": [{"research": {"$gte": 0}}, {"eating": {"$gte": 0}}]}]}'
        tree = json.loads(treestr)
        print tree
        qt = QueryTree(tree, debug=True)
        print "Result = ", qt.sql()


    def test_boolean(self):
        treestr = '{"$and": [{"reading": {"$gte": 0.1}}, {"teaching_or_training": {"$gte": 0.1}}, {"$or": [{"research": {"$gte": 0.05}}, {"eating": {"$gte": 0.2}}]}]}'
        tree = json.loads(treestr)
        print tree
        qt = BooleanQueryTree(tree, debug=True)
        results = qt.sql()
        print "Result = ", results
        self.assertEqual(results,"SELECT v_id, reading, teaching_or_training, research, eating FROM clip_calib_scores WHERE  (reading >= 0.1 AND teaching_or_training >= 0.1 AND  (research >= 0.05 OR eating >= 0.2)) ORDER BY reading, teaching_or_training, research, eating DESC LIMIT 0,50;")

    def test_avg(self):
        treestr = '{"$and": [{"reading": {"$gte": 0}}, {"teaching_or_training": {"$gte": 0}}, {"$or": [{"research": {"$gte": 0}}, {"eating": {"$gte": 0}}]}]}'
        tree = json.loads(treestr)
        print tree
        qt = AvgQueryTree(tree, debug=True)
        results = qt.process()
        print "Result = ", results
        print

    def test_score_fusion(self):
        treestr = '{"$and": [{"reading": {"$gte": 0}}, {"teaching_or_training": {"$gte": 0}}, {"$or": [{"research": {"$gte": 0}}, {"eating": {"$gte": 0}}]}]}'
        tree = json.loads(treestr)
        print tree
        qt = ScoreFusionQueryTree(tree, debug=True)
        results = qt.sql()
        print "Result = ", results
        self.assertEqual(results, "SELECT v_id, reading, teaching_or_training, research, eating, ((reading + teaching_or_training + MAX(research,eating)) / 3) as result FROM clip_calib_scores  ORDER BY result DESC LIMIT 0,50;")
        
        
    def test_rank_fusion(self):
        treestr = '{"$and": [{"reading": {"$gte": 0.1}}, {"teaching_or_training": {"$gte": 0.1}}, {"$or": [{"research": {"$gte": 0.05}}, {"eating": {"$gte": 0.2}}]}]}'
        tree = json.loads(treestr)
        print tree
        qt = RankFusionQueryTree(tree, debug=True)
        results = qt.sql()
        print "Result = ", results
        self.assertEqual(results,"SELECT v_id, reading, teaching_or_training, research, eating, MAX( ( RANK() OVER (ORDER BY reading), RANK() OVER (ORDER BY teaching_or_training), RANK() OVER (ORDER BY MAX(research,eating)) ) ) as result FROM clip_calib_scores  ORDER BY result ASC LIMIT 0,50;")


if __name__ == "__main__":
    unittest.main()

