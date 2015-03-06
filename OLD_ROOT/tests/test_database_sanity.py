"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""
__author__ = 'dhan'
import sys
import os
sys.path.append(os.path.abspath(".."))
import unittest
from json import loads
import pymongo
import bson

class TestDBSanity(unittest.TestCase):
    def setUp(self):
        # Creates app and gets a client to it
        self.conn = pymongo.Connection("localhost")
        self.db = self.conn["smqtk"]

    def testMEDTESTcount(self):
        pass

    def test_videos(self):
        pass

    def test_videos(self):
        pass


if __name__ == "__main__":
    unittest.main()
