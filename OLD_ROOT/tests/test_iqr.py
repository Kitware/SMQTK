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
import WebUI

class TestIQR(unittest.TestCase):
    def setUp(self):
        # Creates app and gets a client to it
        self.app1 = WebUI.app
        self.app1.testing = True
        self.app = self.app1.test_client()


    def tearDown(self):
        WebUI.shutdown()

    def testInitialization(self):
        resp = self.parseResponse("/iqr/init_new_search_session?query='wood and trees'")
        self.assertEqual("uid" in resp, True)

    def testRefinementParameters(self):
        resp = self.parseResponse("/iqr/init_new_search_session?query='wood and trees'")
        self.assertEqual("uid" in resp, True)

        # Check the exclusion of the uid returns error
        rv = self.app.get("/iqr/refine_search")
        obj = loads(rv.data)
        self.assertEqual("error" in obj, True)

        pos = [1,2,3]
        neg = [4,5,6]

        resp2 = self.parseResponse("/iqr/refine_search?uid="+resp["uid"] + "&positive=" + str(pos) + "&negative="+str(neg))

        self.assertEqual("error" in resp, False)

        self.assertEqual(resp2["query"]["positive"], pos)
        self.assertEqual(resp2["query"]["negative"], neg)


    def testGetResults(self):
        resp = self.parseResponse("/iqr/init_new_search_session?query='wood and trees'")
        self.assertEqual("uid" in resp, True)

        # Check the exclusion of the uid returns error
        rv = self.app.get("/iqr/refine_search")
        obj = loads(rv.data)
        self.assertEqual("error" in obj, True)

        pos = [1,2,3]
        neg = [4,5,6]

        resp2 = self.parseResponse("/iqr/refine_search?uid="+resp["uid"] + "&positive=" + str(pos) + "&negative="+str(neg))

        self.assertEqual("error" in resp, False)

        self.assertEqual(resp2["query"]["positive"], pos)
        self.assertEqual(resp2["query"]["negative"], neg)

    def testArchiveSearch(self):
        resp = self.parseResponse("/iqr/init_new_search_session?query='wood and trees'")
        self.assertEqual("uid" in resp, True)

        # Check the exclusion of the uid returns error
        rv = self.app.get("/iqr/refine_search")
        obj = loads(rv.data)
        self.assertEqual("error" in obj, True)

        pos = [1,2,3]
        neg = [4,5,6]

        resp2 = self.parseResponse("/iqr/refine_search?uid="+resp["uid"] + "&positive=" + str(pos) + "&negative="+str(neg))

        self.assertEqual("error" in resp, False)

        self.assertEqual(resp2["query"]["positive"], pos)
        self.assertEqual(resp2["query"]["negative"], neg)



    def parseResponse(self, url, postdata=None, method="get"):
        if method == "get":
            rv = self.app.get(url)

        elif method == "post":
            rv = self.app.post(url,
                           # String conversion required, as the test client ifnores content_type and assumes it is a file
                           data=json.dumps(postdata),
                          content_type='application/json')

        elif method == "put":
            rv = self.app.put(url,
                           # String conversion required, as the test client ifnores content_type and assumes it is a file
                           data=json.dumps(postdata),
                          content_type='application/json')

        elif method == "delete":
            rv = self.app.delete(url)
        else:
            raise("method not supported")

        self.failUnless(rv.status_code == 200, "Http request did not return OK, status: %d, it returned: %s" % (rv.status_code, rv.data))

#        print rv.data

        try:
            obj = loads(rv.data)
        except:
            self.fail("Response not valid json")

        if "error" in obj:
            self.fail("Response retuns error : %s" % obj["error"])

        return obj

    def test_videos(self):
        pass


if __name__ == "__main__":
    unittest.main()
    sys.exit(0)

