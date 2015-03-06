"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""
"""
The tests in this file are for obsolete data and API and may fail
-Djay
"""

import sys
import os
sys.path.append(os.path.abspath(".."))
import WebUI
import unittest
from json import loads

class API_Tests(unittest.TestCase):
    def setUp(self):
        # Creates app and gets a client to it
        self.app1 = WebUI.app
        self.app1.testing = True
        self.app = self.app1.test_client()

    def login_viewer(self):
        # Posts login and password for demo database access
        return self.app.post('/login', data=dict(
            login="demo",
            passwd="jasmine"
        ), follow_redirects=True)


    def testRestURLs(self):
        urls_to_pass = [  "/image",
                                    "/attribute_info",
                                    "/query_score",
                                    "/events",
                                    "/zero_shot",
                                    "/rangy"
                                    ]

        urls_to_fail = ["/something_else"]

        for aurl in urls_to_pass:
            print "Testing: ", aurl ,
            rv = self.app.get(aurl)
            print rv.status_code, " ", aurl
            self.failUnless(rv.status_code == 200 or rv.status_code == 403, aurl)

        # Now test for urls that should not pass
        for aurl in urls_to_fail:
            print "Testing failure of: ", aurl
            rv = self.app.get(aurl)
            self.failUnless(rv.status_code == 404, aurl)

    def testImage(self):
        self.login_viewer()
        obj = self.app.get("/image?id=5167864f0a3ee15fe21b30f6&col=ground_truth")

    def testAttributeInfo(self):
        self.login_viewer()
        obj = self.parseResponse("/attribute_info?label=reading")
        print "Score index: ", obj["scores_index"]
        self.failUnless(obj["scores_index"] == 9)

    def testQueryScore(self):
        self.login_viewer()
        obj = self.parseResponse("/query_score?label=soothing&dataset=algo&min=5.0")
        self.failUnless(obj["count"] == 888)

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




if __name__ == "__main__":
    unittest.main()
