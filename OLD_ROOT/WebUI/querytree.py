"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

"""
Class for loading boolean expression trees
and returning sql to get the results
"""

import pymongo
from WebUI import db
from bson import ObjectId
from WebUI.sqlstore import datastore
from sets import Set

# Also create an instance of sqlite3 connection

class QueryTree(object):
    def __init__(self, tree, algo="avg", indent="", debug=False, mode="dynatree", skip=0, limit=30):
        """
        Incoming tree must have a single entry as key

        """
        self.algo = algo
        self.tree = tree
        self.debug = debug
        self.indent = indent
        self.mode = mode
        self.skip = skip
        self.limit=limit


        if type(tree) == type(u""):
            # leaf
            self.children = []
            self.title = tree
            return

        if mode == "parselogic":
            self.parselogic()
            return

        self.title = tree.keys()[0]
        self.children = []
        self.data = None
        self.indent = indent
        self.debug=debug

        if type(tree[self.title]) == type([]):
            for atree in tree[self.title]:
                self.children.append(self.__class__(atree, algo=self.algo, indent=self.indent+" ", debug=self.debug))
        else:
            self.data = tree[self.title]

    def parselogic(self):
        self.children = []
        self.title = "$" + self.tree["logic"]

        for atree in self.tree["terms"]:
            self.children.append(self.__class__(atree, algo=self.algo, indent=self.indent+" ", debug=self.debug, mode=self.mode))

    def sql(self):
        """
        Creates and returns the sql statement based on the tree loaded
        in the base implementation returns all the columns involved in tree with ordering in the order
        in which they appear
        """

        print "Expression: ", self.expression()

        sql = "SELECT " + ','.join(self.expression()) + " FROM ob"
        return sql

    def expression(self):
        """
        Computes the results based on the tree loaded
        else returns empty results
        Recursively calls the parent if any child nodes are not leaf
        """
        # For now just print the tree

        if self.title[0] == "$":
            if self.debug:
                print self.indent + self.title + ", %d Children"%(len(self.children))
            results = []
            for child in self.children:
                # Determine how to combine results
                aresult = child.expression()
                results = results + aresult
            return results
        else:
            result = self.title
            if self.debug:
                print self.indent + "Processing: " + self.title
                # Query the database
            return [result]
            # Must have children

class AvgQueryTree(QueryTree):
    def process(self):
        """
        Computes the results based on the tree loaded
        else returns empty results
        Recursively calls the parent self if any child nodes are not leaf

        For and the score of the clip is the score of the maximum of the attributes selected
        For each clip, find the score of the attributes of all clips

         - Aggregate the attribute scores for all clips
        """

        # Create a new collection
#        prefix = "score"
#        col = db["fusion"]
#        fieldname = time.t
        # Insert the meta object so that this collection can be garbage collected

        # keep only first few


        # For now just print the tree

        if self.title[0] == "$":
            if self.debug:
                print self.indent + self.title + ", %d Children"%(len(self.children))
            results = []
            sum = 1

            # Evaluate all the non-attribute nodes first
            for child in self.children:
                # Determine how to combine results
                aresult = child.process()
                results.append(aresult)
                sum = sum + aresult

            return sum


        else:
            # Create a collection of scores the threshold doesn't matter

            result = []
            colname = "clip_calib_" + self.title
            count = db[colname].find({"score" : self.data}).count()
            if count > 0:
                for ares in db[colname].find({"score" : self.data}):
                    result.append(ares["clip"])
                # Use the information in .data to query the database to query the data

            result = 1
            if self.debug:
                print self.indent + "Avg: " + self.title
                # Query the database

            return result
            # Must have children

class BooleanQueryTree(QueryTree):
    def sql(self):
        """
        Example query is as follows -
        SELECT scary, working  FROM clip_calib_scores WHERE (scary > 0.1 AND working > 0.1) ORDER BY scary DESC LIMIT 0,50;
        """
        # Recursively invokes expression

        items = ", ".join(self.expression())

        sql = "SELECT v_id, " + items + " FROM ob WHERE " + self.query()
        sql = sql + " ORDER BY " + items + " DESC LIMIT 0,50;"

        return sql

    def query(self):
        """
        Computes the results based on the tree loaded
        else returns empty results
        Recursively calls itself if any child nodes are not leaf
        """
        # For now just print the tree

        if self.title[0] == "$":
            if self.debug:
                print self.indent + self.title + ", %d Children"%(len(self.children))


            results = []

            for child in self.children:
                # Determine how to combine results
                aresult = child.query()
                # Combine using the title operator
                if type(aresult) == type(u"d"):
                    # Got a string
                    results = results + [aresult]
                else:
                    results = results + aresult

            return " (" + ( " " + self.title[1:].upper() + " ").join(results) + ")"
        else:
            key = self.data.keys()[0]
            expr = self.title
            if key == "$gte":
                expr = expr + " >= "
            expr = expr + str(self.data[key])

            return expr


class ScoreFusionQueryTree(QueryTree):
    def sql(self):
        """
        Example query is as follows -
        SELECT scary, working,  (scary + working) /2 as result FROM clip_calib_scores ORDER BY result DESC LIMIT 0,50;
        """
        terms = self.expression()
        # Recursively invokes expression
        items = ", ".join(terms)

        tables = set()
        # for i, item in enumerate(terms):
        #     if item[:2] not in ["bu", "sc", "ob"]:
        #         terms[i]  = "sc." + item

        # print terms

        for anitem in terms:
            if anitem[:2] == "bu":
                tables.add("bu")
            if anitem[:2] == "sc":
                tables.add("sc")
            if anitem[:2] == "ob":
                tables.add("ob")
                ob = 1

        tablestr = ", ".join(tables)

        tableslist = list(tables)

        if len(tableslist) == 1:
            wherejoin = " "

        if len(tables) == 2:
            wherejoin = "WHERE " + tableslist[0] + ".v_id = " + tableslist[1] + ".v_id"

        if len(tables) == 3:
            wherejoin = "WHERE ob.v_id = sc.v_id and ob.v_id = bu.v_id "
        # if(bu = 1)

        sql = "SELECT " + tableslist[0] + ".v_id, " + items + ", " + self.query() + " as result FROM " + tablestr + " " + wherejoin
        sql = sql + " ORDER BY result DESC LIMIT " + str(self.skip) +"," + str(self.limit) + ";"

        return sql

    def query(self):
        """
        Computes the results based on the tree loaded
        else returns empty results
        Recursively calls itself if any child nodes are not leaf
        """
        # For now just print the tree

        if self.title[0] == "$":
            if self.debug:
                print self.indent + self.title + ", %d Children"%(len(self.children))

            results = []
            # raise

            for child in self.children:
                # Determine how to combine results
                aresult = child.query()
                # Combine using the title operator
                print type(aresult), type(u"d")
                print aresult
                if type(aresult) == type(u"d"):
                    # Got a string
                    results = results + [aresult]
                else:
                    results = results + aresult

            if self.title[1:] == "and":
                # Use average
                return "((" + " + ".join(results) + ") / " + str(len(results)) + ") "

            elif self.title[1:] == "or":
                # Use max (easy)
                return "MAX(" + ",".join(results) + ")"

        else:
            return self.title



class RankFusionQueryTree(QueryTree):
    def sql(self):
        """
        Example query is as follows -
        SELECT scary, working,  (scary + working) /2 as result FROM clip_calib_scores ORDER BY result DESC LIMIT 0,50;
        """
        # Recursively invokes expression

        items = ", ".join(self.expression())
        print items
        
        
        sql = "SELECT v_id, " + items + ", MAX( ( RANK() OVER (ORDER BY " + self.query() + " ) as result FROM clip_calib_scores "
        sql = sql + " ORDER BY result ASC LIMIT " + str(self.skip) +"," + str(self.limit) + ";"

        return sql

    def query(self):
        """
        Computes the results based on the tree loaded
        else returns empty results
        Recursively calls itself if any child nodes are not leaf
        """
        # For now just print the tree

        if self.title[0] == "$":
            if self.debug:
                print self.indent + self.title + ", %d Children"%(len(self.children))

            results = []

            for child in self.children:
                # Determine how to combine results
                aresult = child.query()
                # Combine using the title operator
                print type(aresult), type(u"d")
                print aresult
                if type(aresult) == type(u"d"):
                    # Got a string
                    results = results + [aresult]
                else:
                    results = results + aresult

            if self.title[1:] == "and":
                # Use average
#                return "((" + " + ".join(results) + ") / " + str(len(results)) + ") "
                return  "), RANK() OVER (ORDER BY ".join(results) + ") )" 

            elif self.title[1:] == "or":
                # Use max (easy)
                return "MAX(" + ",".join(results) + ")"

        else:
            return self.title

