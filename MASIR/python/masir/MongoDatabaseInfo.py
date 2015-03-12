# coding=utf-8

import pymongo


class MongoDatabaseInfo (object):
    """
    Small class for encapsulating mongo database connection information
    """

    def __init__(self, host, port, db_name):
        self.host = host
        self.port = int(port)
        self.db = db_name

    def new_db_connection(self):
        """
        Return a new Mongo database connection object
        """
        return pymongo.MongoClient(self.host, self.port)[self.db]


def get_db_from_info(info):
    """
    Generate and return a new connection object based the given connection '
    information object

    :param info: Connection information object
    :type info: MongoDatabaseInfo

    :return: Mongo DB connection object
    :rtype: pymongo.Database

    """
    return pymongo.MongoClient(info.host, info.port)[info.db]
