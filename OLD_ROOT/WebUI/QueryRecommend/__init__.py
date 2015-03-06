"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""
__author__ = 'ilseo.kim'
__author__ = 'djay.deo'

# Create flask blueprint to suggest recommended query
import flask
from query_recommend import recommend_query
import os

mod = flask.Blueprint('query_recommend', __name__)

# Shared connection
#conn = pymongo.Connection(flask.current_app.config["MONGO_SERVER"])
#db = conn[flask.current_app.config["MONGO_DB"]]


@mod.route('/query')
def query():
    obj = {}
    obj["query"] = flask.request.args.get("query","")

    if len(obj["query"]) == 0:
        obj["error"] = "empty query"
        return flask.jsonify(obj)

    obj["recommendation"] = recommend_query(obj["query"])
    return flask.jsonify(obj)

