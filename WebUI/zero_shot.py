"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import flask
import os
import json
import pymongo
import WebUI
from .login import role_required
import urllib2
import time
from WebUI.querytree import RankFusionQueryTree, ScoreFusionQueryTree
from WebUI import db, app

import sqlite3


dbpath = os.path.join(app.config['STATIC_DIR'], 'data', 'clip_calib.sqlite')


class datastore:
 
    def __init__(self, path):
        self.data_file = path
 
    def connect(self):
        self.conn = sqlite3.connect(self.data_file)
        return self.conn.cursor()
 
    def disconnect(self):
        self.cursor.close()
 
    def free(self, cursor):
        cursor.close()
 
    def write(self, query, values = ''):
        cursor = self.connect()
        if values != '':
            cursor.execute(query, values)
        else:
            cursor.execute(query)
        self.conn.commit()
        return cursor
 
    def read(self, query, values = ''):
        cursor = self.connect()
        if values != '':
            cursor.execute(query, values)
        else:
            cursor.execute(query)
        return cursor


sqlstore = datastore(dbpath)

mod = flask.Blueprint('zero_shot', __name__, )

frames_sfu = WebUI.db["frames_sfu"]

@mod.route('/')
@role_required("kitwarean")
def zero_shot():
    return flask.render_template("zero_shot.html")

@mod.route('/clip_middle')
@role_required("kitwarean")
def clip_middle():
    """
    Searches and returns the image

    /image?id=5167864f0a3ee15fe21b30f6

    """
    # Get variables
    #    img = request.args.get('img', None)
    #    db = request.args.get('db', None)

    clipname = flask.request.args.get('id', None)

    clipobj = db["clips"].find_one({"id" : clipname}, {"duration" : 1, "preview" : 1})
    if clipobj == None:
        return flask.Response("Error in clipobj")
        flask.abort(403)

    return flask.Response(str(clipobj['preview']), mimetype="image/png")

#    duration = clipobj["duration"]
#
#    frameobj = db["frames"].find_one( { "v_id" : clipname, "duration" : int(duration / 2) }, {"thumb" : 1})
#    if frameobj == None:
#        return flask.Response(str(clipobj['preview']), mimetype="image/png")
#
#    return flask.Response(str(frameobj['thumb']), mimetype="image/png")

@mod.route('/query_clip')
@role_required("kitwarean")
def query_clip():
    """
    On submisison gets the query
    "algo" is avg by default
    "rank_fusion" to be supported soon

    """
    algo = flask.request.args.get("algo", "avg")
    tree = flask.request.args.get('tree', None)

    skip = int(flask.request.args.get('skip', '0'))
    limit = int(flask.request.args.get("limit", '30'))

    # Unescape the incoming tree information
    query = json.loads(urllib2.unquote(tree).decode('utf-8'))

    # Parse the query
    obj = {}
    obj["query"] = {}
    obj["query"]["algo"] = algo
    obj["query"]["tree"] = query
    obj["query"]["treestr"] = json.dumps(query)

    tstart = time.time()
    # Process the tree here
    if algo=="avg":
        # Browse through the tree

        # TODO: Work with trees with depth higher than one

        pass
    elif algo=="rank_fusion":

        pass
    else:
        obj["error"] = "Unknown algo"
        return flask.jsonify(obj)

    engine = RankFusionQueryTree(query)
    obj["clips"] = list(engine.process())[:limit]
    obj["count"] = len(obj["clips"])

    # Send the results back
    tend = time.time()
    obj["query"]["time"] = (tend - tstart)
    return flask.jsonify(obj)

    cur = db[dataset].find(query, { "_id" : 1 }, skip=skip, limit=20)
    obj["count"] = cur.count()

    imgs = {}

    count = 0
    iter = 0
    while 1:
        cur = db[dataset].find(query, { "thumb" : 0, "scores" : { "$slice" : [attr_id, 1]}}, skip=skip, limit=limit).sort("scores.%d" % (attr_id), pymongo.DESCENDING)
        iter = iter + 1
        if count >= limit or iter > 5:
            break;

        for animage in cur:
            # Append to the images
            v_id = "Ground truth"
            if "v_id" in animage:
                v_id = animage["v_id"]

            if v_id in imgs:
                imgs[v_id].append([ str(animage["_id"]), animage["scores"][0]])
            else:
                imgs[v_id] = [[ str(animage["_id"]), animage["scores"][0]], ]
                count = count + 1
                skip = skip + limit

    cliplist = []
    for akey in imgs.keys():
        cliplist.append({'clip_id' : akey, 'high_score' : imgs[akey][0][1], 'images' : imgs[akey]})

    obj["clips"] = sorted(cliplist, key=lambda k: k['high_score'], reverse=True)

    tend = time.time()
    obj["query"] = { 'dataset' : dataset, 'skip' : skip, "time" : (tend - tstart), "tree" : query, "sort" : sort }
    return jsonify(obj)


@mod.route('/query_fusion')
@role_required("kitwarean")
def query_fusion():
    """
    On submisison gets the query
    "algo" is avg by default
    "rank_fusion" to be supported soon

    """
    algo = flask.request.args.get("algo", "scorefusion")
    dataset = flask.request.args.get("dataset", "ignored")
    
    tree = flask.request.args.get('tree', None)

    skip = int(flask.request.args.get('skip', '0'))
    limit = int(flask.request.args.get("limit", '30'))

    # Unescape the incoming tree information
    query = json.loads(urllib2.unquote(tree).decode('utf-8'))

    # Parse the query
    obj = {}
    obj["query"] = {}
    obj["query"]["dataset"] = dataset
    obj["query"]["algo"] = algo
    obj["query"]["tree"] = query
    obj["query"]["treestr"] = json.dumps(query)

    tstart = time.time()

    # Process the tree here
    if algo=="scorefusion":
        # Browse through the tree
        engine = ScoreFusionQueryTree(query)
    elif algo=="rankfusion":
        engine = RankFusionQueryTree(query)
    else:
        obj["error"] = "Unknown algo"
        return flask.jsonify(obj)
    
    obj["sql"] = engine.sql()
    
    cur = sqlstore.connect()    
    cur.execute(obj["sql"])

    rows = cur.fetchall()
    obj["clips"] = []

    for row in rows:
        obj["clips"].append(row[0])

    sqlstore.free(cur)
                
    obj["count"] = len(obj["clips"])

    # Send the results back
    tend = time.time()
    obj["query"]["time"] = (tend - tstart)
    return flask.jsonify(obj)

    cur = db[dataset].find(query, { "_id" : 1 }, skip=skip, limit=20)
    obj["count"] = cur.count()

    imgs = {}

    count = 0
    iter = 0
    while 1:
        cur = db[dataset].find(query, { "thumb" : 0, "scores" : { "$slice" : [attr_id, 1]}}, skip=skip, limit=limit).sort("scores.%d" % (attr_id), pymongo.DESCENDING)
        iter = iter + 1
        if count >= limit or iter > 5:
            break;

        for animage in cur:
            # Append to the images
            v_id = "Ground truth"
            if "v_id" in animage:
                v_id = animage["v_id"]

            if v_id in imgs:
                imgs[v_id].append([ str(animage["_id"]), animage["scores"][0]])
            else:
                imgs[v_id] = [[ str(animage["_id"]), animage["scores"][0]], ]
                count = count + 1
                skip = skip + limit

    cliplist = []
    for akey in imgs.keys():
        cliplist.append({'clip_id' : akey, 'high_score' : imgs[akey][0][1], 'images' : imgs[akey]})

    obj["clips"] = sorted(cliplist, key=lambda k: k['high_score'], reverse=True)

    tend = time.time()
    obj["query"] = { 'dataset' : dataset, 'skip' : skip, "time" : (tend - tstart), "tree" : query, "sort" : sort }
    return jsonify(obj)


