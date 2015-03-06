"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""
__author__ = 'dhan'

from login import login_required
import flask
import json
from WebUI import db, app
import re
import os
import os.path as osp
import sys
import urllib2
import time
from querytree import ScoreFusionQueryTree, RankFusionQueryTree
from sqlstore import medtest_sqlstore, eventkits_sqlstore
from utils import jsonify


mod = flask.Blueprint('vcd', __name__)

# Preload vocabulary
path = osp.join(app.config['STATIC_DIR'], 'data/attributes.js')
fin = open(path,"rt")
#scenes = json.loads(fin.read())

# Preload vocabulary
path = osp.join(app.config['STATIC_DIR'], 'data', 'object_bank_compact_vocabulary.js')
fin = open(path,"rt")
objects = json.loads(fin.read())
scenes = [  "transporting",
            "vacationing",
            "learning",
            "water_activities",
            "congregating",
            "playing_outdoors",
            "has_audience",
            "working",
            "using_tools",
            "trees",
            "vegetation",
            "asphalt",
            "pavement",
            "wood_not_part_of_tree",
            "person",
            "ocean",
            "still_water",
            "snow",
            "clouds",
            "natural_light",
            "sunny",
            "electric_or_indoor_lighting",
            "dirty",
            "natural",
            "manmade",
            "open_area",
            "enclosed_area"]

bu = [  "open_area",
        "blank_screen",
        "crowd",
        "doors_windows_rectangular_objects",
        "caption_title",
        "circular_object_large",
        "hand",
        "group_of_people",
        "vehicle",
        "person_upper_torso",
        "human_face",
        "circular_object_small"]

@mod.route('/', methods=["GET"])
@login_required
def vcdi():
    return flask.render_template("vcdi.html")

@mod.route('/suggest', methods=["GET"])
@login_required
def vcdi_suggest():
    query = flask.request.args.get("term", "")

    banksstr =  flask.request.args.get("banks", '["bu", "sc", "ob"]')
    banks = json.loads(banksstr)

    sample_query = []
    try:
        # if len(query) == 0:
        #     return flask.Response(json.dumps(sample_query),  mimetype='application/json')

        # from operators
        operators = ["and", "or", "not", "(", ")"]
        sample_query = sample_query + [{"category": "Operators", "label": string, "value" : string} for string in operators if query in string]

        if "bu" in banks:
            # from BU clusers
            sample_query = sample_query + [{"category": "BottomupClusters", "label": "bu." + string, "value" : "bu." + string} for string in bu if query in "bu." + string]

        if "ob" in banks:
            # from object bank
            sample_query = sample_query + [{"category": "Objects", "label": "ob." + string, "value" : "ob." + string} for string in objects if query in "ob." + string]

        if "sc" in banks:
            # from scene bank
            sample_query = sample_query + [{"category": "Scenes", "label": "sc." + string, "value" : "sc." + string} for string in scenes if query in "sc." + string]

        # sample_query = sample_query + [{"category": "Banks", "label": "bk." + string, "value" : "bk." + string} for string in banks]

        # if query == "a" or query == "a_":
        #     items = db["asr"].find({}).limit(20)
        # else:
        #     regx = re.compile(query, re.IGNORECASE)
        #     items = db["asr"].find({"words" : regx}).limit(20);
        #
        # # from asr features
        # for item in items:
        #     sample_query.append({"category" : "Audio", "value" : "a_" + item["words"][0], "label" : "a_" + item["words"][0]})
    except:
        pass

    return flask.Response(json.dumps(sample_query),  mimetype='application/json')


@mod.route('/search', methods=["GET", "POST"])
@login_required
def search():
    """
    On submisison gets the query
    "algo" is score fusion
    """
    # Defaults that are fixed for the demo
    algo = flask.request.args.get("algo", "scorefusion")
    dataset = flask.request.args.get("dataset", "MEDTEST")

    skip = int(flask.request.args.get('skip', '0'))
    limit = int(flask.request.args.get("limit", '30'))

    query = flask.request.args.get('query', "")

    # Unescape the incoming tree information
    querye = json.loads(urllib2.unquote(query).decode('utf-8'))

    # Parse the query
    obj = {}
    obj["query"] = {}
    obj["query"]["dataset"] = dataset
    obj["query"]["algo"] = algo
    obj["query"]["tree"] = query
    obj["query"]["treestr"] = json.dumps(query)
    obj["query"]["skip"] = skip
    obj["query"]["limit"] = limit

    if len(querye) == 0:
        obj["error"] = "Empty query"
        return flask.jsonify(obj)

    if dataset == "MEDTEST":
        sqlstore = medtest_sqlstore

    elif dataset == "EVENTKITS":
        sqlstore = eventkits_sqlstore
    else:
        obj["error"] = "Unknown dataset"
        return flask.jsonify(obj)

    tstart = time.time()

    # Process the tree here
    if algo=="scorefusion":
        # Browse through the tree
        engine = ScoreFusionQueryTree(querye, mode="parselogic", skip=skip, limit=limit)
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
    count = 0
    for row in rows:
        # Verify that preview is ready
        clipobj = db.clips.find_one({"id" : row[0]}, {"middle_preview" : 1, "duration" : "1"})
        # if "middle_preview" in clipobj:
        aclip = {}
        aclip["id"] = row[0]
        aclip["score"] = row[-1]
        aclip["attribute_scores"] = row[1:-1]
        count = count + 1
        aclip["rank"] = skip + count
        if "duration" in clipobj:
            aclip["duration"] = clipobj["duration"]
        else:
            aclip["duration"] = "unknown"
        obj["clips"].append(aclip)

    sqlstore.free(cur)

    obj["count"] = len(obj["clips"])

    # Send the results back
    tend = time.time()
    obj["query"]["time"] = (tend - tstart)
    return flask.jsonify(obj)


@mod.route('/info_view')
@login_required
def info_view ():
    # Event kit number (integer) compulsory
    # Either give rank

    obj = {}
    obj["ready"] = 0

    clip_id = flask.request.args.get('clip',"")
    kit = flask.request.args.get('query',"1")
    rank = flask.request.args.get('rank',"1")
    info = bool(flask.request.args.get('info', False))

    obj["query"] = {}
    obj["query"]["kit"] = kit
    obj["query"]["rank"] = rank
    obj["query"]["clip"] = clip_id
    obj["query"]["info"] = info



    if len(clip_id) == 0:
        obj["error"] = "clip not specified, correct way is clip=HVC323096"
        if(info == True):
            return flask.jsonify(obj)
        else:
            # Render empty form
            return flask.render_template("vcd_info_view.html", info=obj, video_url_prefix=flask.current_app.config["VIDEO_URL_PREFIX"])

    # Get kit text
    kit_text = "Video content exploration"
    attribute = "vcd_v" + str(kit)

    # Get the clip id from the result
    obj["clip"] = clip_id
    obj["score"] = 0
    obj["kit_text"] = kit_text
    obj["attribute"] = attribute
    # Need to get the clip in different way

    clip = db["clips"].find_one({attribute : {"$exists" : 1}, "id" : clip_id }, {"_id" : 1, "duration": 1, "scores": 1, "id" : 1})

    if clip == None:
        obj["error"] = "Information not found for that clip"
        if(info == True):
            return flask.jsonify(obj)
        else:
            # Render empty form
            return flask.render_template("vcd_info_view.html", info=obj, video_url_prefix=flask.current_app.config["VIDEO_URL_PREFIX"])

    # Find duration

    if "duration" in clip:
        obj["duration"] = clip["duration"]
    else:
        obj["duration"] = "unknown"

    if info == True:
        return flask.jsonify(obj)
    else:
        obj["ready"] = 1
        return flask.render_template("vcd_info_view.html", info=obj, video_url_prefix=flask.current_app.config["VIDEO_URL_PREFIX"])

@mod.route("/video_names")
@login_required
def vidnames():
    query = flask.request.args.get("term", "")
    dataset = flask.request.args.get("dataset", "MEDTEST")

    sample_query = []
    import re
    regx = re.compile("^" + query , re.IGNORECASE)
    print regx.pattern
    clipnames = db.clips.find({"id" : regx, "vcd_v3" : {"$exists" : 1} },{"id" : 1, "_id" : 0}).limit(20)
    for aclip in clipnames :
        sample_query.append(aclip["id"])

    return flask.Response(json.dumps(sample_query),  mimetype='application/json')


@mod.route('/info')
def info():
    # Event kit number (integer) compulsory
    # Either give rank

    obj = {}

    clip_id = flask.request.args.get('clip',"")
    field = flask.request.args.get('algo',"")
    limit = int(flask.request.args.get("limit", '5'))
    obj["query"] = {}
    obj["query"]["clip"] = clip_id
    obj["query"]["algo"] = field

    if len(clip_id) == 0:
        obj["error"] = "clip not specified, correct way is clip=HVC323096"
        return flask.jsonify(obj)

    if len(field) == 0:
        obj["error"] = "Field not specified, correct way is field=vcd_v1"
        return flask.jsonify(obj)

    # Need to get the clip in different way
    clipobj = db["clips"].find_one({"id" : clip_id }, {"_id" : 1, field : 1, "duration": 1, "id" : 1})

    if clipobj == None:
        obj["error"] = "Clip not found"
        return flask.jsonify(obj)

    if not field in clipobj:
        obj["error"] = str(field) + " result not recorded for clip"
        return flask.jsonify(obj)

    # Get the clip id from the result
    obj["clip"] = clip_id
    obj["algo"] = field
    # obj["fulldata"] = clipobj[field]

    version = int(field[-1])

    if version > 2:
        ev = clipobj[field]["clip"]["evidences"]
    else:
        ev = clipobj[field]["top_attributes"]

    # filter ev so that only top 5 from each group remain

    # find list of groups
    groups = set()
    classified = {}

    for anevidence in ev:
        grp = anevidence["group"]
        if not grp in groups:
            groups.add(grp)
            classified[grp] = []

        if len(classified[grp]) < 5:
            classified[grp].append(anevidence)

    # sort and eliminate the beyond 5 objects in each group


    obj["groups"] = list(groups)
    obj["evidences"] = []
    for agrp in classified:
        obj["evidences"] = obj["evidences"] + classified[agrp]

    if "duration" in clipobj:
        obj["duration"] = clipobj["duration"]
    else:
        obj["duration"] = "unknown"
    return flask.jsonify(obj)


