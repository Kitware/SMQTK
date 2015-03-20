"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

"""
Generic API

To get frames

"""

import flask
import os
import json
import pymongo
import bson
import WebUI
from WebUI.utils import jsonify
import random

mod = flask.Blueprint('api', __name__)

# Shared connection
#conn = pymongo.Connection(flask.current_app.config["MONGO_SERVER"])
#db = conn[flask.current_app.config["MONGO_DB"]]
clips = WebUI.db["clips"]
clipscores = WebUI.db["clip_scores"]
frames = WebUI.db["frames"]

# Preload kit texts
path = os.path.join(WebUI.app.config['STATIC_DIR'], "data/kit_texts.js")
fin = open(path,"rt")
kit_texts = json.loads(fin.read())

# Preload top results
# Mostly obsolete
results = {}
for kit in [1,6,8,9,14]:
    path = os.path.join(WebUI.app.config['STATIC_DIR'], "data/demo3", str(kit) + ".json")
    fin = open(path,"rt")
    val = json.loads(fin.read())
    results[str(kit)] = val
    fin.close()

# Preload vocabulary
path = os.path.join(WebUI.app.config['STATIC_DIR'], "data/similarity.js")
fin = open(path,"rt")
similarity = json.loads(fin.read())

# Preload vocabulary
path = os.path.join(WebUI.app.config['STATIC_DIR'], 'data/vocabulary.js')
fin = open(path,"rt")
vocab = json.loads(fin.read())

@mod.route('/event_results')
def event_results():
    # Event ID
    # DataSet
    # Now obsolete use /event_results_from_training

    obj = {}
    obj["query"] = {}

    kit = flask.request.args.get('kit',"1")
    obj["query"]["kit"] = kit

    training = flask.request.args.get('training',"0Ex")
    obj["query"]["training"] = training

    attribute = "scores.0.E%03d"%(int(kit))
    obj["query"]["attribute"] = attribute

    skip = int(flask.request.args.get('skip', "0"))
    limit = int(flask.request.args.get('limit', "10"))
    obj["query"]["skip"] = skip
    obj["query"]["limit"] = limit

    # Load the clip results
    clipresults = clips.find({attribute : {"$exists" : 1}}, {"_id" : 1, "duration": 1, "scores": 1, "id" : 1}).sort(attribute,pymongo.DESCENDING).skip(skip).limit(limit)
    obj["clips"] = []

    # Load duration
    count = 1
    for aclip in clipresults:
        aclip["rank"] = skip + count
        aclip["score"] = aclip["scores"][0]["E%03d"%int(kit)]
        del aclip["scores"]
        obj["clips"].append(aclip)
        count = count + 1

    # obj["clips"] = [aclip["rank"] =  for aclip in clipresults]
    return jsonify(obj)
@mod.route('/triage_info_from_training')
def triage_info_from_training():
    # Event kit number (integer)
    # algorithm from ["ob2124_max_hik_to_22", "ob2124_max_hik_to_177",  "scene102_avg_hik", "scene102_avg_hik_to_30"]
    # clip id either integer or in the form of  "HVC556585"
    # If strict, will chop any frames that are not in the
    obj = {}
    obj['query'] = {}

    kit = flask.request.args.get('kit',"1")
    obj['query']['kit'] = kit

    clipname = flask.request.args.get('clip',"HVC556585")
    obj['query']['clipname'] = clipname

    training = flask.request.args.get('training',"100Ex")
    obj["query"]["training"] = training

    # Check if clipname is id
    use = "name"
    try:
        clipid = bson.ObjectId(clipname)
        use = "id"
    except:
        use = "name"

    algo = flask.request.args.get('algo',"ob2124_max_hik_to_22")
    strict = flask.request.args.get('strict',"")

    obj["query"]["algo"] = algo
    obj["query"]["strict"] = strict
    obj["query"]["using"] = use

    if use == "name":
        if clipname[:3] != "HVC":
            clipname = "HVC%06d"%(int(clipname))

    try:
        kit = int(kit)
    except:
        obj["error"] = "Kit should be integer"
        return flask.jsonify(obj)

    fieldname = "E%03d_%s"%(int(kit),algo)
    obj["query"]["fieldname"] = fieldname

    valid_algorithms = ["OB_max_avg_positive_hik_21",
        "OB_max_avg_positive_hik_177",
        "ObjectBank_ver2_max_positive_hik_21",
        "ObjectBank_ver2_max_positive_hik_177",
        "sun_attribute_avg_positive_hik_27"]

    if not algo in valid_algorithms:
        obj["error"] = "Unknown algorithm"
        return flask.jsonify(obj)

    if use == "name" :
        data = clips.find_one({"id" : clipname}, {"middle_preview" : 0, "preview" : 0})
    elif use == "id":
        data = clips.find_one({"_id" : id}, {"middle_preview" : 0, "preview" : 0})

    # TODO: find
    if data == None:
        obj["error"] = "Clip not found"
        return flask.jsonify(obj)

    # TODO: find
    if not fieldname in data:
        obj["error"] = "Triage info not found"
        return flask.jsonify(obj)


    del data["_id"]
#    obj["evidences"] = data["evidence"]["clip"]
#     raise Exception()
#     obj["fulldata"] = data
    obj["evidences"] = data[fieldname]["clip"]["evidences"][:5]
    if "duration" in data:
        obj["duration"] = data["duration"]
    else:
        obj["duration"] = "unknown"

    return flask.jsonify(obj)


@mod.route('/event_results_from_training')
def event_results_from_training():
    # Event ID
    # DataSet

    obj = {}
    obj["query"] = {}

    kit = flask.request.args.get('kit',"1")
    obj["query"]["kit"] = kit

    training = flask.request.args.get('training',"100Ex")
    obj["query"]["training"] = training

    attribute = "scores_eval.%s.E%03d."%(training, int(kit))
    obj["query"]["attribute"] = attribute

    skip = int(flask.request.args.get('skip', "0"))
    limit = int(flask.request.args.get('limit', "10"))
    obj["query"]["skip"] = skip
    obj["query"]["limit"] = limit
    obj["query"] ["attribute"]= attribute

    # Load the clip results
    clipresults = clipscores.find({}, {"_id" : 1, attribute: 1, "id" : 1}).sort(attribute,pymongo.DESCENDING).skip(skip).limit(limit)
    obj["clips"] = []

    # Load duration
    count = 1
    for aclip in clipresults:
        aclip["rank"] = skip + count
        aclip["score"] = aclip["scores_eval"][training]["E%03d"%int(kit)]
        aclipobj = clips.find_one({"id" : aclip["id"]},{"duration" : 1})
        if "duration" in aclipobj:
            aclip["duration"] = aclipobj["duration"]
        del aclip["scores_eval"]
        obj["clips"].append(aclip)
        count = count + 1

    return jsonify(obj)



@mod.route('/random_results')
def random_results():
    # skip
    # limit

    obj = {}
    obj["query"] = {}

    skip = int(flask.request.args.get('skip', "0"))
    limit = int(flask.request.args.get('limit', "20"))
    obj["query"]["skip"] = skip
    obj["query"]["limit"] = limit
    obj["clips"] = []

    count = 0
    for i in range(limit/2):
        # Get a random
        skip = int(random.random()*200)

        # Load the clip results
        clipres = clips.find({"dataset": "MEDTEST", "middle_preview" : {"$exists" : 1}},{"id" : 1, "duration" : 1, "_id" : 0}).skip(skip).limit(2)
        for aclip in clipres:
            count = count + 1
            aclip["rank"] = count
            obj["clips"].append(aclip)


    # obj["clips"] = [aclip["rank"] =  for aclip in clipresults]
    return jsonify(obj)



@mod.route('/top100_results')
def top100_results():
    # Event ID
    # DataSet

    obj = {}

    kit = flask.request.args.get('kit',"1")
    skip = int(flask.request.args.get('skip', "0"))
    limit = int(flask.request.args.get('limit', "10"))

    path = os.path.join(WebUI.app.config['STATIC_DIR'], 'data/demo3', kit + '.json')
    try:
        val = results[kit]
        obj["clips"] = val[skip:skip+limit]
    except:
        obj["error"] = "Data not found"

    # Load duration
    for aclip in obj["clips"]:
        clipobj = clips.find_one({"id" : aclip[0]})
        if clipobj == None:
            aclip.append("Unknown")
        else:
            aclip.append(clipobj["duration"])

    obj["query"] = { 'kit' : kit, 'skip' : skip, "path" : path }
    return flask.jsonify(obj)

@mod.route('/triage_info')
def triage_info():
    # Now obsolete use /triage_info_from_training
    # Event kit number (integer)
    # algorithm from ["ob2124_max_hik_to_22", "ob2124_max_hik_to_177",  "scene102_avg_hik", "scene102_avg_hik_to_30"]
    # clip id either integer or in the form of  "HVC556585"
    # If strict, will chop any frames that are not in the

    kit = flask.request.args.get('kit',"1")
    clipname = flask.request.args.get('clip',"HVC556585")
    # Check if clipname is id
    use = "name"
    try:
        clipid = bson.ObjectId(clipname)
        use = "id"
    except:
        use = "name"

    algo = flask.request.args.get('algo',"ob2124_max_hik_to_22")
    strict = flask.request.args.get('strict',"")

    obj = {}

    obj["query"] = {}
    obj["query"]["kit"] = kit
    obj["query"]["algo"] = algo
    obj["query"]["strict"] = strict
    obj["query"]["using"] = use

    if use == "name":
        if clipname[:3] != "HVC":
            clipname = "HVC%06d"%(int(clipname))

    obj["query"]["clip"] = clipname

    try:
        kit = int(kit)
    except:
        obj["error"] = "Kit should be integer"
        return flask.jsonify(obj)

    fieldname = "E%03d_%s"%(int(kit),algo)
    obj["query"]["fieldname"] = fieldname

    valid_algorithms = ["OB_max_avg_positive_hik_21",
        "OB_max_avg_positive_hik_177",
        "ObjectBank_ver2_max_positive_hik_21",
        "ObjectBank_ver2_max_positive_hik_177",
        "sun_attribute_avg_positive_hik_27"]

    if not algo in valid_algorithms:
        obj["error"] = "Unknown algorithm"
        return flask.jsonify(obj)

    if use == "name" :
        data = clips.find_one({"id" : clipname}, {"middle_preview" : 0, "preview" : 0})
    elif use == "id":
        data = clips.find_one({"_id" : id}, {"middle_preview" : 0, "preview" : 0})

    # TODO: find
    if data == None:
        obj["error"] = "Clip not found"
        return flask.jsonify(obj)

    # TODO: find
    if not fieldname in data:
        obj["error"] = "Triage info not found"
        return flask.jsonify(obj)


    del data["_id"]
#    obj["evidences"] = data["evidence"]["clip"]
#     raise Exception()
#     obj["fulldata"] = data
    obj["evidences"] = data[fieldname]["clip"]["evidences"][:5]
    if "duration" in data:
        obj["duration"] = data["duration"]
    else:
        obj["duration"] = "unknown"

    return flask.jsonify(obj)





@mod.route('/frames', methods=["GET", "POST"])
def clip():
    clipname = flask.request.args.get('clip', '003237').zfill(6)

    if clipname[:3] != "HVC":
        clipname= "HVC" + clipname

    framelist = frames.find({'v_id' : clipname}, {"thumb" : 0}).sort("duration")

    obj = {}
    obj['frames'] = []

    durs = set()
    for aframe in framelist:
        if not aframe["duration"] in durs:
            durs.add(aframe["duration"])
            aframe["_id"] = str(aframe["_id"])
            obj['frames'].append(aframe)

    obj["query"] = { "clip" : clipname}
    return flask.jsonify(obj)

@mod.route('/triage_info_view')
def triage_info_view ():
    # Event kit number (integer) compulsory
    # Either give rank

    kit = flask.request.args.get('kit',"1")
    rank = flask.request.args.get('rank',"1")

    # clipname = flask.request.args.get('clip',"HVC556585")
    # algo = flask.request.args.get('algo',"ob2124_max_hik_to_22")
    # strict = flask.request.args.get('strict',"")

    obj = {}

    obj["query"] = {}
    obj["query"]["kit"] = kit
    obj["query"]["rank"] = rank

    try:
        kit = int(kit)
    except:
        obj["error"] = "Kit should be integer"
        return flask.jsonify(obj)

    try:
        rank = int(rank)
    except:
        obj["error"] = "Rank should be integer"
        return flask.jsonify(obj)

    # Get kit text
    kit_text = [i for i in kit_texts if i["kit"] == str(kit)]
    attribute = "scores.0.E%03d"%(int(kit))

    # Need to get the clip in different way
    clip = WebUI.db["clips"].find({attribute : {"$exists" : 1}}, {"_id" : 1, "duration": 1, "scores": 1, "id" : 1}).sort(attribute,pymongo.DESCENDING).skip(rank-1).limit(1)

    # Get the clip id from the result

    obj["clip"] = clip[0]["id"]
    obj["score"] = clip[0]["scores"][0]["E%03d"%(int(kit))]
    obj["kit_text"] = kit_text[0]["name"]

    # Find duration
    clipobj = WebUI.db["clips"].find_one({"id" :obj["clip"]})
    if "duration" in clipobj:
        obj["duration"] = clipobj["duration"]
    else:
        obj["duration"] = "unknown"
    # return flask.jsonify(obj)
    return flask.render_template("triage_info_view.html", triage_info=obj, video_url_prefix=flask.current_app.config["VIDEO_URL_PREFIX"])


@mod.route('/triage_info_view_from_training')
def triage_info_view_from_training():
    # Event kit number (integer) compulsory
    # Either give rank

    kit = flask.request.args.get('kit',"1")
    rank = flask.request.args.get('rank',"1")
    training = flask.request.args.get('training',"100Ex")

    # clipname = flask.request.args.get('clip',"HVC556585")
    # algo = flask.request.args.get('algo',"ob2124_max_hik_to_22")
    # strict = flask.request.args.get('strict',"")

    obj = {}

    obj["query"] = {}
    obj["query"]["kit"] = kit
    obj["query"]["rank"] = rank
    obj["query"]["training"] = training

    try:
        kit = int(kit)
    except:
        obj["error"] = "Kit should be integer"
        return flask.jsonify(obj)

    try:
        rank = int(rank)
    except:
        obj["error"] = "Rank should be integer"
        return flask.jsonify(obj)

    if not training in ["0Ex", "10Ex", "100Ex"]:
        obj["error"] = "Only 0Ex, 10Ex or 100Ex are supported training datasets"
        return flask.jsonify(obj)

    # Get kit text
    kit_text = [i for i in kit_texts if i["kit"] == str(kit)]

    attribute = "scores_eval.%s.E%03d."%(training, int(kit))

    # Load the clip results
    clip = clipscores.find({}, {"_id" : 1, attribute: 1, "id" : 1}).sort(attribute,pymongo.DESCENDING).skip(rank-1).limit(1)

    # Get the clip id from the result
    obj["clip"] = clip[0]["id"]
    obj["score"] = clip[0]["scores_eval"][training]["E%03d"%(int(kit))]
    obj["kit_text"] = kit_text[0]["name"]
    obj["training"] = training;
    # Find duration
    clipobj = WebUI.db["clips"].find_one({"id" :obj["clip"]})
    if "duration" in clipobj:
        obj["duration"] = clipobj["duration"]
    else:
        obj["duration"] = "unknown"
    # return flask.jsonify(obj)
    return flask.render_template("triage_info_view.html", triage_info=obj, video_url_prefix=flask.current_app.config["VIDEO_URL_PREFIX"])


