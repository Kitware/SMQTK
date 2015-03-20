"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""
__author__ = 'dhan'

import flask
from werkzeug.wsgi import wrap_file
import re
import pymongo
import gridfs
import bson
import json
import os
import urllib2
import urllib
import time
from flask.views import MethodView
from WebUI.video_process import process_video

this_dir = os.path.dirname(os.path.abspath(__file__))
mod = flask.Blueprint('upload', __name__,
                template_folder="templates",
                static_folder=os.path.join(this_dir, "static")
                )

conn = pymongo.Connection()
datadb = conn["files"]
@mod.route("/")
def oneshot_index():
    print "URL PREFIX: ", mod.static_url_path
    return flask.render_template("index.html")

@mod.route("/process")
def oneshot_process():
    restype = "video"
    result = dict()
    result["resource"] = restype
    if flask.request.method == "POST":
        result["id"] = flask.request.form["id"]
        result["success"] = "POST successfully submitted"
    else:
        result["id"] = flask.request.args["id"]
        task = process_video.delay(result["id"])
        # Send this to tasks and send back the result
        result["task"] = task.__dict__["id"]
        result["status"] = flask.url_for('task', _external=True) + "?id=" + result["task"]

    return flask.jsonify(result)


@mod.route("/upload", methods=("get", "post"))
def upload():
    restype = "video"
    result = dict()
    result["resource"] = restype

    # Process get request
    if flask.request.method == "GET":
        resid = flask.request.args.get("id", "new")
        if resid == "new":
            result["id"] = str(bson.ObjectId())
        else:
            id = bson.ObjectId(resid)
            gf = gridfs.GridFS(datadb, restype)
            fileobj = gf.get(id)
            data = wrap_file(flask.request.environ, fileobj)
            response = flask.current_app.response_class(
                data,
                mimetype=fileobj.content_type,
                direct_passthrough=True)
            response.content_length = fileobj.length
            response.last_modified = fileobj.upload_date
            response.set_etag(fileobj.md5)
            response.cache_control.max_age = 0
            response.cache_control.s_max_age = 0
            response.cache_control.public = True
            response.headers['Content-Disposition'] = 'attachment; filename=' + fileobj.filename
            response.make_conditional(flask.request)
            return response

        result["success"] = "GET request successful"
        return flask.jsonify(result)

    if flask.request.method == "POST":
        result["id"] = flask.request.form["flowIdentifier"]
        result["success"] = "POST request successful"
        result["current_chunk"] = int(flask.request.form["flowChunkNumber"])
        result["total_chunks"] = int(flask.request.form["flowTotalChunks"])
        result["filename"] = flask.request.form["flowFilename"]
        f = flask.request.files["file"]
        first = False

        if result["current_chunk"] == 1:
            first = True
            result["first"] = 1
            # Create a file
            gf = gridfs.GridFS(datadb, restype)
            afile = gf.new_file(chunk_size=1024*1024, filename=result["filename"], _id=bson.ObjectId(result["id"]))
            afile.write(f.read())
            afile.close()

        if not first:
            obj = {}
            obj["n"] = result["current_chunk"] - 1
            obj["files_id"] = bson.ObjectId(result["id"])
            obj["data"] = bson.Binary(f.read())

            datadb[restype + ".chunks"].insert(obj)
            fileobj = datadb[restype + ".files"].find_one({"_id" : obj["files_id"]})
            datadb[restype + ".files"].update({"_id" : obj["files_id"]}, {"$set" : {"length" : fileobj["length"] + len(obj["data"])}})


        if result["current_chunk"] == result["total_chunks"]:
            last = True
            result["last"] = 1
            # Add the attachment id to the
#             if not sessobj.has_key("attachments"):
#                 sessobj["attachments"] = [ {"ref" : ObjectId(resid), "pos" : 0}]
#                 sessobj.validate()
#                 sessobj.save()
# #                print "Inserted attachments", str(sessobj["attachments"])
#             else:
#                 size_before = len(sessobj["attachments"])
#                 sessobj["attachments"].append({"ref" : ObjectId(resid), "pos" : size_before + 1})
#                 sessobj.validate()
#                 sessobj.save()
# #                print "Appended to  attachments", str(sessobj["attachments"])


        print result

    return flask.jsonify(result)


@mod.route('/oneshot_iqr', methods=['GET', 'POST'])
def oneshot_iqr():
    """
    Starts the 
    Can start with Defaults to 3 flash mob ids
    
    537c11fa0a3ee1079f86d2f8
    537c11fa0a3ee1079f86d2fa
    537c11fa0a3ee1079f86d2fc

    Initialize the search results
    GET params query and feature both strings
    feature must be within known features

    @return:
    """
    obj = {}

    # query : will be event kit in case of triage information
    seedstr = flask.request.args.get("query", "%5B%22537c11fa0a3ee1079f86d2f8%22%2C%20%22537c11fa0a3ee1079f86d2fa%22%2C%20%22537c11fa0a3ee1079f86d2fc%22%5D")
    seeds = json.loads(urllib2.unquote(seedstr))

    # Create a distance kernel 
    
    # uid = backend.init_new_search_session(None, obj["query"], dkms[feature_type])

    obj["seeds"] = seeds
    # obj["uid"] = str(uid)
    # obj["spoe"] = "other"
    # obj["next"] = "http://localhost:5003/iqr/refine_search?" + urllib.urlencode({"uid" : str(uid)})
    return flask.jsonify(obj)

@mod.route('/refine_search', methods=['GET', 'POST'])
def refine_search():
    """
       Corresponds to

       def refine_search(self, search_uuid,
                      refined_positive_ids, refined_negative_ids,
                      removed_positive_ids, removed_negative_ids):

    Accepts the +ves -ves and
    @return:
    """
    obj = {}

    # query : will be event kit in case of triage information
    uid = flask.request.args.get("uid", None)
    qpositive = flask.request.args.get("positive", "[]") # json array
    qnegative = flask.request.args.get("negative", "[]") # json array

    if uid == None:
        obj["error"] = "Missing search ID"

    positive = []
    negative = []

    # Convert from HVC to non HVC
    for apos in json.loads(qpositive):
        if len(apos) == 9:
            positive.append(int(apos[3:]))
        else:
            positive.append(int(apos))

    for apos in json.loads(qnegative):
        if len(apos) == 9:
            negative.append(int(apos[3:]))
        else:
            negative.append(int(apos[3:]))

    obj["query"] = {}
    obj["query"]["uid"] = uid
    obj["query"]["positive"] = positive
    obj["query"]["negative"] = negative

    try:
        ret = backend.refine_iqr_search(uid,positive,negative,[],[])
    except Exception as e:
        obj["error"] = str(type(e)) + ": " + str(e)
        return jsonify(obj)

    obj["host"] = ret[0].host
    obj["port"] = ret[0].port
    obj["name"] = ret[0].name
    obj["collection"] = ret[0].collection
    obj["state"] = "http://localhost:5003/iqr/search_state?" + urllib.urlencode({"uid" : uid})
    obj["results"] = "http://localhost:5003/iqr/search_results?" + urllib.urlencode({"uid" : uid})

    return jsonify(obj)

@mod.route('/search_results', methods=['GET', 'POST'])
def search_results():
    """
    Corresponds to
    Accepts uid and pool index
    @return:
    """
    skip = int(flask.request.args.get("skip", "0"))
    limit = int(flask.request.args.get("limit", "20"))

    obj = {}

    # query : will be event kit in case of triage information
    uidstr = flask.request.args.get("query", None)

    if uidstr == None:
        obj["error"] = "Missing search ID"

    uidstr = json.loads(uidstr)

    obj["query"] = {}
    obj["query"]["uid"] = uidstr
    obj["clips"] = []
    states = backend.get_search_sessions()
    obj["sessions"] = []
    for astate in states:
        obj["sessions"].append(str(astate))
    try:
        uid = uuid.UUID(uidstr)
        state = backend.get_iqr_search_state(uid)
        # use the uid of the state and get the information from the database
        col = str(state.uuid)
        obj["collection"] = col
        searchdb[col].ensure_index([("model_id", pymongo.ASCENDING),("probability", pymongo.DESCENDING) ])
        # Force probabilities
        obj["positives"] = list(state.positives)
        obj["negatives"] = list(state.negatives)
        log = ""
        for id in state.positives:
            # log = log + "Found %d"%(searchdb[col].find({"model_id" : "FUSION", "clip_id" : id}).count()) + ", "
            # res = searchdb[col].update({"model_id" : "FUSION", "clip_id" : id}, {"$set" : { "probability" : 1.0}})
            # log = log + "Done %d"%id + ", "
            news = searchdb[col].find_one({"model_id" : "FUSION", "clip_id" : id})
            news["probability"] = 1.0001
            searchdb[col].save(news)
            log = log + "Now : " + str(news)


        for id in state.negatives:
            # log = log + "Found %d"%(searchdb[col].find({"model_id" : "FUSION", "clip_id" : id}).count()) + ", "
            # res = searchdb[col].update({"model_id" : "FUSION", "clip_id" : id}, {"$set" : { "probability" : 0.0}})
            # log = log + "Done %d"%id + ", "
            news = searchdb[col].find_one({"model_id" : "FUSION", "clip_id" : id})
            news["probability"] = 0.0
            searchdb[col].save(news)
            log = log + "Now : " + str(news)

        obj["log"] = log

        allres = searchdb[col].find({"model_id" : "FUSION"}).sort([("probability", pymongo.DESCENDING)]).skip(skip).limit(limit)
        rank = skip + 1
        for one in allres:
            aclip = {}
            aclip["score"] = one["probability"]
            aclip["id"] = "HVC" + str(one["clip_id"]).zfill(6)
            clipobj = db["clips"].find_one({"id" : "HVC" + str(one["clip_id"]).zfill(6)},{"duration" : 1})
            aclip["duration"] = clipobj["duration"]
            aclip["rank"] = rank
            rank = rank + 1
            obj["clips"].append(aclip)
        obj["count"] = len(obj["clips"])

    except Exception as e:
        obj["error"] = str(type(e)) + ": " + str(e)
        return jsonify(obj)

    obj["next"] = "http://localhost:5003/iqr/search_results?" + urllib.urlencode({"uid" : uid, "skip" : skip+limit } )
    return jsonify(obj)


@mod.route('/search_state', methods=['GET', 'POST'])
def search_state():
    """
    Accepts uuid
    @return:
    """
    obj = {}

    # query : will be event kit in case of triage information
    uidstr = flask.request.args.get("uid", None)

    if uidstr == None:
        obj["error"] = "Missing search ID"

    obj["query"] = {}
    obj["query"]["uid"] = uidstr
    # obj["vids"] = []
    states = backend.get_search_sessions()
    obj["sessions"] = []
    for astate in states:
        obj["sessions"].append(str(astate))
    try:
        uid = uuid.UUID(uidstr)
        state = backend.get_iqr_search_state(uid)
        # use the uid of the state and get the information from the database
        col = str(state.uuid)
        # obj["collection"] = col
        # all = searchdb[col].find()
        # for one in all:
        #     obj["vids"].append(one)

        searchdb[col].ensure_index([("model_id", pymongo.ASCENDING)])
        obj["count"] = searchdb[col].find({"model_id" : "FUSION"}).count()
        # obj["stateobj"] = state
    except Exception as e:
        obj["error"] = str(type(e)) + ": " + str(e)
        return jsonify(obj)

    obj["state"] = "http://localhost:5003/iqr/search_state?" + urllib.urlencode({"uid" : str(uidstr)})

    return jsonify(obj)
