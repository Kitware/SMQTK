"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""
"""
 Blueprint for Query refinement
 __author__ = 'dhan'
"""

import json
import flask
import sys, os
from WebUI import db, conn, app
import urllib2
import urllib
import uuid
import pymongo
from .cache import tc, load_known_features, known_features, mgr
import logging
logger = logging.getLogger("WebUI.iqr")
load_known_features()

import logging
from login import login_required
mod = flask.Blueprint('query_refinement', __name__)

from WebUI.utils import jsonify
import numpy as np

logging.basicConfig(filename='/tmp/sqmtk.log',level=logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s')

from SMQTK_Backend.SmqtkController import SmqtkController as Controller
from SMQTK_Backend.DistanceKernelInterface import DistanceKernel_File_IQR, DistanceKernel_File_Archive
from SMQTK_Backend.FeatureMemory import initFeatureManagerConnection, getFeatureManager
from SMQTK_Backend.FeatureMemory import DistanceKernel, FeatureMemory
#
data_path = app.config["DATA_DIR"]

iqr_kernal = DistanceKernel_File_IQR( data_path + "/clipids_eventkit.txt", data_path + "/bg_flag_eventkit.txt", data_path + "/kernel_eventkit.npy")
medtest_kernal = DistanceKernel_File_Archive( data_path + "/clipids_eventkit.txt", data_path + "/clipids_medtest.txt", data_path + "/kernel_medtest.npy")

config = Controller.generate_config()
# for fusion etc
config.set("smqtk_controller", "mongo_database", "smqtk_controller")
#config.set("ecd_controller", "default_classifier_config_file", os.path.abspath(os.path.dirname(__file__) + "/.
# ./classifier_config.json"))

# TODO: Remove these as no longer needed
config.set('ecd_controller', 'dmi_clip_id_map', data_path + '/clipids_eventkit.txt')
config.set('ecd_controller', 'dmi_bg_data_map', data_path + '/bg_flag_eventkit.txt')
config.set('ecd_controller', 'dmi_kernel_data', data_path + '/kernel_eventkit.npy')
config.set('vcd_controller', 'store_name', data_path + '/vcd_store/store.sqlite')

backend = Controller(config)

localconn = pymongo.Connection("localhost")
searchdb = localconn["smqtk_controller"]

fin = open(data_path + '/clipids_eventkit.txt')
clipids = [ int(aclip) for aclip in fin.readlines()]

logging.log(logging.INFO, "Clipids loaded = %d" %(len(clipids)))

def shutdown():
    backend.shutdown()

@mod.route('/', methods=['GET', 'POST'])
@login_required
def iqr():
    return flask.render_template("iqr.html", known_features=known_features)

@mod.route('/init_new_search_session', methods=['GET', 'POST'])
def init_new_search_session():
    """
    Initialize the search results
    GET params query and feature both strings
    feature must be within known features

    @return:
    """
    request = flask.request

    obj = {}

    # query : will be event kit in case of triage information
    query = request.args.get("query", "")
    feature_type = request.args.get("feature", "csift")

    # Prepare kernal
    if query:
        obj["query"] = urllib2.unquote("query")
        obj["feature"] = feature_type

    assert feature_type in known_features
    dkm = tc.get_dk(feature_type)
    uid = backend.init_new_search_session(None, obj["query"], dkm )

    obj["uid"] = str(uid)
    obj["spoe"] = "other"
    obj["next"] = "http://localhost:5003/iqr/refine_search?" + urllib.urlencode({"uid" : str(uid)})

    return jsonify(obj)

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
            try:
                clipobj = db["clips"].find_one({"id" : "HVC" + str(one["clip_id"]).zfill(6)},{"duration" : 1})
                aclip["duration"] = clipobj["duration"]
            except:
                aclip["duration"] = "unknown"
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

@mod.route('/archive_search', methods=['GET', 'POST'])
def archive_search():
    obj = {}
    obj["query"] = {}
    sessionstr = flask.request.args.get("session", None)
    statestr = flask.request.args.get("state", None)
    obj["query"]["session"] = sessionstr
    obj["query"]["state"] = statestr


    try:
        stateuid = uuid.UUID(statestr)
        sessionuid = uuid.UUID(sessionstr)

        ret = backend.archive_search(sessionuid, stateuid, medtest_kernal)

        obj["host"] = ret[0].host
        obj["port"] = ret[0].port
        obj["name"] = ret[0].name
        obj["collection"] = ret[0].collection

        return jsonify(obj)

    except Exception as e:
        obj["error"] = str(type(e)) + ": " + str(e)
        return jsonify(obj)

@mod.route('/archive_search_results', methods=['GET', 'POST'])
def archive_search_results():
    skip = int(flask.request.args.get("skip", "0"))
    limit = int(flask.request.args.get("limit", "20"))
    query = flask.request.args.get("query", "")

    try:
        uidstr = json.loads(query)
    except:
        uidstr = query

    obj = {}

    if uidstr == None:
        obj["error"] = "Missing search ID"

    obj["query"] = {}
    obj["query"]["uid"] = uidstr
    obj["clips"] = []
    states = backend.get_search_sessions()
    obj["sessions"] = []
    for astate in states:
        obj["sessions"].append(str(astate))
    try:
        uid = uuid.UUID(uidstr)

        allres = searchdb[uidstr].find({"model_id" : "FUSION"}).sort([("probability", pymongo.DESCENDING)]).skip(skip).limit(limit)
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



    return jsonify(obj)


@mod.route('/archive_search_view', methods=['GET', 'POST'])
def archive_search_view():
    sessionstr = flask.request.args.get("session", None)
    statestr = flask.request.args.get("state", None)

    return flask.render_template("archive_search.html", session=sessionstr, state = statestr)


###########################################################################################################
@mod.route('/oneshot_iqr', methods=['GET', 'POST'])
def oneshot_iqr():
    """
    Starts the
    Can start with Defaults to 3 flash mob ids


    Initialize the search results
    GET params query and feature both strings
    feature must be within known features

    @return:
    """
    obj = {}

    default_seeds = ["537d159e0a3ee14ee41a5382", "537d159e0a3ee14ee41a5384", "537d159e0a3ee14ee41a5386"]

    # query : will be event kit in case of triage information
    seedstr = flask.request.args.get("seeds", urllib2.quote(json.dumps(default_seeds)))
    seeds = json.loads(urllib2.unquote(seedstr))
    if len(seeds) == 0:
        seeds = default_seeds

    newid = uuid.uuid4()
    f_name = "csift"
    # Create a dkms and put it in the dict
    datap = os.path.join(data_path, f_name)

    cid_file = os.path.join(datap, 'iqr_train/clipids_eventkit.txt')
    bg_flags_file = os.path.join(datap, 'iqr_train/bg_flag_eventkit.txt')
    kernel_file = os.path.join(datap, 'iqr_train/kernel_eventkit.npy')
    feature_file = os.path.join(datap, "iqr_train/data_eventkit.npy")
    logger.info("Loading feature memory")

    # Store with a timeout of 600 secs
    tc.store_FeatureMemory(str(newid), cid_file, bg_flags_file, feature_file, kernel_file, timeout=600)
    fm = tc.get_fm(str(newid))

    flask.current_app.config["WORK_DIR"]
    current_id = int(max(fm.get_ids()) + 1)
    ids = []
    logger.info("Updating feature memory")
    # Modify it
    for aseed in seeds:
        current_id = current_id + 1
        feature_vec_file = os.path.join(flask.current_app.config["WORK_DIR"], aseed, "csift_flann.txt")
        feature_vec = np.array(np.loadtxt(feature_vec_file))
        fm.update(current_id, feature_vec)
        ids.append(current_id)

    logger.info("Initializing search session")

    # Create a distance kernel
    uid = backend.init_new_search_session(None, "Trial", fm.get_distance_kernel(), state_uuid=newid)
    # backend.refine_iqr_search(uid,ids, [], [], [])
    # logger.info("Requesting refinement with no positives")

    obj["seeds"] = seeds
    obj["uid"] = str(uid)
    obj["ids"] = [str(anid) for anid in ids]
    obj["state"] = "http://localhost:5003/iqr/search_state?" + urllib.urlencode({"uid" : uid})
    obj["results"] = "http://localhost:5003/iqr/search_results?" + urllib.urlencode({"query" : str(uid)})

    return flask.jsonify(obj)
