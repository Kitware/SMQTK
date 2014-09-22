"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

from flask import Flask, url_for, send_file, request, abort, Response, jsonify, render_template
import flask
import pymongo
from bson import ObjectId
import bson
import time
import mimetypes
import json
import urllib
import operator
import os, sys
from werkzeug.routing import BaseConverter
import json

import smqtk_config


mimetypes.add_type('image/png', '.png')
mimetypes.add_type('video/ogg', '.ogv')
mimetypes.add_type('video/webm', '.webm')

app = Flask(__name__,
            static_url_path="",
            static_folder=smqtk_config.STATIC_DIR
            )

app.config.from_object("smqtk_config")
try:
    app.config.from_envvar("smqtk_config")
except RuntimeError:
    # Allowing no config provided in environment
    pass

app.jinja_env.add_extension('jinja2.ext.do')
app.secret_key = app.config["SECRET_KEY"]
#import login
from .login import mod as login_mod, role_required
app.register_blueprint(login_mod)
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

# Shared connection
conn = pymongo.Connection(app.config["MONGO_SERVER"],
                          _connect=False)

db = conn[app.config["MONGO_DB"]]

scores = db["ground_truth"]
attribs = db["attributes"]
eventkits = db["eventkits"]


from .triage import mod
app.register_blueprint(mod)

if app.config["USE_VCDI"]:
    from .vcdi import mod
    app.register_blueprint(mod, url_prefix='/vcdi')

if app.config["USE_CLUSTERING"]:
    from .clustering import mod
    app.register_blueprint(mod, url_prefix='/clusters')

if app.config["USE_SUGGESTIONS"]:
    from .suggestions import mod
    app.register_blueprint(mod, url_prefix='/suggest')

if app.config["USE_IQR"]:
    from .query_refinement import mod
    app.register_blueprint(mod, url_prefix='/iqr')

class RegexConverter(BaseConverter):
    def __init__(self, url_map, *items):
        super(RegexConverter, self).__init__(url_map)
        self.regex = items[0]


app.url_map.converters['regex'] = RegexConverter

if app.config["USE_IQR_ONESHOT"]:
    import oneshot
    app.register_blueprint(oneshot.mod, url_prefix="/oneshot")
    from video_process import celeryapp


from .zero_shot import mod
app.register_blueprint(mod, url_prefix='/zero_shot')

from .QueryRecommend import mod
app.register_blueprint(mod, url_prefix='/query_recommend')

from .mainimage import mod
app.register_blueprint(mod)

from .adjudication import mod
app.register_blueprint(mod, url_prefix='/adjudicate')

@app.route('/attribute_info')
def attribs_info():
    """
    Searches and returns the information of attribute
    """
    label = request.args.get('label', None)
    dist = request.args.get('dist', None)

    if label == None:
        abort(403)


    # Locate the index from label
    attribute = attribs.find_one({"label" : label})

    if attribute == None:
        print "No one found"
        abort(403)

    attr_id = int(attribute["scores_index"])
    # Now perform the searche
    obj = {}
    obj["query"] = { "label" : label, 'dist' : dist}
    obj["scores_index"] = attr_id

    if dist <> None:
        if dist in attribute:
            obj["dist"] = attribute[dist]
        else:
            # Actually throw an error
            pass
    else:
        obj["dist"] = []

    return jsonify(obj)

@app.route('/query_score', methods=["GET", "POST"])
def query_score():

    # Fetch parameters
    if request.method == 'POST':
        label = request.form['label']
        dataset = request.form['dataset']
        min_ = request.form['min']
        skip = int(request.form['skip'])

    else:
        # TODO: work if either a label or an id, or index is present
        label = request.args.get('label', None)
        dataset = request.args.get('dataset', None)

        min_ = request.args.get('min', None)
        max_ = request.args.get('max', None)
        skip = int(request.args.get('skip', '0'))

    max_ = None
    # Get the index of the label
    attribute = attribs.find_one({"label" : label})

    if attribute == None:
        print "Unknown attribute"
        abort(403)

    attr_id = int(attribute["scores_index"])

    # Construct the query
    query = {}

    if dataset == "ground_truth":
        # TODO: add thresholds to the ground truth
        query = { "scores." + str(attr_id) :  1.0}
    elif dataset == "algo":
        # Add minimum threshold
        if min_ <> None:
            query = { "scores." + str(attr_id) :  {"$gte" : float(min_)} }
            # Add to the query
    else:
        print "Unknown dataset"
        abort(403)

    obj = {}

    obj["count"] = db[dataset].find(query, { "thumb" : 0, "scores" : { "$slice" : [attr_id, 1]}}).count()

    tstart = time.time()

    cur = db[dataset].find(query, { "thumb" : 0, "scores" : { "$slice" : [attr_id, 1]}}, skip=skip, limit=20).sort("scores.%d" % (attr_id), pymongo.DESCENDING)

    obj["count"] = cur.count()
    imgs = []

    for animage in cur:
        # Append to the images
        v_id = "Ground truth"
        if "v_id" in animage:
            v_id = animage["v_id"]
        imgs.append([ str(animage["_id"]), animage["scores"][0], v_id])

    obj["images"] = imgs

    tend = time.time()
    obj["query"] = { "label" : label, 'dataset' : dataset, 'min' : min_, "max" : max_, 'skip' : skip, "time" : (tend - tstart)}
    return jsonify(obj)


@app.route('/home')
@app.route('/')
def home():
    if 'user' in flask.session:
        pass
    else:
        flask.flash("You are not logged in..", "info")

    return flask.render_template('home.html')

@app.route('/hovertest')
def hovertest():
    return render_template("hover.html")

@app.route('/queryexpansion')
def queryexpansion():
    return render_template("queryexpansion.html")


@app.route('/eventsearch')
def eventsearch():
    return render_template("event_search.html")


@app.route('/search')
def search():
    # Event ID
    # DataSet

    cmd = request.args.get('cmd', "submit")
    where = request.args.get('where', "MED11-compare")

    obj = {}

    if cmd == "submit":
        what = request.args.get('what', "E007")
        what = int(what[1:])
        obj["query"] = {"cmd" : cmd , "what" : what, "where" : where}
#        try:
#        except:
        result = processes.event_search(what, where)
        obj["group"] = str(result[0])
        obj["task"] = str(result[1].uuid)
        obj["progress"] = url_for("search", _external=True) + "?cmd=progress&what=" + obj["task"]
        obj["results"] = url_for("search", _external=True) + "?cmd=result&what=" + obj["task"]
        #obj["error"] = "Controller raised exception"
        return jsonify(obj)
    elif cmd == "progress":
        what = request.args.get('what')
        obj["query"] = {"cmd" : cmd , "what" : what, "where" : where}
        try:
            result = processes.get_progress(what)
        except:
            obj["error"] = "Controller raised exception"
            return jsonify(obj)

        obj["progress"] = result.progress
        obj["status"] = result.__repr__()
        obj["state"] = result.status

        return jsonify(obj)

    elif cmd == "results":
        what = request.args.get('what')
        obj["query"] = {"cmd" : cmd , "what" : what, "where" : where}
        result = processes.get_search_results(what)
        try:
            result = processes.get_search_results(what)
        except:
            obj["error"] = "Controller raised exception"
            return jsonify(obj)
        sortedres = sorted(result.items(), key=operator.itemgetter(1), reverse=True)
        obj["results"] = sortedres[:50]
        return jsonify(obj)
    else:
        obj["error"] = "Unknown cmd"
        return Response(json.dumps(obj), 403)

@app.route('/research')
@role_required("kitwarean")
def list():
    return render_template("list.html")

@app.route('/upload')
def upload():
    return render_template("upload.html")

@app.route('/triage')
@login.login_required
def triage():
    return render_template("triage.html", video_url_prefix=app.config["VIDEO_URL_PREFIX"])


@app.route('/about')
def help():
    from .utils import get_git_name, get_kernel_timestamp
    return render_template("about.html", git=get_git_name(), host=app.config["MONGO_SERVER"], kernel_timestamp=get_kernel_timestamp())

@app.route('/help')
def about():
    return render_template("help.html")

@app.route('/rangy')
def rangy():
    return render_template("rangy.html")

@app.route('/test')
def test():
    return render_template("testExpression.html")


@app.route('/image')
def image():
    """
    Searches and returns the image

    /image?id=5167864f0a3ee15fe21b30f6

    """
    # Get variables
    #    img = request.args.get('img', None)
    #    db = request.args.get('db', None)

    imgname = request.args.get('id', None)
    colname = request.args.get('col', None)

    if colname == None:
        abort(403)

    if imgname == None:
        abort(403)

    try:
        id = ObjectId(imgname)
    except:
        abort(403)

    col = db[colname]
    docImage = col.find_one({'_id':id}, {"thumb" : 1})

    if docImage == None:
        abort(403)

    return Response(str(docImage['thumb']), mimetype="image/png")

@app.route('/events')
def event_kit():
    # Fetch parameters

    # TODO: work if either a label or an id, or index is present
    kit = request.args.get('kit', None)
    obj = {}

    if kit == None:
        kits = eventkits.find({}, {'kit' : 1, '_id' : 0, 'name': 1})
        obj["list"] = []
        for akit in kits:
            obj["list"].append({'name' : akit['name'], 'kit' : akit["kit"]})
        pass
    else:
        # Get the index of the label
        kitobj = eventkits.find_one({"kit" : kit})

        if kitobj == None:
            print "Unknown event kit"
            abort(403)

        # Construct the query
        obj["name"] = kitobj["name"]
        obj["text"] = kitobj["text"].replace("\n\n", "<br/> <br/>")
        obj["query"] = { "kit" : kit }

    return jsonify(obj)



@app.route('/group_score', methods=["GET", "POST"])
def group_score():

    # Fetch parameters
    if request.method == 'POST':
        label = request.form['label']
        dataset = request.form['dataset']
        min_ = request.form['min']
        skip = int(request.form['skip'])

    else:
        # TODO: work if either a label or an id, or index is present
        label = request.args.get('label', None)
        dataset = request.args.get('dataset', None)

        min_ = request.args.get('min', None)
        max_ = request.args.get('max', None)
        skip = int(request.args.get('skip', '0'))
        limit = int(request.args.get("limit", '20'))

    max_ = None
    # Get the index of the label
    attribute = attribs.find_one({"label" : label})

    if attribute == None:
        print "Unknown attribute"
        abort(403)

    attr_id = int(attribute["scores_index"])

    # Construct the query
    query = {}

    if dataset == "ground_truth":
        # TODO: add thresholds to the ground truth
        query = { "scores." + str(attr_id) :  1.0}
    elif dataset == "algo":
        # Add minimum threshold
        if min_ <> None:
            query = { "scores." + str(attr_id) :  {"$gte" : float(min_)} }
            # Add to the query
    else:
        print "Unknown dataset"
        abort(403)

    obj = {}

    obj["count"] = db[dataset].find(query, { "thumb" : 0, "scores" : { "$slice" : [attr_id, 1]}}).count()

    tstart = time.time()

    imgs = {}

    cur = db[dataset].find(query, { "thumb" : 0, "scores" : { "$slice" : [attr_id, 1]}}, skip=skip, limit=limit).sort("scores.%d" % (attr_id), pymongo.DESCENDING)
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
    obj["query"] = { "label" : label, 'dataset' : dataset, 'min' : min_, "max" : max_, 'skip' : skip, "time" : (tend - tstart)}
    return jsonify(obj)


@app.route('/clip', methods=["GET", "POST"])
def clip():
    clipname = request.args.get('id', '003237')
    use = "name";
    try:
        id = bson.ObjectId(clipname)
        use= "id"
    except:
        pass

    colname = request.args.get('col', "algo")
    preview = request.args.get('preview', "none")

    if preview != "none":
        col = db['clips']
        if use == "name":
            if not clipname[:3] == "HVC":
                clipname = "HVC" + clipname

            docImage = col.find_one({'id':clipname})
        else:
            docImage = col.find_one({'_id':id})

        if docImage == None:
            im = open(os.path.join(app.config['STATIC_DIR'], "data/sprite_dummy.jpg"), "rb")
            return Response(str(im.read()), mimetype="image/jpg")

        if not preview in docImage:
            im = open(os.path.join(app.config['STATIC_DIR'], "data/sprite_dummy.jpg"), "rb")
            return Response(str(im.read()), mimetype="image/jpg")

        return Response(str(docImage[preview]), mimetype="image/jpg")

    clipname = clipname.zfill(6)
    if clipname[:3] == "HVC":
        clipname= clipname[3:]

    frames = db["algo"].find({'v_id' : clipname}, {"scores" : 0, "thumb" : 0}).sort("duration")
#     frames = db["algo"].find({'v_id' : '003237'}, {"scores" : 0, "thumb" : 0}).sort(["duration"])

    obj = {}
    obj['frames'] = []

    durs = set()
    for aframe in frames:
        if not aframe["duration"] in durs:
            durs.add(aframe["duration"])
            aframe["_id"] = str(aframe["_id"])
            obj['frames'].append(aframe)

    obj["query"] = { "id" : clipname, 'dataset' : colname}
    return jsonify(obj)

def get_json_response(view_name, *args, **kwargs):
    '''Calls internal view method, parses json, and returns python dict.'''
    view = flask.current_app.view_functions[view_name]
    res = view(*args, **kwargs)
    #XXX: to avoid the json decode internally for every call,
    #you could make your own jsonify function that used a subclass
    #of Response which has an attribute with the underlying
    #(non-JSON encoded) data
    js = json.loads(res.data)
    return js


if app.config["USE_IQR_ONESHOT"]:
    @app.route("/task", methods=['GET', 'POST'])
    def task():
        # Parse input paramter "time"
        from .video_process import process_for_time

        if flask.request.method == "POST":
            # Submit task
            t = int(flask.request.form.get("time",-1))
            if t < 0:
                return flask.jsonify({"error" : "Invalid form submission, time should be > 0"})
            task = process_for_time.delay(t)
            id = task.__dict__["id"]
            return flask.jsonify( {"task" : id,
                                   "status" : flask.url_for('task') + "?id=" + id})

        if flask.request.method == "GET":
            id = flask.request.args.get("id",None)
            localid = flask.request.args.get("localid",None)
            if id == None :
                # If id is not supplied then look for time
                t = int(flask.request.args.get("time", -1))
                if t < 0:
                    return flask.jsonify({"error" : "Invalid form submission, requires either valid id or time > 0"})

                task = process_for_time.delay(t)
                id = task.__dict__["id"]
                return flask.jsonify( {"task" : id,
                                       "status" : flask.url_for('task',_external=True) + "?id=" + id})



            # Get task progress (and possibly results)
            res = celeryapp.AsyncResult(id)
            import json
            current = 1
            total = 1
            meta = {}
            if res.state == "SUCCESS":
                meta["state"] = 6

            if res.state == "PROGRESS":
                metastr = res.result
                print "METASTR: " + metastr
                meta = json.loads(metastr)
                try:
                    meta = json.loads(metastr)
                except Exception as e:
                    print "EXCEPTIPON: " + e.message
                    meta = {}
                    meta["state"] = 0

                if "state" in meta:
                    if meta["state"] == 4:
                        # Try to estimate the number of files
                        work = app.config["WORK_DIR"] + "/" + meta["id"] + "/work/colordescriptor/456/123456/"
                        images = app.config["WORK_DIR"] + "/" + meta["id"] + "/images/456/123456/"
                        current = len(os.listdir(work))
                        total = len(os.listdir(images)) * 2
                        if current >= total:
                            meta["state"] = 4
                            meta["label"] = "Features computed"

            return flask.jsonify( {"task" : id,
                                   "localid" : localid,
                                   "ready" : res.ready(),
                                   "state" : res.state,
                                   "meta" : meta,
                                   "current" : current,
                                   "total" : total,
                                   "status" : flask.url_for('task', _external=True) + "?id=" + id})

    @app.route("/status")
    def status():
        """
        Return json for the status of workers
        """

        query = flask.request.args.get('q',None)

        response = {}

        i =  celeryapp.control.inspect()
        if query == 'workers':
            response["workers"] = i.registered()
        elif query == 'tasks':
            response["tasks"] = i.active()
        elif query == "both":
            response["workers"] = i.registered()
            response["tasks"] = i.active()
            response["stats"] = i.stats()
        else:
            response["error"] = "Unknown query"

        return flask.jsonify(response)

    @app.route("/dashboard")
    def dashboard():
        #return flask.send_from_directory(app.config['STATIC_DIR'], "dashboard.html")
        return flask.render_template('dashboard.html')

