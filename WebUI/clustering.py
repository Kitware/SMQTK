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
import numpy
import scipy.cluster.hierarchy as hie
from .login import role_required
from random import choice
import StringIO
from PIL import Image

mod = flask.Blueprint('api_clustering', __name__, )

# Shared connection
#conn = pymongo.Connection(flask.current_app.config["MONGO_SERVER"])
#db = conn[flask.current_app.config["MONGO_DB"]]

thispath = os.path.dirname(os.path.abspath(__file__))
linkagepath = os.path.join(WebUI.app.config['STATIC_DIR'], "data/clustering/Z_HI_profile_1_2124.npy")
Z = numpy.load(linkagepath)
tree = hie.to_tree(Z, True)

frames_sfu = WebUI.db["frames_sfu"]

@mod.route('/')
@role_required("kitwarean")
def clustering():
    return flask.render_template("clustering.html")

@mod.route('/get')
@role_required("kitwarean")
def cluster_info():
    clusterid = int(flask.request.args.get('id', '-1'))

    if clusterid < 0:
        # Return root node
        node = tree[0]
    else:
        node = tree[1][clusterid]

    obj = {}
#    obj["query"] = {}
#    obj["query"]["id"] = clusterid

    obj["id"] = node.id
    obj["count"] = node.get_count()
    obj["title"] = "Id: " + str(obj["id"]) + ", Count: " + str(obj["count"])

    # javascript likes isFolder
    obj["isFolder"] = not node.is_leaf()

    # Compile information about the children

    obj["children"] = []

    # Add left if exists
    child_node = node.get_left()
    if(child_node):
        child = {}
        child["id"] = child_node.id
        child["count"] = child_node.get_count()
        child["title"] = "Id: " + str(child["id"]) + ", Count: " + str(child["count"])
        child["isFolder"] = not child_node.is_leaf()
        if child["isFolder"]:
            child["children"] = []
        obj["children"].append(child)

    # Add right child if exists
    child_node = node.get_right()
    if(child_node):
        child = {}
        child["id"] = child_node.id
        child["count"] = child_node.get_count()
        child["title"] = "Id: " + str(child["id"]) + ", Count: " + str(child["count"])
        child["isFolder"] = not child_node.is_leaf()
        if child["isFolder"]:
            child["children"] = []
        obj["children"].append(child)

    return flask.jsonify(obj)

@mod.route('/random_frame')
@role_required("kitwarean")
def random_frame():
    """
    Returns a random frame from the cluster_id (also index) specified
    """

    clusterid = int(flask.request.args.get('clusterid', '0'))
    send_image = bool(flask.request.args.get('img', False))
    gridx = int(flask.request.args.get('gridx', '1'))
    gridy = int(flask.request.args.get('gridy', '1'))

    obj = {}
    obj["query"] = {}
    obj["query"]["clusterid"] = clusterid
    obj["query"]["img"] = send_image
    obj["query"]["gridx"] = gridx
    obj["query"]["gridy"] = gridy

    node = tree[1][clusterid]

    obj["count"] = node.get_count()

    # Get leaves
    # Note: this line needs a scipy fix
    ids = node.pre_order(lambda x: x.id)

    # Select random of that

    if send_image:
        # Query the image
        if gridx == 1 and gridy == 1:
            rand = choice(ids)
            frame = frames_sfu.find_one({'index' : rand}, {"thumb" : 1})
            if aframe == None:
                largeImage = Image.new("RGB", (200, 150))
                buf = StringIO.StringIO()
                largeImage.save(buf, format="JPEG")
                contents = buf.getvalue()
                buf.close()
                return flask.Response(str(contents), mimetype="image/png")
            return flask.Response(str(frame['thumb']), mimetype="image/png")
        else:
            # Note: hardcoded for x size = 200
            # Fetch all the images
            largeImage = Image.new("RGB", (gridx*200, gridy*150))

            for i in range(gridx):
                for j in range(gridy):
                    rand = choice(ids)
                    aframe = frames_sfu.find_one({'index' : rand}, {"thumb" : 1})
                    if aframe == None:
                        continue

                    anImage = Image.open(StringIO.StringIO(aframe["thumb"]))
                    if anImage.size[1] > 150:
                        anImage = anImage.crop((0,0,200,150))
                    # Crop first 15o pixels and paste them in the largeimage
                    largeImage.paste(anImage, (i*200, j*150))
            buf = StringIO.StringIO()
            largeImage.save(buf, format="JPEG")
            contents = buf.getvalue()
            buf.close()

            return flask.Response(str(contents), mimetype="image/png")
    else:
        return flask.jsonify(obj)



