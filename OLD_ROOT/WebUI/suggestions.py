"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

__author__ = 'dhan'

from WebUI import app
import nltk
NLTK_DATAPATH = app.config["NLTK_DATAPATH"]
nltk.data.path = [NLTK_DATAPATH]

from nltk.corpus import wordnet as wn

import flask
import json
import os

mod = flask.Blueprint('api_suggest', __name__)

@mod.route('/suggest_similar', methods=["GET"])
def suggest_similar():
    obj = {}
    obj["query"] = {}
    obj["query"]["search"] = flask.request.args.get("search", "")

    words = obj["query"]["search"].split(" ");
    obj["query"]["word"] = words[0]

    scores = []
    fin = open(os.path.join(app.config['STATIC_DIR'], "data/vocabulary.js"),"r")
    vocab = json.loads(fin.read())

    # Compute dist for each of hte attributes
    for aword in vocab:
        one = wn.synsets(words[0])
        two = wn.synsets(aword)
        if len(one) < 1 or len(two) < 1:
            continue

        score = one[0].path_similarity(two[0])
        scores.append([score, aword])
    # Preload vocabulary

    obj["suggestions"] = sorted(scores, reverse=True)

    return flask.jsonify(obj)
