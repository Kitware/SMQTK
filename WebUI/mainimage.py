"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import flask
import pymongo
import bson
import WebUI

mod = flask.Blueprint('mainimage', __name__)
#oid = OpenID()
from .triage import results

clips = WebUI.db["clips"]
frames = WebUI.db["frames"]

@mod.route('/mainimage', methods=["get"])
def mainimage():
    kits = [{'name' :   "E001: Board Trick", 'kit' : "1" },
           {'name' :   "E006: Birthday party",'kit' : "6" },
           {'name' :   "E008: Flash mob gathering",'kit' : "8" },
            {'name' :   "E009: Getting a vehicle unstuck",'kit' : "9" },
            {'name' :   "E014: Repairing an appliance", 'kit' : "14" }
                    ];
    output = ''

    for akit in kits:
        akit["ranks"] = []
        kitres = results[akit["kit"]]
        rank = 0
        for res in kitres:
            rank = rank + 1
            if rank > 30:
                break
            arank = {}
            arank["label"] = res[2]
            arank["score"] = res[1]
            arank["images"] = []
#            for i in range(5):
#                arank["images"].append( { 'src' : "http://www.osidb.com/free-icons/png/128x128/symbols/char-aum.png",

            # Get the 5 images
            # Read the info    if clipname[:3] != "HVC":

            images = []
            for algo in [ "ob2124_max_hik_to_compact_ver3", "scene102_avg_hik_to_compact_ver3"]:
                fieldname = "E%03d_%s"%(int(akit["kit"]),algo)

                data = clips.find_one({"id" : res[0], fieldname : {"$exists" : 1} }, {fieldname : 1})
                if(data != None):
                    for i in range(2):
                        topsegs = data[fieldname]["top_segments"]
                        #output = output + str(topseg["timestamp"]) + "</br>"
                        images.append(topsegs[i]["timestamp"])
                else:
                    continue

            if len(images) < 4:
                continue
            # Get the duration
            aclip = clips.find_one({'id' : res[0]}, {"thumb" : 0, "preview" : 0})
            middle = int(aclip["duration"] * 0.5)
            # Get the id for middle frame
            images.insert(0, middle)

            cat = ["Middle", "O1", "O2", "S1", "S2"]

            for j in range(5):
                animage = images[j]
                frame = frames.find_one({'v_id' : res[0], "duration" : int(animage)}, {"thumb" : 0})

                if frame == None:
                    arank["images"].append( { 'src' : "",
                                              'time' : str(animage),
                                              'cat' : cat[j]})
                else:
                    arank["images"].append( { 'src' : "image?col=frames&id="+ str(frame["_id"]),
                                           'time' : str(animage),
                                           'cat' : cat[j]})

            akit["ranks"].append(arank)

    return flask.render_template("mainimage.html", kits=kits, output = output, )

