"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import os
import subprocess
#from subprocess import check_output
import inspect
import json
import flask
import datetime
from bson import ObjectId
import os
import os.path as osp
import time

from WebUI import app

def get_git_name():
    curdir = os.getcwd()
#    print curdir
    p = inspect.getfile(get_git_name)
    newdir = os.path.dirname(os.path.abspath(p))
#    print newdir
    out = ''
    try:
        params = ["git", "describe", "--tags", "--always"]
        out = subprocess.Popen(params, stdout=subprocess.PIPE).communicate()[0]
    except:
        # Support for particular case when running on windows 
        os.chdir(newdir)
        params = ["C:/PortableGit-1.7.11/bin/git.exe", "describe", "--tags"]
        out = subprocess.check_output(params)
        os.chdir(curdir)

    os.chdir(newdir)
    return out

class MongoJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        elif isinstance(obj, ObjectId):
            return unicode(obj)
        return json.JSONEncoder.default(self, obj)

def jsonify(*args, **kwargs):
    """ jsonify with support for MongoDB ObjectId
    """
    return flask.Response(json.dumps(dict(*args, **kwargs), cls=MongoJsonEncoder), mimetype='application/json')

def get_kernel_timestamp():
    data_dir = app.config['DATA_DIR']
    return str(time.ctime(os.path.getctime(osp.join(data_dir, "kernel_eventkit.npy"))))

if __name__ == "__main__":
    print get_git_name()
