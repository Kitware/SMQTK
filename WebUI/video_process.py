"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""
__author__ = 'dhan'

import os
import os.path as osp
import flask
from celery import Celery
from celery.task.control import inspect
from celery.result import AsyncResult
import time
import gridfs
import pymongo
import bson
import sys
import json
import glob

from WebUI import app
from common_utils.sprite import create_sprite

def make_celery(app):
    celery = Celery(app.import_name, broker=app.config['CELERY_BROKER_URL'])
    celery.conf.update(app.config)
    TaskBase = celery.Task
    class ContextTask(TaskBase):
        abstract = True
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)
    celery.Task = ContextTask
    return celery

app.config.update(
    CELERY_BROKER_URL='mongodb://%s/celery' % app.config['MONGO_SERVER'],
    CELERY_RESULT_BACKEND='mongodb://%s/celeryresults' % app.config['MONGO_SERVER']
    )

celeryapp = make_celery(app)

def get_celery_worker_status():
    ERROR_KEY = "ERROR"
    try:
        insp = celeryapp.control.inspect()
        d = insp.stats()
        r = insp.registered()
        insp.active()
        if not d:
            d = { ERROR_KEY: 'No running Celery workers were found.' }
    except IOError as e:
        from errno import errorcode
        msg = "Error connecting to the brocker: " + str(e)
        if len(e.args) > 0 and errorcode.get(e.args[0]) == 'ECONNREFUSED':
            msg += ' Check that the RabbitMQ server is running.'
        d = { ERROR_KEY: msg }
    except ImportError as e:
        d = { ERROR_KEY: str(e)}
    return {"stats" : d, "registered" : r}

def save_feature_vector(id, vid_descriptor):
    """
    Saves the feature vector from VCDWorker for video
    """
    fname = os.path.join(app.config["WORK_DIR"], id, vid_descriptor.descriptor_id + ".txt")
    import numpy
    numpy.savetxt(fname, vid_descriptor.feat_vec)


@celeryapp.task()
def process_for_time(a):
    print "Starting .."
    for i in range(a):
        time.sleep(1)
        metastr = str({'current': i, 'total': 5})
        celeryapp.current_task.update_state(state='PROGRESS', meta=metastr)
        print metastr
    return a

@celeryapp.task()
def process_video(a):
    """
    a is the string ObjectId of the video to be processed
    """

    # Setup the task
    from SMQTK_Backend.VCDWorkers import descriptor_modules
    metastr = {'state': 1, 'label': "Inspection done", 'id': a}
    celeryapp.current_task.update_state(state='PROGRESS', meta=json.dumps(metastr))
    print metastr

    # Prepare to fetch the file from database (local pymongo database)
    conn = pymongo.Connection()
    datadb = conn["files"]
    gf = gridfs.GridFS(datadb, "video")
    fileobj = gf.get(bson.ObjectId(a))

    # Create the directory structure for the file
    filename = osp.join(app.config["WORK_DIR"], a, "HVC123456.mp4")
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
        os.makedirs(directory + "/work")
        os.makedirs(directory + "/images")

    # Download the file here for
    fout = open(filename, "wb")
    fout.write(fileobj.read())

    metastr = {'state': 2, 'label': "File Saved", 'id': a}
    celeryapp.current_task.update_state(state='PROGRESS', meta=json.dumps(metastr))
    print metastr

    # Setup color descriptors
    mods = descriptor_modules.get_descriptor_modules()
    cd = mods["colordescriptor"]
    cdconf =  cd.generate_config()

    cdobj = cd(cdconf, osp.join(app.config["WORK_DIR"], a, "work"),
                       osp.join(app.config["WORK_DIR"], a, "images")
                       )

    # Generate frames
    cdobj.generate_frames(filename)

    metastr = {'state': 3, 'label': "Frame extraction complete", 'id': a}
    celeryapp.current_task.update_state(state='PROGRESS', meta=json.dumps(metastr))
    print metastr

    # Create sprite
    create_sprite(osp.join(app.config["WORK_DIR"], a, "images/456/123456"))

    metastr = {'state': 4, 'label': "Sprite creation complete", 'id': a}
    celeryapp.current_task.update_state(state='PROGRESS', meta=json.dumps(metastr))
    print metastr

    # Aggregate features
    cdobj.descriptor_generation(filename)

    metastr = {'state': 5, 'label': "Features generated", 'id': a}
    celeryapp.current_task.update_state(state='PROGRESS', meta=json.dumps(metastr))
    print metastr

    results = cdobj.process_video(filename)
    # print results

    for aresult in results:
        save_feature_vector(a, aresult)

    # Save results
    metastr = {'state': 6, 'label': "Features aggregated", 'id': a}
    celeryapp.current_task.update_state(state='PROGRESS', meta=json.dumps(metastr))
    print "Done ", a

    return a


# USer has uploaded new videos
# computed feature vectors
# Create an object (fm alrady loaded)
# update using new feature vectors
# initialize iqr with this
# First iteration results

# In the next request, user continues with reference to same persitent matrix
# User selects or uploads more videos

# User can reset



