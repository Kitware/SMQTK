"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""
__author__ = 'dhan'
acts = [{"clip": "HVC365862", "elapsed": 0, "time": "17-Oct-2013,21:39,", "query": {"training": "100Ex", "event": 8}, "label": "yes", "op": "add"}, {"clip": "HVC764328", "elapsed": 1.002, "time": "17-Oct-2013,21:39,", "query": {"training": "100Ex", "event": 8}, "label": "no", "op": "add"}, {"clip": "HVC443435", "elapsed": 3.002, "time": "17-Oct-2013,21:39,", "query": {"training": "100Ex", "event": 8}, "label": "yes", "op": "add"}, {"clip": "HVC067623", "elapsed": 4.003, "time": "17-Oct-2013,21:39,", "query": {"training": "100Ex", "event": 8}, "label": "star", "op": "add"}, {"clip": "HVC298260", "elapsed": 7.009, "time": "17-Oct-2013,21:39,", "query": {"training": "100Ex", "event": 8}, "label": "yes", "op": "add"}, {"clip": "HVC661164", "elapsed": 8.01, "time": "17-Oct-2013,21:39,", "query": {"training": "100Ex", "event": 8}, "label": "yes", "op": "add"}]

import json

import sys
import os
sys.path.append(os.path.abspath(".."))
from WebUI.adjudication import make_csv


print make_csv(acts, "Djay")


