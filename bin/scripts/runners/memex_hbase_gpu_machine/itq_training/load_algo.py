import json
import os

import smqtk.algorithms.nn_index.lsh.itq


THIS_DIR = os.path.dirname(__file__)


def load_algo(m=smqtk.algorithms.nn_index.lsh.itq):
    with open(os.path.join(THIS_DIR, "itq_config.json")) as f:
        itq_config = json.load(f)
    itq_index = m.ITQNearestNeighborsIndex.from_config(itq_config)
    return itq_index
