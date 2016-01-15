import json

import smqtk.algorithms.nn_index.lsh.functors.itq
import smqtk.utils.jsmin as jsmin


def load_algo(m=smqtk.algorithms.nn_index.lsh.functors.itq):
    with open("itq_config.json") as f:
        itq_config = json.loads(jsmin.jsmin(f.read()))
    itq_index = m.ITQNearestNeighborsIndex.from_config(itq_config)
    return itq_index
