import json
import smqtk.algorithms.nn_index.lsh.itq


JSON_CONFIG_FP = "/data/kitware/smqtk/image_cache_cnn_compute/itq_model/256-bit/itq_config.json"


def load_algo(m=smqtk.algorithms.nn_index.lsh.itq):
    with open(JSON_CONFIG_FP) as f:
        itq_config = json.load(f)
    itq_index = m.ITQNearestNeighborsIndex.from_config(itq_config)
    return itq_index
