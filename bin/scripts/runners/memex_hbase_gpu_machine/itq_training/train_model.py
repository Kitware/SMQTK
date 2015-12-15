from smqtk.utils.bin_utils import logging, initialize_logging
if not logging.getLogger().handlers:
    initialize_logging(logging.getLogger(), logging.DEBUG)
log = logging.getLogger(__name__)

log.info("Loading descriptor elements")
import cPickle
import json
from smqtk.representation.descriptor_element.postgres_element import PostgresDescriptorElement
d_config = json.load(open('/data/kitware/smqtk/image_cache_cnn_compute/psql_descriptor_config.localhost.json'))
d_type_str = open("/data/kitware/smqtk/image_cache_cnn_compute/descriptor_type_name.txt").read().strip()
d_elements = []
with open("/data/kitware/smqtk/image_cache_cnn_compute/descriptor_uuid_set.1m_train_sample.pickle") as f:
    for uuid in cPickle.load(f):
        d_elements.append(
            PostgresDescriptorElement.from_config(d_config, d_type_str, uuid)
        )

log.info("Loading ITQ index algo")
from load_algo import load_algo
itq_index = load_algo()

log.info("Building index")
itq_index.build_index(d_elements)
