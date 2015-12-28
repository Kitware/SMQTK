import json

from smqtk.representation import DescriptorElementFactory
from smqtk.utils.bin_utils import logging, initialize_logging
from smqtk.utils.jsmin import jsmin

from load_algo import load_algo


if not logging.getLogger().handlers:
    initialize_logging(logging.getLogger(), logging.DEBUG)
log = logging.getLogger(__name__)


log.info("Loading descriptor elements")
df_config = json.loads(jsmin(open('descriptor_factory_config.json').read()))
factory = DescriptorElementFactory.from_config(df_config)

d_type_str = open("descriptor_type_name.txt").read().strip()
d_elements = []
with open("descriptor_uuids.train.txt") as f:
    for uuid in (l.strip() for l in f):
        d_elements.append(factory(d_type_str, uuid))

log.info("Sorting descriptors by UUID")
d_elements.sort(key=lambda e: e.uuid())


log.info("Loading ITQ index algo")
itq_index = load_algo()

log.info("Building index")
itq_index.build_index(d_elements)
