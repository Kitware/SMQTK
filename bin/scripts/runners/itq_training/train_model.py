import json

from smqtk.representation import DescriptorElementFactory
from smqtk.utils.bin_utils import logging, initialize_logging
from smqtk.utils.jsmin import jsmin

from load_algo import load_algo


if not logging.getLogger().handlers:
    initialize_logging(logging.getLogger(), logging.DEBUG)
log = logging.getLogger(__name__)


log.info("Loading descriptor elements")
d_type_str = open("descriptor_type_name.txt").read().strip()
df_config = json.loads(jsmin(open('descriptor_factory_config.json').read()))
factory = DescriptorElementFactory.from_config(df_config)

#
# Sample code for finding non-NaN descriptors in parallel
#
# def add_non_nan_uuid(uuid):
#     d = factory.new_descriptor(d_type_str, uuid)
#     if d.vector().sum() > 0:
#         return uuid
#     return None
#
# import multiprocessing
# p = multiprocessing.Pool()
# non_nan_uuids = \
#     p.map(add_non_nan_uuid,
#           (l.strip() for l in open('descriptor_uuids.txt')))

d_elements = []
with open("descriptor_uuids.train.txt") as f:
    for uuid in (l.strip() for l in f):
        d_elements.append(factory(d_type_str, uuid))

# log.info("Sorting descriptors by UUID")
# d_elements.sort(key=lambda e: e.uuid())


log.info("Loading ITQ index algo")
itq_index = load_algo()

# Assuming ITQ functor, which needs fitting
itq_index.lsh_functor.fit(d_elements)

log.info("Building index")
# includes adding to adding to configured descriptor and hash indexes
itq_index.build_index(d_elements)
