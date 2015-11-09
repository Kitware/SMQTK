from smqtk.utils.bin_utils import logging, initialize_logging
if not logging.getLogger().handlers:
    initialize_logging(logging.getLogger(), logging.DEBUG)
log = logging.getLogger(__name__)


DESCRIPTORS_PICKLE = "/data/kitware/smqtk/image_cache_cnn_compute/DescriptorElement.1mil_sample.pickle"


log.info("Loading descriptor elements")
import cPickle
with open(DESCRIPTORS_PICKLE) as f:
    descr_elements = cPickle.load(f)


log.info("Building index")
from load_algo import load_algo
itq_index = load_algo()
x = itq_index.build_index(descr_elements)
