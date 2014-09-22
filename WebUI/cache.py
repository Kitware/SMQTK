from WebUI import app
from SMQTK_Backend.FeatureMemory import \
    initFeatureManagerConnection, \
    getFeatureManager, \
    ReadWriteLock, \
    DummyRWLock
import logging

logger = logging.getLogger("WebUI.Cache")
logger.setLevel(logging.INFO)

import os
# Connect with the feature memory server
mgr_srvr_addr = ('127.0.0.1', 30000)
initFeatureManagerConnection(mgr_srvr_addr, 'test')
mgr = getFeatureManager(mgr_srvr_addr)

# Get timed cache
# noinspection PyUnresolvedReferences
#: :type: SMQTK_Backend.FeatureMemory.TimedCache
tc = mgr.get_common_tc()
data_path = app.config['DATA_DIR']


# Load the distance kernel matrices
known_features = ["csift", "hog_2x2", "isa", "mfcc_4096",
                  "OB_max_avg_positive"]


def load_feature_dk(f_name):
    try:
        datap = os.path.join(data_path, f_name)
        # Create a feature map->featureMemory and pass through update process
        cid_file = os.path.join(datap, 'iqr_train/clipids_eventkit.txt')
        bg_flags_file = os.path.join(datap, 'iqr_train/bg_flag_eventkit.txt')
        kernel_file = os.path.join(datap, 'iqr_train/kernel_eventkit.npy')

        logging.debug(kernel_file)

        # dkm = mgr.symmetric_dk_from_file(cid_file, kernel_file, bg_flags_file)
        tc.store_DistanceKernel_symmetric(f_name, cid_file, kernel_file,
                                          bg_flags_file)
    except KeyError:
        # warn if you want to
        logger.error("ERROR while loading files")
        pass
    except Exception as e:
        logger.error(e.message)


def load_known_features():
    for afeature in known_features:
        # Since both the main server and celery workers call this cache loading
        # method, we now acquire the singleton TimedCache's RLock before
        # attempting to load a kernel.
        with tc:
            if not afeature in tc.keys():
                logger.info("Loading feature %s" % afeature)
                load_feature_dk(afeature)
            else:
                logger.info("Feature %s already loaded" % afeature)


if __name__ == "__main__":
    logging.basicConfig()
    load_known_features()
