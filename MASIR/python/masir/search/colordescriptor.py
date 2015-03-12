# coding=utf-8

import logging
import multiprocessing.pool
import numpy
import numpy.matlib
import os
import os.path as osp
import PIL.Image
import subprocess
import tempfile

from . import FeatureDescriptor
from ._colordescriptor import DescriptorIO
from ._colordescriptor import encode_FLANN

from masir.FeatureMemory import FeatureMemory


__all__ = [
    'ColorDescriptor_CSIFT',
    'ColorDescriptor_TCH',
]


#
# Async helper functions
#

def _cd_async_image_feature((self, t_id, log_id, uid, filepath)):
    """ Function for asynchronous processing of descriptor image features """
    log = logging.getLogger(log_id)
    # TODO: Use multiprocessing.current_process to get identifier, not t_id
    try:
        log.info("[thread-%d] %s", t_id, (uid, filepath))
        feat = self.image_feature(filepath)
        # Skip image if we generate an invalid feature
        if numpy.isnan(feat.sum()):
            log.info("[thread-%d] generated feature contained NaNs. Skipping.",
                     t_id)
            return None, None
        return uid, feat
    except Exception, ex:
        log.warn("[thread-%d] Feature generation failed for descriptor '%s'. "
                 "Skipping image. Error message: %s",
                 t_id, self.name, str(ex))
        return None, None


# noinspection PyAbstractClass
class ColorDescriptor (FeatureDescriptor):

    # BACKGROUND_RATIO = 0.05  # 5%
    BACKGROUND_RATIO = 0.40  # 40%

    # Convenience accessor for CSIFT sub-class
    #: :type: type
    CSIFT = None
    # Convenience accessor for TCH sub-class
    #: :type: type
    TCH = None

    def __init__(self, base_data_dir, base_work_dir):
        """
        Initialize a colorDescriptor feature computation
        """
        super(ColorDescriptor, self).__init__(base_data_dir, base_work_dir)

    @property
    def name(self):
        return "ColorDescriptorAbstract"

    @property
    def ids_file(self):
        return osp.join(self._data_dir, "id_map.npy")

    @property
    def bg_flags_file(self):
        return osp.join(self._data_dir, "bg_flags.npy")

    @property
    def feature_data_file(self):
        return osp.join(self._data_dir, "feature_data.npy")

    @property
    def kernel_data_file(self):
        return osp.join(self._data_dir, "kernel_data.npy")

    @staticmethod
    def _image_properties(image_file):
        """
        Use file command to extract required metadata from an image file

        :param image_file: Image file to extract properties from
        :type image_file: str

        :return: A dictionary of properties. Expected keys: width, height
        :rtype: dict of (str, unknown)

        """
        if not osp.isfile(image_file):
            raise ValueError("Image file provided did not exist (given: %s)"
                             % image_file)

        img = PIL.Image.open(image_file)
        width, height = img.size

        return {
            'width': width,
            'height': height,
        }

    def _compute_cd_feature(self, image_file,
                            descriptor_type, codebook, ffile,
                            frame_num=0):
        """
        Compute the colorDescriptor features for a descriptor type, and return


        :param image_file: Image file to generate descriptors for
        :type image_file: str
        :param descriptor_type: 'csift' or 'tch'
        :type descriptor_type: str
        :param codebook: Path to the codebook file for the given descriptor type
        :type codebook: str
        :param ffile: Path to the flann index file for the given descriptor type
        :type ffile: str

        :return: Per-quadrant L1 normalized descriptor matrix for image file
            (total outgoing histogram feature sum of 1.0)
        :rtype: numpy.matrixlib.defmatrix.matrix

        """
        image_file = osp.abspath(osp.expanduser(image_file))

        img_props = self._image_properties(image_file)
        w = img_props['width']
        h = img_props['height']

        # For pixel sample grid, we want to take at a maximum of 50,
        # sample points in longest direction with at least a 6 pixel spacing. We
        # will take fewer sample points to ensure the 6 pixel minimum spacing.
        # (magic numbers are a result of tuning, see Sangmin)
        #
        # Using min instead of max due to images that are long and thin, and
        # vice versa, which, with max, would cause some quadrants to have no
        # detections (see spHist below)
        #
        # ds_spacing = max(int(max(w, h) / 50.0), 6)
        ds_spacing = max(int(min(w, h) / 50.0), 6)
        self._log.debug('ds_spacing = %d', ds_spacing)

        # TODO: Could take in more than one image, then making the following
        # section a loop, pulling the descriptor_matrix out for use after the
        # loop
        # {

        # temporary output file
        tmp_fd, tmp_file = tempfile.mkstemp()
        self._log.debug("using temp file: %s", tmp_file)

        def tmp_clean():
            self._log.debug("cleaning temp file: %s", tmp_file)
            os.remove(tmp_file)
            os.close(tmp_fd)

        cmd = ['colorDescriptor', image_file,
               '--detector', 'densesampling',
               '--ds_spacing', str(ds_spacing),
               '--descriptor', descriptor_type,
               '--output', tmp_file]

        p = subprocess.Popen(cmd,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()

        if p.returncode != 0:
            tmp_clean()
            raise RuntimeError("Failed to fun colorDescriptor executable for "
                               "file \"%(file)s\" (command: %(cmd)s)\n"
                               "Output:\n%(out)s\n"
                               "Error :\n%(err)s"
                               % {"file": image_file,
                                  "cmd": cmd,
                                  "out": out,
                                  "err": err})
        else:
            # Read in descriptor output from file and convert to matrix form
            info, descriptors = DescriptorIO.readDescriptors(tmp_file)
            tmp_clean()

            # number of descriptors from
            n = info.shape[0]
            descriptor_matrix = numpy.hstack((
                numpy.zeros((n, 1)) + frame_num,
                info[:, 0:2],
                descriptors
            ))

        self._log.debug("descriptor_matrix :: %s\n%s",
                        descriptor_matrix.shape, descriptor_matrix)

        # } //end loop section

        # encode + quantize result
        quantized = encode_FLANN.quantizeResults3(descriptor_matrix,
                                                  codebook, ffile)
        self._log.debug("quantized :: %s\n%s",
                        quantized.shape, quantized)

        # sp_hist is int64 type
        sp_hist = encode_FLANN.build_sp_hist2(quantized)
        self._log.debug("sphist :: %s\n%s",
                        sp_hist.shape, sp_hist)
        self._log.debug("sphist sums: \n%s", sp_hist.sum(axis=1))

        # Result of build_sp_hist is an 8xN matrix, where each row is a
        # clip-level feature for a spacial region. Final feature product
        # will composed of 4 of the 8 vectors (full image + image thirds)

        # normalizing each "quadrant" so their sum is a quarter of the feature
        # total (this 4x multiplier on each vector norm)
        q1 = sp_hist[0] / (sp_hist[0].sum()*4.0)
        q2 = sp_hist[5] / (sp_hist[5].sum()*4.0)
        q3 = sp_hist[6] / (sp_hist[6].sum()*4.0)
        q4 = sp_hist[7] / (sp_hist[7].sum()*4.0)

        feature = numpy.hstack((q1, q2, q3, q4))

        # Diagnostic introspection logging
        # self._log.info("feature :: %s shape=%s dtype=%s",
        #                type(feature), feature.shape, feature.dtype)
        # self._log.info("normalized max: %s", feature.max())
        # self._log.info("normalized sum: %s", feature.sum())
        # self._log.info("normalized sum 1/4: %s", feature[:4096].sum())
        # self._log.info("normalized sum 2/4: %s", feature[4096*1:4096*2].sum())
        # self._log.info("normalized sum 3/4: %s", feature[4096*2:4096*3].sum())
        # self._log.info("normalized sum 4/4: %s", feature[4096*3:4096*4].sum())

        return feature

    # Use of numpy.matlib causes havoc in PyCharm's syntax parser
    # noinspection PyNoneFunctionAssignment,PyTypeChecker
    # noinspection PyProtectedMember,PyAttributeOutsideInit
    def generate_feature_data(self, ingest_manager, **kwds):
        """ Generate feature data files for ColorDescriptor CSIFT descriptor.

        Works over image files currently.

        Additional key-word arguments:
            parallel: number of parallel sub-processes to utilize. If not
                      provided, uses all available cores.

        :raises ValueError: When there are no images in the given ingest.

        :param ingest_manager: The ingest to create data files over.
        :type ingest_manager: IngestManager

        """
        self._log.info("Generating %s data files for given ingest",
                       self.__class__.__name__)

        parallel = kwds.get('parallel', None)

        if not len(ingest_manager):
            raise ValueError("No images in given ingest. No processing to do")

        self._log.info("Generating features asynchronously")
        args = []
        for i, (uid, filepath) in enumerate(ingest_manager.iteritems()):
            args.append((self, i, self._log.name, uid, filepath))

        pool = multiprocessing.Pool(processes=parallel)
        map_results = pool.map_async(_cd_async_image_feature, args).get()
        r_dict = dict(map_results)
        pool.close()
        pool.join()

        # Filter failed executions -- dict-ifying will cull duplicated None keys
        # do we only have to remove a lingering None key if there is one.
        if None in r_dict:
            del r_dict[None]
        if not map_results:
            raise RuntimeError("All images in ingest failed ColorDescriptor "
                               "feature generation. Cannot proceed.")

        # due to raise conditions above, can assume that there will be at least
        # one feature in r_dict
        self._log.info("Constructing feature matrix and idx-to-uid map")
        num_features = len(r_dict)
        sorted_uids = sorted(r_dict.keys())
        feature_length = len(r_dict[sorted_uids[0]])
        idx2uid_map = numpy.empty(num_features, dtype=numpy.uint32)
        feature_mat = numpy.matlib.empty((num_features, feature_length))
        for idx, uid in enumerate(sorted_uids):
            idx2uid_map[idx] = uid
            feature_mat[idx] = r_dict[uid]

        # flag a leading percentage of the collected IDs as background data
        # (flagging leading vs. random is more deterministic)
        self._log.info("Constructing BG flags map")
        pivot = int(num_features * self.BACKGROUND_RATIO)
        idx2bg_map = numpy.empty(num_features, dtype=numpy.bool)
        bg_clip_ids = set()
        for idx, item_id in enumerate(idx2uid_map):
            if idx < pivot:
                idx2bg_map[idx] = True
                bg_clip_ids.add(item_id)
            else:
                idx2bg_map[idx] = False

        # Construct a dummy FeatureMemory for the purpose of distance kernel
        # generation
        self._log.info("Generating distance kernel")
        dummy_dk = numpy.matlib.empty((num_features, num_features),
                                      dtype=numpy.bool)
        fm = FeatureMemory(idx2uid_map, bg_clip_ids,
                           feature_mat, dummy_dk)
        kernel_mat = fm._generate_distance_kernel_matrix()

        self._log.info("Saving out data files")
        numpy.save(self.ids_file, idx2uid_map)
        numpy.save(self.bg_flags_file, idx2bg_map)
        numpy.save(self.feature_data_file, feature_mat)
        numpy.save(self.kernel_data_file, kernel_mat)


# noinspection PyPep8Naming
class ColorDescriptor_CSIFT (ColorDescriptor):
    """
    ColorDescriptor detector with CSIFT descriptor. Distance kernel metric: HIK
    """

    @property
    def name(self):
        return "colordescriptor-csift"

    def __init__(self, base_data_dir, base_work_dir):
        super(ColorDescriptor_CSIFT, self).__init__(base_data_dir,
                                                    base_work_dir)

        self._data_dir = osp.join(self._data_dir, 'colordescriptor-csift')
        self._work_dir = osp.join(self._work_dir, 'colordescriptor-csift')
        if not osp.isdir(self._data_dir):
            os.makedirs(self._data_dir)
        if not osp.isdir(self._work_dir):
            os.makedirs(self._work_dir)

        # FLANN data support
        self._flann_csift_codebook = osp.join(osp.dirname(__file__),
                                              "_colordescriptor",
                                              "csift_codebook_med12.txt")
        self._flann_csift_ffile = osp.join(osp.dirname(__file__),
                                           "_colordescriptor",
                                           "csift.flann")

    def image_feature(self, image_file):
        """ Compute the feature vector for the given image file

        :param image_file:
        :type image_file:

        :return: Image level feature vector
        :rtype: numpy.matrixlib.defmatrix.matrix

        """
        return self._compute_cd_feature(image_file, 'csift',
                                        self._flann_csift_codebook,
                                        self._flann_csift_ffile)


# noinspection PyPep8Naming
class ColorDescriptor_TCH (ColorDescriptor):
    """
    ColorDescriptor detector with TCH descriptor. Distance kernel metric: HIK
    """

    @property
    def name(self):
        return "colordescriptor-tch"

    def __init__(self, base_data_dir, base_work_dir):
        super(ColorDescriptor_TCH, self).__init__(base_data_dir, base_work_dir)

        self._data_dir = osp.join(self._data_dir, 'colordescriptor-tch')
        self._work_dir = osp.join(self._work_dir, 'colordescriptor-tch')
        if not osp.isdir(self._data_dir):
            os.makedirs(self._data_dir)
        if not osp.isdir(self._work_dir):
            os.makedirs(self._work_dir)

        # FLANN data support
        self._flann_tch_codebook = osp.join(osp.dirname(__file__),
                                            "_colordescriptor",
                                            "tch_codebook_med12.txt")
        self._flann_tch_ffile = osp.join(osp.dirname(__file__),
                                         "_colordescriptor",
                                         "tch.flann")

    def image_feature(self, image_file):
        """ Compute the feature vector for the given image file

        :param image_file:
        :type image_file:

        :return: Image level feature vector
        :rtype: numpy.matrixlib.defmatrix.matrix

        """
        return self._compute_cd_feature(image_file, 'transformedcolorhistogram',
                                        self._flann_tch_codebook,
                                        self._flann_tch_ffile)


# Convenience accessors as class properties on the base class
ColorDescriptor.CSIFT = ColorDescriptor_CSIFT
ColorDescriptor.TCH = ColorDescriptor_TCH
