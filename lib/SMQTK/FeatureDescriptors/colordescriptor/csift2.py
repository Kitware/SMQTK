import abc
import logging
import multiprocessing
import numpy
import os.path as osp
import pyflann
import scipy.cluster.vq

from SMQTK.FeatureDescriptors import FeatureDescriptor
from SMQTK.utils import safe_create_dir, SimpleTimer

from . import utils


class DummyResult (object):
    """
    Mock AsyncResult with preexisting info and descriptor matrices
    """
    def __init__(self, info, descriptors):
        self._i = info
        self._d = descriptors

    def get(self):
        return self._i, self._d


class ColorDescriptor_Base (FeatureDescriptor):
    """
    Simple implementation of ColorDescriptor feature descriptor utility for
    feature generation over images and videos.

    This was started as an attempt at gaining a deeper understanding of what's
    going on with this feature descriptor's use and how it applied to later use
    in an classifiers.

    Codebook generated via kmeans given a set of input data. FLANN index model
    used for quantization, buily using auto-tuning (picks the best indexing
    algorithm of linear, kdtree, kmeans, or combined), and using the Chi-Squared
    distance function.

    """

    # colordescriptor executable that should be on the PATH
    PROC_COLORDESCRIPTOR = 'colorDescriptor'

    # Distance function to use in FLANN indexing. See FLANN documentation for
    # available distance function types (under the MATLAB section reference for
    # valid string identifiers)
    FLANN_DISTANCE_FUNCTION = 'chi_square'

    def __init__(self, data_directory, work_directory):
        super(ColorDescriptor_Base, self).__init__(data_directory,
                                                   work_directory)
        # Cannot pre-load FLANN stuff because odd things happen when processing/
        # threading. Loading index file is fast anyway.
        self._codebook = None
        if self.has_model:
            self._codebook = numpy.load(self.codebook_filepath)

    @property
    def codebook_filepath(self):
        return osp.join(self.data_directory,
                        "%s.codebook.npy" % (self.descriptor_type(),))

    @property
    def flann_index_filepath(self):
        return osp.join(self.data_directory,
                        "%s.flann_index.dat" % (self.descriptor_type(),))

    @property
    def has_model(self):
        has_model = (osp.isfile(self.codebook_filepath)
                     and osp.isfile(self.flann_index_filepath))
        # Load the codebook model if not already loaded. FLANN index will be
        # loaded when needed to prevent thread/subprocess memory issues.
        if self._codebook is None and has_model:
            self._codebook = numpy.load(self.codebook_filepath)
        return has_model

    @abc.abstractmethod
    def descriptor_type(self):
        """
        :return: String descriptor type as used by colorDescriptor
        :rtype: str
        """
        return

    @abc.abstractmethod
    def _generate_descriptor_matrices(self, *data_items):
        """
        Generate info and descriptor matrices based on ingest type.

        :param data_items: DataFile elements to generate combined info and
            descriptor matrices for.
        :type data_items: tuple of SMQTK.utils.DataFile.DataFile

        :return: Combined info and descriptor matrices for all base images
        :rtype: (numpy.core.multiarray.ndarray, numpy.core.multiarray.ndarray)

        """
        pass

    def _get_checkpoint_dir(self, data):
        """
        The directory that contains checkpoint material for a given data element

        :param data: Data element
        :type data: SMQTK.utils.DataFile.DataFile

        :return: directory path
        :rtype: str

        """
        d = osp.join(self.work_directory, *data.split_md5sum(8)[:-1])
        safe_create_dir(d)
        return d

    def _get_checkpoint_info_descriptors_file(self, data, frame=None):
        """
        Get the standard path to a data element's computed descriptor output,
        which for colorDescriptor consists of two matrices: info and descriptors

        :param data: Data element
        :type data: SMQTK.utils.DataFile.DataFile

        :param frame: frame within the data file
        :type frame: int

        :return: Paths to info and descriptor checkpoint numpy files
        :rtype: (str, str)

        """
        d = self._get_checkpoint_dir(data)
        return (
            osp.join(d, "%s.info.%d.npy" % (data.md5sum, frame or 0)),
            osp.join(d, "%s.descriptors.%d.npy" % (data.md5sum, frame or 0))
        )

    def _get_checkpoint_feature_file(self, data, frame=None):
        """
        Return the standard path to a data element's computed feature checkpoint
        file relative to our current working directory.

        :param data: Data element
        :type data: SMQTK.utils.DataFile.DataFile

        :param frame: frame within the data file
        :type frame: int

        :return: Standard path to where the feature checkpoint file for this
            given data element.
        :rtype: str

        """
        return osp.join(self._get_checkpoint_dir(data),
                        "%s.feature.npy" % data.md5sum)

    def generate_model(self, data_list, parallel=None, **kwargs):
        """
        Generate this feature detector's data-model given a file ingest. This
        saves the generated model to the currently configured data directory.

        For colorDescriptor, we generate raw features over the ingest data,
        compute a codebook via kmeans, and then create an index with FLANN via
        the "autotune" algorithm to intelligently pick the fastest indexing
        method.

        :param data_list: List of input data elements to generate model with.
        :type data_list: list of SMQTK.utils.DataFile.DataFile
            or tuple of SMQTK.utils.DataFile.DataFile

        :param parallel: Optionally specification of how many processors to use
            when pooling sub-tasks. If None, we attempt to use all available
            cores.
        :type parallel: int


        Additional optional key-word arguments
        ======================================

        :param kmeans_k: Centroids to generate. Default of 1024
        :type kmeans_k: int

        :param kmeans_iter: Number of times to run the kmeans algorithms, using
            the centroids from the best run. Default of 5.
        :type kmeans_iter: int

        :param kmeans_threshold: Distortion difference termination threshold.
            KMeans algorithm terminates during a run if the centroid distortion
            since the last iteration is less than this threshold. Default of
            1e-5.
        :type kmeans_threshold: float

        :param flann_target_precision: Target precision percent to tune index
            for. Default is 0.99 (99% accuracy).
        :type flann_target_precision: float

        :param flann_sample_fraction: Fraction of input data to use for index
            auto tuning. Default is 1.0 (100%).
        :type flann_sample_fraction: float

        """
        if self.has_model:
            self.log.warn("ColorDescriptor model for descriptor type '%s' "
                          "already generated!", self.descriptor_type())
            return

        pyflann.set_distance_type(self.FLANN_DISTANCE_FUNCTION)
        flann = pyflann.FLANN()

        if not osp.isfile(self.codebook_filepath):
            self.log.info("Did not find existing ColorDescriptor codebook for "
                          "descriptor '%s'.", self.descriptor_type())

            # generate descriptors
            with SimpleTimer("Generating descriptor matrices...", self.log):
                info, descriptors = \
                    self._generate_descriptor_matrices(*data_list)

            # compute centroids (codebook) with kmeans
            # - NOT performing whitening, as this transforms the feature space
            #   in such aray that newly computed features cannot be applied to
            #   the generated codebook as the same exact whitening
            #   transformation would need to be applied in order for the
            #   comparison to the codebook centroids to be valid.
            with SimpleTimer("Computing scipy.cluster.vq.kmeans...", self.log):
                codebook, distortion = scipy.cluster.vq.kmeans(
                    descriptors,
                    kwargs.get('kmeans_k', 1024),
                    kwargs.get('kmeans_iter', 5),
                    kwargs.get('kmeans_threshold', 1e-5)
                )
                self.log.debug("KMeans result distortion: %f", distortion)
                # Alternate kmeans implementations: OpenCV, sklearn, pyflann
            with SimpleTimer("Saving generated codebook...", self.log):
                numpy.save(self.codebook_filepath, codebook)
        else:
            self.log.info("Found existing codebook file.")
            codebook = numpy.load(self.codebook_filepath)

        # create FLANN index
        # - autotune will force select linear search if there are < 1000 words
        #   in the codebook vocabulary.
        if self.log.getEffectiveLevel() <= logging.DEBUG:
            log_level = 'info'
        else:
            log_level = 'warning'
        with SimpleTimer("Building FLANN index...", self.log):
            params = flann.build_index(codebook, **{
                "target_precision": kwargs.get("flann_target_precision", 0.99),
                "sample_fraction": kwargs.get("flann_sample_fraction", 1.0),
                "log_level": log_level,
                "algorithm": "autotuned"
            })
            # TODO: Save params dict as JSON?
        with SimpleTimer("Saving FLANN index to file...", self.log):
            flann.save_index(self.flann_index_filepath)

        # save generation results to class for immediate feature computation use
        self._codebook = codebook

    def compute_feature(self, data, no_checkpoint=False):
        """
        Given some kind of data, process and return a feature vector as a Numpy
        array.

        :raises RuntimeError: Feature extraction failure of some kind.

        :param data: Some kind of input data for the feature descriptor. This is
            descriptor dependent.
        :type data:
            SMQTK.utils.DataFile.DataFile or SMQTK.utils.VideoFile.VideoFile

        :param no_checkpoint: Normally, we produce a checkpoint file, which
            contains the numpy feature vector for a given video so that it may
            be loaded instead of re-computed if the same video is visited again.
            If this is True, we do not save such a file to our work directory.

        :return: Feature vector. This is a histogram of N bins where N is the
            number of centroids in the codebook. Bin values is percent
            composition, not absolute counts.
        :rtype: numpy.ndarray

        """
        checkpoint_filepath = self._get_checkpoint_feature_file(data)
        if osp.isfile(checkpoint_filepath):
            self.log.debug("Found checkpoint feature vector file, loading and "
                           "returning.")
            return numpy.load(checkpoint_filepath)

        if not self.has_model:
            raise RuntimeError("No model currently loaded! Check the existance "
                               "or, or generate, model files!\n"
                               "Codebook path: %s\n"
                               "FLANN Index path: %s"
                               % (self.codebook_filepath,
                                  self.flann_index_filepath))

        self.log.debug("Computing descriptors...")
        info, descriptors = self._generate_descriptor_matrices(data)

        # Quantization
        # - loaded the model at class initialization if we had one
        pyflann.set_distance_type(self.FLANN_DISTANCE_FUNCTION)
        flann = pyflann.FLANN()
        flann.load_index(self.flann_index_filepath, self._codebook)
        idxs, dists = flann.nn_index(descriptors)

        # Create histogram
        # - See numpy note about ``bins`` to understand why the +1 is necessary
        h, _ = numpy.histogram(idxs,
                               bins=numpy.arange(self._codebook.shape[0] + 1))
        self.log.debug("Quantization histogram: %s", h)
        # Normalize histogram into relative frequencies
        # - Not using /= on purpose. h is originally int32 coming out of
        #   histogram. /= would keep int32 type when we want it to be
        #   transformed into a float type by the division.
        if h.sum():
            h = h / float(h.sum())
        else:
            h = numpy.zeros(h.shape, h.dtype)
        self.log.debug("Normalized histogram: %s", h)

        if not no_checkpoint:
            if not osp.isdir(osp.dirname(checkpoint_filepath)):
                safe_create_dir(osp.dirname(checkpoint_filepath))
            numpy.save(checkpoint_filepath, h)

        return h


# noinspection PyAbstractClass
class ColorDescriptor_Image (ColorDescriptor_Base):

    def _generate_descriptor_matrices(self, *data_items):
        """
        Generate info and descriptor matrices based on ingest type.

        :param data_items: DataFile elements to generate combined info and
            descriptor matrices for.
        :type data_items: tuple of SMQTK.utils.DataFile.DataFile

        :return: Combined info and descriptor matrices for all base images
            vertically stacked.
        :rtype: (numpy.core.multiarray.ndarray, numpy.core.multiarray.ndarray)

        """
        assert len(data_items), "No data given to process."

        if len(data_items) == 1:
            # Check for checkpoint files
            return utils.generate_descriptors(
                self.PROC_COLORDESCRIPTOR, data_items[0].filepath,
                self.descriptor_type(),
                *self._get_checkpoint_info_descriptors_file(data_items[0])
            )
        else:
            # compute and V-stack matrices for all given images
            pool = multiprocessing.Pool(processes=self.PARALLEL)

            # Mapping of UID to async processing result
            #: :type: dict of (int, multiprocessing.pool.ApplyResult)
            r_map = {}
            with SimpleTimer("Computing descriptors async...", self.log):
                for di in data_items:
                    args = (self.PROC_COLORDESCRIPTOR, di.filepath,
                            self.descriptor_type())
                    args += self._get_checkpoint_info_descriptors_file(di)
                    r_map[di.uid] = pool.apply_async(
                        utils.generate_descriptors, args
                    )

            # Each result is a tuple of two ndarrays: info and descriptor
            # matrices.
            with SimpleTimer("Combining results...", self.log):
                combined_info = None
                combined_desc = None
                for uid in sorted(r_map.keys()):
                    i, d = r_map[uid].get()
                    if combined_info is None:
                        combined_info = i
                        combined_desc = d
                    else:
                        combined_info = numpy.vstack((combined_info, i))
                        combined_desc = numpy.vstack((combined_desc, d))

            pool.close()
            pool.join()
            return combined_info, combined_desc


# noinspection PyAbstractClass
class ColorDescriptor_Video (ColorDescriptor_Base):

    FRAME_EXTRACTION_PARAMS = {
        "second_offset": 0.2,       # Start 20% in
        "second_interval": 2,       # Sample every 2 seconds
        "max_duration": 0.6,        # Cover middle 60% of video
        "output_image_ext": 'png'   # Output PNG files
    }

    def _generate_descriptor_matrices(self, *data_items):
        """
        Generate info and descriptor matrices based on ingest type.

        :param data_items: DataFile elements to generate combined info and
            descriptor matrices for.
        :type data_items: tuple of SMQTK.utils.VideoFile.VideoFile

        :return: Combined info and descriptor matrices for all base images
            vertically stacked.
        :rtype: (numpy.core.multiarray.ndarray, numpy.core.multiarray.ndarray)

        """
        # For each video, extract frames and submit colorDescriptor processing
        # jobs for each frame, combining all results into a single matrix for
        # return.
        pool = multiprocessing.Pool(processes=self.PARALLEL)
        #: :type: dict of (tuple of (int, int), multiprocessing.pool.ApplyResult)
        r_map = {}
        with SimpleTimer("Extracting frames and submitting descriptor jobs..."):
            for di in data_items:
                p = dict(self.FRAME_EXTRACTION_PARAMS)
                p['second_offset'] = di.metadata().duration * p['second_offset']
                p['max_duration'] = di.metadata().duration * p['max_duration']
                fm = di.frame_map(**self.FRAME_EXTRACTION_PARAMS)
                for frame, imgPath in fm.iteritems():
                    r_map[di.uid, frame] = pool.apply_async(
                        utils.generate_descriptors,
                        args=(self.PROC_COLORDESCRIPTOR, imgPath,
                              self.descriptor_type())
                    )

        # Each result is a tuple of two ndarrays: info and descriptor matrices
        with SimpleTimer("Combining results...", self.log):
            combined_info = None
            combined_desc = None
            for uid, frame in sorted(r_map.keys()):
                i, d = r_map[uid, frame].get()
                if combined_info is None:
                    combined_info = i
                    combined_desc = d
                else:
                    combined_info = numpy.vstack((combined_info, i))
                    combined_desc = numpy.vstack((combined_desc, d))

        pool.close()
        pool.join()
        return combined_info, combined_desc


valid_descriptor_types = [
    'csift',
    'transformedcolorhistogram'
]


def create_image_descriptor_class(descriptor_type_str):
    """
    Create and return a ColorDescriptor class that operates over Image files
    using the given descriptor type.
    """
    assert descriptor_type_str in valid_descriptor_types, \
        "Given ColorDescriptor type was not valid! Given: %s. Expected one " \
        "of: %s" % (descriptor_type_str, valid_descriptor_types)

    class _cd_image_impl (ColorDescriptor_Image):
        def descriptor_type(self):
            """
            :rtype: str
            """
            return descriptor_type_str

    _cd_image_impl.__name__ = "ColorDescriptor_Image_%s" % descriptor_type_str
    return _cd_image_impl


def create_video_descriptor_class(descriptor_type_str):
    """
    Create and return a ColorDescriptor class that operates over Video files
    using the given descriptor type.
    """
    assert descriptor_type_str in valid_descriptor_types, \
        "Given ColorDescriptor type was not valid! Given: %s. Expected one " \
        "of: %s" % (descriptor_type_str, valid_descriptor_types)

    class _cd_video_impl (ColorDescriptor_Video):
        def descriptor_type(self):
            """
            :rtype: str
            """
            return descriptor_type_str

    _cd_video_impl.__name__ = "ColorDescriptor_Video_%s" % descriptor_type_str
    return _cd_video_impl


ColorDescriptor_Image_csift = create_image_descriptor_class('csift')
ColorDescriptor_Image_transformedcolorhistogram = \
    create_image_descriptor_class('transformedcolorhistogram')

ColorDescriptor_Video_csift = create_video_descriptor_class('csift')
ColorDescriptor_Video_transformedcolorhistogram = \
    create_video_descriptor_class('transformedcolorhistogram')
