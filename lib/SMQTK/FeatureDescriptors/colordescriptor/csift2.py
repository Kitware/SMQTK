import abc
import logging
import multiprocessing
import multiprocessing.pool
import numpy
import os
import os.path as osp
import pyflann
import scipy.cluster.vq
import tempfile

from SMQTK.FeatureDescriptors import FeatureDescriptor
from SMQTK.utils import safe_create_dir, SimpleTimer

from . import utils


class ColorDescriptor_Base (FeatureDescriptor):
    """
    Simple implementation of ColorDescriptor feature descriptor utility for
    feature generation over images and videos.

    This was started as an attempt at gaining a deeper understanding of what's
    going on with this feature descriptor's use and how it applied to later use
    in an indexer.

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

    # Total number of descriptors to use from input data to generate codebook
    # model. Fewer than this may be used if the data set is small, but if it is
    # greater, we randomly sample down to this count (occurs on a per element
    # basis).
    CODEBOOK_DESCRIPTOR_LIMIT = 1000000.

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
    def _generate_descriptor_matrices(self, *data_items, **kwargs):
        """
        Generate info and descriptor matrices based on ingest type.

        :param data_items: DataFile elements to generate combined info and
            descriptor matrices for.
        :type data_items: tuple of SMQTK.utils.DataFile.DataFile

        :param limit: Limit the number of descriptor entries to this amount.
        :type limit: int

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
        d = osp.join(self.work_directory, *data.split_md5sum(8))
        safe_create_dir(d)
        return d

    def _get_standard_info_descriptors_filepath(self, data, frame=None):
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
        if frame is not None:
            return (
                osp.join(d, "%s.info.%d.npy" % (data.md5sum, frame)),
                osp.join(d, "%s.descriptors.%d.npy" % (data.md5sum, frame))
            )
        else:
            return (
                osp.join(d, "%s.info.npy" % data.md5sum),
                osp.join(d, "%s.descriptors.npy" % data.md5sum)
            )

    def _get_checkpoint_feature_file(self, data):
        """
        Return the standard path to a data element's computed feature checkpoint
        file relative to our current working directory.

        :param data: Data element
        :type data: SMQTK.utils.DataFile.DataFile

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
        # Set an arbitrary limit of 1000000 descriptors across all data elements
        # num of descriptors to take per element = 1000000 / N

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
            with SimpleTimer("Generating descriptor matrices...", self.log.debug):
                descriptors_checkpoint = osp.join(self.work_directory,
                                                  "model_descriptors.npy")

                if osp.isfile(descriptors_checkpoint):
                    self.log.debug("Found existing computed descriptors work "
                                   "file for model generation.")
                    descriptors = numpy.load(descriptors_checkpoint)
                else:
                    self.log.debug("Computing model descriptors")
                    _, descriptors = \
                        self._generate_descriptor_matrices(
                            *data_list,
                            limit=self.CODEBOOK_DESCRIPTOR_LIMIT
                        )
                    _, tmp = tempfile.mkstemp(dir=self.work_directory,
                                              suffix='.npy')
                    self.log.debug("Saving model-gen info/descriptor matrix")
                    numpy.save(tmp, descriptors)
                    os.rename(tmp, descriptors_checkpoint)

            # Compute centroids (codebook) with kmeans
            # - NOT performing whitening, as this transforms the feature space
            #   in such a way that newly computed features cannot be applied to
            #   the generated codebook as the same exact whitening
            #   transformation would need to be applied in order for the
            #   comparison to the codebook centroids to be valid.
            # - Alternate kmeans implementations: OpenCV, sklearn, pyflann
            with SimpleTimer("Computing scipy.cluster.vq.kmeans...",
                             self.log.debug):
                codebook, distortion = scipy.cluster.vq.kmeans(
                    descriptors,
                    kwargs.get('kmeans_k', 1024),
                    kwargs.get('kmeans_iter', 5),
                    kwargs.get('kmeans_threshold', 1e-5)
                )
                self.log.debug("KMeans result distortion: %f", distortion)
            # with SimpleTimer("Computing pyflann.FLANN.hierarchical_kmeans...",
            #                  self.log.debug):
            #     # results in 1009 clusters (should, anyway, given the
            #     # function's comment)
            #     codebook2 = flann.hierarchical_kmeans(descriptors, 64, 16, 5)
            with SimpleTimer("Saving generated codebook...", self.log.debug):
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
        with SimpleTimer("Building FLANN index...", self.log.debug):
            params = flann.build_index(codebook, **{
                "target_precision": kwargs.get("flann_target_precision", 0.99),
                "sample_fraction": kwargs.get("flann_sample_fraction", 1.0),
                "log_level": log_level,
                "algorithm": "autotuned"
            })
            # TODO: Save params dict as JSON?
        with SimpleTimer("Saving FLANN index to file...", self.log.debug):
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
        h, _ = numpy.histogram(idxs, bins=self._codebook.shape[0] + 1)
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

    def _generate_descriptor_matrices(self, *data_items, **kwargs):
        """
        Generate info and descriptor matrices based on ingest type.

        :param data_items: DataFile elements to generate combined info and
            descriptor matrices for.
        :type data_items: tuple of SMQTK.utils.DataFile.DataFile

        :param limit: Limit the number of descriptor entries to this amount.
        :type limit: int

        :return: Combined info and descriptor matrices for all base images
        :rtype: (numpy.core.multiarray.ndarray, numpy.core.multiarray.ndarray)

        """
        if not len(data_items):
            raise ValueError("No data given to process.")

        inf = float('inf')
        descriptor_limit = kwargs.get('limit', inf)
        per_item_limit = numpy.floor(float(descriptor_limit) / len(data_items))

        if len(data_items) == 1:
            # Check for checkpoint files
            info_fp, desc_fp = \
                self._get_standard_info_descriptors_filepath(data_items[0])
            utils.generate_descriptors(
                self.PROC_COLORDESCRIPTOR, data_items[0].filepath,
                self.descriptor_type(), info_fp, desc_fp, per_item_limit
            )
            return numpy.load(info_fp), numpy.load(desc_fp)
        else:
            # compute and V-stack matrices for all given images
            pool = multiprocessing.Pool(processes=self.PARALLEL)

            # Mapping of UID to tuple containing:
            #   (info_fp, desc_fp, async processing result)
            r_map = {}
            with SimpleTimer("Computing descriptors async...", self.log.debug):
                for di in data_items:
                    info_fp, desc_fp = \
                        self._get_standard_info_descriptors_filepath(di)
                    args = (self.PROC_COLORDESCRIPTOR, di.filepath,
                            self.descriptor_type(), info_fp, desc_fp)
                    r = pool.apply_async(utils.generate_descriptors, args)
                    r_map[di.uid] = (info_fp, desc_fp, r)
            pool.close()

            # Pass through results from descriptor generation, aggregating
            # matrix shapes.
            # - Transforms r_map into:
            #       UID -> (info_fp, desc_fp, starting_row, SubSampleIndices)
            self.log.debug("Constructing information for super matrices...")
            s_keys = sorted(r_map.keys())
            running_height = 0  # info and desc heights congruent
            i_width = d_width = None
            for uid in s_keys:
                ifp, dfp, r = r_map[uid]
                i_shape, d_shape = r.get()
                if None in (i_width, d_width):
                    i_width = i_shape[1]
                    d_width = d_shape[1]

                ssi = None
                if i_shape[0] > per_item_limit:
                    # pick random indices to subsample down to size limit
                    ssi = sorted(
                        numpy.random.permutation(i_shape[0])[:per_item_limit]
                    )

                r_map[uid] = (ifp, dfp, running_height, ssi)
                running_height += min(i_shape[0], per_item_limit)
            pool.join()

            # Asynchronously load files, inserting data into master matrices
            self.log.debug("Building super matrices...")
            master_info = numpy.zeros((running_height, i_width), dtype=float)
            master_desc = numpy.zeros((running_height, d_width), dtype=float)
            tp = multiprocessing.pool.ThreadPool(processes=self.PARALLEL)
            for uid in s_keys:
                ifp, dfp, sR, ssi = r_map[uid]
                tp.apply_async(ColorDescriptor_Image._thread_load_matrix,
                               args=(ifp, master_info, sR, ssi))
                tp.apply_async(ColorDescriptor_Image._thread_load_matrix,
                               args=(dfp, master_desc, sR, ssi))
            tp.close()
            tp.join()
            return master_info, master_desc

    @staticmethod
    def _thread_load_matrix(filepath, m, sR, subsample=None):
        """
        load a numpy matrix from ``filepath``, inserting the loaded matrix into
        ``m`` starting at the row ``sR``.

        If subsample has a value, it will be a list if indices to
        """
        n = numpy.load(filepath)
        if subsample:
            n = n[subsample, :]
        m[sR:sR+n.shape[0], :n.shape[1]] = n


# noinspection PyAbstractClass
class ColorDescriptor_Video (ColorDescriptor_Base):

    FRAME_EXTRACTION_PARAMS = {
        "second_offset": 0.2,       # Start 20% in
        "second_interval": 2,       # Sample every 2 seconds
        "max_duration": 0.6,        # Cover middle 60% of video
        "output_image_ext": 'png'   # Output PNG files
    }

    def _generate_descriptor_matrices(self, *data_items, **kwargs):
        """
        Generate info and descriptor matrices based on ingest type.

        :param data_items: DataFile elements to generate combined info and
            descriptor matrices for.
        :type data_items: tuple of SMQTK.utils.VideoFile.VideoFile

        :param limit: Limit the number of descriptor entries to this amount.
        :type limit: int

        :return: Combined info and descriptor matrices for all base images
        :rtype: (numpy.core.multiarray.ndarray, numpy.core.multiarray.ndarray)

        """
        descriptor_limit = kwargs.get('limit', float('inf'))
        # With videos, an "item" is one video, so, collect for a while video
        # as normal, then subsample from the full video collection.
        per_item_limit = numpy.floor(float(descriptor_limit) / len(data_items))

        # For each video, extract frames and submit colorDescriptor processing
        # jobs for each frame, combining all results into a single matrix for
        # return.
        pool = multiprocessing.Pool(processes=self.PARALLEL)

        # Mapping of [UID] to [frame] to tuple containing:
        #   (info_fp, desc_fp, async processing result)
        r_map = {}
        with SimpleTimer("Extracting frames and submitting descriptor jobs...",
                         self.log.debug):
            for di in data_items:
                r_map[di.uid] = {}
                p = dict(self.FRAME_EXTRACTION_PARAMS)
                p['second_offset'] = di.metadata().duration * p['second_offset']
                p['max_duration'] = di.metadata().duration * p['max_duration']
                fm = di.frame_map(**self.FRAME_EXTRACTION_PARAMS)
                for frame, imgPath in fm.iteritems():
                    info_fp, desc_fp = \
                        self._get_standard_info_descriptors_filepath(di, frame)
                    r = pool.apply_async(
                        utils.generate_descriptors,
                        args=(self.PROC_COLORDESCRIPTOR, imgPath,
                              self.descriptor_type(), info_fp, desc_fp)
                    )
                    r_map[di.uid][frame] = (info_fp, desc_fp, r)
        pool.close()

        # Each result is a tuple of two ndarrays: info and descriptor matrices
        with SimpleTimer("Collecting shape information for super matrices...",
                         self.log.debug):
            running_height = 0
            i_width = d_width = None

            # Transform r_map[uid] into:
            #   (info_mat_files, desc_mat_files, sR, ssi_list)
            #   -> files in frame order
            uids = sorted(r_map)
            for uid in uids:
                video_num_desc = 0
                video_info_mat_fps = []  # ordered list of frame info mat files
                video_desc_mat_fps = []  # ordered list of frame desc mat files
                for frame in sorted(r_map[uid]):
                    ifp, dfp, r = r_map[uid][frame]
                    i_shape, d_shape = r.get()
                    if None in (i_width, d_width):
                        i_width = i_shape[1]
                        d_width = d_shape[1]

                    video_info_mat_fps.append(ifp)
                    video_desc_mat_fps.append(dfp)
                    video_num_desc += i_shape[0]

                    # If combined descriptor height exceeds the per-item limit,
                    # generate a random subsample index list
                    ssi = None
                    if video_num_desc > per_item_limit:
                        ssi = sorted(
                            numpy.random.permutation(video_num_desc)[:per_item_limit]
                        )
                        video_num_desc = len(ssi)

                    r_map[uid] = (video_info_mat_fps, video_desc_mat_fps,
                                  running_height, ssi)
                    running_height += video_num_desc
        pool.join()

        with SimpleTimer("Building master descriptor matrices...",
                         self.log.debug):
            master_info = numpy.zeros((running_height, i_width), dtype=float)
            master_desc = numpy.zeros((running_height, d_width), dtype=float)
            tp = multiprocessing.pool.ThreadPool(processes=self.PARALLEL)
            for uid in uids:
                info_fp_list, desc_fp_list, sR, ssi = r_map[uid]
                tp.apply_async(ColorDescriptor_Video._thread_load_matrices,
                               args=(master_info, info_fp_list, sR, ssi))
                tp.apply_async(ColorDescriptor_Video._thread_load_matrices,
                               args=(master_desc, desc_fp_list, sR, ssi))
            tp.close()
            tp.join()

        return master_info, master_desc

    @staticmethod
    def _thread_load_matrices(m, file_list, sR, subsample=None):
        """
        load numpy matrices from files in ``file_list``, concatenating them
        vertically. If a list of row indices is provided in ``subsample`` we
        subsample those rows out of the concatenated matrix. This matrix is then
        inserted into ``m`` starting at row ``sR``.
        """
        c = numpy.load(file_list[0])
        for i in range(1, len(file_list)):
            c = numpy.vstack((c, numpy.load(file_list[i])))
        if subsample:
            c = c[subsample, :]
        m[sR:sR+c.shape[0], :c.shape[1]] = c


# Begin automatic class type creation
valid_descriptor_types = [
    'rgbhistogram',
    'opponenthistogram',
    'huehistogram',
    'nrghistogram',
    'transformedcolorhistogram',
    'colormoments',
    'colormomentinvariants',
    'sift',
    'huesift',
    'hsvsift',
    'opponentsift',
    'rgsift',
    'csift',
    'rgbsift',
]


def _create_image_descriptor_class(descriptor_type_str):
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


def _create_video_descriptor_class(descriptor_type_str):
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


# In order to allow multiprocessing, class types must be concretely assigned to
# variables in the module. Dynamic generation causes issues with pickling (the
# default data transmission protocol).

ColorDescriptor_Image_rgbhistogram = _create_image_descriptor_class('rgbhistogram')
ColorDescriptor_Image_opponenthistogram = _create_image_descriptor_class('opponenthistogram')
ColorDescriptor_Image_huehistogram = _create_image_descriptor_class('huehistogram')
ColorDescriptor_Image_nrghistogram = _create_image_descriptor_class('nrghistogram')
ColorDescriptor_Image_transformedcolorhistogram = _create_image_descriptor_class('transformedcolorhistogram')
ColorDescriptor_Image_colormoments = _create_image_descriptor_class('colormoments')
ColorDescriptor_Image_colormomentinvariants = _create_image_descriptor_class('colormomentinvariants')
ColorDescriptor_Image_sift = _create_image_descriptor_class('sift')
ColorDescriptor_Image_huesift = _create_image_descriptor_class('huesift')
ColorDescriptor_Image_hsvsift = _create_image_descriptor_class('hsvsift')
ColorDescriptor_Image_opponentsift = _create_image_descriptor_class('opponentsift')
ColorDescriptor_Image_rgsift = _create_image_descriptor_class('rgsift')
ColorDescriptor_Image_csift = _create_image_descriptor_class('csift')
ColorDescriptor_Image_rgbsift = _create_image_descriptor_class('rgbsift')

ColorDescriptor_Video_rgbhistogram = _create_video_descriptor_class('rgbhistogram')
ColorDescriptor_Video_opponenthistogram = _create_video_descriptor_class('opponenthistogram')
ColorDescriptor_Video_huehistogram = _create_video_descriptor_class('huehistogram')
ColorDescriptor_Video_nrghistogram = _create_video_descriptor_class('nrghistogram')
ColorDescriptor_Video_transformedcolorhistogram = _create_video_descriptor_class('transformedcolorhistogram')
ColorDescriptor_Video_colormoments = _create_video_descriptor_class('colormoments')
ColorDescriptor_Video_colormomentinvariants = _create_video_descriptor_class('colormomentinvariants')
ColorDescriptor_Video_sift = _create_video_descriptor_class('sift')
ColorDescriptor_Video_huesift = _create_video_descriptor_class('huesift')
ColorDescriptor_Video_hsvsift = _create_video_descriptor_class('hsvsift')
ColorDescriptor_Video_opponentsift = _create_video_descriptor_class('opponentsift')
ColorDescriptor_Video_rgsift = _create_video_descriptor_class('rgsift')
ColorDescriptor_Video_csift = _create_video_descriptor_class('csift')
ColorDescriptor_Video_rgbsift = _create_video_descriptor_class('rgbsift')


cd_type_list = [
    ColorDescriptor_Image_rgbhistogram,
    ColorDescriptor_Video_rgbhistogram,
    ColorDescriptor_Image_opponenthistogram,
    ColorDescriptor_Video_opponenthistogram,
    ColorDescriptor_Image_huehistogram,
    ColorDescriptor_Video_huehistogram,
    ColorDescriptor_Image_nrghistogram,
    ColorDescriptor_Video_nrghistogram,
    ColorDescriptor_Image_transformedcolorhistogram,
    ColorDescriptor_Video_transformedcolorhistogram,
    ColorDescriptor_Image_colormoments,
    ColorDescriptor_Video_colormoments,
    ColorDescriptor_Image_colormomentinvariants,
    ColorDescriptor_Video_colormomentinvariants,
    ColorDescriptor_Image_sift,
    ColorDescriptor_Video_sift,
    ColorDescriptor_Image_huesift,
    ColorDescriptor_Video_huesift,
    ColorDescriptor_Image_hsvsift,
    ColorDescriptor_Video_hsvsift,
    ColorDescriptor_Image_opponentsift,
    ColorDescriptor_Video_opponentsift,
    ColorDescriptor_Image_rgsift,
    ColorDescriptor_Video_rgsift,
    ColorDescriptor_Image_csift,
    ColorDescriptor_Video_csift,
    ColorDescriptor_Image_rgbsift,
    ColorDescriptor_Video_rgbsift,
]
