import abc
import logging
import json
import math
import mimetypes
import multiprocessing
import multiprocessing.pool
import os
import os.path as osp
import subprocess
import sys
import tempfile

import numpy
import sklearn.cluster

from smqtk.algorithms.descriptor_generator import DescriptorGenerator
from smqtk.representation.data_element.file_element import DataFileElement
from smqtk.utils import file_utils, SimpleTimer, video_utils
from smqtk.utils.string_utils import partition_string
from smqtk.utils.video_utils import get_metadata_info

# Attempt importing utilities module. If not, flag descriptor as unusable.
from . import utils


# Requires FLANN bindings
try:
    import pyflann
except ImportError:
    pyflann = None


# noinspection PyPep8Naming
class ColorDescriptor_Base (DescriptorGenerator):
    """
    Simple implementation of ColorDescriptor feature descriptor utility for
    feature generation over images and videos.

    This was started as an attempt at gaining a deeper understanding of what's
    going on with this feature descriptor's use and how it applied to later use
    in an indexer.

    Codebook generated via kmeans given a set of input data. FLANN index model
    used for quantization, built using auto-tuning (picks the best indexing
    algorithm of linear, kdtree, kmeans, or combined), and using the Chi-Squared
    distance function.

    """

    # Name/Path to the colorDescriptor executable to use. By default we assume
    # its accessible on the PATH.
    EXE = 'colorDescriptor'

    @classmethod
    def is_usable(cls):
        """
        Check whether this descriptor is available for use.

        :return: Boolean determination of whether this implementation is usable.
        :rtype: bool

        """
        log = logging.getLogger('.'.join([ColorDescriptor_Base.__module__,
                                          ColorDescriptor_Base.__name__,
                                          "is_usable"]))

        if not hasattr(ColorDescriptor_Base, "_is_usable_cache"):
            # Base assumption that it is available, now lets prove its false.
            ColorDescriptor_Base._is_usable_cache = True

            if pyflann is None:
                # missing FLANN bindings dependency
                log.debug("could not import FLANN bindings (pyflann)")
                ColorDescriptor_Base._is_usable_cache = False
            else:
                # Check for colorDescriptor executable on the path
                log_file = open(tempfile.mkstemp()[1], 'w')
                try:
                    # This should try to print out the CLI options return with
                    # code 1.
                    subprocess.call([cls.EXE, '--version'],
                                    stdout=log_file, stderr=log_file)
                    # it is known that colorDescriptor has a return code of 1 no
                    # matter if it exited "successfully" or not, which is not
                    # helpful, I know.
                except OSError:
                    log.debug("Could not locate colorDescriptor executable. "
                              "Make sure that its on the PATH! See "
                              "smqtk/descriptor_generator/colordescriptor/"
                              "INSTALL.md for help.")
                    # If there was anything written to the log file, output it.
                    log_file.flush()
                    if log_file.tell():
                        with open(log_file.name) as f:
                            log.debug("STDOUT and STDERR output from attempted "
                                      "colorDescriptor call:\n%s", f.read())
                    ColorDescriptor_Base._is_usable_cache = False
                finally:
                    log_file.close()
                    os.remove(log_file.name)

                # Checking if DescriptorIO is importable
                if not utils.has_colordescriptor_module():
                    log.debug("Could not import DescriptorIO. Make sure that "
                              "the colorDescriptor package is on the "
                              "PYTHONPATH! See "
                              "smqtk/descriptor_generator/colordescriptor/"
                              "INSTALL.md for help.")
                    ColorDescriptor_Base._is_usable_cache = False

        return ColorDescriptor_Base._is_usable_cache

    def __init__(self, model_directory, work_directory,
                 model_gen_descriptor_limit=1000000,
                 kmeans_k=1024, flann_distance_metric='hik',
                 flann_target_precision=0.95,
                 flann_sample_fraction=0.75,
                 flann_autotune=False,
                 random_seed=None, use_spatial_pyramid=False,
                 parallel=None):
        """
        Initialize a new ColorDescriptor interface instance.

        :param model_directory: Path to the directory to store/read data model
            files on the local filesystem. Relative paths are treated relative
            to the current working directory.
        :type model_directory: str | unicode

        :param work_directory: Path to the directory in which to place
            temporary/working files. Relative paths are treated relative to
            the current working directory.
        :type work_directory: str | unicode

        :param model_gen_descriptor_limit: Total number of descriptors to use
            from input data to generate codebook model. Fewer than this may be
            used if the data set is small, but if it is greater, we randomly
            sample down to this count (occurs on a per element basis).
        :type model_gen_descriptor_limit: int

        :param kmeans_k: Centroids to generate. Default of 1024
        :type kmeans_k: int

        :param flann_distance_metric: Distance function to use in FLANN
            indexing. See FLANN documentation for available distance function
            types (under the MATLAB section reference for valid string
            identifiers)
        :type flann_distance_metric: str

        :param flann_target_precision: Target precision percent to tune index
            for. Default is 0.95 (95% accuracy). For some codebooks, if this is
            too close to 1.0, the FLANN library may non-deterministically
            overflow, causing an infinite loop requiring a SIGKILL to stop.
        :type flann_target_precision: float

        :param flann_sample_fraction: Fraction of input data to use for index
            auto tuning. Default is 0.75 (75%).
        :type flann_sample_fraction: float

        :param flann_autotune: Have FLANN module use auto-tuning algorithm to
            find an optimal index representation and parameter set.
        :type flann_autotune: bool

        :param random_seed: Optional value to seed components requiring random
            operations.
        :type random_seed: None or int

        :param use_spatial_pyramid: Use spacial pyramids when quantizing low
            level descriptors during feature computation.
        :type use_spatial_pyramid: bool

        :param parallel: Specific number of threads/cores to use when performing
            asynchronous activities. When None we will use all cores available.
        :type parallel: int | None

        """
        super(ColorDescriptor_Base, self).__init__()

        # TODO: Because of the FLANN library non-deterministic overflow issue,
        #       an alternative must be found before this can be put into
        #       production. Suggest saving/using sk-learn MBKMeans class? Can
        #       the class be regenerated from an existing codebook?
        self._model_dir = model_directory
        self._work_dir = work_directory

        self._model_gen_descriptor_limit = model_gen_descriptor_limit

        self._kmeans_k = int(kmeans_k)
        self._flann_distance_metric = flann_distance_metric
        self._flann_target_precision = float(flann_target_precision)
        self._flann_sample_fraction = float(flann_sample_fraction)
        self._flann_autotune = bool(flann_autotune)
        self._use_sp = use_spatial_pyramid
        self._rand_seed = None if random_seed is None else int(random_seed)

        if self._rand_seed is not None:
            numpy.random.seed(self._rand_seed)

        self.parallel = parallel

        # Cannot pre-load FLANN stuff because odd things happen when processing/
        # threading. Loading index file is fast anyway.
        self._codebook = None
        if self.has_model:
            self._codebook = numpy.load(self.codebook_filepath)

    def get_config(self):
        """
        Return a JSON-compliant dictionary that could be passed to this class's
        ``from_config`` method to produce an instance with identical
        configuration.

        Keys of the returned dictionary are based on the initialization argument
        names.

        :return: JSON type compliant configuration dictionary.
        :rtype: dict

        """
        return {
            "model_directory": self._model_dir,
            "work_directory": self._work_dir,
            "model_gen_descriptor_limit": self._model_gen_descriptor_limit,
            "kmeans_k": self._kmeans_k,
            "flann_distance_metric": self._flann_distance_metric,
            "flann_target_precision": self._flann_target_precision,
            "flann_sample_fraction": self._flann_sample_fraction,
            "flann_autotune": self._flann_autotune,
            "random_seed": self._rand_seed,
            "use_spatial_pyramid": self._use_sp,
        }

    @property
    def name(self):
        if self._use_sp:
            return '_'.join([self.__class__.__name__, 'spatial'])
        else:
            return self.__class__.__name__

    @property
    def codebook_filepath(self):
        file_utils.safe_create_dir(self._model_dir)
        return osp.join(self._model_dir,
                        "%s.codebook.npy" % (self.descriptor_type(),))

    @property
    def flann_index_filepath(self):
        file_utils.safe_create_dir(self._model_dir)
        return osp.join(self._model_dir,
                        "%s.flann_index.dat" % (self.descriptor_type(),))

    @property
    def flann_params_filepath(self):
        file_utils.safe_create_dir(self._model_dir)
        return osp.join(self._model_dir,
                        "%s.flann_params.json" % (self.descriptor_type(),))

    @property
    def has_model(self):
        has_model = (osp.isfile(self.codebook_filepath) and
                     osp.isfile(self.flann_index_filepath))
        # Load the codebook model if not already loaded. FLANN index will be
        # loaded when needed to prevent thread/subprocess memory issues.
        if self._codebook is None and has_model:
            self._codebook = numpy.load(self.codebook_filepath)
        return has_model

    @property
    def temp_dir(self):
        return file_utils.safe_create_dir(osp.join(self._work_dir, 'temp_files'))

    @abc.abstractmethod
    def descriptor_type(self):
        """
        :return: String descriptor type as used by colorDescriptor
        :rtype: str
        """
        return

    @abc.abstractmethod
    def _generate_descriptor_matrices(self, data_set, **kwargs):
        """
        Generate info and descriptor matrices based on ingest type.

        :param data_set: Iterable of data elements to generate combined info
            and descriptor matrices for.
        :type item_iter: collections.Set[smqtk.representation.DataElement]

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
        :type data: smqtk.representation.DataElement

        :return: directory path
        :rtype: str

        """
        d = osp.join(self._work_dir, *partition_string(str(data.uuid()), 10))
        file_utils.safe_create_dir(d)
        return d

    def _get_standard_info_descriptors_filepath(self, data, frame=None):
        """
        Get the standard path to a data element's computed descriptor output,
        which for colorDescriptor consists of two matrices: info and descriptors

        :param data: Data element
        :type data: smqtk.representation.DataElement

        :param frame: frame within the data file
        :type frame: int

        :return: Paths to info and descriptor checkpoint numpy files
        :rtype: (str, str)

        """
        d = self._get_checkpoint_dir(data)
        if frame is not None:
            return (
                osp.join(d, "%s.info.%d.npy" % (str(data.uuid()), frame)),
                osp.join(d, "%s.descriptors.%d.npy" % (str(data.uuid()), frame))
            )
        else:
            return (
                osp.join(d, "%s.info.npy" % str(data.uuid())),
                osp.join(d, "%s.descriptors.npy" % str(data.uuid()))
            )

    def _get_checkpoint_feature_file(self, data):
        """
        Return the standard path to a data element's computed feature checkpoint
        file relative to our current working directory.

        :param data: Data element
        :type data: smqtk.representation.DataElement

        :return: Standard path to where the feature checkpoint file for this
            given data element.
        :rtype: str

        """
        if self._use_sp:
            return osp.join(self._get_checkpoint_dir(data),
                            "%s.feature.sp.npy" % str(data.uuid()))
        else:
            return osp.join(self._get_checkpoint_dir(data),
                            "%s.feature.npy" % str(data.uuid()))

    def generate_model(self, data_set, **kwargs):
        """
        Generate this feature detector's data-model given a file ingest. This
        saves the generated model to the currently configured data directory.

        For colorDescriptor, we generate raw features over the ingest data,
        compute a codebook via kmeans, and then create an index with FLANN via
        the "autotune" or linear algorithm to intelligently pick the fastest
        indexing method.

        :param data_set: Set of input data elements to generate the model
            with.
        :type data_set: collections.Set[smqtk.representation.DataElement]

        """
        if self.has_model:
            self._log.warn("ColorDescriptor model for descriptor type '%s' "
                           "already generated!", self.descriptor_type())
            return

        # Check that input data is value for processing through colorDescriptor
        valid_types = self.valid_content_types()
        invalid_types_found = set()
        for di in data_set:
            if di.content_type() not in valid_types:
                invalid_types_found.add(di.content_type())
        if invalid_types_found:
            self._log.error("Found one or more invalid content types among "
                            "input:")
            for t in sorted(invalid_types_found):
                self._log.error("\t- '%s", t)
            raise ValueError("Discovered invalid content type among input "
                             "data: %s" % sorted(invalid_types_found))

        if not osp.isfile(self.codebook_filepath):
            self._log.info("Did not find existing ColorDescriptor codebook for "
                           "descriptor '%s'.", self.descriptor_type())

            # generate descriptors
            with SimpleTimer("Generating descriptor matrices...",
                             self._log.info):
                descriptors_checkpoint = osp.join(self._work_dir,
                                                  "model_descriptors.npy")

                if osp.isfile(descriptors_checkpoint):
                    self._log.debug("Found existing computed descriptors work "
                                    "file for model generation.")
                    descriptors = numpy.load(descriptors_checkpoint)
                else:
                    self._log.debug("Computing model descriptors")
                    _, descriptors = \
                        self._generate_descriptor_matrices(
                            data_set,
                            limit=self._model_gen_descriptor_limit
                        )
                    _, tmp = tempfile.mkstemp(dir=self._work_dir,
                                              suffix='.npy')
                    self._log.debug("Saving model-gen info/descriptor matrix")
                    numpy.save(tmp, descriptors)
                    os.rename(tmp, descriptors_checkpoint)

            # Compute centroids (codebook) with kmeans
            with SimpleTimer("Computing sklearn.cluster.MiniBatchKMeans...",
                             self._log.info):
                kmeans_verbose = self._log.getEffectiveLevel <= logging.DEBUG
                kmeans = sklearn.cluster.MiniBatchKMeans(
                    n_clusters=self._kmeans_k,
                    init_size=self._kmeans_k*3,
                    random_state=self._rand_seed,
                    verbose=kmeans_verbose,
                    compute_labels=False,
                )
                kmeans.fit(descriptors)
                codebook = kmeans.cluster_centers_
            with SimpleTimer("Saving generated codebook...", self._log.debug):
                numpy.save(self.codebook_filepath, codebook)
        else:
            self._log.info("Found existing codebook file.")
            codebook = numpy.load(self.codebook_filepath)

        # create FLANN index
        # - autotune will force select linear search if there are < 1000 words
        #   in the codebook vocabulary.
        pyflann.set_distance_type(self._flann_distance_metric)
        flann = pyflann.FLANN()
        if self._log.getEffectiveLevel() <= logging.DEBUG:
            log_level = 'info'
        else:
            log_level = 'warning'
        with SimpleTimer("Building FLANN index...", self._log.info):
            p = {
                "target_precision": self._flann_target_precision,
                "sample_fraction": self._flann_sample_fraction,
                "log_level": log_level,
            }
            if self._flann_autotune:
                p['algorithm'] = "autotuned"
            if self._rand_seed is not None:
                p['random_seed'] = self._rand_seed
            flann_params = flann.build_index(codebook, **p)
        with SimpleTimer("Saving FLANN index to file...", self._log.debug):
            # Save FLANN index data binary
            flann.save_index(self.flann_index_filepath)
            # Save out log of parameters
            with open(self.flann_params_filepath, 'w') as ofile:
                json.dump(flann_params, ofile, indent=4, sort_keys=True)

        # save generation results to class for immediate feature computation use
        self._codebook = codebook

    def _compute_descriptor(self, data):
        """
        Given some kind of data, process and return a feature vector as a Numpy
        array.

        :raises RuntimeError: Feature extraction failure of some kind.

        :param data: Some kind of input data for the feature descriptor. This is
            descriptor dependent.
        :type data: smqtk.representation.DataElement

        :return: Feature vector. This is a histogram of N bins where N is the
            number of centroids in the codebook. Bin values is percent
            composition, not absolute counts.
        :rtype: numpy.ndarray

        """
        super(ColorDescriptor_Base, self)._compute_descriptor(data)

        checkpoint_filepath = self._get_checkpoint_feature_file(data)
        # if osp.isfile(checkpoint_filepath):
        #     return numpy.load(checkpoint_filepath)

        if not self.has_model:
            raise RuntimeError("No model currently loaded! Check the existence "
                               "or, or generate, model files!\n"
                               "Codebook path: %s\n"
                               "FLANN Index path: %s"
                               % (self.codebook_filepath,
                                  self.flann_index_filepath))

        self._log.debug("Computing descriptors for data UID[%s]...",
                        data.uuid())
        info, descriptors = self._generate_descriptor_matrices({data})

        # Load FLANN components
        pyflann.set_distance_type(self._flann_distance_metric)
        flann = pyflann.FLANN()
        flann.load_index(self.flann_index_filepath, self._codebook)

        if not self._use_sp:
            ###
            # Codebook Quantization
            #
            # - loaded the model at class initialization if we had one
            self._log.debug("Quantizing descriptors")

            try:
                # If the distance method is HIK, we need to treat it special
                # since that method produces a similarity score, not a distance
                # score.
                #
                if self._flann_distance_metric == 'hik':
                    # This searches for all NN instead of minimum between n and
                    # the number of descriptors and keeps the last one because
                    # hik is a similarity score and not a distance, which is
                    # also why the values in dists is flipped below.
                    #: :type: numpy.core.multiarray.ndarray, numpy.core.multiarray.ndarray
                    idxs = flann.nn_index(descriptors,
                                          self._codebook.shape[0])[0]
                    # Only keep the last index for each descriptor return
                    idxs = numpy.array([i_array[-1] for i_array in idxs])
                else:
                    # :type: numpy.core.multiarray.ndarray, numpy.core.multiarray.ndarray
                    idxs = flann.nn_index(descriptors, 1)[0]
            except AssertionError:

                self._log.error("Codebook shape  : %s", self._codebook.shape)
                self._log.error("Descriptor shape: %s", descriptors.shape)

                raise

            # Create histogram
            # - Using explicit bin slots to prevent numpy from automatically
            #   creating tightly constrained bins. This would otherwise cause
            #   histograms between two inputs to be non-comparable (unaligned
            #   bins).
            # - See numpy note about ``bins`` to understand why the +1 is
            #   necessary
            # - Learned from spatial implementation that we could feed multiple
            #   neighbors per descriptor into here, leading to a more populated
            #   histogram.
            #   - Could also possibly weight things based on dist from
            #     descriptor?
            #: :type: numpy.core.multiarray.ndarray
            h = numpy.histogram(idxs,  # indices are all integers
                                bins=numpy.arange(self._codebook.shape[0]+1))[0]
            # self._log.debug("Quantization histogram: %s", h)
            # Normalize histogram into relative frequencies
            # - Not using /= on purpose. h is originally int32 coming out of
            #   histogram. /= would keep int32 type when we want it to be
            #   transformed into a float type by the division.
            if h.sum():
                # noinspection PyAugmentAssignment
                h = h / float(h.sum())
            else:
                h = numpy.zeros(h.shape, h.dtype)
            # self._log.debug("Normalized histogram: %s", h)

        else:
            ###
            # Spatial Pyramid Quantization
            #
            self._log.debug("Quantizing descriptors using spatial pyramid")
            ##
            # Quantization factor - number of nearest codes to be saved
            q_factor = 10
            ##
            # Concatenating spatial information to descriptor vectors to format:
            #   [ x y <descriptor> ]
            self._log.debug("Creating combined descriptor matrix")
            m = numpy.concatenate((info[:, :2],
                                   descriptors), axis=1)
            ##
            # Creating quantized vectors, consisting vector:
            #   [ x y c_1 ... c_qf dist_1 ... dist_qf ]
            # which has a total size of 2+(qf*2)
            #
            # Sangmin's code included the distances in the quantized vector, but
            # then also passed this vector into numpy's histogram function with
            # integral bins, causing the [0,1] to be heavily populated, which
            # doesn't make sense to do.
            #   idxs, dists = flann.nn_index(m[:, 2:], q_factor)
            #   q = numpy.concatenate([m[:, :2], idxs, dists], axis=1)
            self._log.debug("Computing nearest neighbors")
            if self._flann_distance_metric == 'hik':
                # Query full ordering of code indices
                idxs = flann.nn_index(m[:, 2:], self._codebook.shape[0])[0]
                # Extract the right-side block for use in building histogram
                # Order doesn't actually matter in the current implementation
                #   because index relative position is not being weighted.
                idxs = idxs[:, -q_factor:]
            else:
                idxs = flann.nn_index(m[:, 2:], q_factor)[0]
            self._log.debug("Creating quantization matrix")
            # This matrix consists of descriptor (x,y) position + near code
            #   indices.
            q = numpy.concatenate([m[:, :2], idxs], axis=1)
            ##
            # Build spatial pyramid from quantized matrix
            self._log.debug("Building spatial pyramid histograms")
            hist_sp = self._build_sp_hist(q, self._codebook.shape[0])
            ##
            # Combine each quadrants into single vector
            self._log.debug("Combining global+thirds into final histogram.")
            f = sys.float_info.min  # so as we don't div by 0 accidentally

            def rf_norm(hist):
                return hist / (float(hist.sum()) + f)
            h = numpy.concatenate([rf_norm(hist_sp[0]),
                                   rf_norm(hist_sp[5]),
                                   rf_norm(hist_sp[6]),
                                   rf_norm(hist_sp[7])],
                                  axis=1)
            # noinspection PyAugmentAssignment
            h /= h.sum()

        self._log.debug("Saving checkpoint feature file")
        if not osp.isdir(osp.dirname(checkpoint_filepath)):
            file_utils.safe_create_dir(osp.dirname(checkpoint_filepath))
        numpy.save(checkpoint_filepath, h)

        return h

    @staticmethod
    def _build_sp_hist(feas, bins):
        """
        Build spatial pyramid from quantized data. We expect feature matrix
        to be in the following format:

            [[ x y c_1 ... c_n ]
             [ ... ]
             ... ]

        NOTES:
            - See encode_FLANN.py for original implementation this was adapted
                from.

        :param feas: Feature matrix with the above format.
        :type feas: numpy.core.multiarray.ndarray

        :param bins: number of bins for the spatial histograms. This should
            probably be the size of the codebook used when generating quantized
            descriptors.
        :type bins: int

        :return: Matrix of 8 rows representing the histograms for the different
            spatial regions
        :rtype: numpy.core.multiarray.ndarray

        """
        bins = numpy.arange(0, bins+1)
        cordx = feas[:, 0]
        cordy = feas[:, 1]
        feas = feas[:, 2:]

        # hard quantization
        # global histogram
        #: :type: numpy.core.multiarray.ndarray
        hist_sp_g = numpy.histogram(feas, bins=bins)[0]
        hist_sp_g = hist_sp_g[numpy.newaxis]
        # 4 quadrants
        # noinspection PyTypeChecker
        midx = numpy.ceil(cordx.max()/2)
        # noinspection PyTypeChecker
        midy = numpy.ceil(cordy.max()/2)
        lx = cordx < midx
        rx = cordx >= midx
        uy = cordy < midy
        dy = cordy >= midy
        # logging.error("LXUI: %s,%s", lx.__repr__(), uy.__repr__())
        # logging.error("Length LXUI: %s,%s", lx.shape, uy.shape)
        # logging.error("feas dimensions: %s", feas.shape)

        #: :type: numpy.core.multiarray.ndarray
        hist_sp_q1 = numpy.histogram(feas[lx & uy], bins=bins)[0]
        #: :type: numpy.core.multiarray.ndarray
        hist_sp_q2 = numpy.histogram(feas[rx & uy], bins=bins)[0]
        #: :type: numpy.core.multiarray.ndarray
        hist_sp_q3 = numpy.histogram(feas[lx & dy], bins=bins)[0]
        #: :type: numpy.core.multiarray.ndarray
        hist_sp_q4 = numpy.histogram(feas[rx & dy], bins=bins)[0]
        hist_sp_q1 = hist_sp_q1[numpy.newaxis]
        hist_sp_q2 = hist_sp_q2[numpy.newaxis]
        hist_sp_q3 = hist_sp_q3[numpy.newaxis]
        hist_sp_q4 = hist_sp_q4[numpy.newaxis]

        # 3 layers
        # noinspection PyTypeChecker
        ythird = numpy.ceil(cordy.max()/3)
        l1 = cordy <= ythird
        l2 = (cordy > ythird) & (cordy <= 2*ythird)
        l3 = cordy > 2*ythird
        #: :type: numpy.core.multiarray.ndarray
        hist_sp_l1 = numpy.histogram(feas[l1], bins=bins)[0]
        #: :type: numpy.core.multiarray.ndarray
        hist_sp_l2 = numpy.histogram(feas[l2], bins=bins)[0]
        #: :type: numpy.core.multiarray.ndarray
        hist_sp_l3 = numpy.histogram(feas[l3], bins=bins)[0]
        hist_sp_l1 = hist_sp_l1[numpy.newaxis]
        hist_sp_l2 = hist_sp_l2[numpy.newaxis]
        hist_sp_l3 = hist_sp_l3[numpy.newaxis]
        # concatenate
        hist_sp = numpy.vstack((hist_sp_g, hist_sp_q1, hist_sp_q2,
                                hist_sp_q3, hist_sp_q4, hist_sp_l1,
                                hist_sp_l2, hist_sp_l3))
        return hist_sp

    def _get_data_temp_path(self, de):
        """
        Standard method of generating/getting a data element's temporary file
        path.

        :param de: DataElement instance to generate/get temporary file path.
        :type de: smqtk.representation.DataElement

        :return: Path to the element's temporary file.
        :rtype: str

        """
        temp_dir = None
        # Shortcut when data element is from file, since we are not going to
        # write to / modify the file.
        if not isinstance(de, DataFileElement):
            temp_dir = self.temp_dir
        return de.write_temp(temp_dir)


# noinspection PyAbstractClass,PyPep8Naming
class ColorDescriptor_Image (ColorDescriptor_Base):

    def valid_content_types(self):
        """
        :return: A set valid MIME type content types that this descriptor can
            handle.
        :rtype: set[str]
        """
        return {'image/bmp', 'image/jpeg', 'image/png', 'image/tiff'}

    def _generate_descriptor_matrices(self, data_set, **kwargs):
        """
        Generate info and descriptor matrices based on ingest type.

        :param data_set: Iterable of data elements to generate combined info
            and descriptor matrices for.
        :type item_iter: collections.Set[smqtk.representation.DataElement]

        :param limit: Limit the number of descriptor entries to this amount.
        :type limit: int

        :return: Combined info and descriptor matrices for all base images
        :rtype: (numpy.core.multiarray.ndarray, numpy.core.multiarray.ndarray)

        """
        if not data_set:
            raise ValueError("No data given to process.")

        inf = float('inf')
        descriptor_limit = kwargs.get('limit', inf)
        per_item_limit = numpy.floor(float(descriptor_limit) / len(data_set))

        if len(data_set) == 1:
            # because an iterable doesn't necessarily have a next() method
            di = iter(data_set).next()
            # Check for checkpoint files
            info_fp, desc_fp = \
                self._get_standard_info_descriptors_filepath(di)
            # Save out data bytes to temporary file
            temp_img_filepath = self._get_data_temp_path(di)
            try:
                # Generate descriptors
                utils.generate_descriptors(
                    self.EXE, temp_img_filepath,
                    self.descriptor_type(), info_fp, desc_fp, per_item_limit
                )
            finally:
                # clean temp file
                di.clean_temp()
            return numpy.load(info_fp), numpy.load(desc_fp)
        else:
            # compute and V-stack matrices for all given images
            pool = multiprocessing.Pool(processes=self.parallel)

            # Mapping of UID to tuple containing:
            #   (info_fp, desc_fp, async processing result, tmp_clean_method)
            r_map = {}
            with SimpleTimer("Computing descriptors async...", self._log.debug):
                for di in data_set:
                    # Creating temporary image file from data bytes
                    tmp_img_fp = self._get_data_temp_path(di)

                    info_fp, desc_fp = \
                        self._get_standard_info_descriptors_filepath(di)
                    args = (self.EXE, tmp_img_fp,
                            self.descriptor_type(), info_fp, desc_fp)
                    r = pool.apply_async(utils.generate_descriptors, args)
                    r_map[di.uuid()] = (info_fp, desc_fp, r, di.clean_temp)
            pool.close()

            # Pass through results from descriptor generation, aggregating
            # matrix shapes.
            # - Transforms r_map into:
            #       UID -> (info_fp, desc_fp, starting_row, SubSampleIndices)
            self._log.debug("Constructing information for super matrices...")
            s_keys = sorted(r_map.keys())
            running_height = 0  # info and desc heights congruent

            i_width = None
            d_width = None

            for uid in s_keys:
                ifp, dfp, r, tmp_clean_method = r_map[uid]

                # descriptor generation may have failed for this ingest UID
                try:
                    i_shape, d_shape = r.get()
                except RuntimeError, ex:
                    self._log.warning("Descriptor generation failed for "
                                      "UID[%s], skipping its inclusion in "
                                      "model: %s", uid, str(ex))
                    r_map[uid] = None
                    continue
                finally:
                    # Done with image file, so remove from filesystem
                    tmp_clean_method()

                if d_width is None and d_shape[0] != 0:
                    i_width = i_shape[1]
                    d_width = d_shape[1]

                # skip this result if it generated no descriptors
                if d_shape[1] == 0:
                    continue

                ssi = None
                if i_shape[0] > per_item_limit:
                    # pick random indices to subsample down to size limit
                    ssi = sorted(
                        numpy.random.permutation(i_shape[0])[:per_item_limit]
                    )

                # Only keep this if any descriptors were generated
                r_map[uid] = (ifp, dfp, running_height, ssi)
                running_height += min(i_shape[0], per_item_limit)
            pool.join()

            # Asynchronously load files, inserting data into master matrices
            self._log.debug("Building super matrices...")
            master_info = numpy.zeros((running_height, i_width), dtype=float)
            master_desc = numpy.zeros((running_height, d_width), dtype=float)
            tp = multiprocessing.pool.ThreadPool(processes=self.parallel)
            for uid in s_keys:
                if r_map[uid]:
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


# noinspection PyAbstractClass,PyPep8Naming
class ColorDescriptor_Video (ColorDescriptor_Base):

    # # Custom higher limit for video since, ya know, they have multiple frames.
    CODEBOOK_DESCRIPTOR_LIMIT = 1500000

    FRAME_EXTRACTION_PARAMS = {
        "second_offset": 0.0,       # Start at beginning
        "second_interval": 0.5,     # Sample every 0.5 seconds
        "max_duration": 1.0,        # Cover full duration
        "output_image_ext": 'png',  # Output PNG files
        "ffmpeg_exe": "ffmpeg",
    }

    def valid_content_types(self):
        """
        :return: A set valid MIME type content types that this descriptor can
            handle.
        :rtype: set[str]
        """
        # At the moment, assuming ffmpeg can decode all video types, which it
        # probably cannot, but we'll filter this down when it becomes relevant.
        # noinspection PyUnresolvedReferences
        # TODO: GIF support?
        return set([x for x in mimetypes.types_map.values()
                    if x.startswith('video')])

    def _generate_descriptor_matrices(self, data_set, **kwargs):
        """
        Generate info and descriptor matrices based on ingest type.

        :param data_set: Iterable of data elements to generate combined info
            and descriptor matrices for.
        :type item_iter: collections.Set[smqtk.representation.DataElement]

        :param limit: Limit the number of descriptor entries to this amount.
        :type limit: int

        :return: Combined info and descriptor matrices for all base images
        :rtype: (numpy.core.multiarray.ndarray, numpy.core.multiarray.ndarray)

        """
        descriptor_limit = kwargs.get('limit', float('inf'))
        # With videos, an "item" is one video, so, collect for a while video
        # as normal, then subsample from the full video collection.
        per_item_limit = numpy.floor(float(descriptor_limit) / len(data_set))

        # If an odd number of jobs, favor descriptor extraction
        if self.parallel:
            descr_parallel = int(max(1, math.ceil(self.parallel/2.0)))
            extract_parallel = int(max(1, math.floor(self.parallel/2.0)))
        else:
            cpuc = multiprocessing.cpu_count()
            descr_parallel = int(max(1, math.ceil(cpuc/2.0)))
            extract_parallel = int(max(1, math.floor(cpuc/2.0)))

        # For each video, extract frames and submit colorDescriptor processing
        # jobs for each frame, combining all results into a single matrix for
        # return.
        pool = multiprocessing.Pool(processes=descr_parallel)

        # Mapping of [UID] to [frame] to tuple containing:
        #   (info_fp, desc_fp, async processing result)
        r_map = {}
        with SimpleTimer("Extracting frames and submitting descriptor jobs...",
                         self._log.debug):
            for di in data_set:
                r_map[di.uuid()] = {}
                tmp_vid_fp = self._get_data_temp_path(di)
                p = dict(self.FRAME_EXTRACTION_PARAMS)
                vmd = get_metadata_info(tmp_vid_fp)
                p['second_offset'] = vmd.duration * p['second_offset']
                p['max_duration'] = vmd.duration * p['max_duration']
                fm = video_utils.ffmpeg_extract_frame_map(
                    self._work_dir,
                    tmp_vid_fp,
                    parallel=extract_parallel,
                    **p
                )

                # Compute descriptors for extracted frames.
                for frame, imgPath in fm.iteritems():
                    info_fp, desc_fp = \
                        self._get_standard_info_descriptors_filepath(di, frame)
                    r = pool.apply_async(
                        utils.generate_descriptors,
                        args=(self.EXE, imgPath,
                              self.descriptor_type(), info_fp, desc_fp)
                    )
                    r_map[di.uuid()][frame] = (info_fp, desc_fp, r)

                # Clean temporary video file file while computing descriptors
                # This does not remove the extracted frames that the underlying
                #   detector/descriptor is working on.
                di.clean_temp()
        pool.close()

        # Each result is a tuple of two ndarrays: info and descriptor matrices
        with SimpleTimer("Collecting shape information for super matrices...",
                         self._log.debug):
            running_height = 0

            i_width = None
            d_width = None

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

                    # Descriptor generation may have failed for this UID
                    try:
                        i_shape, d_shape = r.get()
                    except RuntimeError, ex:
                        self._log.warning('Descriptor generation failed for '
                                          'frame %d in video UID[%s]: %s',
                                          frame, uid, str(ex))
                        r_map[uid] = None
                        continue

                    if d_width is None and d_shape[0] != 0:
                        i_width = i_shape[1]
                        d_width = d_shape[1]

                    # Skip if there were no descriptors generated for this
                    # frame
                    if d_shape[1] == 0:
                        continue

                    video_info_mat_fps.append(ifp)
                    video_desc_mat_fps.append(dfp)
                    video_num_desc += d_shape[0]

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
        del pool

        with SimpleTimer("Building master descriptor matrices...",
                         self._log.debug):
            master_info = numpy.zeros((running_height, i_width), dtype=float)
            master_desc = numpy.zeros((running_height, d_width), dtype=float)
            tp = multiprocessing.pool.ThreadPool(processes=self.parallel)
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

    # noinspection PyPep8Naming
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

    # noinspection PyPep8Naming
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
