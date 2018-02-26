"""KWCNN descriptor generator definition."""
from collections import deque
import os
import io
import itertools
import logging
import multiprocessing
import multiprocessing.pool

import numpy
import PIL.Image
import PIL.ImageFile
import six
# noinspection PyUnresolvedReferences
from six.moves import range, zip

from smqtk.algorithms.descriptor_generator import (DescriptorGenerator,
                                                   DFLT_DESCRIPTOR_FACTORY)

from smqtk.utils.bin_utils import report_progress
from smqtk.utils.parallel import parallel_map

try:
    from six.moves import cPickle as pickle
except ImportError:
    import pickle

try:
    import kwcnn
except ImportError as ex:
    logging.getLogger(__name__).warning("Failed to import KWCNN module: %s",
                                        str(ex))
    kwcnn = None


__author__ = 'jason.parham@kitware.com,paul.tunison@kitware.com'

__all__ = [
    "KWCNNDescriptorGenerator",
]


DEFAULT_MODEL_FILEPATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'kwcnnmodel.npy'
)


class KWCNNDescriptorGenerator (DescriptorGenerator):
    """KWCNN descriptor generator to compute autoencoder-generated features."""

    @classmethod
    def is_usable(cls):
        """Return a boolean for if the descriptor generator is usable."""
        valid = kwcnn is not None
        if not valid:
            cls.get_logger().debug("KWCNN python module cannot be imported")
        return valid

    def __init__(self, network_model_filepath=DEFAULT_MODEL_FILEPATH,
                 batch_size=32, use_gpu=False, gpu_device_id=0,
                 network_is_greyscale=False, load_truncated_images=False,
                 pixel_rescale=None, input_scale=None,
                 pre_initialize_network=True):
        """
        Create a KWCNN CNN descriptor generator.

        :param network_model_filepath: The path to the trained KWCNN ``.npy``
            model file to use.
        :type network_model_filepath: str

        :param batch_size: The maximum number of images to process in one feed
            forward of the network. This is especially important for GPUs since
            they can only process a batch that will fit in the GPU memory space.
        :type batch_size: int

        :param use_gpu: If KWCNN should try to use the GPU
        :type use_gpu: bool

        :param gpu_device_id: Integer ID of the GPU device to use. Only used if
            ``use_gpu`` is True.
        :type gpu_device_id: int

        :param network_is_greyscale: If the network is expecting a greyscale
            format of pixels instead of BGR.
        :type network_is_greyscale: bool

        :param load_truncated_images: If we should be lenient and force loading
            of truncated image bytes. This is False by default.
        :type load_truncated_images: bool

        :param pixel_rescale: Re-scale image pixel values before being
            transformed by kwcnn (before mean subtraction, etc)
            into the given tuple ``(min, max)`` range. By default, images are
            loaded in the ``[0, 255]`` range. Refer to the image mean being used
            for desired input pixel scale.
        :type pixel_rescale: None | (float, float)

        :param input_scale: Optional floating-point scalar value to scale values
            of kwcnn network input data AFTER mean subtraction. This value is
            directly multiplied against the pixel values.
        :type input_scale: None | float

        """
        super(KWCNNDescriptorGenerator, self).__init__()

        self.batch_size = int(batch_size)

        self.use_gpu = bool(use_gpu)
        self.gpu_device_id = int(gpu_device_id)
        self.gpu_device_tag = 'gpu%d' % (self.gpu_device_id, )

        self.network_is_greyscale = bool(network_is_greyscale)
        # assert self.network_is_greyscale is False, 'Only color model supported'  # NOQA
        self.load_truncated_images = bool(load_truncated_images)
        self.pixel_rescale = pixel_rescale
        self.input_scale = input_scale

        self.network_model_filepath = str(network_model_filepath)

        assert self.batch_size > 0, \
            "Batch size must be greater than 0 (got %d)" % self.batch_size
        assert self.gpu_device_id >= 0, \
            "GPU Device ID must be greater than 0 (got %d)" % self.gpu_device_id

        # Network setup variables
        self.data = None
        self.model = None
        self.network = None

        self._setup_network(pre_initialize_network=pre_initialize_network)

    def __getstate__(self):
        """Get the state of the descriptor generator."""
        return self.get_config()

    def __setstate__(self, state):
        """Set the state of the descriptor generator."""
        self.__dict__.update(state)
        self._setup_network()

    def _setup_network(self, pre_initialize_network=True):
        """Initialize KWCNN data, model, and network objects."""
        # Check KWCNN
        try:
            assert kwcnn is not None
        except AssertionError:
            self._log.error("KWCNN python module not imported")
            raise

        # Check Theano CPU/GPU state vs. configured
        try:
            if self.use_gpu:
                self._log.debug("Using GPU")
                assert kwcnn.tpl._lasagne.USING_GPU
                assert kwcnn.tpl._lasagne.USING_DEVICE == self.gpu_device_tag
            else:
                self._log.debug("Using CPU")
                assert not kwcnn.tpl._lasagne.USING_GPU
        except AssertionError:
            self._log.error("Theano misconfigured for specified device!")
            url = "http://deeplearning.net/software/theano/library/config.html#environment-variables"  # NOQA
            self._log.error("Review Theano documentation here: %s" % (url, ))

            self._log.error("Requested configuration:")
            # Check the configuration requested by the SMQTK configuration
            self._log.error("\t\t Using CPU       : %s", not self.use_gpu)
            self._log.error("\t\t Using GPU       : %s", self.use_gpu)
            self._log.error("\t\t Using GPU Device: %s", self.gpu_device_tag)

            self._log.error("Theano configuration:")
            # Check the configuration reported by imported Theano
            self._log.error("\t Imported theano module configuration")
            self._log.error("\t\t Using CPU       : %s", not kwcnn.tpl._lasagne.USING_GPU)
            self._log.error("\t\t Using GPU       : %s", kwcnn.tpl._lasagne.USING_GPU)
            self._log.error("\t\t Using GPU Device: %s", kwcnn.tpl._lasagne.USING_DEVICE)

            # Check the $HOME/.theanorc file for configuration
            self._log.error("\t $HOME/.theanorc configuration file")
            theanorc_filepath = os.path.join("~", ".theanorc")
            theanorc_filepath = os.path.expanduser(theanorc_filepath)
            if os.path.exists(theanorc_filepath):
                with open(theanorc_filepath, "r") as theanorc_file:
                    for line in theanorc_file.readlines():
                        self._log.error("\t\t %s" % (line.strip(), ))
            else:
                self._log.error("\t\t NO CONFIGURATION FILE")

            # Check the $THEANO_FLAGS environment variable for configuration
            self._log.error("\t $THEANO_FLAGS environment variable")
            theano_flags = os.environ.get("THEANO_FLAGS", "").strip()
            if len(theano_flags) > 0:
                theano_flag_list = theano_flags.split(",")
                for theano_flag in theano_flag_list:
                    self._log.error("\t\t %s" % (theano_flag.strip(), ))
            else:
                self._log.error("\t\t NO ENVIRONMENT VARIABLE")
            # Raise RuntimeError for the user to address the configuration issue
            raise RuntimeError("Theano misconfigured for specified device")

        # Define model definition
        class OLCD_AutoEncoder_Model(kwcnn.core.KWCNN_Auto_Model):  # NOQA
            """FCNN Model."""

            def __init__(model, *args, **kwargs):
                """FCNN init."""
                super(OLCD_AutoEncoder_Model, model).__init__(*args, **kwargs)
                model.greyscale = kwargs.get("greyscale", False)
                model.bottleneck = kwargs.get("bottleneck", 64)
                model.trimmed = kwargs.get("trimmed", False)

            def _input_shape(model):
                return (64, 64, 1) if model.greyscale else (64, 64, 3)

            def architecture(model, batch_size, in_width, in_height,
                             in_channels, out_classes):
                """FCNN architecture."""
                input_height, input_width, input_channels = model._input_shape()
                _nonlinearity = kwcnn.tpl._lasagne.nonlinearities.LeakyRectify(
                    leakiness=(1. / 10.)
                )

                layer_list = [
                    kwcnn.tpl._lasagne.layers.InputLayer(
                        shape=(None, input_channels, input_height, input_width)
                    )
                ]

                for index in range(2):
                    layer_list.append(
                        kwcnn.tpl._lasagne.layers.batch_norm(
                            kwcnn.tpl._lasagne.Conv2DLayer(
                                layer_list[-1],
                                num_filters=16,
                                filter_size=(3, 3),
                                nonlinearity=_nonlinearity,
                                W=kwcnn.tpl._lasagne.init.Orthogonal(),
                            )
                        )
                    )

                layer_list.append(
                    kwcnn.tpl._lasagne.MaxPool2DLayer(
                        layer_list[-1],
                        pool_size=(2, 2),
                        stride=(2, 2),
                    )
                )

                for index in range(3):
                    layer_list.append(
                        kwcnn.tpl._lasagne.layers.batch_norm(
                            kwcnn.tpl._lasagne.Conv2DLayer(
                                layer_list[-1],
                                num_filters=32,
                                filter_size=(3, 3),
                                nonlinearity=_nonlinearity,
                                W=kwcnn.tpl._lasagne.init.Orthogonal(),
                            )
                        )
                    )

                layer_list.append(
                    kwcnn.tpl._lasagne.MaxPool2DLayer(
                        layer_list[-1],
                        pool_size=(2, 2),
                        stride=(2, 2),
                    )
                )

                for index in range(2):
                    layer_list.append(
                        kwcnn.tpl._lasagne.layers.batch_norm(
                            kwcnn.tpl._lasagne.Conv2DLayer(
                                layer_list[-1],
                                num_filters=32,
                                filter_size=(3, 3),
                                nonlinearity=_nonlinearity,
                                W=kwcnn.tpl._lasagne.init.Orthogonal(),
                            )
                        )
                    )

                l_reshape0 = kwcnn.tpl._lasagne.layers.ReshapeLayer(
                    layer_list[-1],
                    shape=([0], -1),
                )

                if model.trimmed:
                    return l_reshape0

                l_bottleneck = kwcnn.tpl._lasagne.layers.DenseLayer(
                    l_reshape0,
                    num_units=model.bottleneck,
                    nonlinearity=kwcnn.tpl._lasagne.nonlinearities.tanh,
                    W=kwcnn.tpl._lasagne.init.Orthogonal(),
                    name="bottleneck",
                )

                return l_bottleneck

        # Create KWCNN Data, Model, and Network primitives
        self._log.debug("Initializing KWCNN Data")
        self.data = kwcnn.core.KWCNN_Data()

        # Create trimmed model, if it does not exist
        self._log.debug("Initializing KWCNN Model")

        USE_TRIMMED_NETWORK = True

        if USE_TRIMMED_NETWORK:
            trimmed_filepath = self.network_model_filepath
            trimmed_filepath = trimmed_filepath.replace('.npy', '.trimmed.npy')
            if not os.path.exists(trimmed_filepath):
                with open(self.network_model_filepath, 'rb') as model_file:
                    model_dict = pickle.load(model_file)
                key_list = ['best_weights', 'best_fit_weights']
                for key in key_list:
                    layer_list = model_dict[key]
                    model_dict[key] = layer_list[:-2]
                with open(trimmed_filepath, 'wb') as model_file:
                    pickle.dump(model_dict, model_file,
                                protocol=pickle.HIGHEST_PROTOCOL)
            self.network_model_filepath = trimmed_filepath

        # Load model
        self.model = OLCD_AutoEncoder_Model(self.network_model_filepath,
                                            greyscale=self.network_is_greyscale,
                                            trimmed=USE_TRIMMED_NETWORK)

        self._log.debug("Initializing KWCNN Network")
        self.network = kwcnn.core.KWCNN_Network(self.model, self.data)

        # Pre-initialize network during network setup
        if pre_initialize_network:
            # Get the input shape for the KWCNN model
            self.input_shape = self.model._input_shape()
            input_height, input_width, input_channels = self.input_shape
            # Create a temporary numpy array of empty data of correct shape
            temp_shape = (16, input_height, input_width, input_channels, )
            temp_arrays = numpy.zeros(temp_shape, dtype=numpy.float32)
            # Give the dummy data to the KWCNN data object
            self.data.set_data_list(temp_arrays, quiet=True)
            # Test with dummy data, which will compile and load the model
            self._log.debug("Building and compiling KWCNN model...")
            self.network.test(quiet=True)  # Throw away output
            self._log.debug("done")

    def get_config(self):
        """
        Get the configuration dictionary for the descriptor generator.

        Return a JSON-compliant dictionary that could be passed to this class's
        ``from_config`` method to produce an instance with identical
        configuration.

        In the common case, this involves naming the keys of the dictionary
        based on the initialization argument names as if it were to be passed
        to the constructor via dictionary expansion.

        :return: JSON type compliant configuration dictionary.
        :rtype: dict
        """
        return {
            "network_model_filepath": self.network_model_filepath,
            "batch_size": self.batch_size,
            "use_gpu": self.use_gpu,
            "gpu_device_id": self.gpu_device_id,
            "network_is_greyscale": self.network_is_greyscale,
            "load_truncated_images": self.load_truncated_images,
            "pixel_rescale": self.pixel_rescale,
            "input_scale": self.input_scale,
        }

    def valid_content_types(self):
        """
        Return the MIME types that are valid for this descriptor generator.

        :return: A set valid MIME type content types that this descriptor can
            handle.
        :rtype: set[str]
        """
        return {
            "image/tiff",
            "image/png",
            "image/jpeg",
        }

    def _compute_descriptor(self, data):
        raise NotImplementedError("Shouldn't get here as "
                                  "compute_descriptor[_async] is being "
                                  "overridden")

    def compute_descriptor(self, data, descr_factory=DFLT_DESCRIPTOR_FACTORY,
                           overwrite=False):
        """
        Given some data, return a descriptor element containing a descriptor
        vector.

        :raises RuntimeError: Descriptor extraction failure of some kind.
        :raises ValueError: Given data element content was not of a valid type
            with respect to this descriptor.

        :param data: Some kind of input data for the feature descriptor.
        :type data: smqtk.representation.DataElement

        :param descr_factory: Factory instance to produce the wrapping
            descriptor element instance. The default factory produces
            ``DescriptorMemoryElement`` instances by default.
        :type descr_factory: smqtk.representation.DescriptorElementFactory

        :param overwrite: Whether or not to force re-computation of a descriptor
            vector for the given data even when there exists a precomputed
            vector in the generated DescriptorElement as generated from the
            provided factory. This will overwrite the persistently stored vector
            if the provided factory produces a DescriptorElement implementation
            with such storage.
        :type overwrite: bool

        :return: Result descriptor element. UUID of this output descriptor is
            the same as the UUID of the input data element.
        :rtype: smqtk.representation.DescriptorElement

        """
        m = self.compute_descriptor_async([data], descr_factory, overwrite,
                                          procs=1)
        return m[data.uuid()]

    def compute_descriptor_async(self, data_iter,
                                 descr_factory=DFLT_DESCRIPTOR_FACTORY,
                                 overwrite=False, procs=None, **kwds):
        """
        Asynchronously compute feature data for multiple data items.

        :param data_iter: Iterable of data elements to compute features for.
            These must have UIDs assigned for feature association in return
            value.
        :type data_iter: collections.Iterable[smqtk.representation.DataElement]

        :param descr_factory: Factory instance to produce the wrapping
            descriptor element instance. The default factory produces
            ``DescriptorMemoryElement`` instances by default.
        :type descr_factory: smqtk.representation.DescriptorElementFactory

        :param overwrite: Whether or not to force re-computation of a descriptor
            vectors for the given data even when there exists precomputed
            vectors in the generated DescriptorElements as generated from the
            provided factory. This will overwrite the persistently stored
            vectors if the provided factory produces a DescriptorElement
            implementation such storage.
        :type overwrite: bool

        :param procs: Optional specification of how many processors to use
            when pooling sub-tasks. If None, we attempt to use all available
            cores.
        :type procs: int | None

        :raises ValueError: An input DataElement was of a content type that we
            cannot handle.

        :return: Mapping of input DataElement UUIDs to the computed descriptor
            element for that data. DescriptorElement UUID's are congruent with
            the UUID of the data element it is the descriptor of.
        :rtype: dict[collections.Hashable,
                     smqtk.representation.DescriptorElement]

        """
        # Create DescriptorElement instances for each data elem.
        #: :type: dict[collections.Hashable, smqtk.representation.DataElement]
        data_elements = {}
        #: :type: dict[collections.Hashable, smqtk.representation.DescriptorElement]
        descr_elements = {}
        self._log.debug("Checking content types; aggregating data/descriptor "
                        "elements.")
        prog_rep_state = [0] * 7
        for data in data_iter:
            ct = data.content_type()
            if ct not in self.valid_content_types():
                self._log.error("Cannot compute descriptor from content type "
                                "'%s' data: %s)" % (ct, data))
                raise ValueError("Cannot compute descriptor from content type "
                                 "'%s' data: %s)" % (ct, data))
            data_elements[data.uuid()] = data
            descr_elements[data.uuid()] = \
                descr_factory.new_descriptor(self.name, data.uuid())
            report_progress(self._log.debug, prog_rep_state, 1.0)
        self._log.debug("Given %d unique data elements", len(data_elements))

        # Reduce procs down to the number of elements to process if its smaller
        if len(data_elements) < (procs or multiprocessing.cpu_count()):
            procs = len(data_elements)
        if procs == 0:
            raise ValueError("No data elements provided")

        # For thread safely, only use .append() and .popleft() (queue)
        uuid4proc = deque()

        def check_get_uuid(descriptor_elem):
            if overwrite or not descriptor_elem.has_vector():
                # noinspection PyUnresolvedReferences
                uuid4proc.append(descriptor_elem.uuid())

        # Using thread-pool due to in-line function + updating local deque
        p = multiprocessing.pool.ThreadPool(procs)
        try:
            p.map(check_get_uuid, six.itervalues(descr_elements))
        finally:
            p.close()
            p.join()
        del p
        self._log.debug("%d descriptors already computed",
                        len(data_elements) - len(uuid4proc))

        if uuid4proc:
            self._log.debug("Converting deque to tuple for segmentation")
            uuid4proc = tuple(uuid4proc)

            # Split UUIDs into groups equal to our batch size, and an option
            # tail group that is less than our batch size.
            tail_size = len(uuid4proc) % self.batch_size
            batch_groups = (len(uuid4proc) - tail_size) // self.batch_size
            self._log.debug("Processing %d batches of size %d", batch_groups,
                            self.batch_size)
            if tail_size:
                self._log.debug("Processing tail group of size %d", tail_size)

            if batch_groups:
                for g in range(batch_groups):
                    self._log.debug("Starting batch: %d of %d",
                                    g + 1, batch_groups)
                    batch_uuids = \
                        uuid4proc[g * self.batch_size:(g + 1) * self.batch_size]
                    self._process_batch(batch_uuids, data_elements,
                                        descr_elements, procs)

            if tail_size:
                batch_uuids = uuid4proc[-tail_size:]
                self._log.debug("Starting tail batch (size=%d)",
                                len(batch_uuids))
                self._process_batch(batch_uuids, data_elements, descr_elements,
                                    procs)

        self._log.debug("forming output dict")
        return dict((data_elements[k].uuid(), descr_elements[k])
                    for k in data_elements)

    def _process_batch(self, uuids4proc, data_elements, descr_elements, procs):
        """
        Run a number of data elements through the network in batches.

        Compute results of elements based on the number of UUIDs given,
        returning the vectors of descriptors

        :param uuids4proc: UUIDs of the source data to run in the network as a
            batch.
        :type uuids4proc: collections.Sequence[collections.Hashable]

        :param data_elements: Mapping of UUID to data element for input data.
        :type data_elements: dict[collections.Hashable,
                                  smqtk.representation.DataElement]

        :param descr_elements: Mapping of UUID to descriptor element based on
            input data elements.
        :type descr_elements: dict[collections.Hashable,
                                   smqtk.representation.DescriptorElement]

        :param procs: The number of asynchronous processes to run for loading
            images. This may be None to just use all available cores.
        :type procs: None | int

        """
        self._log.debug("Loading image pixel arrays")
        uid_num = len(uuids4proc)
        p = multiprocessing.Pool(procs)
        img_arrays = p.map(
            _process_load_img_array,
            zip(
                (data_elements[uid] for uid in uuids4proc),
                itertools.repeat(self.network_is_greyscale, uid_num),
                itertools.repeat(self.input_shape, uid_num),
                itertools.repeat(self.load_truncated_images, uid_num),
                itertools.repeat(self.pixel_rescale, uid_num),
            )
        )
        p.close()
        p.join()

        self._log.debug("Loading image numpy array into KWCNN Data object")
        self.data.set_data_list(img_arrays, quiet=True)

        self._log.debug("Performing forward inference using KWCNN Network")
        test_results = self.network.test(quiet=True)
        descriptor_list = test_results['probability_list']

        self._log.debug("transform network output into descriptors")
        for uid, v in zip(uuids4proc, descriptor_list):
            if v.ndim > 1:
                # In case kwcnn generates multidimensional array (rows, 1, 1)
                descr_elements[uid].set_vector(numpy.ravel(v))
            else:
                descr_elements[uid].set_vector(v)


def _process_load_img_array(data_element, network_is_greyscale, input_shape,
                            load_truncated_images, pixel_rescale):
    """
    Helper function for multiprocessing image data loading.

    :param network_is_greyscale: If PIL should load the images in greyscale.

    :param load_truncated_images: If PIL should be allowed to load truncated
        image data. If false, and exception will be raised when encountering
        such imagery.

    :param pixel_rescale: A post-load tuple specifying a (min, max) scaling
        applied to the pixel values

    :return: Pre-processed numpy array.

    """
    PIL.ImageFile.LOAD_TRUNCATED_IMAGES = load_truncated_images
    img = PIL.Image.open(io.BytesIO(data_element.get_bytes()))
    mode = "L" if network_is_greyscale else "RGB"
    if img.mode != mode:
        img = img.convert(mode)
    # KWCNN natively uses uint8 or float32 types
    input_size = input_shape[:2]
    if img.size != input_size:
        img = img.resize(input_size, PIL.Image.LANCZOS)
    try:
        # This can fail if the image is truncated and we're not allowing the
        # loading of those images
        img_a = numpy.array(img, dtype=numpy.float32)
        img_a = img_a.reshape(input_shape)
    except:
        logging.getLogger(__name__).error(
            "Failed array-ifying data element. Image may be truncated: %s",
            data_element
        )
        raise
    # Convert to BGR
    if not network_is_greyscale:
        img_a = img_a[:, :, ::-1]
    message = "Loaded invalid greyscale image with shape %s" % (img_a.shape, )
    assert img_a.ndim == 3 and img_a.shape == input_shape, message
    if pixel_rescale:
        pmin, pmax = min(pixel_rescale), max(pixel_rescale)
        r = pmax - pmin
        img_a = (img_a / (255. / r)) + pmin
    return img_a
