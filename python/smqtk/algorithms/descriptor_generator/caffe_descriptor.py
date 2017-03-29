from collections import deque
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
from six.moves import range

from smqtk.algorithms.descriptor_generator import \
    DescriptorGenerator, \
    DFLT_DESCRIPTOR_FACTORY
from smqtk.representation.data_element import from_uri
from smqtk.utils.bin_utils import report_progress

try:
    import caffe
except ImportError as ex:
    logging.getLogger(__name__).warning("Failed to import caffe module: %s",
                                        str(ex))
    caffe = None


__author__ = 'paul.tunison@kitware.com, jacob.becker@kitware.com'

__all__ = [
    "CaffeDescriptorGenerator",
]


class CaffeDescriptorGenerator (DescriptorGenerator):
    """
    Compute images against a Caffe model, extracting a layer as the content
    descriptor.
    """

    @classmethod
    def is_usable(cls):
        valid = caffe is not None
        if not valid:
            cls.get_logger().debug("Caffe python module cannot be imported")
        return valid

    def __init__(self, network_prototxt_uri, network_model_uri, image_mean_uri,
                 return_layer='fc7',
                 batch_size=1, use_gpu=False, gpu_device_id=0,
                 network_is_bgr=True, data_layer='data',
                 load_truncated_images=False, pixel_rescale=None,
                 input_scale=None):
        """
        Create a Caffe CNN descriptor generator

        :param network_prototxt_uri: URI to the text file defining the
            network layout.
        :type network_prototxt_uri: str

        :param network_model_uri: URI to the trained ``.caffemodel``
            file to use.
        :type network_model_uri: str

        :param image_mean_uri: URI to the image mean ``.binaryproto`` or
            ``.npy`` file.
        :type image_mean_uri: str | file | StringIO.StringIO

        :param return_layer: The label of the layer we take data from to compose
            output descriptor vector.
        :type return_layer: str

        :param batch_size: The maximum number of images to process in one feed
            forward of the network. This is especially important for GPUs since
            they can only process a batch that will fit in the GPU memory space.
        :type batch_size: int

        :param use_gpu: If Caffe should try to use the GPU
        :type use_gpu: bool

        :param gpu_device_id: Integer ID of the GPU device to use. Only used if
            ``use_gpu`` is True.
        :type gpu_device_id: int

        :param network_is_bgr: If the network is expecting BGR format pixels.
            For example, the BVLC default caffenet does (thus the default is
            True).
        :type network_is_bgr: bool

        :param data_layer: String label of the network's data layer.
            We assume its 'data' by default.
        :type data_layer: str

        :param load_truncated_images: If we should be lenient and force loading
            of truncated image bytes. This is False by default.
        :type load_truncated_images: bool

        :param pixel_rescale: Re-scale image pixel values before being
            transformed by caffe (before mean subtraction, etc)
            into the given tuple ``(min, max)`` range. By default, images are
            loaded in the ``[0, 255]`` range. Refer to the image mean being used
            for desired input pixel scale.
        :type pixel_rescale: None | (float, float)

        :param input_scale: Optional floating-point scalar value to scale values
            of caffe network input data AFTER mean subtraction. This value is
            directly multiplied against the pixel values.
        :type input_scale: None | float

        """
        super(CaffeDescriptorGenerator, self).__init__()

        self.network_prototxt_uri = str(network_prototxt_uri)
        self.network_model_uri = str(network_model_uri)
        self.image_mean_uri = image_mean_uri

        self.return_layer = str(return_layer)
        self.batch_size = int(batch_size)

        self.use_gpu = bool(use_gpu)
        self.gpu_device_id = int(gpu_device_id)

        self.network_is_bgr = bool(network_is_bgr)
        self.data_layer = str(data_layer)

        self.load_truncated_images = bool(load_truncated_images)
        self.pixel_rescale = pixel_rescale
        self.input_scale = input_scale

        assert self.batch_size > 0, \
            "Batch size must be greater than 0 (got %d)" \
            % self.batch_size
        assert self.gpu_device_id >= 0, \
            "GPU Device ID must be greater than 0 (got %d)" \
            % self. gpu_device_id

        # Network setup variables
        self.network = None
        self.net_data_shape = ()
        self.transformer = None

        self._setup_network()

    def __getstate__(self):
        return self.get_config()

    def __setstate__(self, state):
        # This works because configuration parameters exactly match up with
        # instance attributes.
        self.__dict__.update(state)
        self._setup_network()

    def _setup_network(self):
        """
        Initialize Caffe and the network
        """
        if self.use_gpu:
            self._log.debug("Using GPU")
            caffe.set_device(self.gpu_device_id)
            caffe.set_mode_gpu()
        else:
            self._log.debug("using CPU")
            caffe.set_mode_cpu()

        # Questions:
        #   - ``caffe.TEST`` indicates phase of either TRAIN or TEST
        self._log.debug("Initializing network")
        network_prototxt_element = from_uri(self.network_prototxt_uri)
        network_model_element = from_uri(self.network_model_uri)
        self._log.debug("Loading Caffe network from network/model configs")
        self.network = caffe.Net(network_prototxt_element.write_temp(),
                                 caffe.TEST,
                                 weights=network_model_element.write_temp())
        network_prototxt_element.clean_temp()
        network_model_element.clean_temp()
        # Assuming the network has a 'data' layer and notion of data shape
        self.net_data_shape = self.network.blobs[self.data_layer].data.shape
        self._log.debug("Network data shape: %s", self.net_data_shape)

        # Crating input data transformer
        self._log.debug("Initializing data transformer")
        self.transformer = caffe.io.Transformer(
            {self.data_layer: self.network.blobs[self.data_layer].data.shape}
        )
        self._log.debug("Initializing data transformer -> %s",
                        self.transformer.inputs)

        self._log.debug("Loading image mean")
        image_mean_elem = from_uri(self.image_mean_uri)
        image_mean_bytes = image_mean_elem.get_bytes()
        try:
            a = numpy.load(io.BytesIO(image_mean_bytes))
            self._log.info("Loaded image mean from numpy bytes")
        except IOError:
            self._log.debug("Image mean file not a numpy array, assuming "
                            "URI to protobuf binary.")
            # noinspection PyUnresolvedReferences
            blob = caffe.proto.caffe_pb2.BlobProto()
            blob.ParseFromString(image_mean_bytes)
            a = numpy.array(caffe.io.blobproto_to_array(blob))
            assert a.shape[0] == 1, \
                "Input image mean blob protobuf consisted of more than one " \
                "image. Not sure how to handle this yet."
            a = a.reshape(a.shape[1:])
            self._log.info("Loaded image mean from protobuf bytes")
        assert a.shape[0] in [1, 3], \
            "Currently asserting that we either get 1 or 3 channel images. " \
            "Got a %d channel image." % a[0]
        # TODO: Instead of always using pixel mean, try to use image-mean if
        #       given. Might have to rescale if image/data layer shape is
        #       different.
        a_mean = a.mean(1).mean(1)
        self._log.debug("Initializing data transformer -- mean")
        self.transformer.set_mean(self.data_layer, a_mean)

        self._log.debug("Initializing data transformer -- transpose")
        self.transformer.set_transpose(self.data_layer, (2, 0, 1))
        if self.network_is_bgr:
            self._log.debug("Initializing data transformer -- channel swap")
            self.transformer.set_channel_swap(self.data_layer, (2, 1, 0))
        if self.input_scale:
            self._log.debug("Initializing data transformer -- input scale")
            self.transformer.set_input_scale(self.data_layer, self.input_scale)

    def get_config(self):
        """
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
            "network_prototxt_uri": self.network_prototxt_uri,
            "network_model_uri": self.network_model_uri,
            "image_mean_uri": self.image_mean_uri,
            "return_layer": self.return_layer,
            "batch_size": self.batch_size,
            "use_gpu": self.use_gpu,
            "gpu_device_id": self.gpu_device_id,
            "network_is_bgr": self.network_is_bgr,
            "data_layer": self.data_layer,
            "load_truncated_images": self.load_truncated_images,
            "pixel_rescale": self.pixel_rescale,
            "input_scale": self.input_scale,
        }

    def valid_content_types(self):
        """
        :return: A set valid MIME type content types that this descriptor can
            handle.
        :rtype: set[str]
        """
        return {
            'image/tiff',
            'image/png',
            'image/jpeg',
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
                                        descr_elements, procs, kwds.get('use_mp', True))

            if tail_size:
                batch_uuids = uuid4proc[-tail_size:]
                self._log.debug("Starting tail batch (size=%d)",
                                len(batch_uuids))
                self._process_batch(batch_uuids, data_elements, descr_elements,
                                    procs, kwds.get('use_mp', True))

        self._log.debug("forming output dict")
        return dict((data_elements[k].uuid(), descr_elements[k])
                    for k in data_elements)

    def _process_batch(self, uuids4proc, data_elements, descr_elements, procs, use_mp):
        """
        Run a number of data elements through the network, based on the number
        of UUIDs given, returning the vectors of

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

        :param use_mp: Whether or not to use a multiprocessing pool or a thread pool.
        :type use_mp: bool

        """
        self._log.debug("Updating network data layer shape (%d images)",
                        len(uuids4proc))
        self.network.blobs[self.data_layer].reshape(len(uuids4proc),
                                                    *self.net_data_shape[1:4])

        self._log.debug("Loading image pixel arrays")
        uid_num = len(uuids4proc)

        if use_mp:
            p = multiprocessing.Pool(procs)
        else:
            p = multiprocessing.pool.ThreadPool(procs)

        img_arrays = p.map(
            _process_load_img_array,
            zip(
                (data_elements[uid] for uid in uuids4proc),
                itertools.repeat(self.transformer, uid_num),
                itertools.repeat(self.data_layer, uid_num),
                itertools.repeat(self.load_truncated_images, uid_num),
                itertools.repeat(self.pixel_rescale, uid_num),
            )
        )
        p.close()
        p.join()

        self._log.debug("Loading image bytes into network layer '%s'",
                        self.data_layer)
        self.network.blobs[self.data_layer].data[...] = img_arrays

        self._log.debug("Moving network forward")
        self.network.forward()
        descriptor_list = self.network.blobs[self.return_layer].data

        self._log.debug("extracting return layer '%s' into descriptors",
                        self.return_layer)
        for uid, v in zip(uuids4proc, descriptor_list):
            if v.ndim > 1:
                # In case caffe generates multidimensional array (rows, 1, 1)
                descr_elements[uid].set_vector(numpy.ravel(v))
            else:
                descr_elements[uid].set_vector(v)


def _process_load_img_array((data_element, transformer,
                             data_layer, load_truncated_images,
                             pixel_rescale)):
    """
    Helper function for multiprocessing image data loading

    :param data_element: DataElement providing the bytes
    :type data_element: smqtk.representation.DataElement

    :param transformer: Caffe Transformer instance for pre-processing
    :type transformer: caffe.io.Transformer

    :param load_truncated_images: If PIL should be allowed to load truncated
        image data. If false, and exception will be raised when encountering
        such imagery.

    :return: Pre-processed numpy array.

    """
    PIL.ImageFile.LOAD_TRUNCATED_IMAGES = load_truncated_images
    img = PIL.Image.open(io.BytesIO(data_element.get_bytes()))
    if img.mode != "RGB":
        img = img.convert("RGB")
    # Caffe natively uses float types (32-bit)
    try:
        # This can fail if the image is truncated and we're not allowing the
        # loading of those images
        img_a = numpy.asarray(img, numpy.float32)
    except:
        logging.getLogger(__name__).error(
            "Failed array-ifying data element. Image may be truncated: %s",
            data_element
        )
        raise
    assert img_a.ndim == 3, \
        "Loaded invalid RGB image with shape %s" \
        % img_a.shape
    if pixel_rescale:
        pmin, pmax = min(pixel_rescale), max(pixel_rescale)
        r = pmax - pmin
        img_a = (img_a / (255. / r)) + pmin
    img_at = transformer.preprocess(data_layer, img_a)
    return img_at
