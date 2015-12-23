from collections import deque
import io
import multiprocessing
import multiprocessing.pool

import numpy
import PIL.Image
import PIL.ImageFile

from smqtk.algorithms import DescriptorGenerator

try:
    import caffe
except ImportError:
    caffe = None


__author__ = ['paul.tunison@kitware.com, jacob.becker@kitware.com']

__all__ = [
    "CaffeDescriptorGenerator",
]


class CaffeDescriptorGenerator (DescriptorGenerator):

    @classmethod
    def is_usable(cls):
        valid = caffe is not None
        if not valid:
            cls.logger().debug("Caffe python module not importable")
        return valid

    def __init__(self, network_prototxt_filepath, network_model_filepath,
                 image_mean_filepath,
                 return_layer='fc7',
                 batch_size=1, use_gpu=False, gpu_device_id=0,
                 network_is_bgr=True, data_layer='data',
                 load_truncated_images=False):
        """
        Create a Caffe CNN descriptor generator

        :param network_prototxt_filepath: Path to the text file defining the
            network layout.
        :type network_prototxt_filepath: str

        :param network_model_filepath: The path to the trained ``.caffemodel``
            file to use.
        :type network_model_filepath: str

        :param image_mean_filepath: Path to the image mean ``.binaryproto``
            file.
        :type image_mean_filepath: str

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

        """
        super(CaffeDescriptorGenerator, self).__init__()

        self.network_prototxt_filepath = str(network_prototxt_filepath)
        self.network_model_filepath = str(network_model_filepath)
        self.image_mean_filepath = str(image_mean_filepath)

        self.return_layer = str(return_layer)
        self.batch_size = int(batch_size)

        self.use_gpu = bool(use_gpu)
        self.gpu_device_id = int(gpu_device_id)

        self.network_is_bgr = bool(network_is_bgr)
        self.data_layer = str(data_layer)

        self.load_truncated_images = bool(load_truncated_images)

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
        self.network = caffe.Net(self.network_prototxt_filepath,
                                 self.network_model_filepath,
                                 caffe.TEST)
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
        try:
            a = numpy.load(self.image_mean_filepath)
        except IOError:
            self._log.debug("Image mean file not a numpy array, assuming "
                            "protobuf binary.")
            blob = caffe.proto.caffe_pb2.BlobProto()
            with open(self.image_mean_filepath, 'rb') as f:
                blob.ParseFromString(f.read())
            a = numpy.array(caffe.io.blobproto_to_array(blob))
            assert a.shape[0] == 1, \
                "Input image mean blob protobuf consisted of more than one " \
                "image. Not sure how to handle this yet."
            a = a.reshape(a.shape[1:])
        assert a.shape[0] in [1, 3], \
            "Currently asserting that we either get 1 or 3 channel images. " \
            "Got a %d channel image." % a[0]
        a_mean = a.mean(1).mean(1)
        self._log.debug("Initializing data transformer -- mean")
        self.transformer.set_mean(self.data_layer, a_mean)

        self._log.debug("Initializing data transformer -- transpose")
        self.transformer.set_transpose(self.data_layer, (2, 0, 1))
        if self.network_is_bgr:
            self._log.debug("Initializing data transformer -- channel swap")
            self.transformer.set_channel_swap(self.data_layer, (2, 1, 0))
        self._log.debug("Initializing data transformer -- raw scale")
        self.transformer.set_raw_scale(self.data_layer, 255.0)

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
            "network_prototxt_filepath": self.network_prototxt_filepath,
            "network_model_filepath": self.network_model_filepath,
            "image_mean_filepath": self.image_mean_filepath,
            "return_layer": self.return_layer,
            "batch_size": self.batch_size,
            "use_gpu": self.use_gpu,
            "gpu_device_id": self.gpu_device_id,
            "network_is_bgr": self.network_is_bgr,
            "data_layer": self.data_layer,
            "load_truncated_images": self.load_truncated_images,
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

    def compute_descriptor(self, data, descr_factory, overwrite=False):
        """
        Given some kind of data, return a descriptor element containing a
        descriptor vector.

        This abstract super method should be invoked for common error checking.

        :raises RuntimeError: Descriptor extraction failure of some kind.
        :raises ValueError: Given data element content was not of a valid type
            with respect to this descriptor.

        :param data: Some kind of input data for the feature descriptor.
        :type data: smqtk.representation.DataElement

        :param descr_factory: Factory instance to produce the wrapping
            descriptor element instance.
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
        return m[data]

    def compute_descriptor_async(self, data_iter, descr_factory,
                                 overwrite=False, procs=None, **kwds):
        """
        Asynchronously compute feature data for multiple data items.

        :param data_iter: Iterable of data elements to compute features for.
            These must have UIDs assigned for feature association in return
            value.
        :type data_iter: collections.Iterable[smqtk.representation.DataElement]

        :param descr_factory: Factory instance to produce the wrapping
            descriptor element instances.
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
        :type procs: int

        :return: Mapping of input DataElement instances to the computed
            descriptor element.
            DescriptorElement UUID's are congruent with the UUID of the data
            element it is the descriptor of.
        :rtype: dict[smqtk.representation.DataElement,
                     smqtk.representation.DescriptorElement]

        """
        # Create DescriptorElement instances for each data elem.
        #: :type: dict[collections.Hashable, smqtk.representation.DataElement]
        data_elements = {}
        #: :type: dict[collections.Hashable, smqtk.representation.DescriptorElement]
        descr_elements = {}
        self._log.debug("Checking content types; aggregating data/descriptor "
                        "elements.")
        for d in data_iter:
            ct = d.content_type()
            if ct not in self.valid_content_types():
                raise ValueError("Cannot compute descriptor of content type "
                                 "'%s'" % ct)
            data_elements[d.uuid()] = d
            descr_elements[d.uuid()] = descr_factory.new_descriptor(self.name, d.uuid())
        self._log.debug("Given %d unique data elements", len(data_elements))

        # Reduce procs down to the number of elements to process if its smaller
        if len(data_elements) < (procs or multiprocessing.cpu_count()):
            procs = len(data_elements)

        # For thread safely, only use .append() and .popleft() (queue)
        uuid4proc = deque()

        def check_get_uuid(d):
            if overwrite or not d.has_vector():
                # noinspection PyUnresolvedReferences
                uuid4proc.append(d.uuid())

        p = multiprocessing.pool.ThreadPool(procs)
        try:
            p.map(check_get_uuid, descr_elements.itervalues())
        finally:
            p.close()
            p.join()
        self._log.debug("Converting deque to tuple for segmentation")
        uuid4proc = tuple(uuid4proc)

        if uuid4proc:
            # Split UUIDs into groups equal to our batch size, and an optioan
            # tail group that is less than our batch size.
            tail_size = len(uuid4proc) % self.batch_size
            batch_groups = (len(uuid4proc) - tail_size) // self.batch_size
            self._log.debug("Processing %d batches of size %d", batch_groups,
                            self.batch_size)
            if tail_size:
                self._log.debug("Processing tail group of size %d", tail_size)

            if batch_groups:
                for g in xrange(batch_groups):
                    self._log.debug("Starting batch: %d", g)
                    batch_uuids = \
                        uuid4proc[g*self.batch_size:(g+1)*self.batch_size]
                    self._process_batch(batch_uuids, data_elements,
                                        descr_elements, procs)

            if tail_size:
                batch_uuids = uuid4proc[-tail_size:]
                self._log.debug("Starting tail batch (size=%d)",
                                len(batch_uuids))
                self._process_batch(batch_uuids, data_elements, descr_elements,
                                    procs)

        self._log.debug("forming output dict")
        return dict((data_elements[k], descr_elements[k])
                    for k in data_elements)

    def _process_batch(self, uuids4proc, data_elements, descr_elements, procs):
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

        """
        self._log.debug("Updating network data layer shape (%d images)",
                        len(uuids4proc))
        self.network.blobs[self.data_layer].reshape(len(uuids4proc),
                                                    *self.net_data_shape[1:4])

        # Load data from images into data layer via transformer
        # for i, uid in enumerate(uuids4proc):
        #     img = PIL.Image.open(io.BytesIO(data_elements[uid].get_bytes()))
        #     # Will throw IOError for truncated imagery when we're not
        #     # allowing it. Otherwise we would try to array-ify it and get an
        #     # empty array, breaking the ``Transformer.preprocess`` method.
        #     img.load()
        #     # Make into RGB form so we get an array of an expected shape and
        #     # format.
        #     if img.mode != "RGB":
        #         img = img.convert("RGB")
        #     img_a = numpy.asarray(img)
        #     # Set into network
        #     self.network.blobs[self.data_layer].data[i][...] = \
        #         self.transformer.preprocess(self.data_layer, img_a)

        self._log.debug("Loading image pixel arrays")
        uid_num = len(uuids4proc)
        p = multiprocessing.Pool(procs)
        img_arrays = p.map(
            _process_load_img_array,
            zip(
                [data_elements[uid] for uid in uuids4proc],
                [self.transformer]*uid_num,
                [self.data_layer]*uid_num,
                [self.load_truncated_images]*uid_num,
            )
        )
        p.close()
        p.join()

        self._log.debug("Loading image bytes into network layer '%s'",
                        self.data_layer)
        def set_net_data((i, a)):
            self.network.blobs[self.data_layer].data[i][...] = a
        p = multiprocessing.pool.ThreadPool(procs)
        p.map(set_net_data, enumerate(img_arrays))
        p.close()
        p.join()

        self._log.debug("Moving network forward")
        self.network.forward()

        self._log.debug("extracting return layer '%s' into descriptors",
                        self.return_layer)
        for uid, v in zip(uuids4proc, self.network.blobs[self.return_layer].data):
            descr_elements[uid].set_vector(v)


def _process_load_img_array((data_element, transformer,
                             data_layer, load_truncated_images)):
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
    img_a = numpy.asarray(img)
    assert img_a.ndim == 3, \
        "Loaded invalid RGB image with shape %s" \
        % img_a.shape
    img_at = transformer.preprocess(data_layer, img_a)
    return img_at
