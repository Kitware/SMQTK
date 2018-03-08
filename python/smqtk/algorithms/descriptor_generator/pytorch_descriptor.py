from collections import deque
import io
import logging
import multiprocessing
import multiprocessing.pool

import numpy
from PIL import Image
import six
# noinspection PyUnresolvedReferences
from six.moves import range, zip

from smqtk.algorithms.descriptor_generator import \
    DescriptorGenerator, \
    DFLT_DESCRIPTOR_FACTORY

from smqtk.utils.bin_utils import report_progress

try:
    import torch
except ImportError as ex:
    logging.getLogger(__name__).warning("Failed to import pytorch module: %s",
                                        str(ex))
    torch = None
else:
    import torch.utils.data as data
    from torch.autograd import Variable
    
try:
    import torchvision
except ImportError as ex:
    logging.getLogger(__name__).warning("Failed to import pytorch module: %s",
                                        str(ex))
    torchvision = None
else:
    from torchvision import transforms


__author__ = 'bo.dong@kitware.com'

__all__ = [
    "PytorchDescriptorGenerator",
]

class PytorchDataLoader(data.Dataset):
    def __init__(self, file_list, resize_val, uuid4proc, transform=None):

        self._file_list = file_list
        self._resize_val = resize_val
        self._uuid4proc = uuid4proc
        self._transform = transform

    def __getitem__(self, index):
        img = Image.open(io.BytesIO(self._file_list[self._uuid4proc[index]].get_bytes()))
        img = img.resize((self._resize_val, self._resize_val), Image.BILINEAR).convert('RGB')

        if self._transform is not None:
            img = self._transform(img)

        return img, self._uuid4proc[index]

    def __len__(self):
        return len(self._uuid4proc)


class PytorchDescriptorGenerator (DescriptorGenerator):
    """
    Compute images against a pytorch model. The pytorch model
    outputs the desired features for the input images.
    """

    @classmethod
    def is_usable(cls):
        valid = torch is not None and torchvision is not None
        if not valid:
            cls.get_logger().debug("Pytorch or torchvision (or both) python \
            module cannot be imported")
        return valid

    def __init__(self, model_cls, model_uri, transform, resize_val,
                 batch_size=1, use_gpu=False, gpu_device_id=0):
        """
        Create a pytorch CNN descriptor generator

        :param model_cls: model definition class.
        :type model_cls: str

        :param model_uri: URI to the trained ``.pt`` file to use.
        :type model_uri: None | str

        :param transform: torchvision transform module for preprocess the image.
        :type transform: torchvision.transform

        :param resize_val: Resize the input image to the resize_val x resize_val.
        :type resize-val: int

        :param batch_size: The maximum number of images to process in one feed
            forward of the network. This is especially important for GPUs since
            they can only process a batch that will fit in the GPU memory space.
        :type batch_size: int

        :param use_gpu: If pytorch should try to use the GPU
        :type use_gpu: bool

        :param gpu_device_id: Integer ID of the GPU device to use. Only used if
            ``use_gpu`` is True.
        :type gpu_device_id: None | int

        """
        super(PytorchDescriptorGenerator, self).__init__()

        self._model_cls = model_cls
        self._model_uri = model_uri
        self._transform = transform
        self._resize_val = resize_val
        self._batch_size = int(batch_size)
        self._use_gpu = bool(use_gpu)
        self._gpu_device_id = gpu_device_id

        assert self._batch_size > 0, \
            "Batch size must be greater than 0 (got {})".format(\
                self._batch_size)

        if self._use_gpu:
            GPU_list = [x for x in range(torch.cuda.device_count())]
            if self._gpu_device_id is None:
                self._gpu_device_id = GPU_list
            else:
                self._gpu_device_id = int(self._gpu_device_id)
                assert self._gpu_device_id in GPU_list, \
                    "GPU Device ID must be in GPU_list {} (got {})".format(GPU_list, self._gpu_device_id)
                self._gpu_device_id = [self._gpu_device_id]

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
        Initialize pytorch network
        """
        self._model_cls.eval()

        if self._use_gpu:
            self._log.debug("Using GPU")
            self._model_cls.cuda(self._gpu_device_id[0])
            self._model_cls = torch.nn.DataParallel(self._model_cls, device_ids=self._gpu_device_id)
        else:
            self._log.debug("using CPU")

        if self._model_uri is not None:
            self._log.debug("load the trained model: {}".format(self._model_uri))
            snapshot = torch.load(self._model_uri)
            self._model_cls.load_state_dict(snapshot['state_dict'])

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
            "model_cls": self._model_cls,
            "model_uri": self._model_uri,
            "transform": self._transform,
            "resize_val": self._resize_val,
            "batch_size": self._batch_size,
            "use_gpu": self._use_gpu,
            "gpu_device_id": self._gpu_device_id,
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

        if len(data_elements) == 0:
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
            kwargs = {'num_workers': procs if procs is not None else multiprocessing.cpu_count(), 'pin_memory': True}
            data_loader_cls = PytorchDataLoader(file_list=data_elements,resize_val=self._resize_val,\
                                                  uuid4proc=uuid4proc, transform=self._transform)
            data_loader = torch.utils.data.DataLoader(data_loader_cls, batch_size=self._batch_size,\
                                                      shuffle=False, **kwargs)

            self._log.debug("Extract pytorch features")
            for (data, uuids) in data_loader:
                if self._use_gpu:
                    data = data.cuda()

                input = Variable(data)
                pytorch_f = self._model_cls(input)

                for idx, uuid in enumerate(uuids):
                    descr_elements[uuid] = pytorch_f.data.cpu().numpy()[idx]

        self._log.debug("forming output dict")
        return dict((data_elements[k].uuid(), descr_elements[k])
                    for k in data_elements)
