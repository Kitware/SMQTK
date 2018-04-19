from collections import deque
import os
import logging
import multiprocessing
import multiprocessing.pool
import numpy as np
from tqdm import tqdm
from skimage.transform import resize
import _pickle as pickle


import six
from six.moves import range

from smqtk.algorithms.descriptor_generator import \
    DescriptorGenerator, \
    DFLT_DESCRIPTOR_FACTORY

from smqtk.utils.bin_utils import report_progress
from smqtk.pytorch_model import get_pytorchmodel_element_impls
from smqtk.algorithms.descriptor_generator.pytorch_descriptor import PytorchDataLoader
from smqtk.utils.image_utils import overlay_saliency_map
from smqtk.representation.data_element.memory_element import DataMemoryElement
from smqtk.representation.data_set.file_set import DataFileSet

try:
    import torch
except ImportError as ex:
    logging.getLogger(__name__).warning("Failed to import pytorch module: %s",
                                        str(ex))
    torch = None
else:
    import torch.utils.data as data
    from torch.autograd import Variable
    import torch.nn as nn
    import torch.nn.functional as F

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
    "PytorchProSaliencyDescriptorGenerator",
    "ProMaskSaliencyDataset",
]

def generate_masks(mask_num, grid_num, image_size=(224, 224)):
    if not os.path.isfile('masks_{}_{}.npy'.format(mask_num, grid_num)):
        cell_size = np.ceil(224 / grid_num)
        up_size = (grid_num + 1) * cell_size

        grid = np.random.randint(0, 2, (mask_num, grid_num, grid_num)).astype('float32')

        masks = np.empty((mask_num, 224, 224))
        for i in tqdm(range(mask_num), desc='Generating masks'):
            x, y = np.random.randint(0, cell_size, 2)
            masks[i, :, :] = resize(grid[i], (up_size, up_size), order=3,
                                      mode='reflect')[x:x + image_size[0], y:y + image_size[1]]
        masks = masks.reshape(-1, 1, *image_size)
        masks.tofile('masks_{}_{}.npy'.format(mask_num, grid_num))
    else:
        masks = np.fromfile('masks_{}_{}.npy'.format(mask_num, grid_num)).reshape(-1, 1, *image_size)

    masks = torch.from_numpy(masks).float().cuda()
    return masks

class TensorDataset(data.Dataset):
    def __init__(self, tensor):
        self._tensor = tensor

    def __getitem__(self, index):
        return self._tensor[index]

    def __len__(self):
        return self._tensor.size(0)

class ProMaskSaliencyDataset(data.Dataset):
    def __init__(self, masks, classifier, batch_size):

        self._filters = masks
        self._query_f = None
        self._img_set = None
        self._process_img = None
        self._process_imgbatch = None
        self._classifier = classifier
        self._batch_size = batch_size
        self._filters_num = masks.size(0)


    @classmethod
    def get_logger(cls):
        """
        :return: logging object for this class
        :rtype: logging.Logger
        """
        return logging.getLogger('.'.join((cls.__module__, cls.__name__)))

    @property
    def _log(self):
        """
        :return: logging object for this class as a property
        :rtype: logging.Logger
        """
        return self.get_logger()

    @property
    def query_f(self):
        return self._query_f

    @query_f.setter
    def query_f(self, val):
        self._query_f = val

    @property
    def image_set(self):
        return self._img_set

    @image_set.setter
    def image_set(self, val):
        self._img_set = val

    @property
    def process_img(self):
        return self._process_img

    @process_img.setter
    def process_img(self, val):
        if not isinstance(val, torch.Tensor):
            raise TypeError("{} has to be torch.Tensor!".format(val))
        if val.dim() != 3:
            raise ValueError("{} has to be 3 dimensions!".format(val))

        self._process_img = val
        self._img_set = [(self._process_img, -1)]

    @property
    def process_imgbatch(self):
        return self._process_imgbatch

    @process_imgbatch.setter
    def process_imgbatch(self, val):
        if not isinstance(val, torch.Tensor):
            raise TypeError("{} has to be torch.Tensor!".format(val))
        if val.dim() != 4:
            raise ValueError("{} has to be 4 dimensions!".format(val))

        self._process_imgbatch = val
        self._img_set = [(val[i], -1) for i in range(val.size(0))]


    def __getitem__(self, index):
        cur_img, _ = self._img_set[index]

        def obtain_masked_img_targetP(img):
            masked_imgs = torch.mul(self._filters.cuda(), img.cuda())

            # masked_image loader
            kwargs = {'shuffle': False}
            masked_imgs_loader = torch.utils.data.DataLoader(
                    TensorDataset(masked_imgs), batch_size=self._batch_size, **kwargs)

            #obtain masked image's probability of the query image
            sim = []
            for m_img in tqdm(masked_imgs_loader, total=len(masked_imgs_loader), desc='Predicting masked images'):
                #matched_f = nn.Softmax(dim=1)(self._classifier(Variable(m_img))[1])
                matched_f = self._classifier(Variable(m_img))[0]
                # matched_f = F.normalize(matched_f, p=2, dim=1)
                # matched_f = self._classifier(Variable(m_img))[0]
                query_f = Variable(torch.from_numpy(self._query_f).unsqueeze(0).cuda())
                sim.append((query_f * matched_f).sum(1))

            sim = torch.cat(sim)

            return sim

        tc_p = obtain_masked_img_targetP(cur_img)

        res_sa = torch.matmul(tc_p.data, self._filters.view(self._filters_num, -1)).view(
            (cur_img.size(1), cur_img.size(2)))

        return res_sa.cpu().numpy()

    def __len__(self):
        return len(self._img_set)


class PytorchProSaliencyDescriptorGenerator (DescriptorGenerator):
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

    def __init__(self, model_cls_name, model_uri=None, mask_num=4000, grid_num=13, resize_val=224,
                 batch_size=1, use_gpu=False, in_gpu_device_id=None, saliency_store_uri='./sa_map',
                 saliency_uuid_dict_file=None):
        """
        Create a pytorch CNN descriptor generator

        :param model_cls_name: model definition name.
        :type model_cls_name: str

        :param model_uri: URI to the trained ``.pt`` file to use.
        :type model_uri: None | str

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
        super(PytorchProSaliencyDescriptorGenerator, self).__init__()


        self.model_cls_name = model_cls_name
        self.model_uri = model_uri
        self.mask_num = mask_num
        self.grid_num = grid_num
        self.resize_val = resize_val
        self.batch_size = int(batch_size)
        self.use_gpu = bool(use_gpu)
        self.in_gpu_device_id = in_gpu_device_id
        self.saliency_store_uri = saliency_store_uri
        self.saliency_uuid_dict_file = saliency_uuid_dict_file
        # initialize_logging(self._log, logging.DEBUG)

        assert self.batch_size > 0, \
            "Batch size must be greater than 0 (got {})".format(self.batch_size)

        if self.use_gpu:
            gpu_list = [x for x in range(torch.cuda.device_count())]
            if self.in_gpu_device_id is None:
                self.gpu_device_id = gpu_list
            else:
                self.gpu_device_id = int(self.in_gpu_device_id)
                assert self.gpu_device_id in gpu_list, \
                    "GPU Device ID must be in gpu_list {} (got {})".format(gpu_list, self.gpu_device_id)
                self.gpu_device_id = [self.gpu_device_id]

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
        pt_model = get_pytorchmodel_element_impls()[self.model_cls_name]()

        self.model_cls = pt_model.model_def()
        if self.model_cls is None:
            raise ValueError("Model class cannot be None!!!")

        self.transform = pt_model.transforms()
        if self.transform is None:
            raise ValueError("Transform cannot be None!!!")

        self.model_cls.eval()
        for p in self.model_cls.parameters():
            p.requires_grad = False

        if self.use_gpu:
            self._log.debug("Using GPU")
            self.model_cls.cuda(self.gpu_device_id[0])
            self.model_cls = torch.nn.DataParallel(self.model_cls, device_ids=self.gpu_device_id)
        else:
            self._log.debug("using CPU")

        if self.model_uri is not None:
            self._log.debug("load the trained model: {}".format(self.model_uri))
            snapshot = torch.load(self.model_uri)
            self.model_cls.load_state_dict(snapshot['state_dict'])

        masks = generate_masks(self.mask_num, self.grid_num, image_size=(self.resize_val, self.resize_val))
        self.saliency_generator = ProMaskSaliencyDataset(masks, self.model_cls, self.batch_size)
        self.data_set = DataFileSet(root_directory=self.saliency_store_uri)

        self._sm_uuid_dict = {}
        if os.path.isfile(self.saliency_uuid_dict_file):
            with open(self.saliency_uuid_dict_file, 'rb') as f:
                self._sm_uuid_dict = pickle.load(f)

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
            'model_cls_name': self.model_cls_name,
            'model_uri': self.model_uri,
            'mask_num' : self.mask_num,
            'grid_num' : self.grid_num,
            'resize_val': self.resize_val,
            'batch_size': self.batch_size,
            'use_gpu': self.use_gpu,
            'in_gpu_device_id': self.in_gpu_device_id,
            'saliency_store_uri': self.saliency_store_uri,
            'saliency_uuid_dict_file': self.saliency_uuid_dict_file
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
                           overwrite=False, query_f=None, query_uuid=None):
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
                                          procs=1, query_f=query_f, query_uuid=query_uuid)
        return m[data.uuid()]

    def compute_descriptor_async(self, data_iter,
                                 descr_factory=DFLT_DESCRIPTOR_FACTORY,
                                 overwrite=False, procs=None, query_f=None, query_uuid=None,
                                 **kwds):
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
        for d in data_iter:
            ct = d.content_type()
            if ct not in self.valid_content_types():
                self._log.error("Cannot compute descriptor from content type "
                                "'%s' data: %s)" % (ct, d))
                raise ValueError("Cannot compute descriptor from content type "
                                 "'%s' data: %s)" % (ct, d))
            data_elements[d.uuid()] = d
            descr_elements[d.uuid()] = descr_factory.new_descriptor(self.name, d.uuid())
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
            # set the query flag accordingly
            saliency_flag = False
            if query_f is not None:
                if query_uuid is None:
                    raise ValueError('Query uuid has to be provided!')
                saliency_flag = True

            self._log.debug("Converting deque to tuple for segmentation")
            kwargs = {'num_workers': procs if procs is not None
                        else multiprocessing.cpu_count(), 'pin_memory': True}
            data_loader_cls = PytorchDataLoader(file_list=data_elements, resize_val=self.resize_val,
                                                uuid4proc=uuid4proc, transform=self.transform, saliency_info=True)
            data_loader = torch.utils.data.DataLoader(data_loader_cls, batch_size=self.batch_size,
                                                      shuffle=False, **kwargs)

            self._log.debug("Extract pytorch features")

            for (d, uuids, resized_org_img, (w, h)) in tqdm(data_loader, total=len(data_loader), desc='extracting feature'):
                # use output probability as the feature vector
                if self.use_gpu:
                    # pytorch_f = nn.Softmax(1)(self.model_cls(Variable(d.cuda()))[1])
                    pytorch_f = self.model_cls(Variable(d.cuda()))[0]
                    # pytorch_f = F.normalize(pytorch_f, p=2, dim=1)
                else:
                    pytorch_f = nn.Softmax(1)(self.model_cls(Variable(d))[1])
                    raise ValueError('Need to use GPU')

                # estimated probablity saliency maps
                if saliency_flag:
                    self.saliency_generator.query_f = query_f
                    q_uuid = query_uuid

                    self.saliency_generator.process_imgbatch = d

                for idx, uuid in enumerate(uuids):
                    f_vec = pytorch_f.data.cpu().numpy()[idx]
                    descr_elements[uuid].set_vector(f_vec)

                    if saliency_flag:
                        if (uuid, q_uuid) in self._sm_uuid_dict and \
                                self.data_set.has_uuid(self._sm_uuid_dict[(uuid, q_uuid)]):
                            descr_elements[uuid].update_saliency_map({q_uuid : self._sm_uuid_dict[(uuid, q_uuid)]})
                        else:
                            ## generate the saliency map
                            #-------------------------------------------------------
                            sa_map = self.saliency_generator[idx]

                            # write out the top K saliency maps
                            dme = DataMemoryElement(bytes=overlay_saliency_map(sa_map, resized_org_img[idx], w[idx], h[idx]), content_type='image/png')
                            self.data_set.add_data(dme)

                            descr_elements[uuid].update_saliency_map({q_uuid : dme.uuid()})
                            self._sm_uuid_dict[(uuid, q_uuid)] = dme.uuid()
                            #-------------------------------------------------------

        self._log.debug('write out saliency uuid dict')
        with open(self.saliency_uuid_dict_file, 'wb') as f:
            pickle.dump(self._sm_uuid_dict, f)


        self._log.debug("forming output dict")
        return dict((data_elements[k].uuid(), descr_elements[k])
                    for k in data_elements)
