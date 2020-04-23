from smqtk.algorithms.descriptor_generator import DescriptorGenerator, \
    DFLT_DESCRIPTOR_FACTORY
from torchvision import models, transforms
from torch.utils.data import DataLoader
from .utils import PytorchImagedataset  
import torch
import torchvision 
from collections import deque
from torch.autograd import Variable
import multiprocessing

try:
    import torch
    import torchvision
except ImportError as ex:
    logging.getLogger(__name__).warning("Failed to import torch/torchvision \
                                     module: %s", str(ex))
    torch = None
    torchvision = None

__all__ = [
    "PytorchModelDescriptor",
]


class PytorchModelDescriptor (DescriptorGenerator):
    """
    Compute images against a PyTorch model, extracting a layer as the content
    descriptor.
    """


    @classmethod
    def is_usable(cls):
        valid = torch is not None or torchvision is not None 
        if not valid:
            cls.get_logger().debug("PyTorch and torchvision cannot be imported")
        return valid


    def truncate_pytorch_model(self, model, return_layer_list):
        """
        Given a pytorch model and label of layer, the function returns a 
        model truncated at return layer.
        :param model: The pytorch model that needs to be truncated at 
               a certain return layer in network.
        :type model: torch.nn
        :param return_layer_list: List of return layers in hierarchical order.
        :type return_layer_list: List of string [str, str, ...]
        
        :return seq_mod,t1_model: Last sequential block of network 
                in present state.
        :return model,trunc_model: Model truncated until last sequential block 
        :rtype: torch.nn    
        """
        if len(return_layer_list) == 2:
            t1_model, _ = self.truncate_pytorch_model(model, 
                                                  [return_layer_list[0]]) 
            sub_module_list = [_ for _ in t1_model.named_children()]
            for inx, lay in enumerate(sub_module_list):
                if return_layer_list[1] == lay[0]:
                    sub_pos = inx
            trunc_pos = len(sub_module_list) - (sub_pos+1)
            model_sub_ = torch.nn.Sequential(*(list(t1_model.children()))
                                                  [:-trunc_pos])
            setattr(locals().get("model"), 'classifier', model_sub_)
            return t1_model, model
        else:
            module_list = list(model.__dict__['_modules'])
            layer_position = (module_list.index(return_layer_list[0])) 
            if len(module_list) == layer_position:
                return model, model
            else:
                trunc_model = torch.nn.Sequential(*(list(model.children())
                                                       [:layer_position+1]))
                try:
                    seq_mod = torch.nn.Sequential((list(model.children())
                                                       [layer_position]))
                except IndexError:
                    seq_mod = None
                return seq_mod, trunc_model

    def check_model_dict(self, model, return_key):
        """
        Checks model dictionary to see if the top return layer is present.
        :param model: Base model to be checked for presense of layer
        :type model: torch.nn
        :param return_keys: Label of top return layer for feature 
               collection.
        :type return_keys: str
        """
        try:
            if return_key is not '':
                assert (getattr(model,"__dict__")).get("_modules")[return_key]
                #model_dict = getattr(model,"__dict__")
                #model = model_dict.get("_modules")[return_key]
        except KeyError:
            self._log.info("KeyError: Given return layer is \
                                               not present in model")

    def __init__(self, 
                 model_name = 'resnet18',
                 return_layer = 'avgpool', 
                 custom_model_arch = None, 
                 weights_filepath = None, 
                 norm_mean = None, 
                 norm_std = None, 
                 use_gpu = True,
                 batch_size = 32,
                 pretrained = True):
        """
        Create a PyTorch CNN descriptor generator
        :param model_name: Name of model on PyTorch library,
            for example: 'resnet50', 'vgg16'.
        :type model_name: str
        :param return_layer: The label of the layer we take data from 
               to compose output descriptor vector.
        :type return_layer: str 
        :param custom_model_arch: Method that implements a custom Pytorch
            model.
        :type custom_model_arch: torch.nn
        :param weights_filepath: Absolute file path to weights of a custom 
               model custom_model_arch.
        :type weights_filepath: str
        :param norm_mean: Mean for normalizing images across three channels.
        :type norm_mean: List [float, float, float].
        :param norm_std: Standard deviation for normalizing images across 
               three channels.
        :type norm_std: List [float, float, float].
        :param use_gpu: If Caffe should try to use the GPU
        :type use_gpu: bool
        :param batch_size: The maximum number of images to process in one feed
            forward of the network. This is especially important for GPUs since
            they can only process a batch that will fit in the GPU memory
            space.
        :type batch_size: int
        :param pretrained: The network is loaded with pretrained weights 
               available on torchvision instead of custom weights.
        :type pretrained: bool
        """
        self.model_name = model_name
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224,224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(norm_mean, norm_std)])
        self.batch_size = batch_size
        self.return_layer = return_layer
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.use_gpu = use_gpu
        self.pretrained = pretrained
        self.weights_filepath = weights_filepath
        self.custom_model_arch = custom_model_arch 
        if not custom_model_arch:
            try: 
                assert model_name in models.__dict__.keys()
            except AssertionError:
                self._log.info("Invalid model name, model not present \
                             in torchvision. Please load network architecture")
                self._log.info("Available models include:{}"
                       .format([s for s in models.__dict__.keys() \
                                              if not "__" in s])) 
            model = getattr(models, self.model_name)(self.pretrained)
            ret_para = [k for k in self.return_layer.split('.')]
            self.check_model_dict(model, ret_para[0])
            try:
                _, new_model = self.truncate_pytorch_model(model, ret_para)
                assert new_model
                model = new_model
            except AssertionError:
                self._log.info("Invalid return layer label selected model:{}"\
                                                            .format(model))
                raise AssertionError
        else:
             model = custom_model_arch
        if (not self.pretrained) and (self.weights_filepath):
            checkpoint = torch.load(self.weights_filepath)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            model.load_state_dict(checkpoint)
        model.eval()
        if self.use_gpu:
            try:
                model = model.cuda()
                self.model = torch.nn.DataParallel(model)
            except AssertionError:
                self.model = model 
                self._log.info("Cannot load PyTorch model to GPU, running on CPU")

    def __getstate__(self):
        return self.get_config()

    def __setstate__(self, state):
        # This works because configuration parameters exactly match up with
        # instance attributes
        self.__dict__.update(state)
        self._setup_network()

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
            'model_name': self.model_name,
            'return_layer': self.return_layer,
            'custom_model_arch': self.custom_model_arch,
            'weights_filepath': self.weights_filepath,
            'norm_mean': self.norm_mean,
            'norm_std': self.norm_std,
            'use_gpu': self.use_gpu,
            'batch_size': self.batch_size,
            'pretrained': self.pretrained,
        }

    def valid_content_types(self):
        """
        :return: A set valid MIME type content types that this descriptor can
            handle.
        :rtype: set[str]
        """
        return {
            'image/bmp',
            'image/tiff',
            'image/png',
            'image/jpeg',
        }

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
        m = self.compute_descriptor_async([data], descr_factory)
        return m[data.uuid()]

    def _compute_descriptor(self, data):
        raise NotImplementedError("Shouldn't get here as "
                                  "compute_descriptor[_async] is being "
                                  "overridden")

    def compute_descriptor_async(self, data_set, descriptor_elem_factory= 
                                 DFLT_DESCRIPTOR_FACTORY, overwrite=False):
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
        :raises ValueError: An input DataElement was of a content type that we
            cannot handle.
        :return: Mapping of input DataElement UUIDs to the computed descriptor
            element for that data. DescriptorElement UUID's are congruent with
            the UUID of the data element it is the descriptor of.
        :rtype: dict[collections.Hashable,
                     smqtk.representation.DescriptorElement]
        """
        self.data_elements = {}
        self.descr_elements = {}
        self.uuid4proc = deque()
        for d in data_set:
            ct = d.content_type()
            if ct not in self.valid_content_types():
                self._log.error("Cannot compute descriptor from content type "
                                "'%s' data: %s)" % (ct, d))
                raise ValueError("Cannot compute descriptor from content type "
                                 "'%s' data: %s)" % (ct, d))
            self.data_elements[d.uuid()] = d
            self.descr_elements[d.uuid()] = descriptor_elem_factory \
                               .new_descriptor(self.name, d.uuid()) 
            self.uuid4proc.append(d.uuid())       
        self._log.debug("Given %d unique data elements", len(self.data_elements))
        if len(self.data_elements) == 0:
            raise ValueError("No data elements provided") 
        if self.uuid4proc:
            self._log.debug("Converting deque to tuple for segmentation")
            kwargs = {'num_workers': multiprocessing.cpu_count(), 
                                              'pin_memory': True}
            data_loader_cls = PytorchImagedataset(self.data_elements, 
                                             self.uuid4proc, self.transforms)
            data_loader = DataLoader(data_loader_cls, 
                         batch_size=self.batch_size, shuffle=False, **kwargs)
            self._log.debug("Extracting PyTorch features")
            for (d, uuids) in data_loader:
                if self.use_gpu:
                    d = d.cuda()
                # Test for speed Variable and no_grad 
                pytorch_f = self.model(Variable(d)).squeeze()
                if len(pytorch_f.shape) < 2:
                    pytorch_f = pytorch_f.unsqueeze(0)
                if len(pytorch_f.shape) > 2:
                    pytorch_f = pytorch_f.view(pytorch_f.shape(0), (pytorch_f.shape(1)*pytorch_f.shape(2)))
                [self.descr_elements[uuid].set_vector(
                               pytorch_f.data.cpu().numpy()[idx]) 
                               for idx, uuid in enumerate(uuids)]
        self._log.debug("forming output dict")
        return dict((self.data_elements[k].uuid(), self.descr_elements[k])
                    for k in self.data_elements)

