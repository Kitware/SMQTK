from smqtk.algorithms.descriptor_generator import DescriptorGenerator, \
    DFLT_DESCRIPTOR_FACTORY
from smqtk.utils.cli import ProgressReporter

from collections import deque
import multiprocessing
import multiprocessing.pool
import six
import logging 

try:
    import torch
    import torchvision
    from torch.utils.data import DataLoader
    from torch.autograd import Variable
    from .utils import PytorchImagedataset
except ImportError as ex:
    logging.warning("Failed to import torch/torchvision "
                                     "module: %s", str(ex))
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
        valid = (torch is None) or (torchvision is None) 
        if valid:
            cls.get_logger().debug("PyTorch or torchvision cannot be imported")
        return (not valid)


    def truncate_pytorch_model(self, model, t1_model):
        """
        Given a pytorch model and label of layer, the function returns a 
        model truncated at return layer.
        :param model: The pytorch model that needs to be truncated at 
               a certain return layer in network.
        :type model: torch.nn.Sequential
        :param t1_model: The pytorch sequential block of layers containing
               the final return layer key.
        :type t1_model: torch.nn.Sequential
 
        :return model: Model truncated till the given sub module return
                layer 
        :rtype: torch.nn.Sequential    
        """
        # Extract children of submodule return_key1
        sub_module_list = [_ for _ in t1_model.named_children()]
        for inx, lay in enumerate(sub_module_list):
            if self.return_layer[1] == lay[0]:
                sub_pos = inx
                break
        trunc_pos = len(sub_module_list) - (sub_pos+1)
        model_sub_ = torch.nn.Sequential(*(list(t1_model.children()))
                                                  [:-trunc_pos])
        setattr(locals().get("model"), self.return_layer[0], model_sub_)
        return model

    def check_model_truncate(self, model):
        """
        Checks model dictionary to see if the top return layer is present.
        :param model: Base model to be checked for presense of layer
        :type model: torch.nn
        :param model: The final model truncated to layer return_key2 if 
                      present, otherwise the model to return_key1.
        :type model: torch.nn.Sequential
        """
        try:
        # We currently support iterating through only two levels of the network 
            assert len(self.return_layer) < 3
            if self.return_layer[0] is not '':
                assert model._modules[self.return_layer[0]]
                module_list = list(model.__dict__['_modules'])
                layer_position = (module_list.index(self.return_layer[0]))
                if len(self.return_layer) == 1:
                    # If return_key1 is the last submodule
                    if len(module_list) == layer_position:
                        return model
                    else:
                        # If no submodule i.e return_key2 return 
                        # truncated model
                        model = torch.nn.Sequential(*(list(model.children())
                                                       [:layer_position+1]))
                # Return the last submodule that needs to be truncated further.
                if len(self.return_layer) == 2:
                    last_stage = torch.nn.Sequential((list(model.children())
                                                       [layer_position]))[0]
                    # If we want to truncate submodule return_key1   
                    model = self.truncate_pytorch_model(model, last_stage) 
            return model      
        except KeyError:
            self._log.error("Given return layer is "
                                  "invalid:{}".format(self.return_layer))
            raise

    def __init__(self, 
                 model_name = 'resnet18', return_layer = 'avgpool', 
                 custom_model_arch = None, weights_filepath = None,
                 input_dim = (224, 224), norm_mean = [0.485, 0.456, 0.406], 
                 norm_std = [0.229, 0.224, 0.225], use_gpu = True,
                 batch_size = 32, pretrained = True):
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
        :param input_dim: Image height and width of an input image.
        :type input_dim: (int, int)
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
            torchvision.transforms.Resize((input_dim[0], input_dim[1])),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(norm_mean, norm_std)])
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.use_gpu = use_gpu
        self.pretrained = pretrained
        self.weights_filepath = weights_filepath
        self.custom_model_arch = custom_model_arch 
        # Check if user wants to load custom model or a model from torchvision
        if not custom_model_arch:
            try: 
                assert model_name in torchvision.models.__dict__.keys()
            except AssertionError:
                self._log.error("Invalid model name, model not present "
                             "in torchvision. Please load network architecture")
                self._log.info("Available models include:{}"
                       .format([s for s in torchvision.models.__dict__.keys() 
                                              if not "__" in s]))
                raise 
            # Loading model from torchvision library
            model = getattr(torchvision.models, self.model_name)(self.pretrained)
        else:
             # If custom architecture 
             model = custom_model_arch
        if (not self.pretrained) and (self.weights_filepath):
            checkpoint = torch.load(self.weights_filepath)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            model.load_state_dict(checkpoint)
        if (not self.pretrained) and (not self.weights_filepath):
            self._log.error("Network might be loaded with junk weights")
            raise ValueError
        self.return_layer = [k for k in return_layer.split('.')]
        # We currently support iterating through only two levels of the network 
        # i.e return_layer1 and return_layer2
        # Check if return_layer1 is present in model and truncate the sub 
        # module containing return_key2.
        model = self.check_model_truncate(model)
        model.eval()

        if self.use_gpu:
            try:
                model = model.cuda()
                self.model = torch.nn.DataParallel(model)
            except ValueError:
                self.model = model
                self._log.info("Cannot load PyTorch model to GPU, running on CPU")

        try:
            assert model
        except AssertionError:
            self._log.info("Selected model{}".format(sub_model))
            raise ("Model could not be loaded")

    def __getstate__(self):
        return self.get_config()

    def _setup_network(self):
        pass
        #raise NotImplementedError("Nada")

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
            'return_layer': '.'.join(self.return_layer),
            'custom_model_arch': self.custom_model_arch,
            'weights_filepath': self.weights_filepath,
            'input_dim': self.input_dim,
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

    def check_get_uuid(self, descriptor_elem):
        if self.overwrite or not descriptor_elem.has_vector():
            self.uuid4proc.append(descriptor_elem.uuid())

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
        data_elements = {}
        descr_elements = {}
        pr = ProgressReporter(self._log.debug, 1.0).start()
        for d in data_set:
            ct = d.content_type()
            if ct not in self.valid_content_types():
                self._log.error("Cannot compute descriptor from content type "
                                "'%s' data: %s)" % (ct, d))
                raise ValueError("Cannot compute descriptor from content type "
                                 "'%s' data: %s)" % (ct, d))
            data_elements[d.uuid()] = d
            descr_elements[d.uuid()] = descriptor_elem_factory \
                               .new_descriptor(self.name, d.uuid()) 
            pr.increment_report()
        pr.report()
        self.overwrite = overwrite 
        self.uuid4proc = deque()

        procs = multiprocessing.cpu_count()
        if len(data_elements) < procs:
            procs = len(data_elements)
        if procs == 0:
            raise ValueError("No data elements provided")
        # Using thread-pool due to in-line function + updating local deque
        p = multiprocessing.pool.ThreadPool(procs)
        try:
            p.map(self.check_get_uuid, six.itervalues(descr_elements))
        except AttributeError:
            p.close()
            p.join()
        del p
        self._log.debug("%d descriptors already computed",
                     len(data_elements) - len(self.uuid4proc))
        self._log.debug("Given %d unique data elements", 
                                     len(data_elements))
        if len(data_elements) == 0:
            raise ValueError("No data elements provided") 

        if self.uuid4proc:
            kwargs = {'num_workers': procs, 'pin_memory': True}
            data_loader_cls = PytorchImagedataset(data_elements, 
                                   self.uuid4proc, self.transforms)
            data_loader = DataLoader(data_loader_cls, 
                         batch_size=self.batch_size, shuffle=False, **kwargs)
            self._log.debug("Extracting PyTorch features")
            for (d, uuids) in data_loader:
                if self.use_gpu:
                    d = d.cuda()
                pytorch_f = self.model(Variable(d)).squeeze()
                if len(pytorch_f.shape) < 2:
                    pytorch_f = pytorch_f.unsqueeze(0)
                if len(pytorch_f.shape) > 2:
                    import numpy
                    pytorch_f = pytorch_f.view(pytorch_f.shape[0],
                               (numpy.prod(pytorch_f.shape[1:])))
                [descr_elements[uuid].set_vector(
                               pytorch_f.data.cpu().numpy()[idx]) 
                               for idx, uuid in enumerate(uuids)]
        self._log.debug("forming output dict")
        return dict((data_elements[k].uuid(), descr_elements[k])
                    for k in data_elements)

def _process_load_img_array(image_pil, transforms = None):
    """
    Helper function for multiprocessing image data loading

    """
    if transforms:
        image_pil = transforms(image_pil)
    return torchvision.transforms.ToPILImage(image_pil) 
    
