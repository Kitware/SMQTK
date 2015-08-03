import logging
import numpy
import os.path as osp
try:
    import sklearn.cluster
except:
    sklearn.cluster = None

from smqtk.quantization import Quantization
from smqtk.data_rep.descriptor_element_impl.local_elements import DescriptorMemoryElement
from smqtk.utils import SimpleTimer
from smqtk.utils.data_rep_utils import npy_array_to_descriptor_element_md5, iterable_npy_array_to_iterable_desc_elem_md5

class SKLearn_Cluster_MiniBatch_Kmeans (Quantization):
    def __init__(self,
        work_dir,
        quantization_filepath,
        label,
        kmeans_k = 256,
        init_size = 768,
        random_seed = 42,
        compute_labels = False,
        verbose = 0):

        # Call super class constructor to set common members (work directory, 
        # location where the filepath is saved, and label).
        super(SKLearn_Cluster_MiniBatch_Kmeans, self).__init__(work_dir, quantization_filepath, label)
        self._kmeans_k = kmeans_k
        self._init_size = init_size
        self._rand_seed = random_seed
        self._compute_labels = compute_labels
        self._verbose = None
        self._quantization = None
        self._quantization_numpy = None
        self._data_element_type = None
        self._descriptor_element_factory = None
        self._log = logging.getLogger('.'.join([Quantization.__module__,
                                            Quantization.__name__]))

    @classmethod
    def is_usable(cls):
        return sklearn.cluster is not None
    
    @property
    def has_quantization(self):
        # See if there is already a file saved for this quantization
        self._has_quantization = osp.isfile(self._quantization_filepath + ".npy")

        # If there is a quantization file already and this quantization
        # has not been implemented, load it and return
        if self._quantization is None and self._has_quantization:
            self._quantization_numpy = numpy.load(self._quantization_filepath + ".npy")
            self._quantization = iterable_npy_array_to_iterable_desc_elem_md5(self._data_element_type, 
                self._descriptor_element_factory, self._label, self._quantization_numpy)

        return self._has_quantization

    def generate_quantization(self, descriptor_matrices, descriptor_element_factory, data_element_type):
        # TODO -- Validity checks
        self._descriptor_element_factory = descriptor_element_factory
        self._data_element_type = data_element_type
        # Check for existance of quantization
        if self.has_quantization:
            self._log.warn("Quantization already generated at %s for %s. Returning." % (self._quantization_filepath, self._label))
            return

        # Compute centroids with kmeans
        with SimpleTimer("Computing sklearn.cluster.MiniBatchKMeans...", self._log.info):
            self._verbose = self._log.getEffectiveLevel <= logging.DEBUG
            kmeans = sklearn.cluster.MiniBatchKMeans(
                n_clusters=self._kmeans_k,
                init_size=self._kmeans_k*3,
                random_state=self._rand_seed,
                verbose=self._verbose,
                compute_labels=False)
            kmeans.fit(descriptor_matrices.vector())
            quantization = kmeans.cluster_centers_
            
        with SimpleTimer("Saving generated quantization...", self._log.info):
            numpy.save(self._quantization_filepath, quantization)

        # To use the indexing plugins, we need to turn the quantization into an iterable of
        # DataElements from its natural numpy state.
        quantization_iter = iterable_npy_array_to_iterable_desc_elem_md5(self._data_element_type, self._descriptor_element_factory, self._label, quantization)

        self._quantization_numpy = quantization
        self._quantization = quantization_iter
        return 

cbook_type_list = [
    SKLearn_Cluster_MiniBatch_Kmeans
]