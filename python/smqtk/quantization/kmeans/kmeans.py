import logging
import numpy
import os.path as osp
import sklearn.cluster

from smqtk.quantization import Quantization
from smqtk.data_rep.descriptor_element_impl.local_elements import DescriptorMemoryElement
from smqtk.utils import SimpleTimer

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
        self._log = logging.getLogger('.'.join([Quantization.__module__,
                                            Quantization.__name__]))

    @classmethod
    def is_usable(cls):
        # TODO -- define usability criteria
        return True

    @property
    def has_quantization(self):
        # See if there is already a file saved for this quantization
        self._has_quantization = osp.isfile(self._quantization_filepath + ".npy")

        # If there is a quantization file already and this quantization
        # has not been implemented, load it and return
        if self._quantization is None and self._has_quantization:
            self._quantization = numpy.load(self._quantization_filepath + ".npy")
        return self._has_quantization
    

    def generate_quantization(self, descriptor_matrices):
        # TODO -- Validity checks

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

            kmeans.fit(descriptor_matrices)
            quantization = kmeans.cluster_centers_
            
        with SimpleTimer("Saving generated quantization...", self._log.info):
            numpy.save(self._quantization_filepath, quantization)

        self._quantization = quantization

cbook_type_list = [
    SKLearn_Cluster_MiniBatch_Kmeans
]