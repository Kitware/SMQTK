import logging
import numpy
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

        super(Quantization, self).__init__()
        self._kmeans_k = kmeans_k
        self._quantization_filepath = quantization_filepath
        self._init_size = init_size
        self._rand_seed = random_seed
        self._compute_labels = compute_labels
        self._verbose = None

        self._log = logging.getLogger('.'.join([Quantization.__module__,
                                            Quantization.__name__]))

    @classmethod
    def is_usable(cls):
        # TODO -- define usability criteria
        return True

    def generate_quantization(self, descriptor_matrices):
        # Validity checks
        elements = []
#        super(Quantization, self).generate_quantization
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