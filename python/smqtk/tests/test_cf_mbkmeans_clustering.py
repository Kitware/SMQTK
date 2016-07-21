import unittest

import nose.tools
import numpy
from sklearn.cluster import MiniBatchKMeans

from smqtk.compute_functions import mb_kmeans_build_apply
from smqtk.representation.descriptor_index.memory import \
    MemoryDescriptorIndex
from smqtk.representation.descriptor_element.local_elements import \
    DescriptorMemoryElement


class TestMBKMClustering (unittest.TestCase):
    """ Test class to be picked up by nosetests """

    def test_clustering_equal_descriptors(self):
        # Test that clusters of descriptor of size  n-features are correctly
        # clustered together.
        print "Creating dummy descriptors"
        n_features = 8
        n_descriptors = 20

        index = MemoryDescriptorIndex()
        c = 0
        for i in range(n_features):
            v = numpy.ndarray(8)
            v[...] = 0
            v[i] = 1
            for j in range(n_descriptors):
                d = DescriptorMemoryElement('test', c)
                d.set_vector(v)
                index.add_descriptor(d)
                c += 1

        print "Creating test MBKM"
        mbkm = MiniBatchKMeans(n_features, batch_size=12, verbose=True,
                               compute_labels=False, random_state=0)

        # Initial fit with half of index
        d_classes = mb_kmeans_build_apply(index, mbkm, n_descriptors)

        # There should be 20 descriptors per class
        for c in d_classes:
            nose.tools.assert_equal(
                len(d_classes[c]),
                n_descriptors,
                "Cluster %s did not have expected number of descriptors "
                "(%d != %d)"
                % (c, n_descriptors, len(d_classes[c]))
            )

            # Each descriptor in each cluster should be equal to the other
            # descriptors in that cluster
            uuids = list(d_classes[c])
            v = index[uuids[0]].vector()
            for uuid in uuids[1:]:
                v2 = index[uuid].vector()
                numpy.testing.assert_array_equal(v, v2,
                                                 "vector in cluster %d did not "
                                                 "match other vectors "
                                                 "(%s != %s)"
                                                 % (c, v, v2))

    def test_empty_index(self):
        # what happens when function given an empty descriptor index
        index = MemoryDescriptorIndex()
        mbkm = MiniBatchKMeans()
        d = mb_kmeans_build_apply(index, mbkm, 0)
        nose.tools.assert_false(d)
