from __future__ import print_function

import unittest

import numpy
from sklearn.cluster import MiniBatchKMeans

from smqtk.compute_functions import mb_kmeans_build_apply
from smqtk.representation.descriptor_set.memory import \
    MemoryDescriptorSet
from smqtk.representation.descriptor_element.local_elements import \
    DescriptorMemoryElement


class TestMBKMClustering (unittest.TestCase):

    def test_clustering_equal_descriptors(self):
        # Test that clusters of descriptor of size  n-features are correctly
        # clustered together.
        print("Creating dummy descriptors")
        n_features = 8
        n_descriptors = 20

        desr_set = MemoryDescriptorSet()
        c = 0
        for i in range(n_features):
            v = numpy.ndarray((8,))
            v[...] = 0
            v[i] = 1
            for j in range(n_descriptors):
                d = DescriptorMemoryElement('test', c)
                d.set_vector(v)
                desr_set.add_descriptor(d)
                c += 1

        print("Creating test MBKM")
        mbkm = MiniBatchKMeans(n_features, batch_size=12, verbose=True,
                               compute_labels=False, random_state=0)

        # Initial fit with half of desr_set
        d_classes = mb_kmeans_build_apply(desr_set, mbkm, n_descriptors)

        # There should be 20 descriptors per class
        for c in d_classes:
            self.assertEqual(
                len(d_classes[c]),
                n_descriptors,
                "Cluster %s did not have expected number of descriptors "
                "(%d != %d)"
                % (c, n_descriptors, len(d_classes[c]))
            )

            # Each descriptor in each cluster should be equal to the other
            # descriptors in that cluster
            uuids = list(d_classes[c])
            v = desr_set[uuids[0]].vector()
            for uuid in uuids[1:]:
                v2 = desr_set[uuid].vector()
                numpy.testing.assert_array_equal(v, v2,
                                                 "vector in cluster %d did not "
                                                 "match other vectors "
                                                 "(%s != %s)"
                                                 % (c, v, v2))

    def test_emptyset(self):
        # what happens when function given an empty descriptor set
        descr_set = MemoryDescriptorSet()
        mbkm = MiniBatchKMeans()
        d = mb_kmeans_build_apply(descr_set, mbkm, 0)
        self.assertFalse(d)
