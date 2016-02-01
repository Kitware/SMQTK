import unittest

import multiprocessing
import multiprocessing.pool
import nose.tools as ntools
import numpy

from smqtk.algorithms.classifier.libsvm import LibSvmClassifier
from smqtk.representation import \
    ClassificationElementFactory, \
    DescriptorElementFactory
from smqtk.representation.classification_element.memory import \
    MemoryClassificationElement
from smqtk.representation.descriptor_element.local_elements import \
    DescriptorMemoryElement


if LibSvmClassifier.is_usable():

    class TestLibSvmClassifier (unittest.TestCase):

        def tearDown(self):
            # Clear MemoryElement content
            DescriptorMemoryElement.MEMORY_CACHE = {}

        def test_simple_classification(self):
            """
            simple LibSvmClassifier test - 2-class

            Test libSVM classification functionality using random constructed
            data, training the y=0.5 split
            """
            DIM = 2
            N = 1000
            POS_LABEL = 'positive'
            p = multiprocessing.pool.ThreadPool()
            d_factory = DescriptorElementFactory(DescriptorMemoryElement, {})
            c_factory = ClassificationElementFactory(MemoryClassificationElement, {})

            def make_element((i, v)):
                d = d_factory.new_descriptor('test', i)
                d.set_vector(v)
                return d

            # Constructing artificial descriptors
            x = numpy.random.rand(N, DIM)
            x_pos = x[x[:, 1] <= 0.45]
            x_neg = x[x[:, 1] >= 0.55]

            d_pos = p.map(make_element, enumerate(x_pos))
            d_neg = p.map(make_element, enumerate(x_neg, start=N//2))

            # Create/Train test classifier
            classifier = LibSvmClassifier(
                train_params={
                    '-t': 0,  # linear kernel
                    '-b': 1,  # enable probability estimates
                    '-c': 2,  # SVM-C parameter C
                    '-q': '',  # quite mode
                },
                normalize=None,  # DO NOT normalize descriptors
            )
            classifier.train({POS_LABEL: d_pos}, d_neg)

            # Test classifier
            x = numpy.random.rand(N, DIM)
            x_pos = x[x[:, 1] <= 0.45]
            x_neg = x[x[:, 1] >= 0.55]

            d_pos = p.map(make_element, enumerate(x_pos, N))
            d_neg = p.map(make_element, enumerate(x_neg, N + N//2))

            d_pos_sync = {}  # for comparing to async
            for d in d_pos:
                c = classifier.classify(d, c_factory)
                ntools.assert_equal(c.max_label(),
                                    POS_LABEL,
                                    "Found False positive: %s :: %s" %
                                    (d.vector(), c.get_classification()))
                d_pos_sync[d] = c

            d_neg_sync = {}
            for d in d_neg:
                c = classifier.classify(d, c_factory)
                ntools.assert_equal(c.max_label(),
                                    LibSvmClassifier.NEGATIVE_LABEL,
                                    "Found False negative: %s :: %s" %
                                    (d.vector(), c.get_classification()))
                d_neg_sync[d] = c

            # test that async classify produces the same results
            m_pos = classifier.classify_async(d_pos, c_factory)
            ntools.assert_equal(m_pos, d_pos_sync,
                                "Async computation of pos set did not yield "
                                "the same results as synchronous "
                                "classification.")
            m_neg = classifier.classify_async(d_neg, c_factory)
            ntools.assert_equal(m_neg, d_neg_sync,
                                "Async computation of neg set did not yield "
                                "the same results as synchronous "
                                "classification.")

            combined_truth = dict(d_pos_sync.iteritems())
            combined_truth.update(d_neg_sync)
            m_combined = classifier.classify_async(d_pos + d_neg,
                                                   c_factory)
            ntools.assert_equal(m_combined, combined_truth,
                                "Async computation of all test descriptors "
                                "did not yield the same results as "
                                "synchronous classification.")
            # trying diff input order for kicks
            m_combined = classifier.classify_async(d_neg + d_pos,
                                                   c_factory)
            ntools.assert_equal(m_combined, combined_truth,
                                "Async computation of all test descriptors "
                                "(mixed order) did not yield the same results "
                                "as synchronous classification.")

            # Closing resources
            p.close()
            p.join()

        def test_simple_multiclass_classification(self):
            """
            simple LibSvmClassifier test - 3-class

            Test libSVM classification functionality using random constructed
            data, training the y=0.33 and y=.66 split
            """
            DIM = 2
            N = 1000
            P1_LABEL = 'p1'
            P2_LABEL = 'p2'
            p = multiprocessing.pool.ThreadPool()
            d_factory = DescriptorElementFactory(DescriptorMemoryElement, {})
            c_factory = ClassificationElementFactory(MemoryClassificationElement, {})
            di = 0

            def make_element((i, v)):
                d = d_factory.new_descriptor('test', i)
                d.set_vector(v)
                return d

            # Constructing artificial descriptors
            x = numpy.random.rand(N, DIM)
            x_p1 = x[x[:, 1] <= 0.30]
            x_p2 = x[(x[:, 1] >= 0.36) & (x[:, 1] <= 0.63)]
            x_neg = x[x[:, 1] >= 0.69]

            d_p1 = p.map(make_element, enumerate(x_p1, di))
            di += len(d_p1)
            d_p2 = p.map(make_element, enumerate(x_p2, di))
            di += len(d_p2)
            d_neg = p.map(make_element, enumerate(x_neg, di))
            di += len(d_neg)

            # Create/Train test classifier
            classifier = LibSvmClassifier(
                train_params={
                    '-t': 0,  # linear kernel
                    '-b': 1,  # enable probability estimates
                    '-c': 2,  # SVM-C parameter C
                    '-q': ''  # quite mode
                },
                normalize=None,  # DO NOT normalize descriptors
            )
            classifier.train({P1_LABEL: d_p1, P2_LABEL: d_p2}, d_neg)

            # Test classifier
            x = numpy.random.rand(N, DIM)
            x_p1 = x[x[:, 1] <= 0.30]
            x_p2 = x[(x[:, 1] >= 0.36) & (x[:, 1] <= 0.63)]
            x_neg = x[x[:, 1] >= 0.69]

            d_p1 = p.map(make_element, enumerate(x_p1, di))
            di += len(d_p1)
            d_p2 = p.map(make_element, enumerate(x_p2, di))
            di += len(d_p2)
            d_neg = p.map(make_element, enumerate(x_neg, di))
            di += len(d_neg)

            d_p1_sync = {}
            for d in d_p1:
                c = classifier.classify(d, c_factory)
                ntools.assert_equal(c.max_label(),
                                    P1_LABEL,
                                    "Incorrect %s label: %s :: %s" %
                                    (P1_LABEL, d.vector(),
                                     c.get_classification()))
                d_p1_sync[d] = c

            d_p2_sync = {}
            for d in d_p2:
                c = classifier.classify(d, c_factory)
                ntools.assert_equal(c.max_label(),
                                    P2_LABEL,
                                    "Incorrect %s label: %s :: %s" %
                                    (P2_LABEL, d.vector(),
                                     c.get_classification()))
                d_p2_sync[d] = c

            d_neg_sync = {}
            for d in d_neg:
                c = classifier.classify(d, c_factory)
                ntools.assert_equal(c.max_label(),
                                    LibSvmClassifier.NEGATIVE_LABEL,
                                    "Incorrect %s label: %s :: %s" %
                                    (LibSvmClassifier.NEGATIVE_LABEL,
                                     d.vector(),
                                     c.get_classification()))
                d_neg_sync[d] = c

            # test that async classify produces the same results
            # -- p1
            async_p1 = classifier.classify_async(d_p1, c_factory)
            ntools.assert_equal(async_p1, d_p1_sync,
                                "Async computation of p1 set did not yield "
                                "the same results as synchronous computation.")
            # -- p2
            async_p2 = classifier.classify_async(d_p2, c_factory)
            ntools.assert_equal(async_p2, d_p2_sync,
                                "Async computation of p2 set did not yield "
                                "the same results as synchronous computation.")
            # -- neg
            async_neg = classifier.classify_async(d_neg, c_factory)
            ntools.assert_equal(async_neg, d_neg_sync,
                                "Async computation of neg set did not yield "
                                "the same results as synchronous computation.")
            # -- combined
            sync_combined = dict(d_p1_sync.iteritems())
            sync_combined.update(d_p2_sync)
            sync_combined.update(d_neg_sync)
            async_combined = classifier.classify_async(d_p1 + d_p2 + d_neg,
                                                       c_factory)
            ntools.assert_equal(async_combined, sync_combined,
                                "Async computation of all test descriptors "
                                "did not yield the same results as "
                                "synchronous classification.")
            # -- combined -- different order
            async_combined = classifier.classify_async(d_p2 + d_neg + d_p1,
                                                       c_factory)
            ntools.assert_equal(async_combined, sync_combined,
                                "Async computation of all test descriptors "
                                "(mixed order) did not yield the same results "
                                "as synchronous classification.")

            # Closing resources
            p.close()
            p.join()
