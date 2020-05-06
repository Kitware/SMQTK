from __future__ import (absolute_import, division, print_function)

import unittest

import unittest.mock as mock
import multiprocessing
import multiprocessing.pool
import numpy
import pytest
import six
from six.moves import cPickle, zip

from smqtk.algorithms.classifier import Classifier
from smqtk.algorithms.classifier.libsvm import LibSvmClassifier
from smqtk.representation import \
    ClassificationElementFactory, \
    DescriptorElementFactory
from smqtk.representation.classification_element.memory import \
    MemoryClassificationElement
from smqtk.representation.descriptor_element.local_elements import \
    DescriptorMemoryElement
from smqtk.utils.configuration import configuration_test_helper


@pytest.mark.skipif(not LibSvmClassifier.is_usable(),
                    reason="LibSvmClassifier does not report as usable.")
class TestLibSvmClassifier (unittest.TestCase):

    def test_impl_findable(self):
        self.assertIn(LibSvmClassifier, Classifier.get_impls())

    @mock.patch('smqtk.algorithms.classifier.libsvm.LibSvmClassifier'
                '._reload_model')
    def test_configuration(self, m_inst_load_model):
        """ Test configuration handling for this implementation.

        Mocking out model loading when given URIs if they happen to point to
        something.
        """
        ex_model_uri = 'some model uri'
        ex_labelmap_uri = 'some label map uri'
        ex_trainparams = {'-s': 8,  '-t': -10,  '-b': 42,  '-c': 7.2}
        ex_normalize = 2

        c = LibSvmClassifier(ex_model_uri, ex_labelmap_uri,
                             train_params=ex_trainparams,
                             normalize=ex_normalize)
        for inst in configuration_test_helper(c):  # type: LibSvmClassifier
            assert inst.svm_model_uri == ex_model_uri
            assert inst.svm_label_map_uri == ex_labelmap_uri
            assert inst.train_params == ex_trainparams
            assert inst.normalize == ex_normalize

    def test_no_save_model_pickle(self):
        # Test model preservation across pickling even without model cache
        # file paths set.
        classifier = LibSvmClassifier(
            train_params={
                '-t': 0,  # linear kernel
                '-b': 1,  # enable probability estimates
                '-c': 2,  # SVM-C parameter C
                '-q': '',  # quite mode
            },
            normalize=None,  # DO NOT normalize descriptors
        )
        self.assertTrue(classifier.svm_model is None)
        # Empty model should not trigger __LOCAL__ content in pickle
        self.assertNotIn('__LOCAL__', classifier.__getstate__())
        _ = cPickle.loads(cPickle.dumps(classifier))

        # train arbitrary model (same as ``test_simple_classification``)
        DIM = 2
        N = 1000
        POS_LABEL = 'positive'
        NEG_LABEL = 'negative'
        d_factory = DescriptorElementFactory(DescriptorMemoryElement, {})
        c_factory = ClassificationElementFactory(
            MemoryClassificationElement, {}
        )

        def make_element(iv):
            i, v = iv
            d = d_factory.new_descriptor('test', i)
            d.set_vector(v)
            return d

        # Constructing artificial descriptors
        x = numpy.random.rand(N, DIM)
        x_pos = x[x[:, 1] <= 0.45]
        x_neg = x[x[:, 1] >= 0.55]
        p = multiprocessing.pool.ThreadPool()
        d_pos = p.map(make_element, enumerate(x_pos))
        d_neg = p.map(make_element, enumerate(x_neg, start=N//2))
        p.close()
        p.join()

        # Training
        classifier.train({POS_LABEL: d_pos, NEG_LABEL: d_neg})

        # Test original classifier
        # - Using classification method implemented by the subclass directly
        #   in order to test simplest scope possible.
        t_v = numpy.random.rand(DIM)
        c_expected = list(classifier._classify_arrays([t_v]))[0]

        # Should see __LOCAL__ content in pickle state now
        p_state = classifier.__getstate__()
        self.assertIn('__LOCAL__', p_state)
        self.assertIn('__LOCAL_LABELS__', p_state)
        self.assertIn('__LOCAL_MODEL__', p_state)
        self.assertTrue(len(p_state['__LOCAL_LABELS__']) > 0)
        self.assertTrue(len(p_state['__LOCAL_MODEL__']) > 0)

        # Restored classifier should classify the same test descriptor the
        # same
        #: :type: LibSvmClassifier
        classifier2 = cPickle.loads(cPickle.dumps(classifier))
        c_post_pickle = list(classifier2._classify_arrays([t_v]))[0]
        # There may be floating point error, so extract actual confidence
        # values and check post round
        c_pp_positive = c_post_pickle[POS_LABEL]
        c_pp_negative = c_post_pickle[NEG_LABEL]
        c_e_positive = c_expected[POS_LABEL]
        c_e_negative = c_expected[NEG_LABEL]
        self.assertAlmostEqual(c_e_positive, c_pp_positive, 5)
        self.assertAlmostEqual(c_e_negative, c_pp_negative, 5)

    def test_simple_classification(self):
        """
        simple LibSvmClassifier test - 2-class

        Test libSVM classification functionality using random constructed
        data, training the y=0.5 split
        """
        DIM = 2
        N = 1000
        POS_LABEL = 'positive'
        NEG_LABEL = 'negative'
        p = multiprocessing.pool.ThreadPool()
        d_factory = DescriptorElementFactory(DescriptorMemoryElement, {})

        def make_element(iv):
            i, v = iv
            elem = d_factory.new_descriptor('test', i)
            elem.set_vector(v)
            return elem

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
        classifier.train({POS_LABEL: d_pos, NEG_LABEL: d_neg})

        # Test classifier
        x = numpy.random.rand(N, DIM)
        x_pos = x[x[:, 1] <= 0.45]
        x_neg = x[x[:, 1] >= 0.55]

        # Test that examples expected to classify to the positive class are,
        # and same for those expected to be in the negative class.
        c_map_pos = list(classifier._classify_arrays(x_pos))
        for v, c_map in zip(x_pos, c_map_pos):
            assert c_map[POS_LABEL] > c_map[NEG_LABEL], \
                "Found False positive: {} :: {}" \
                .format(v, c_map)

        c_map_neg = list(classifier._classify_arrays(x_neg))
        for v, c_map in zip(x_neg, c_map_neg):
            assert c_map[NEG_LABEL] > c_map[POS_LABEL], \
                "Found False negative: {} :: {}" \
                .format(v, c_map)

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
        P3_LABEL = 'p3'
        p = multiprocessing.pool.ThreadPool()
        d_factory = DescriptorElementFactory(DescriptorMemoryElement, {})
        di = 0

        def make_element(iv):
            i, v = iv
            elem = d_factory.new_descriptor('test', i)
            elem.set_vector(v)
            return elem

        # Constructing artificial descriptors
        x = numpy.random.rand(N, DIM)
        x_p1 = x[x[:, 1] <= 0.30]
        x_p2 = x[(x[:, 1] >= 0.36) & (x[:, 1] <= 0.63)]
        x_p3 = x[x[:, 1] >= 0.69]

        d_p1 = p.map(make_element, enumerate(x_p1, di))
        di += len(d_p1)
        d_p2 = p.map(make_element, enumerate(x_p2, di))
        di += len(d_p2)
        d_p3 = p.map(make_element, enumerate(x_p3, di))
        di += len(d_p3)

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
        classifier.train({P1_LABEL: d_p1, P2_LABEL: d_p2, P3_LABEL: d_p3})

        # Test classifier
        x = numpy.random.rand(N, DIM)
        x_p1 = x[x[:, 1] <= 0.30]
        x_p2 = x[(x[:, 1] >= 0.36) & (x[:, 1] <= 0.63)]
        x_p3 = x[x[:, 1] >= 0.69]

        # Test that examples expected to classify to certain classes are.
        c_map_p1 = list(classifier._classify_arrays(x_p1))
        for v, c_map in zip(x_p1, c_map_p1):
            assert c_map[P1_LABEL] > max(c_map[P2_LABEL], c_map[P3_LABEL]), \
                "Incorrect {} label: {} :: {}".format(P1_LABEL, v, c_map)

        c_map_p2 = list(classifier._classify_arrays(x_p2))
        for v, c_map in zip(x_p2, c_map_p2):
            assert c_map[P2_LABEL] > max(c_map[P1_LABEL], c_map[P3_LABEL]), \
                "Incorrect {} label: {} :: {}".format(P2_LABEL, v, c_map)

        c_map_p3 = list(classifier._classify_arrays(x_p3))
        for v, c_map in zip(x_p3, c_map_p3):
            assert c_map[P3_LABEL] > max(c_map[P1_LABEL], c_map[P2_LABEL]), \
                "Incorrect {} label: {} :: {}".format(P3_LABEL, v, c_map)

        # Closing resources
        p.close()
        p.join()
