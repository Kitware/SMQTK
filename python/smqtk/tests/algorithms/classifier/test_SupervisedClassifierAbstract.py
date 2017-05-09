import unittest

import nose.tools

from smqtk.algorithms.classifier import SupervisedClassifier


class DummySupervisedClassifier (SupervisedClassifier):

    EXPECTED_LABELS = ['constant']
    EXPECTED_HAS_MODEL = False

    @classmethod
    def is_usable(cls):
        return True

    def get_config(self):
        return {}

    def get_labels(self):
        return self.EXPECTED_LABELS

    def _classify(self, d):
        return {d.uuid(): d.vector().tolist()}

    def has_model(self):
        return self.EXPECTED_HAS_MODEL

    def train(self, class_examples=None, **kwds):
        # Return super-method result in its attempt to unify mappings
        class_examples = \
            super(DummySupervisedClassifier, self).train(class_examples, **kwds)
        return class_examples


class TestSupervisedClassifierAbstractClass (unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_classifier = DummySupervisedClassifier()

    def test_train_hasModel(self):
        # Calling the train method should fail the class also reports that it
        # already has a model. Shouldn't matter what is passed to the method (or
        # lack of things passed to the method).
        self.test_classifier.EXPECTED_HAS_MODEL = True
        nose.tools.assert_raises(
            RuntimeError,
            self.test_classifier.train
        )

    #
    # Testing train abstract function functionality. Method currently does not
    # care what the value for labels are.
    #

    def test_train_noModel_noExamples(self):
        self.test_classifier.EXPECTED_HAS_MODEL = False
        nose.tools.assert_raises(
            ValueError,
            self.test_classifier.train
        )

    def test_train_noModel_oneExample_classExamples(self):
        self.test_classifier.EXPECTED_HAS_MODEL = False
        input_class_examples = {
            'label_1': [0, 1, 2],
        }
        nose.tools.assert_raises(
            ValueError,
            self.test_classifier.train, input_class_examples
        )

    def test_train_noModel_oneExample_kwargs(self):
        self.test_classifier.EXPECTED_HAS_MODEL = False
        nose.tools.assert_raises(
            ValueError,
            self.test_classifier.train, label_1=[0, 1]
        )

    def test_train_noModel_classExamples_only(self):
        self.test_classifier.EXPECTED_HAS_MODEL = False
        input_class_examples = {
            'label_1': [0, 1, 2, 3],
            'label_2': [3, 4, 5, 6],
        }
        m = self.test_classifier.train(input_class_examples)
        nose.tools.assert_equal(m, input_class_examples)

    def test_train_noModel_kwargs_only(self):
        self.test_classifier.EXPECTED_HAS_MODEL = False

        e = {
            'label_1': [0, 1, 2, 3, 4],
            'label_2': [3, 4, 5, 6, 7],
        }

        m = self.test_classifier.train(label_1=e['label_1'], label_2=e['label_2'])
        nose.tools.assert_equal(m, e)

    def test_train_noModel_combined(self):
        self.test_classifier.EXPECTED_HAS_MODEL = False

        expected = {
            'label_1': [0, 1, 2, 3, 4],
            'label_2': [3, 4, 5, 6, 7],
            'label_3': [8, 9, 10, 11],
            'special symbolLabel +here': [5, 1, 76, 8, 9, 2, 5],
        }

        class_examples = {
            'label_1': expected['label_1'],
            'special symbolLabel +here': expected['special symbolLabel +here'],
        }
        label_2 = expected['label_2']
        label_3 = expected['label_3']
        m = self.test_classifier.train(class_examples, label_2=label_2,
                                       label_3=label_3)
        nose.tools.assert_equal(m, expected)
