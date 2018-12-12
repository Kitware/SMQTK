import unittest

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

    def _train(self, class_examples, **extra_params):
        # Return super-method result in its attempt to unify mappings
        return class_examples


class TestSupervisedClassifierAbstractClass (unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_classifier = DummySupervisedClassifier()

    def test_train_hasModel(self):
        # Calling the train method should fail the class also reports that it
        # already has a model. Shouldn't matter what is passed to the method
        # (or lack of things passed to the method).
        self.test_classifier.EXPECTED_HAS_MODEL = True
        self.assertRaises(
            RuntimeError,
            self.test_classifier.train, {}
        )

    #
    # Testing train abstract function functionality. Method currently does not
    # care what the value for labels are.
    #

    def test_train_noModel_noExamples(self):
        self.test_classifier.EXPECTED_HAS_MODEL = False
        self.assertRaises(
            ValueError,
            self.test_classifier.train, {}
        )

    def test_train_noModel_oneExample_classExamples(self):
        self.test_classifier.EXPECTED_HAS_MODEL = False
        input_class_examples = {
            'label_1': [0, 1, 2],
        }
        self.assertRaises(
            ValueError,
            self.test_classifier.train, input_class_examples
        )

    def test_train_noModel_classExamples_only(self):
        self.test_classifier.EXPECTED_HAS_MODEL = False
        input_class_examples = {
            'label_1': [0, 1, 2, 3],
            'label_2': [3, 4, 5, 6],
            'label_3': [8, 9, 10, 11],
            'special symbolLabel +here': [5, 1, 76, 8, 9, 2, 5],
        }
        # Intentionally not passing DescriptorElements here.
        # noinspection PyTypeChecker
        m = self.test_classifier.train(class_examples=input_class_examples)
        self.assertEqual(m, input_class_examples)
