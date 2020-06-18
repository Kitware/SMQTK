from smqtk.algorithms.classifier import Classifier, SupervisedClassifier


STUB_CLASSIFIER_MOD_PATH = __name__


class DummyClassifier (Classifier):
    def get_config(self):
        """
        Return a JSON-compliant dictionary that could be passed to this
        class's ``from_config`` method to produce an instance with identical
        configuration.

        In the common case, this involves naming the keys of the dictionary
        based on the initialization argument names as if it were to be passed
        to the constructor via dictionary expansion.

        :return: JSON type compliant configuration dictionary.
        :rtype: dict

        """
        # No constructor, no config
        return {}

    @classmethod
    def is_usable(cls):
        """
        Check whether this class is available for use.

        Since certain plugin implementations may require additional
        dependencies that may not yet be available on the system, this method
        should check for those dependencies and return a boolean saying if the
        implementation is usable.

        NOTES:
            - This should be a class method
            - When an implementation is deemed not usable, this should emit a
                warning detailing why the implementation is not available for
                use.

        :return: Boolean determination of whether this implementation is
                 usable.
        :rtype: bool

        """
        return True

    def _classify_arrays(self, array_iter):
        for _ in array_iter:
            yield {
                'negative': 0.5,
                'positive': 0.5,
            }

    def get_labels(self):
        """
        Get the sequence of class labels that this classifier can classify
        descriptors into. This includes the negative label.

        :return: Sequence of possible classifier labels.
        :rtype: collections.abc.Sequence[collections.abc.Hashable]

        :raises RuntimeError: No model loaded.

        """
        return ['negative', 'positive']


class DummySupervisedClassifier (SupervisedClassifier):
    """
    Supervise classifier stub implementation.
    """

    @classmethod
    def is_usable(cls):
        return True

    def get_config(self):
        pass

    def has_model(self):
        pass

    def _train(self, class_examples, **extra_params):
        pass

    def get_labels(self):
        pass

    def _classify_arrays(self, array_iter):
        pass
