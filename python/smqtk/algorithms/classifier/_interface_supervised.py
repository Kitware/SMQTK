import abc

from smqtk.utils import merge_dict

from ._interface_classifier import Classifier


class SupervisedClassifier (Classifier):
    """
    Class of classifiers that are trainable via supervised training, i.e. are
    given specific descriptor examples for class labels.
    """

    @abc.abstractmethod
    def has_model(self):
        """
        :return: If this instance currently has a model loaded. If no model is
            present, classification of descriptors cannot happen (needs to be
            trained).
        :rtype: bool
        """

    def train(self, class_examples=None, **kwds):
        """
        Train the supervised classifier model.

        If a model is already loaded, we will raise an exception in order to
        prevent accidental overwrite.

        If the same label is provided to both ``class_examples`` and ``kwds``,
        the examples given to the reference in ``kwds`` will prevail.

        :param class_examples: Dictionary mapping class labels to iterables of
            DescriptorElement training examples.
        :type class_examples: dict[collections.Hashable,
                 collections.Iterable[smqtk.representation.DescriptorElement]]

        :param kwds: Keyword assignment of labels to iterables of
            DescriptorElement training examples. Keyword provided iterables
            are used in place of class iterables provided in ``class_examples``
            when there are conflicting keys.
        :type kwds:
            collections.Iterable[smqtk.representation.DescriptorElement]

        :raises ValueError: There were no class examples provided.
        :raises ValueError: Less than 2 classes were given.
        :raises RuntimeError: A model already exists in this instance.Following
            through with training would overwrite this model. Throwing an
            exception for information protection.

        """
        if self.has_model():
            raise RuntimeError("Instance currently has a model. Halting "
                               "training to prevent overwrite of existing "
                               "trained model.")

        if class_examples is None:
            class_examples = {}

        merged = {}
        merge_dict(merged, class_examples)
        merge_dict(merged, kwds)

        if not merged:
            raise ValueError("No class examples were provided.")
        elif len(merged) < 2:
            raise ValueError("Need 2 or more classes for training. Given %d."
                             % len(merged))

        # TODO(paul.tunison): Check that the same values/descriptors are not
        #   assigned to multiple labels?

        return self._train(merged)

    @abc.abstractmethod
    def _train(self, class_examples):
        """
        Internal method that trains the classifier implementation.

        This method is called after checking that there is not already a model
        trained, thus it can be assumed that no model currently exists.

        The class labels will have already been checked before entering this
        method, so it can be assumed that the ``class_examples`` will container
        at least two classes.

        :param class_examples: Dictionary mapping class labels to iterables of
            DescriptorElement training examples.
        :type class_examples: dict[collections.Hashable,
                 collections.Iterable[smqtk.representation.DescriptorElement]]

        """
