import abc

from smqtk.algorithms import SmqtkAlgorithm
from smqtk.utils import (
    bin_utils,
    parallel,
)

from ._defaults import DFLT_CLASSIFIER_FACTORY


class Classifier (SmqtkAlgorithm):
    """
    Interface for algorithms that classify input descriptors into discrete
    labels and/or label confidences.
    """

    def classify(self, d, factory=DFLT_CLASSIFIER_FACTORY, overwrite=False):
        """
        Classify the input descriptor against one or more discrete labels,
        outputting a ClassificationElement containing the classification
        result.

        We return confidence values for each label the configured model
        contains. Implementations may act in a discrete manner whereby only one
        label is marked with a ``1`` value (others being ``0``), or in a
        continuous manner whereby each label is given a confidence-like value
        in the [0, 1] range.

        The returned ``ClassificationElement`` will have the same UUID as the
        input ``DescriptorElement``.

        :param d: Input descriptor to classify
        :type d: smqtk.representation.DescriptorElement

        :param factory: Classification element factory. The default factory
            yields MemoryClassificationElement instances.
        :type factory: smqtk.representation.ClassificationElementFactory

        :param overwrite: Recompute classification of the input descriptor and
            set the results to the ClassificationElement produced by the
            factory.
        :type overwrite: bool

        :raises ValueError: The given descriptor element did not have a vector
            to operate on.
        :raises RuntimeError: Could not perform classification for some reason
            (see message in raised exception).

        :return: Classification result element
        :rtype: smqtk.representation.ClassificationElement

        """
        if not d.has_vector():
            raise ValueError("Given DescriptorElement does not contain a "
                             "vector to classify.")
        c_elem = factory.new_classification(self.name, d.uuid())
        if overwrite or not c_elem.has_classifications():
            c = self._classify(d)
            c_elem.set_classification(c)

        return c_elem

    def classify_async(self, d_iter, factory=DFLT_CLASSIFIER_FACTORY,
                       overwrite=False, procs=None, use_multiprocessing=False,
                       ri=None):
        """
        Asynchronously classify the DescriptorElements in the given iterable.

        :param d_iter: Iterable of DescriptorElements
        :type d_iter:
            collections.Iterable[smqtk.representation.DescriptorElement]

        :param factory: Classifier element factory to use for element
            generation. The default factory yields MemoryClassificationElement
            instances.
        :type factory: smqtk.representation.ClassificationElementFactory

        :param overwrite: Recompute classification of the input descriptor and
            set the results to the ClassificationElement produced by the
            factory.
        :type overwrite: bool

        :param procs: Explicit number of cores/thread/processes to use.
        :type procs: None | int

        :param use_multiprocessing: Use multiprocessing instead of threading.
        :type use_multiprocessing: bool

        :param ri: Progress reporting interval in seconds. Set to a value > 0 to
            enable. Disabled by default.
        :type ri: float | None

        :return: Mapping of input DescriptorElement instances to the computed
            ClassificationElement. ClassificationElement UUID's are congruent
            with the UUID of the DescriptorElement
        :rtype: dict[smqtk.representation.DescriptorElement,
                     smqtk.representation.ClassificationElement]

        """
        self._log.debug("Async classifying descriptors")
        ri = ri and ri > 0 and ri

        def work(d_elem):
            return d_elem, self.classify(d_elem, factory, overwrite)

        classifications = parallel.parallel_map(
            work, d_iter,
            cores=procs,
            ordered=False,
            use_multiprocessing=use_multiprocessing,
        )

        r_state = [0] * 7
        if ri:
            r_progress = bin_utils.report_progress
        else:
            def r_progress(*_):
                return

        d2c_map = {}
        for d, c in classifications:
            d2c_map[d] = c

            r_progress(self._log.debug, r_state, ri)

        return d2c_map

    #
    # Abstract methods
    #

    @abc.abstractmethod
    def get_labels(self):
        """
        Get the sequence of class labels that this classifier can classify
        descriptors into. This includes the negative label.

        :return: Sequence of possible classifier labels.
        :rtype: collections.Sequence[collections.Hashable]

        :raises RuntimeError: No model loaded.

        """

    @abc.abstractmethod
    def _classify(self, d):
        """
        Internal method that constructs the label-to-confidence map (dict) for
        a given DescriptorElement.

        The passed descriptor element is guaranteed to have a vector to extract.
        It is not extracted yet due to the philosophy of waiting until the
        vector is immediately needed. This moment is thus determined by the
        implementing algorithm.

        :param d: DescriptorElement containing the vector to classify.
        :type d: smqtk.representation.DescriptorElement

        :raises RuntimeError: Could not perform classification for some reason
            (see message in raised exception).

        :return: Dictionary mapping trained labels to classification confidence
            values
        :rtype: dict[collections.Hashable, float]

        """
