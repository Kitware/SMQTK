import abc
import logging
import multiprocessing.pool
import os
import traceback

from smqtk.algorithms import SmqtkAlgorithm
from smqtk.utils.plugin import get_plugins


__author__ = 'paul.tunison@kitware.com, jacob.becker@kitware.com'


class Classifier (SmqtkAlgorithm):
    """
    Interface for algorithms that classify input descriptors into discrete
    labels and/or label confidences.
    """

    def classify(self, d, factory, overwrite=False):
        """
        Classify the input descriptor against one or more discrete labels,
        outputting a ClassificationElement containing the classification result.


        We return confidence values for each label the configured model
        contains. Implementations may act in a discrete manner whereby only one
        label is marked with a ``1`` value (others being ``0``), or in a
        continuous manner whereby each label is given a confidence-like value in
        the [0, 1] range.

        :param d: Input descriptor to classify
        :type d: smqtk.representation.DescriptorElement

        :param factory: Classification element factory
        :type factory: smqtk.representation.ClassificationElementFactory

        :param overwrite: Recompute classification of the input descriptor and
            set the results to the ClassificationElement produced by the
            factory.
        :type overwrite: bool

        :raises RuntimeError: Could not perform classification for some reason
            (see message).

        :return: Classification result element
        :rtype: smqtk.representation.ClassificationElement

        """
        c_elem = factory.new_classification(self.name, d.uuid())
        if overwrite or not c_elem.has_classifications():
            c = self._classify(d)
            c_elem.set_classification(c)
        else:
            self._log.debug("Found existing classification in generated "
                            "element")

        return c_elem

    def classify_async(self, d_iter, factory, overwrite=False, procs=None):
        """
        Asynchronously classify the DescriptorElements in the given iterable.

        :param d_iter: Iterable of DescriptorElements
        :type d_iter:
            collections.Iterable[smqtk.representation.DescriptorElement]

        :param factory: Classifier element factory to use for element generation
        :type factory: smqtk.representation.ClassificationElementFactory

        :param overwrite:
        :type overwrite:

        :param procs:
        :type procs:

        :return: Mapping of input DescriptorElement instances to the computed
            ClassificationElement. ClassificationElement UUID's are congruent
            with the UUID of the DescriptorElement
        :rtype:

        """
        self._log.info("Async classifying descriptors")

        # Mapping of DataElement to async processing result
        ar_map = {}
        # Mapping of DescriptorElement to its associated ClassificationElement
        #: :type: dict[smqtk.representation.DescriptorElement, smqtk.representation.ClassificationElement]
        d2c_map = {}

        self._log.info("Queueing async work")
        procs = procs and int(procs)
        pool = multiprocessing.pool.ThreadPool(procs)
        for d in d_iter:
            d2c_map[d] = factory.new_classification(self.name, d.uuid())
            if overwrite or not d2c_map[d].has_classifications():
                ar_map[d] = pool.apply_async(_async_helper, args=(self, d))
        pool.close()

        self._log.info("Collecting results")
        failures = False
        p_T = 0.0
        p_inc = 0.1
        for i, (d, ar) in enumerate(ar_map.iteritems()):
            c = ar.get()
            if c is None:
                failures = True
                continue
            else:
                d2c_map[d].set_classification(c)

            # progress reporting
            p = float(i+1) / len(ar_map)
            if p >= p_T:
                self._log.debug("Progress: [%d/%d] %3.3f%%",
                                i+1, len(ar_map),
                                float(i + 1) / (len(ar_map)) * 100)
                p_T += p_inc
        pool.join()

        if failures:
            raise RuntimeError("Failure occurred during descriptor "
                               "classification. See logging.")

        return d2c_map

    #
    # Abstract methods
    #

    @abc.abstractmethod
    def get_labels(self):
        """
        Get the sequence of class labels that this classifier can classify
        descriptors into..

        :return: Sequence of possible classifier labels.
        :rtype: collections.Sequence[collections.Hashable]

        :raises RuntimeError: No model loaded.

        """

    @abc.abstractmethod
    def _classify(self, d):
        """
        Internal method that defines thh generation of the classification map
        for a given DescriptorElement. This returns a dictionary mapping
        integer labels to a floating point value.

        :param d: DescriptorElement containing the vector to classify.
        :type d: smqtk.representation.DescriptorElement

        :raises RuntimeError: Could not perform classification for some reason
            (see message).

        :return: Dictionary mapping trained labels to classification confidence
            values
        :rtype: dict[collections.Hashable, float]

        """


def _async_helper(c_inst, data):
    """
    Helper method for asynchronously producing a descriptor vector.

    :param data: Data to generate feature over
    :type data: smqtk.representation.DataElement

    :param c_inst: Classifier algorithm instance
    :type c_inst: Classifier

    :return: UID and associated feature vector
    :rtype: numpy.core.multiarray.ndarray or None
    """
    log = logging.getLogger("_async_feature_generator_helper")
    try:
        # noinspection PyProtectedMember
        return c_inst._classify(data)
    except Exception, ex:
        log.error("[%s] Failed feature generation\n"
                  "Error: %s\n"
                  "Traceback:\n"
                  "%s",
                  data, str(ex), traceback.format_exc())
        return None


def get_classifier_impls(reload_modules=False):
    """
    Discover and return discovered ``Classifier`` classes. Keys in the returned
    map are the names of the discovered classes, and the paired values are the
    actual class type objects.

    We search for implementation classes in:
        - modules next to this file this function is defined in (ones that begin
          with an alphanumeric character),
        - python modules listed in the environment variable ``CLASSIFIER_PATH``
            - This variable should contain a sequence of python module
              specifications, separated by the platform specific PATH separator
              character (``;`` for Windows, ``:`` for unix)

    Within a module we first look for a helper variable by the name
    ``CLASSIFIER_CLASS``, which can either be a single class object or an
    iterable of class objects, to be specifically exported. If the variable is
    set to None, we skip that module and do not import anything. If the variable
    is not present, we look at attributes defined in that module for classes
    that descend from the given base class type. If none of the above are found,
    or if an exception occurs, the module is skipped.

    :param reload_modules: Explicitly reload discovered modules from source.
    :type reload_modules: bool

    :return: Map of discovered class object of type ``Classifier``
        whose keys are the string names of the classes.
    :rtype: dict[str, type]

    """
    this_dir = os.path.abspath(os.path.dirname(__file__))
    env_var = "CLASSIFIER_PATH"
    helper_var = "CLASSIFIER_CLASS"
    return get_plugins(__name__, this_dir, env_var, helper_var, Classifier,
                       reload_modules=reload_modules)
