import collections
from copy import deepcopy
import logging
import os

import numpy.linalg

from smqtk.algorithms import Classifier
from smqtk.representation.descriptor_element import elements_to_matrix

try:
    import svm
    import svmutil
except ImportError:
    svm = None
    svmutil = None


__author__ = "paul.tunison@kitware.com"


class LibSvmClassifier (Classifier):
    """
    Classifier that uses libSVM for support-vector machine functionality.
    """

    @classmethod
    def is_usable(cls):
        """
        Check whether this class is available for use.

        This implementation required the libSVM python bindings to be installed
        and loadable.

        :return: Boolean determination of whether this implementation is usable.
        :rtype: bool

        """
        return None not in {svm, svmutil}

    # noinspection PyDefaultArgument
    def __init__(self,
                 svm_model_fp=None,
                 train_params={
                     '-s': 0,  # C-SVC, assumed default if not provided
                     '-t': 0,  # linear kernel
                     '-b': 1,  # enable probability estimates
                     '-c': 2,  # SVM parameter C
                     '-g': 0.0078125,  # initial gamma (1 / 128)
                 },
                 train_vector_norm_ord=2,
                 ):
        """
        Initialize the classifier with an empty or existing model.

        Model file paths are optional. If they are given and the file(s) exist,
        we will load them. If they do not, we treat the path(s) as the output
        path(s) for saving a model after calling ``train``. If this is None
        (default), no model is loaded nor output via training, thus any model
        trained will only exist in memory during the lifetime of this instance.

        :param svm_model_fp: Path to the libSVM model file.
        :type svm_model_fp: None | str

        :param train_params: SVM parameters used for training. See libSVM
            documentation for parameter flags and values.
        :type train_params: dict[str, int|float]

        :param train_vector_norm_ord: Vector normalization level (1 = L1,
            2 = L2, etc.)
        :type train_vector_norm_ord: int

        """
        super(LibSvmClassifier, self).__init__()

        self.svm_model_fp = svm_model_fp
        self.train_params = train_params
        self.train_vector_norm_ord = train_vector_norm_ord

        # generated parameters
        self.svm_model = None

        if self.svm_model_fp and os.path.isfile(self.svm_model_fp):
            self.svm_model = svmutil.svm_load_model(self.svm_model_fp)

    @staticmethod
    def _gen_param_string(params):
        """
        Make a single string out of a parameters dictionary
        """
        return ' '.join((str(k)+' '+str(v) for k, v in params.iteritems()))

    def get_config(self):
        """
        Return a JSON-compliant dictionary that could be passed to this class's
        ``from_config`` method to produce an instance with identical
        configuration.

        :return: JSON type compliant configuration dictionary.
        :rtype: dict

        """
        return {
            "svm_model_fp": self.svm_model_fp,
            "train_params": self.train_params,
            "train_vector_norm_ord": self.train_vector_norm_ord,
        }

    def train(self, *class_descriptors, **kwds):
        """
        Train the SVM classifier model.

        Positional arguments to this method should be iterables of
        DescriptorElement instances. Each iterable is assigned an integer label
        starting with ``0``.

        Negative examples are required for SVM training. Unless the
        ``negatives`` keyword argument is provided, we assume the last
        positional argument is the iterable of negative examples. Negative
        examples  are given the ``-1`` label.

        If this instance was constructed with a model filepath, the trained
        model will be saved to that path. If a model is already loaded, we will
        raise an exception in order to prevent accidental overwrite.

        :param class_descriptors: Sequence of DescriptorElement iterables for
            each class to train for, including the negative example iterable if
            not provided via the ``negatives`` keyword argument
        :type class_descriptors: collections.Sequence[
                                    collections.Iterable[
                                        smqtk.representation.DescriptorElement]]

        :raises RuntimeError: A model file path was configured and has been
            loaded. Following through with training would overwrite this model
            permanently on disk.

        """
        if os.path.isfile(self.svm_model_fp or '') and self.svm_model:
            raise RuntimeError("Halting training to prevent overwrite of "
                               "existing trained model @ %s", self.svm_model_fp)

        NEGATIVES = 'negatives'

        # stuff for debug reporting
        etm_ri = None
        param_debug = {'-q': ''}
        if self._log.getEffectiveLevel() <= logging.DEBUG:
            etm_ri = 1.0
            param_debug = {}

        # Collect class and negative groups
        if NEGATIVES in kwds:
            negatives = kwds[NEGATIVES]
        else:
            negatives = class_descriptors[-1]
            class_descriptors = class_descriptors[:-1]

        # Form libSVM problem input
        self._log.debug("Formatting problem input")
        train_labels = []
        train_vectors = []
        train_group_sizes = []
        for i, g in enumerate(class_descriptors):
            self._log.debug('-- class %d', i)
            # requires a sequence, so making the iterable ``g`` a tuple
            if not isinstance(g, collections.Sequence):
                g = tuple(g)
            train_group_sizes.append(len(g))
            x = elements_to_matrix(g, report_interval=etm_ri)
            # L2 normalize each descriptor before training
            # noinspection PyTypeChecker
            n = numpy.linalg.norm(x, self.train_vector_norm_ord, axis=1)
            n[n == 0] = 1.  # replace 0's with 1's, preventing div-by-zero
            x /= n.reshape((n.size, 1))  # reshape acting as transpose

            train_labels.extend([i]*x.shape[0])
            train_vectors.extend(x.tolist())

            del g

        self._log.debug('-- negatives (-1)')
        if not isinstance(negatives, collections.Sequence):
            negatives = tuple(negatives)
        x = elements_to_matrix(negatives, report_interval=etm_ri)
        # noinspection PyTypeChecker
        n = numpy.linalg.norm(x, self.train_vector_norm_ord, axis=1)
        n[n == 0] = 1.  # replace 0's with 1's, preventing div-by-zero
        x /= n.reshape((n.size, 1))  # reshape acting as transpose
        train_labels.extend([-1]*x.shape[0])
        train_vectors.extend(x.tolist())
        del x, n

        self._log.debug("Forming train params")
        #: :type: dict
        params = deepcopy(self.train_params)
        params.update(param_debug)
        # Only need to calculate positive class weights when C-SVC type
        if '-s' not in params or int(params['-s']) == 0:
            for i, n in enumerate(train_group_sizes):
                params['-w'+str(i)] = \
                    max(1.0, len(negatives) / float(n))

        self._log.debug("Making parameters obj")
        svm_params = svmutil.svm_parameter(self._gen_param_string(params))
        self._log.debug("Creating SVM problem")
        svm_problem = svm.svm_problem(train_labels, train_vectors)
        self._log.debug("Training SVM model")
        self.svm_model = svmutil.svm_train(svm_problem, svm_params)

        if self.svm_model_fp:
            svmutil.svm_save_model(self.svm_model_fp, self.svm_model)

    def classify(self, d, factory):
        pass
