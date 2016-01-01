import cPickle
import collections
from copy import deepcopy
import ctypes
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
    def __init__(self, svm_model_fp=None, svm_label_map_fp=None,
                 train_params={
                     '-s': 0,  # C-SVC, assumed default if not provided
                     '-t': 0,  # linear kernel
                     '-b': 1,  # enable probability estimates
                     '-c': 2,  # SVM parameter C
                     # '-g': 0.0078125,  # initial gamma (1 / 128)
                 },
                 normalize=None,
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

        :param svm_label_map_fp: Path to the pickle file containing this model's
            output labels.
        :type svm_label_map_fp: None | str

        :param train_params: SVM parameters used for training. See libSVM
            documentation for parameter flags and values.
        :type train_params: dict[str, int|float]

        :param normalize: Normalize input vectors to training and
            classification methods using ``numpy.linalg.norm``. This may either
            be  ``None``, disabling normalization, or any valid value that could
            be passed to the ``ord`` parameter in ``numpy.linalg.norm`` for 1D
            arrays. This is ``None`` by default (no normalization).
        :type normalize: None | int | float | str

        """
        super(LibSvmClassifier, self).__init__()

        self.svm_model_fp = svm_model_fp
        self.svm_label_map_fp = svm_label_map_fp
        self.train_params = train_params
        self.normalize = normalize
        # Validate normalization parameter by trying it on a random vector
        if normalize is not None:
            self._norm_vector(numpy.random.rand(8))

        # generated parameters
        #: :type: svm.svm_model
        self.svm_model = None
        # dictionary mapping SVM integer labels to semantic labels
        #: :type: dict[int, collections.Hashable]
        self.svm_label_map = None

        self._reload_model()

    def __getstate__(self):
        return self.get_config()

    def __setstate__(self, state):
        self.svm_model_fp = state['svm_model_fp']
        self.svm_label_map_fp = state['svm_label_map_fp']
        self.train_params = state['train_params']
        self.normalize = state['normalize']

        # C libraries/pointers don't survive across processes.
        self.svm_model = None
        self._reload_model()

    def _reload_model(self):
        """
        Reload SVM model from configured file path.
        """
        if self.svm_model_fp and os.path.isfile(self.svm_model_fp):
            self.svm_model = svmutil.svm_load_model(self.svm_model_fp)
        if self.svm_label_map_fp and os.path.isfile(self.svm_label_map_fp):
            with open(self.svm_label_map_fp, 'rb') as f:
                self.svm_label_map = cPickle.load(f)

    @staticmethod
    def _gen_param_string(params):
        """
        Make a single string out of a parameters dictionary
        """
        return ' '.join((str(k)+' '+str(v) for k, v in params.iteritems()))

    def _norm_vector(self, v):
        """
        Class standard array normalization. Normalized along max dimension (a=0
        for a 1D array, a=1 for a 2D array, etc.).

        :param v: Vector to normalize
        :type v: numpy.ndarray

        :return: Returns the normalized version of input array ``v``.
        :rtype: numpy.ndarray

        """
        if self.normalize is not None:
            n = numpy.linalg.norm(v, self.normalize, v.ndim - 1,
                                  keepdims=True)
            # replace 0's with 1's, preventing div-by-zero
            n[n == 0.] = 1.
            return v / n

        # Normalization off
        return v

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
            "svm_label_map_fp": self.svm_label_map_fp,
            "train_params": self.train_params,
            "normalize": self.normalize,
        }

    def has_model(self):
        """
        :return: If this instance currently has a model loaded. If no model is
            present, classification of descriptors cannot happen.
        :rtype: bool
        """
        return None not in (self.svm_model, self.svm_label_map)

    def train(self, positive_classes, negatives):
        """
        Train the SVM classifier model.

        The class label ``negative`` is reserved for the negative class.

        If this instance was constructed with a model filepaths, the trained
        model and labels will be saved to those paths. If a model is already
        loaded, we will raise an exception in order to prevent accidental
        overwrite.

        :param positive_classes: Dictionary mapping positive class labels to
            iterables of DescriptorElement training examples.
        :type positive_classes:
            dict[collections.Hashable,
                 collections.Iterable[smqtk.representation.DescriptorElement]]

        :raises ValueError: The ``negative`` label was found in the
            ``positive_classes`` dictionary. This is reserved for the negative
            example class.
        :raises RuntimeError: A model file path was configured and has been
            loaded. Following through with training would overwrite this model
            permanently on disk.

        """
        # Offset from 0 for positive class labels to use
        # - not using label of 0 because we think libSVM wants positive labels
        CLASS_LABEL_OFFSET = 1
        NEG_LABEL = "negative"

        if self.has_model():
            raise RuntimeError("Halting training to prevent overwrite of "
                               "existing trained model @ %s", self.svm_model_fp)

        if NEG_LABEL in positive_classes:
            raise ValueError("Found '%s' label in positive_classes map. "
                             "This label is reserved for negative class."
                             % NEG_LABEL)

        # Stuff for debug reporting
        etm_ri = None
        param_debug = {'-q': ''}
        if self._log.getEffectiveLevel() <= logging.DEBUG:
            etm_ri = 1.0
            param_debug = {}

        # Form libSVM problem input values
        self._log.debug("Formatting problem input")
        train_labels = []
        train_vectors = []
        train_group_sizes = []
        self.svm_label_map = {}
        # Making SVM label assignment deterministic to alphabetic order
        for i, l in enumerate(sorted(positive_classes), CLASS_LABEL_OFFSET):
            # Map integer SVM label to semantic label
            self.svm_label_map[i] = l

            self._log.debug('-- class %d (%s)', i, l)
            # requires a sequence, so making the iterable ``g`` a tuple
            g = positive_classes[l]
            if not isinstance(g, collections.Sequence):
                g = tuple(g)

            train_group_sizes.append(float(len(g)))
            x = elements_to_matrix(g, report_interval=etm_ri)
            x = self._norm_vector(x)
            train_labels.extend([i]*x.shape[0])
            train_vectors.extend(x.tolist())
            del g, x

        self._log.debug('-- negatives (-1)')
        # Map integer SVM label to semantic label
        self.svm_label_map[-1] = NEG_LABEL
        # requires a sequence, so making the iterable ``negatives`` a tuple
        if not isinstance(negatives, collections.Sequence):
            negatives = tuple(negatives)
        negatives_size = float(len(negatives))
        x = elements_to_matrix(negatives, report_interval=etm_ri)
        x = self._norm_vector(x)
        train_labels.extend([-1]*x.shape[0])
        train_vectors.extend(x.tolist())
        del negatives, x

        self._log.debug("Training elements: %d labels, %d vectors",
                        len(train_labels), len(train_vectors))

        self._log.debug("Forming train params")
        #: :type: dict
        params = deepcopy(self.train_params)
        params.update(param_debug)
        # Only need to calculate positive class weights when C-SVC type
        if '-s' not in params or int(params['-s']) == 0:
            for i, n in enumerate(train_group_sizes, CLASS_LABEL_OFFSET):
                params['-w'+str(i)] = \
                    max(1.0, negatives_size / float(n))

        self._log.debug("Making parameters obj")
        svm_params = svmutil.svm_parameter(self._gen_param_string(params))
        self._log.debug("Creating SVM problem")
        svm_problem = svm.svm_problem(train_labels, train_vectors)
        self._log.debug("Training SVM model")
        self.svm_model = svmutil.svm_train(svm_problem, svm_params)
        self._log.debug("Training SVM model -- Done")

        if self.svm_label_map_fp:
            self._log.debug("saving file -- labels -- %s", self.svm_label_map_fp)
            with open(self.svm_label_map_fp, 'wb') as f:
                cPickle.dump(self.svm_label_map, f)
        if self.svm_model_fp:
            self._log.debug("saving file -- model -- %s", self.svm_model_fp)
            svmutil.svm_save_model(self.svm_model_fp, self.svm_model)

    def get_labels(self):
        """
        Get the sequence of integer labels that this classifier can classify
        descriptors into. The last label is the negative label.

        :return: Sequence of positive integer labels, and the negative label.
        :rtype: collections.Sequence[int]

        :raises RuntimeError: No model loaded.

        """
        if not self.has_model():
            raise RuntimeError("No model loaded")
        return self.svm_label_map.values()

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
        if not self.has_model():
            raise RuntimeError("No SVM model present for classification")

        # Get and normalize vector
        v = d.vector().astype(float)
        v = self._norm_vector(v)
        v, idx = svm.gen_svm_nodearray(v.tolist())

        # Effectively reproducing the body of svmutil.svm_predict in order to
        # simplify and get around excessive prints
        svm_type = self.svm_model.get_svm_type()
        nr_class = self.svm_model.get_nr_class()
        c = dict((l, 0.) for l in self.get_labels())

        if self.svm_model.is_probability_model():
            if svm_type in [svm.NU_SVR, svm.EPSILON_SVR]:
                nr_class = 0
            prob_estimates = (ctypes.c_double * nr_class)()
            svm.libsvm.svm_predict_probability(self.svm_model, v,
                                               prob_estimates)
            # Update dict
            for l, p in zip(self.svm_model.get_labels(),
                            prob_estimates[:nr_class]):
                c[self.svm_label_map[l]] = p
        else:
            if svm_type in (svm.ONE_CLASS, svm.EPSILON_SVR, svm.NU_SVC):
                nr_classifier = 1
            else:
                nr_classifier = nr_class*(nr_class-1)//2
            dec_values = (ctypes.c_double * nr_classifier)()
            label = svm.libsvm.svm_predict_values(self.svm_model, v, dec_values)
            # Update dict
            c[self.svm_label_map[label]] = 1.

        assert len(c) == len(self.svm_label_map)
        return c
