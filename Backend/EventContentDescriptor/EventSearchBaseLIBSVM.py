"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import event_agent_generator as eag


class EventSearchBaseLIBSVM ():
    """
    Process specifically responsible for executing search engine queries.
    """

    NAME = "BaseEventSearchProcess"

    def __init__(self, filepath_model):
        """
        Create a new search base classifier

        :param filepath_model: filepath of the model container file
        """

        self._filepath_model = filepath_model

        if filepath_model.find('approx') >= 0:
            use_approx_model = True
        else:
            use_approx_model = False
        self._use_approx_model = use_approx_model

        self._model = eag.load_ECD_model_libSVM(filepath_model, use_approx_model)

    @property
    def filepath_model(self):
        return self._filepath_model

    @property
    def svm_model(self):
        """
        @return: pointer for the SVM model
        """
        return self._model['svm_model']

    @property
    def is_approx_model(self):
        """
        @return: bool, whether the model is approximate model
        """
        return self._use_approx_model

    @property
    def target_class(self):
        """
        @return: target class label
        """
        return self._model['target_class']

    @property
    def SVs(self):
        """
        @return: numpy 2D matrix of support vectors
        """
        return self._model['SVs']

    @property
    def str_kernel(self):
        """
        @return: name of the kernel
        """
        return self._model['str_kernel']

    def search(self, clipids, features, preprocess=True, report_margins_recounting=False):
        """ Compute scores for clips

        @param clipids: list of clipids
        @param features: features for clipids
        @type features: row-wise 2D numpy.array. i-th row in features array corresponds to i-th clipid
        @param preprocess: preprocess features based on kernel type
        @return: dictionary with 'probs','margins', (optional)'margins_recounting' which are each numpy array of floats
        """

        outputs = eag.apply_ECD_model_libSVM(self._model, features, self.is_approx_model,
                                             preprocess=preprocess,
                                             report_margins_recounting=report_margins_recounting)

        return outputs





