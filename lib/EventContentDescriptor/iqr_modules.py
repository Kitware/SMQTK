import logging
import libsvm_tools as svmtools
import numpy as np
import svm
import svmutil


def iqr_model_train(matrix_kernel_train, labels_train, idx2clipid,
                    svm_para='-w1 50 -t 4 -b 1 -c 1'):
    """
    Light-weighted SVM learning module for online IQR

    @param matrix_kernel_train: n-by-n square numpy array with kernel values
        between training data
    @param labels_train: row-wise labels of training data (1 or True indicates
        positive, 0 or False otherwise
    @param idx2clipid: idx2clipid(row_idx) returns the clipid for the 0-base row
        in matrix
    @param svm_para: (optional) SVM learning parameter

    @rtype: dictionary with 'clipids_SV': list of clipids for support vectors
    @return: output as a dictionary with 'clipids_SV'

    """
    log = logging.getLogger('SMQTK.iqr_model_train')

    # Precomputed kernels need to set serial number to each row of the kernel
    matrix_kernel_train = np.vstack((np.arange(1, len(matrix_kernel_train)+1),
                                     matrix_kernel_train)).T
    problem = svm.svm_problem(labels_train.tolist(),
                              matrix_kernel_train.tolist(),
                              isKernel=True)
    log.debug("Done problem")
    svm_param = svm.svm_parameter(svm_para)
    log.debug("Done svm_param")

    # train model
    model = svmutil.svm_train(problem, svm_param)
    log.debug("Done train model")

    # check learning failure
    if model.l == 0:
        raise Exception('svm model learning failure')
    log.debug("Done checking learning failure")

    n_SVs = model.l
    clipids_SVs = []
    idxs_train_SVs = svmtools.get_SV_idxs_nonlinear_svm(model)
    for i in range(n_SVs):
        _idx_1base = idxs_train_SVs[i]
        _idx_0base = _idx_1base - 1
        clipids_SVs.append(idx2clipid[_idx_0base])
        model.SV[i][0].value = i+1 # within SVM model, index needs to be 1-base
    log.debug("Done collecting support vector IDs")

    output = dict()
    output['model'] = model
    output['clipids_SVs'] = clipids_SVs

    return output


def iqr_model_test(svm_model, matrix_kernel_test, clipids_test, sv_labels,
                   target_class=1):
    """
    Apply an SVM model on test data

    @param svm_model: Loaded SVM model to run with.

    @param matrix_kernel_test: n-by-m kernel matrix where n (row) is
        |SVs| & m (col) is |test data|
    @type matrix_kernel_test: numpy.core.multiarray.ndarray

    @param clipids_test: list of clipids ordered

    @param target_class: positive class id. Default = 1.

    @return: dictionary with 'probs','clipids'
    @rtype: dictionary with 'probs' (np.array), 'clipids' (int list)

    """
    log = logging.getLogger("SMQTK.iqr_model_test")

    weights = svmtools.get_SV_weights_nonlinear_svm(svm_model,
                                                    target_class=target_class)
    # case when the label of positive data was 2nd in SVM model symbol list
    # since platt scaling was parameterized for negative data, swap probs
    idx_target = svmtools.get_column_idx_for_class(svm_model, target_class)
    if idx_target == 1:
        weights *= -1.0

    # compute margins
    margins = weights[0] * matrix_kernel_test[0]
    for i in range(1, matrix_kernel_test.shape[0]):
        margins += weights[i] * matrix_kernel_test[i]
    if matrix_kernel_test.ndim == 1:   # case where there was single test data
        margins = np.array([margins])  # make a single number margin into an np array

    # compute probs, using platt scaling
    log.debug("Computing probabilities via Platt Scaling")
    rho = svm_model.rho[0]
    probA = svm_model.probA[0]
    probB = svm_model.probB[0]
    probs = 1.0 / (1.0 + np.exp((margins - rho) * probA + probB))
    del margins

    # # case when the label of positive data was 2nd in SVM model symbol list
    # # since platt scaling was parameterized for negative data, swap probs
    # idx_target = svmtools.get_column_idx_for_class(svm_model, target_class)
    # if idx_target == 1:
    #     probs = 1.0 - probs

    output = dict()
    output['probs'] = probs
    output['clipids'] = clipids_test
    return output
