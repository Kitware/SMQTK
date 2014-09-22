import EventContentDescriptor.event_agent_generator as eag
import svm
import svmutil
import libsvm_tools as svmtools
import numpy as np

def iqr_model_train(filepath_model,
                    matrix_kernel_train, labels_train, idx2clipid,
                    svm_para = '-w1 50 -t 4 -b 1 -c 1'):
    """
    Light-weighted SVM learning module for online IQR

    @param filepath_model: a full path to save the learned SVM model
    @param matrix_kernel_train: n-by-n square numpy array with kernel values between training data
    @param labels_train: row-wise labels of training data (1 or True indicates positive, 0 or False otherwise
    @param idx2clipid: idx2clipid(row_idx) returns the clipid for the 0-base row in matrix
    @param svm_para: (optional) SVM learning parameter
    @rtype: dictionary with 'clipids_SV': list of clipids for support vectors
    @return: output as a dictionary with 'clipids_SV'
    """

    # set training inputs
    matrix_kernel_train = np.vstack((np.arange(1, len(matrix_kernel_train)+1), matrix_kernel_train)).T
    print "Done matrix_kernel_train"
    problem = svm.svm_problem(labels_train.tolist(), matrix_kernel_train.tolist(), isKernel=True)
    print "Done problem"
    svm_param = svm.svm_parameter(svm_para)
    print "Done svm_param"

    # train model
    model = svmutil.svm_train(problem, svm_param)
    print "Done train model"

    # release memory
    del problem
    del svm_param
    print "Done release memory"

    # check learning failure
    if model.l == 0:
        raise Exception('svm model learning failure')

    n_SVs = model.l
    clipids_SVs = []
    idxs_train_SVs = svmtools.get_SV_idxs_nonlinear_svm(model)
    for i in range(n_SVs):
        _idx_1base = idxs_train_SVs[i]
        _idx_0base = _idx_1base - 1
        clipids_SVs.append(idx2clipid[_idx_0base])
        model.SV[i][0].value = i+1 # within SVM model, index needs to be 1-base

    print "Done checking learning failure"

    svmutil.svm_save_model(filepath_model, model)

    output = dict()
    output['model'] = model
    output['clipids_SVs'] = clipids_SVs

    return output


def iqr_model_test(filepath_model, matrix_kernel_test, clipids_test, target_class=1):
    """
    Apply an SVM model on test data

    @param filepath_model: a full path to load the learned SVM model
    @param matrix_kernel_test: n-by-m kernel maxtrix where n (row) is |SVs| & m (col) is |test data|
    @type matrix_kernel_test: 2D numpy.array
    @param clipids_test: list of clipids ordered
    @param target_class: positive class id. Default = 1.
    @return: dictionary with 'probs','clipids'
    @rtype: dictionary with 'probs' (np.array), 'clipids' (int list)
    """

    model = svmutil.svm_load_model(filepath_model)
    weights = svmtools.get_SV_weights_nonlinear_svm(model, target_class=target_class)

    # compute margins
    margins = weights[0] * matrix_kernel_test[0]
    for i in range(1, matrix_kernel_test.shape[0]):
        margins += weights[i] * matrix_kernel_test[i]
    if matrix_kernel_test.ndim == 1:   # case where there was single test data
        margins = np.array([margins])  # make a single number margin into an np array

    # compute probs, using platt scaling
    rho = model.rho[0]
    probA = model.probA[0]
    probB = model.probB[0]
    probs = 1.0 / (1.0 + np.exp((margins - rho) * probA + probB))
    del margins

    # case when the label of positive data was 2nd in SVM model symbol list
    # since platt scaling was parameterized for negative data, swap probs
    idx_target = svmtools.get_column_idx_for_class(model, target_class)
    if idx_target == 1:
        probs   = 1.0 - probs

    output = dict()
    output['probs'] = probs
    output['clipids'] = clipids_test
    return output









