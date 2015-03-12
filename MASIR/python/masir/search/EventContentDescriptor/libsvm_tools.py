"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import os
import numpy as np
import csv
import svmutil
import kernels
import math

__all__ = ['load_libsvm_data', 'write_libsvm_kernel_matrix', 'write_libsvm_input',
           'replace_labels_libsvm', 'parse_deva_detection_file', 'get_weight_vector_svm_model_linear']


def load_libsvm_data_help(data_file_name):
    """
    svm_read_problem(data_file_name) -> [y, x]

    Read LIBSVM-format data from data_file_name and return labels y
    and data instances x.
    """
    prob_y = []
    prob_x = []
    for line in open(data_file_name):
        line = line.split(None, 1)
        # In case an instance with all zero features
        if len(line) == 1: line += ['']
        label, features = line
        xi = {}
        for e in features.split():
            ind, val = e.split(":")
            xi[int(ind)] = float(val)
        prob_y += [float(label)]
        prob_x += [xi]
    return (prob_y, prob_x)


def load_libsvm_data(file_libsvm):
    """
    load libsvm basic format (one data per row) into numpy matrices, and also parse labels
    """
    [labels, data_] = load_libsvm_data_help(file_libsvm)
    def convert_datum(datum_in):
        dim = max(datum_in.keys())
        data_out = []
        for d in range(1, int(dim+1)):
            data_out.append(datum_in[d])
        return data_out

    data_converted = map(convert_datum, data_)
    data = np.matrix(data_converted)
    return (labels, data)


def write_libsvm_kernel_matrix(labels, idxs, matrix_np,
                               fileout):
    """
    Given all the information needed to write a kernel matrix in libsvm,
    write it out to a target file
    """

    fout = open(fileout, 'w')

    [n, dim] = matrix_np.shape
    for i in range(n): # every line
        fout.write('%d '%labels[i])
        fout.write('0:%d '%idxs[i])
        for j in range(dim):
            fout.write('%d:%g '%(j+1, matrix_np[i,j]))
        fout.write('\n')

    fout.close()


def write_libsvm_input(labels, data_rowwise, file_out, f_print_labels, f_print_value):
    n_ = len(labels)
    [n,dim] = data_rowwise.shape

    if n_ != n:
        print 'Input error: n_ != n: %d != %d'%(n_,n)
        return

    fout = open(file_out, 'w')
    for i in range(n):
        i_label = labels[i]
        i_str = ''
        i_str += (f_print_labels(i_label) + ' ')
        for j in range(1,1+dim):
            i_str += (str(j) + ':' + f_print_value(data_rowwise[i,j-1]) + ' ')
        fout.write('%s\n'%i_str)

    fout.close()


def retrieve_labels(filein):
    """
    Return only the labels from SVM files
    """

    fin_ = open(filein, 'r')
    fin  = csv.reader(fin_, delimiter=' ')

    labels = []
    for line in fin:
        labels.append(line[0])
    fin_.close()

    return labels

def retrieve_scores_libsvm(filein, label):
    """
    Return the scores of a particular label from a libsvm output file
    """
    fin_ = open(filein, 'r')
    fin  = csv.reader(fin_, delimiter= ' ')

    str_label = str(label)

    line_header = fin.next()
    id_column = None
    for (e, id) in zip(line_header, range(len(line_header))):
        if e == str_label:
            id_column = id
            break;
    scores = []
    for line in fin:
        scores.append(float(line[id_column]))
    fin_.close()

    return scores


def replace_labels_libsvm(labels_new, filein, fileout):
    """
    Only replace the labels of libSVM files, to avoid duplicate computations
    """
    fin_ = open(filein, 'r')
    fin  = csv.reader(fin_, delimiter=' ')
    fout = open(fileout, 'w')

    count = 0
    for line in fin:
        line[0] = str(labels_new[count]) # replace the label to a new one
        fout.write(' '.join(line))
        fout.write('\n')
        count += 1

    fin_.close()
    fout.close()


def write_deva_detection_file(clipids, scores, eid, fileout, write_mode='w'):
    """
    Given all score information, write deva detection file
    """
    fout = open(fileout, write_mode)

    clipids_scores = zip(clipids, scores)

    if write_mode == 'w':
        fout.write('"TrialID","Score"\n')

    for (clipid, score) in clipids_scores:
        lineid = "%06d.E%03d"%(clipid, eid)
        fout.write('"%s","%f"\n'%(lineid, score))

    fout.close()

def parse_deva_detection_file(file_in):
    """
    Read a DEVA detection file, and outputs a CLIPSx3 matrix
    where
    Col 1 : clipid
    Col 2 : target event
    Col 3 : score
    """
    # file = os.path.join('C:', 'aladdin_deva_test\\DEVA_cli_sysdir\\Ilseo-2011-08-19-in\\HoG3D\\DEV1_HoG3D_HIK_full.csv')
    fin_ = open(file_in, 'r')
    fin  = csv.reader(fin_, delimiter=',')
    lines = [line for line in fin]
    lines = lines[1::] # get rid of the first line header (perhaps better way to do this?)

    mat = np.zeros([len(lines), 3])
    count = 0
    for line in lines:
        mat[count][0] = int(line[0][0:6])
        mat[count][1] = int(line[0][-3::])
        mat[count][2] = float(line[1])
        count += 1
    return mat


def convert_SV_to_nparray(SVdict):
    """
    Convert support vector (SV) of libSVM model in dict type, to numpy array
    """
    key_max = max(SVdict.keys())
    nparray = np.zeros(key_max+1)
    for (k,v) in SVdict.iteritems():
        if k >= 0:
            nparray[k] = v
    return nparray



def get_weight_vector_svm_model_linear(file_svm_model_linear, target_class = 1):
    """
    Given a filename for a linear SVM model (in libSVM format),
    return a linear weight vector

    Two-class SVM is assumed, where weights will be translated to represent target_class (default 1)
    """
    svm_model_linear = svmutil.svm_load_model(file_svm_model_linear)
    SVs = svm_model_linear.get_SV() # list of dict
    sv_coef = svm_model_linear.get_sv_coef()
    labels = svm_model_linear.get_labels()

    # Find target class idx
    idx_target = None
    for idx in range(len(labels)):
        if labels[idx] == target_class:
            idx_target = idx
            break
    assert(not(idx_target==None))

    multiplier = 1.0
    if idx_target == 1:
        multiplier *= -1.0

    # weight vector to be returned
    w = None
    for (wi, svi) in zip(sv_coef, SVs):
        vi = convert_SV_to_nparray(svi)
        wvi = wi*vi
        if w==None:
            w = wvi
        else:
            w += wvi
    w *= multiplier

    return w

def get_linear_coeffs_from_svm_model_nonlinear(model_libsvm, SVs, target_class = 1):
    """
    Given non-linear libsvm model and support vectors,
    computed weighted sum of SVs using libsvm weights for the target class

    @param model_libsvm: loaded nonlinear libsvm model
    @param SVs: support vectors in numpy array format (row-wise)
    @param target_class: target class label (default 1)
    @return: weighted sum of SVs using libsvm weights
    """

    # Find target class idx within model
    labels = model_libsvm.get_labels()
    idx_target = None
    for idx in range(len(labels)):
        if labels[idx] == target_class:
            idx_target = idx
            break
    assert(not(idx_target==None))

    # if target class idx is 1 (not 0), then, flip the sign of weighted sum
    multiplier = 1.0
    if idx_target == 1:
        multiplier *= -1.0

    # get SV coefficients
    sv_coefs = model_libsvm.get_sv_coef()

    # compute weighted sum
    wsum = None
    for (wi, svi) in zip(sv_coefs, SVs):
        if wsum == None:
            wsum = wi * svi
        else:
            wsum += (wi * svi)

    wsum *= multiplier

    return wsum


def prepare_precomputed_matrix(data):
    """
    Converts a numpy 2D array data, into a form that can be used as an input for libSVM

    @param data: numpy 2D array with size being *testdata(row)-by-training(col)*, with kernel values
    @return: return a data list which is ready to be directly used as an input for libsvm
    """

    n = data.shape[0]
    indices = np.array([range(1,n+1)]).T
    return (np.concatenate([indices, data], axis=1).tolist())


#def get_precomputed_matrix_from_svm_model(svm_model, func_kernel, data_train, data_test,
#                                          flag_prob=True):
#    """
#    Given a non-linear SVM model (in libSVM format),
#    returns a pre-computed kernel matrix, and optional probabilities.
#    NOTE: this implementation is not memory efficient, although simple. Consider using 'apply_compact_svm'.
#    NOTE: NOT TESTED!!
#
#    @param svm_model: parsed svm_model (via provided API from libSVM)
#    @param func_kernel: func_kernel(x1,x2) computes the kernel value between x1 & x2
#    @param data_train: n-by-d row-wise data matrix
#    @param data_test: m-by-d row-wise data matrix
#    @return: (matrix_kernel, scores) where matrix_kernel is in libSVM list of list format
#    """
#
#    print 'libsvm_tools.get_precomputed_matrix_from_svm_model: \n not tested..! Remove this msg after testing'
#
#    # row ids of support vectors within training matrix
#    # idxs were stored in SVM model as 1-base
#    idxs_train_SVs = map(lambda x: int(x[0]), svm_model.get_SV())
#
#    # compute kernel matrix, but, only against SVs from training data (to save computation)
#    n = data_train.shape[0]
#    m = data_test.shape[0]
#    matrix_kernel = np.zeros((m, n+1)).tolist()
#    for i in range(m):
#        for j in idxs_train_SVs:
#            matrix_kernel[i][j] = func_kernel(data_test[i], data_train[j-1])
#
#    scores = None
#    if flag_prob:
#        options_te = '-b 1 '
#        scores = np.array(svmutil.svm_predict_simple(matrix_kernel, svm_model, options_te))
#
#    return (matrix_kernel, scores)


def get_SV_idxs_nonlinear_svm(svm_model):
    """ From nonlinear SVM model, get idxs of SVs (w.r.t. training data)
    """
    idxs_train_SVs = map(lambda x: int(x[0]), svm_model.get_SV())
    return idxs_train_SVs

# def get_SV_weights_nonlinear_svm(svm_model, target_class = 1):
#     """ From nonlinear SVM model, get weights for SVs, for the given target_class
#     """
#     idx_target = get_column_idx_for_class(svm_model, target_class)
#     weights = np.array([ws[idx_target] for ws in svm_model.get_sv_coef()])
#     return weights

def get_SV_weights_nonlinear_svm(svm_model, target_class=1, flag_manual_sign_flip=False):
    """ From nonlinear SVM model, get weights for SVs, for the given target_class.
    Only works for 1-vs-all training.
    @todo: this implementation is not fully vetted, although it seems to be working during ECD learning
    """

    # general implementation not working anymore with libSVM 3.12
    # idx_target = get_column_idx_for_class(svm_model, target_class)
    # weights = np.array([ws[idx_target] for ws in svm_model.get_sv_coef()])

    weights = (np.array(svm_model.get_sv_coef())).flatten()
    if flag_manual_sign_flip:
        idx_target = get_column_idx_for_class(svm_model, target_class)
        if idx_target != 0:
            weights *= -1

    return weights

def get_compact_nonlinear_svm(svm_model, data_train_orig):
    """
    Given a non-linear SVM model, remove zero-weighted SVs, and also produce compact training data with SVs only

    @param svm_model: loaded (non-linear) svm model
    @param data_train_orig: n-by-d row-wise training data matrix in numpy format
    @return: an updatedsvm_model & support vectors sub-selected from data_train_orig
    """

    n_SVs = svm_model.l
    idxs_train_SVs = get_SV_idxs_nonlinear_svm(svm_model)
    [_, d] = data_train_orig.shape

    SVs = np.zeros([n_SVs, d]) # initialize memory

    for i in range(n_SVs):
        idx = idxs_train_SVs[i]
        svm_model.SV[i][0].value = i+1 #idx is 1-base
        SVs[i] = data_train_orig[idx-1]# use 0-base

    return (svm_model, SVs)


def write_compact_nonlinear_svm(file_compact_svm, target_class,
                                file_svm_model, svm_model=None,
                                file_SVs=None, SVs=None,
                                str_kernel=None):
    """
    Writes a textfile with all the necessary file locations for (nonlinear) libSVM agent
    All the component files of 'file_compact_svm' will be written in the same directory

    @param file_compact_svm: file to be written with all the information below
    @param target_class: integer target class, e.g., 0 or 30.
    @param file_svm_model: filename to the compact svm model to be written
    @param file_SVs: filename to the support vectors (only applicable if nonlinear SVM)
    @param str_kernel: string of kernel function to be used (e.g., kernels.ngd etc)
    @param svm_model: actual svm_model from get_compact_nonlinear_svm, which will be saved at file_svm_model (if not already)
    @param SVs: actual support vectors in numpy format to be saved (if not already), generated by get_compact_linear_svm
    @return: 1 if success
    """

    dir_compact = os.path.dirname(file_compact_svm)

    if svm_model:
        svmutil.svm_save_model(os.path.join(dir_compact, file_svm_model), svm_model)
    if SVs is not None:
        np.save(os.path.join(dir_compact, file_SVs), SVs)

    with open(file_compact_svm, 'wb') as fin:
        fin.write('file_svm_model=%s\n'%file_svm_model)
        fin.write('target_class=%d\n'%target_class)
        if file_SVs:
            fin.write('file_SVs=%s\n'%file_SVs)
        if str_kernel:
            fin.write('str_kernel=%s\n'%str_kernel)
        fin.flush()

def parse_compact_nonlinear_svm(file_compact_svm, flag_load_model=True):
    """
    Parse configurations and/or actual models, based on
    a config file written by write_compact_nonlinear_svm.
    """

    print 'Loading (compact) nonlinear SVM configuration:\n%s...'%file_compact_svm

    model = dict()
    model['file_svm_model'] = None
    model['svm_model'] = None
    model['target_class'] = None
    model['file_SVs'] = None
    model['SVs'] = None
    model['str_kernel'] = None
    model['func_kernel'] = None

    model_keys = model.keys()

    with open(file_compact_svm) as fin:
        for line in fin:
            strs = line.strip().split('=')
            if len(strs) == 2:
                key = strs[0].strip()
                if key in model_keys:
                    model[key] = strs[1].strip()

    # string to integer
    model['target_class'] = int(model['target_class'])
    print model

    if flag_load_model:
        print '... finding kernel..'

        model['func_kernel'] = getattr(kernels, model['str_kernel'])

        dir_compact = os.path.dirname(file_compact_svm)
        print '... loading SVM model..'
        model['svm_model'] = svmutil.svm_load_model(os.path.join(dir_compact, model['file_svm_model']))

        print '... loading SVs (may take some time)..'
        tmp = os.path.join(dir_compact, model['file_SVs'])
        if not os.path.exists(tmp):
            tmp += '.npy'
        model['SVs'] = np.load(tmp)

    return model

def get_column_idx_for_class(model, target_class):
    """
    Given a libSVM model, find the 0-base column index of the corresponding target_class label
    This is necessary since labels can be encoded in arbitrary order in libSVM models

    @param model: libSVM model
    @param target_class: integer label
    @return: index of the target_class
    """
    idx_target = None
    for idx in range(model.nr_class):
        if model.label[idx] == target_class:
            idx_target = idx
            break
    return idx_target


#def default_aladdin_performance_error(labels, scores, target_label):
#    import aladdin_fold_learn_tools as toolbox
#    pfa_window = 0.02
#    (avg_pfa, _) = toolbox.average_errorRate_targetRatio(labels, scores,
#                                                         pfa_window, ratio=12.5, target_class=target_label)
#    return avg_pfa



def apply_common_nonlinear_svm(model, kernel_matrix, kernel_matrix_recounting = None,
                               target_class = 1, model_is_compact = True):
    """ Common routine to apply libSVM on test data, once the input data is structured in common format.
    This uses a custom test implementation which bypasses libsvm routine (faster).
    This implementation is very memory intensive for recounting to have kernel_matrix_recounting ready.
    @todo: This routine will generalize to full SVM as well, and used within EAG training CV as well

    @param model: libsvm model
    @param kernel_matrix: 2D array of kernel values, rows are test data, cols are training data (maybe SVs)
    @type kernel_matrix: numpy.array

    @param kernel_matrix_recounting: (optional) 3D array of kernel values, dim0: test data, dim1: training data, dim2: feature dimensions
    @type kernel_matrix_recounting: numpy.array

    @param target_class: the target class encoded in model, default = 1.
    @type target_class: int
    @param model_is_compact: Set to True (default),if compact SVM (only SVs are embedded among all used training examples).
                             If 'full' SVM model is used, then, set to False for correct behavior.
    @type model_is_compact: bool
    @return: dictionary with 'probs','margins', and optional 'margins_mer'
    @rtype: dictionary with multiple numpy.array entries
    """

    idx_target = get_column_idx_for_class(model, target_class)

    if not model_is_compact:
        idxs_SVs = get_SV_idxs_nonlinear_svm(model)   # 1 base
        idxs_SVs = [ (_idx - 1) for _idx in idxs_SVs] # 0 base
        kernel_matrix = kernel_matrix[:, idxs_SVs]
        if kernel_matrix_recounting is not None:
            kernel_matrix_recounting = kernel_matrix_recounting[:, idxs_SVs, :]

    # compute margins
    # this part needs to be updated, to select SV row/columns
    weights = get_SV_weights_nonlinear_svm(model, target_class = target_class)
    margins = np.dot(kernel_matrix, weights)

    # compute probs, using platt scaling
    rho   = model.rho[0]
    probA = model.probA[0]
    probB = model.probB[0]
    probs   = 1.0 / (1.0 + np.exp((margins - rho) * probA + probB))

    # compute margins_recoutning
    margins_recounting = None
    if kernel_matrix_recounting is not None:
        tmp = kernel_matrix_recounting.shape
        margins_recounting = np.zeros((tmp[0], tmp[2]))
        for i in range(tmp[0]):
            margins_recounting[i] = np.dot(kernel_matrix_recounting[i].T, weights)

    if idx_target == 1:
        margins = -margins
        probs   = 1.0 - probs
        if margins_recounting is not None:
            margins_recounting = -margins_recounting

    outputs = dict()
    outputs['margins'] = margins
    outputs['probs']   = probs
    outputs['margins_recounting'] = margins_recounting

    return outputs



def apply_full_nonlinear_svm(model, data, report_margins_recounting=False):
    """ Apply parsed full SVM model (original libSVM model with embedded SVs)
    This is a custom implementation which bypasses libsvm routine (faster).
    @param model: model parsed by event_agent_generator.parse_full_SVM_model
    @param data: row-wise data vector/matrix in numpy format
    @type data: numpy.array
    @param report_margins_recounting: if True, report bin-wise contribution towards margin for every data_test
    @type report_margins_recounting: bool
    @return: dictionary with 'probs' and 'margins', which are each numpy array of floats
    @rtype: dictionary of float numpy arrays

    todo: add mer
    """

    print '#training samples loaded by full SVM model: %d' %model['train_data'].shape[0]
    matrices = kernels.compute_kernel_matrix(data, model['train_data'],
                                             func_kernel = model['func_kernel'],
                                             recounting = report_margins_recounting)

    outputs = apply_common_nonlinear_svm(model['svm_model'],
                                         kernel_matrix = matrices['kernel_matrix'],
                                         kernel_matrix_recounting = matrices['kernel_matrix_recounting'],
                                         target_class = model['target_class'],
                                         model_is_compact = False)

    return outputs



def apply_common_nonlinear_svm_memory_light(model, func_kernel, SVs, data,
                                            target_class=1,
                                            report_margins_recounting=False):
    """ Common routine to apply nonlinear compact libSVM on test data,
    Uses smaller memory foot print during recounting, than 'apply_common_nonlinear_svm'
    @param model: libsvm model
    @param func_kernel: kernel function
    @param SVs: row-wise support vector matrix
    @param data: test data in numpy format
    @param target_class: target class
    @type target_class: int
    @param report_margins_recounting: if True, report recounting per data as well
    @return: dictionary with 'probs','margins', and optional 'margins_mer'
    @rtype: dictionary with multiple numpy.array entries
    """

    # get SV weights
    weights = get_SV_weights_nonlinear_svm(model, target_class=target_class)

    # compute kernel_matrix and kernel_matrix_recounting
    # in memory efficient way

    n1 = data.shape[0]
    dim = len(data[0])
    n2 = SVs.shape[0]

    # kernel matrix is |data|-by-|SVs|
    kernel_matrix = np.zeros((n1, n2))

    margins_recounting = None
    if report_margins_recounting:
        margins_recounting = np.zeros((n1, dim))

    _tmp_in = np.zeros((1, dim))
    for i in range(n1):
        _tmp_in[0] = data[i]
        # _tmp_out['kernel_matrix']: 1-by-|SVs|
        # _tmp_out['kernel_matrix_recounting']: 1 x|SVs| x dim
        _tmp_out = kernels.compute_kernel_matrix(_tmp_in, SVs, func_kernel=func_kernel,
                                                 recounting=report_margins_recounting)
        kernel_matrix[i] = _tmp_out['kernel_matrix'][0]
        if report_margins_recounting:
            margins_recounting[i] = np.dot(_tmp_out['kernel_matrix_recounting'][0].T, weights)

    # this part needs to be updated further for more generalization, to select SV row/columns
    margins = np.dot(kernel_matrix, weights)

    # compute probs, using platt scaling
    rho = model.rho[0]
    probA = model.probA[0]
    probB = model.probB[0]
    probs   = 1.0 / (1.0 + np.exp((margins - rho) * probA + probB))

    idx_target = get_column_idx_for_class(model, target_class)
    if idx_target == 1:
        margins = -margins
        probs = 1.0 - probs
        if margins_recounting is not None:
            margins_recounting = -margins_recounting

    outputs = dict()
    outputs['margins'] = margins
    outputs['probs'] = probs
    outputs['margins_recounting'] = margins_recounting

    return outputs


def apply_compact_nonlinear_svm(model, data, use_approx = False,
                                report_margins_recounting=False):
    """ Apply parsed compact SVM model to new data.
    This is a custom implementation which bypasses libsvm routine (faster).

    @param model: model parsed from  'parse_compact_nonlinear_svm'
    @param data: row-wise data vector/matrix in numpy format
    @type data: numpy.array
    @param report_margins_recounting: if True, report bin-wise contribution towards margin for every data_test
    @type report_margins_recounting: bool
    @return: dictionary with 'probs','margins','margins_recounting' which are each numpy array of floats
    @rtype: dictionary of multiple numpy.array
    """

    if use_approx:
        svm_model_approx = compute_approx_nonlinear_SVM(model, model['SVs'])
        outputs = apply_approx_nonlinear_SVM(svm_model_approx, data,
                                             report_margins_recounting = report_margins_recounting)
    else:
        # handle report_margins_recounting

        if not report_margins_recounting:
            # speedy version without MER
            matrices = kernels.compute_kernel_matrix(data, model['SVs'], func_kernel=model['func_kernel'],
                                                     recounting=report_margins_recounting)
            outputs = apply_common_nonlinear_svm(model['svm_model'],
                                                 kernel_matrix=matrices['kernel_matrix'],
                                                 kernel_matrix_recounting=matrices['kernel_matrix_recounting'],
                                                 target_class=model['target_class'])
        else:
            # memory light implementation to deal with MER
            outputs = apply_common_nonlinear_svm_memory_light(model['svm_model'], model['func_kernel'],
                                                              model['SVs'], data,
                                                              target_class=model['target_class'],
                                                              report_margins_recounting=report_margins_recounting)

    return outputs


def learn_compact_nonlinear_svm(file_libsvm_model0,
                                file_SVs,
                                file_libsvm_model1,
                                file_svm_compact,
                                str_kernel, options_train,
                                target_class,
                                labels,
                                data, file_data,
                                kernel_matrix, file_kernel_matrix, kernel_matrix_type, flag_kernel_compute,
                                splits, func_sort, logfile):
    """
    @param file_libsvm_model0: file path for the leanred SVM model to be saved in libsvm format
    @param file_SVs: filename of support vectors to be saved in numpy format
    @param file_libsvm_model1: file path for compact SVM, still stored in libsvm format
    @param file_svm_compact: file path to the full compact svm model to be written (with many other info)
    @param str_kernel: string of kernel function to be used (e.g., kernels.ngd etc)
    @param options_train: list of lisbsvm training strings to be tried, e.g., ['-b 1 -c 1','-b 1 -c 1000']
    @param options_test: libsvm test option string to be used during cross-validation, e.g., '-b 1'
    @param target_class: target positive class
    @param labels: ground truth labels in integers.
                   Positive integers for event kit positives, Negatives for event kit negs, zero for None.
    @param data: training data, numpy row-wise matrix. If None and kernel_matrix does not exist, then, read from file_data
    @param file_data: file path to the input training 'data'. If data is None, then read from this file
    @param kernel_matrix: kernel matrix
    @param file_kernel_matrix: if kernel matrix is None, and this path is not, then, loaded from this file.
                               if flag_kernel_compute==True, then, computed kernel is saved to this file.
    @param kernel_matrix_type: 'numpy' (square numpy matrix) or 'libsvm' (2dim list-type ready for libsvm)
    @param flag_kernel_compute: if True, re-compute kernel matrix
    @param splits: integer-based splits in numpy vector, e.g., [1 1 2 2 3 3] for 6 data in three splits
    @param file_scores: if not None, save the scores generated by SVM during cross validation
    @param func_error: func_error(labels, scores, target_label) outputs error to sort svm parameters
    @param logfile: log file where info will be written, e.g., the pairs of options_train & func_error outputs
    """

    _log = None
    if logfile:
        _log = open(logfile, 'wb')

    # Learn nonlinear SVM model & save this initial model (before compactization) in libsvm format
    model0 = None
    # add training code with full data training
    svmutil.svm_save_model(file_libsvm_model0, model0)
    _log.write('Saved initial nonlinear SVM (model0) at: %s\n'%file_libsvm_model0)

    # computed compact SVM model 'model1'
    (model1, SVs) = get_compact_nonlinear_svm(model0, data)

    # write compact SVM model, with all the required information
    write_compact_nonlinear_svm(file_svm_compact, target_class,
                                file_libsvm_model1, svm_model=model1,
                                file_SVs=file_SVs, SVs=SVs,
                                str_kernel=str_kernel)
    _log.write('Saved compact nonlinear SVM at: %s\n'%file_svm_compact)
    _log.close()



#########################################################################
# Approximate non-linear SVM model
# - only valid for testing
# - built from a compact SVM
# - Based on Maji's paper on efficient approximation of additive models
#########################################################################


def compute_approx_nonlinear_SVM(svm_model_compact, n_approx=500, verbose = False):
    """
    Given a non-linear SVM model, remove zero-weighted SVs, and also produce compact training data with SVs only
    Based on Maji's paper on efficient approximation of additive models.

    @param svm_model: loaded (non-linear) svm model, by 'parse_compact_nonlinear_svm'
    @param SVs: support vectors (NOTE: do processing as needed a priori)
    @n_approx: the scope of approximation, i.e., the number of bins to approximate each dimension, higher more accurate & slower/memory-intensive
    @return: approximate SVM model
    """

    # MODEL OUTPUT
    svm_model_approx = dict()
    svm_model_approx['str_kernel'] = svm_model_compact['str_kernel']
    str_kernel = svm_model_approx['str_kernel']
    svm_model_approx['target_class'] = svm_model_compact['target_class']
    model_orig = svm_model_compact['svm_model'] # SVM model
    svm_model_approx['rho']   = model_orig.rho[0]
    svm_model_approx['probA'] = model_orig.probA[0]
    svm_model_approx['probB'] = model_orig.probB[0]
    svm_model_approx['target_index'] = get_column_idx_for_class(model_orig, svm_model_compact['target_class'])
    svm_model_approx['n_approx'] = n_approx

    SVs = svm_model_compact['SVs']
    feat_dim = SVs.shape[1] # dimension of features
    vals_max = np.amax(SVs, axis=0)
    vals_min = np.amin(SVs, axis=0)

    # approximation grid map
    input_grids = np.zeros((n_approx, feat_dim))
    for i in range(feat_dim):
        input_grids[:,i] = np.linspace(vals_min[i], vals_max[i], num=n_approx)

    # step size for each bin
    step_sizes = (input_grids[1,:] - input_grids[0,:])
    for i in range(feat_dim):
        step_sizes[i] = input_grids[1,i] - input_grids[0,i]

    # SVM model coefficients for SVs
    n_SVs = model_orig.l
    _sv_coef = model_orig.get_sv_coef()
    sv_coef = np.zeros(n_SVs)
    for (k, v) in enumerate(_sv_coef):
        sv_coef[k] = v[0]


    func_additive_func = None
    if str_kernel == 'hik':
        func_additive_func = lambda x,y: np.amin(np.vstack((x,y)), axis=0)
    else:
        # ADD MORE additive functions based on kernel function here
        raise Exception('Unknown kernel function'%str_kernel)

    # output grid map for all input grid map values
    output_grids = np.zeros((n_approx, feat_dim))
    for i in range(feat_dim):
        if (verbose) == True and (i%200 == 0):
            print 'libsvmtools.compute_approx: computing feature dim i=%d / %d'%(i, feat_dim)
        for j in range(n_approx):
            output_grids[j,i] = (sv_coef * func_additive_func((np.ones(n_SVs)*input_grids[j,i]), SVs[:,i])).sum()
#            for k in range(n_SVs):
#                output_grids[j,i] +=  sv_coef[k][0] * func_additive_func(input_grids[j,i], SVs[k,i])

    svm_model_approx['input_grids']  = input_grids
    svm_model_approx['output_grids'] = output_grids
    svm_model_approx['vals_max']     = vals_max
    svm_model_approx['vals_min']     = vals_min
    svm_model_approx['step_sizes']   = step_sizes


    return svm_model_approx


def linear_interpolate(x0, y0, x1, y1, x):
    """
    Given (x0, y0), and (x1, y1), and an x in-between,
    predict y via interpolation
    """
    if x1 == x0:
        y = 0
    else:
        y = y0 + (y1-y0)*((x-x0)/(x1-x0))
    return y


def apply_approx_nonlinear_SVM(svm_model_approx, data_test,
                               report_margins_recounting=False, verbose = False):
    """
    Apply approximate SVM model, learned from 'compute_approx_nonlinear_SVM'
    @param svm_model: loaded (non-linear) svm model
    @param data_test: row-wise test data in numpy array format
    @param report_margins_recounting: if True, report bin-wise contribution towards margin for every data_test
    @type report_margins_recounting: bool
    @return: dictionary of results, such as 'probs', 'margins', 'margins_mer'
    @rtype: dict of numpy arrays
    """

    # number of data
    n = data_test.shape[0]

    input_grids  = svm_model_approx['input_grids']
    output_grids = svm_model_approx['output_grids']
    feature_dim  = output_grids.shape[1]
    n_bins       = output_grids.shape[0]
    vals_min     = svm_model_approx['vals_min']
    vals_max     = svm_model_approx['vals_max']
    step_sizes   = svm_model_approx['step_sizes']

    eps = math.pow(2.0, -52)

    # bin-wise contribution towards margin, for every data
    margins_recounting = np.zeros((n,feature_dim))

    for (i, data) in enumerate(data_test):
        if (verbose == True) and (i%100 ==0):
            print 'libsvmtools.apply_approx_nonlinear_SVM: i= %d / %d'%(i, len(data))
        for k in range(feature_dim):
            if step_sizes[k] < eps: # constant along that dimension
                margins_recounting[i,k] = output_grids[0,k]
                #margins[i] += output_grids[0,k]
            else:
                v = data[k]
                if v >= vals_max[k]:
                    margins_recounting[i,k] = output_grids[n_bins-1, k]
                elif data[k] < vals_min[k]:
                    margins_recounting[i,k] = linear_interpolate(0,0, vals_min[k], output_grids[0,k], v)
                else:
                    idx_map   = int(math.floor((v - vals_min[k]) /  step_sizes[k]))
                    try:
                        margins_recounting[i,k] = linear_interpolate(input_grids[idx_map,k],   output_grids[idx_map,k],
                                                          input_grids[idx_map+1,k], output_grids[idx_map+1,k],
                                                          v)
                    except:
                        idx_map = len(input_grids) - 2
                        margins_recounting[i,k] = linear_interpolate(input_grids[idx_map,k],   output_grids[idx_map,k],
                                                          input_grids[idx_map+1,k], output_grids[idx_map+1,k],
                                                          v)

    # margins per data
    margins = np.zeros(n) - svm_model_approx['rho']
    margins += np.sum(margins_recounting, axis=1)

    # probs through platt scaling
    probs = 1.0 / (1.0 + np.exp((margins * svm_model_approx['probA']) + svm_model_approx['probB']))

    if svm_model_approx['target_index'] == 1:
        probs = 1.0 - probs
        margins = -margins
        margins_recounting = -margins_recounting

    outputs = dict()
    outputs['probs'] = probs
    outputs['margins'] = margins
    if report_margins_recounting:
        outputs['margins_recounting'] = margins_recounting

    return outputs


def write_approx_nonlinear_SVM(filename, svm_model_approx):
    import cPickle
    with open(filename, 'wb') as fout:
        cPickle.dump(svm_model_approx, fout)

def load_approx_nonlinear_SVM(filename):
    import cPickle
    fin = open(filename, 'rb')
    svm_model_approx = cPickle.load(fin)
    fin.close()
    return svm_model_approx

def write_approx_nonlinear_SVM_numpy(filename, svm_model_approx):
    """
    Write Approximate Nonlinear SVM
    """

    str_kernel = svm_model_approx['str_kernel']
    param_array0 = np.zeros((1, svm_model_approx['input_grids'].shape[1]))
    idx_kernel = None
    for (str_kernel, idx) in kernels.IDXS_KERNELS:
        if str_kernel == str_kernel:
            idx_kernel = idx
            break

    param_array0[0,0] = idx_kernel
    param_array0[0,1] = svm_model_approx['target_class']
    param_array0[0,2] = svm_model_approx['target_index']
    param_array0[0,3] = svm_model_approx['rho']
    param_array0[0,4] = svm_model_approx['probA']
    param_array0[0,5] = svm_model_approx['probB']
    param_array0[0,6] = svm_model_approx['n_approx']
    param_array1 = np.vstack((param_array0,
                              svm_model_approx['vals_max'],
                              svm_model_approx['vals_min'],
                              svm_model_approx['step_sizes'],
                              svm_model_approx['input_grids'],
                              svm_model_approx['output_grids']))

    np.save(filename, param_array1)

def load_approx_nonlinear_SVM_numpy(filename):
    """
    Load Approximate Nonlinear SVM
    """

    param_array1 = np.load(filename)
    idx_kernel = param_array1[0,0]
    str_kernel = None
    for (_str_kernel, idx) in kernels.IDXS_KERNELS:
        if idx == idx_kernel:
            str_kernel = _str_kernel
            break

    svm_model_approx = dict()
    svm_model_approx['str_kernel'] = str_kernel
    svm_model_approx['target_class'] = int(param_array1[0,1])
    svm_model_approx['target_index'] = int(param_array1[0,2])
    svm_model_approx['rho']   = param_array1[0,3]
    svm_model_approx['probA'] = param_array1[0,4]
    svm_model_approx['probB'] = param_array1[0,5]
    svm_model_approx['n_approx'] = n_approx = int(param_array1[0,6])
    svm_model_approx['vals_max'] = param_array1[1,:]
    svm_model_approx['vals_min'] = param_array1[2,:]
    svm_model_approx['step_sizes']   = param_array1[3,:]
    svm_model_approx['input_grids']  = param_array1[4:(4+n_approx), :]
    svm_model_approx['output_grids'] = param_array1[(4+n_approx):, :]

    return svm_model_approx


############# TEST CODE BELOW ###############

# import easyio
#
#def test2_compact_libsvm():
#    # The data paths only work for Sangmin's Lelex machine
#    test_dir_data  = 'E:/Aladdin_data/clips_hog3d'
#    test_dir_model = 'E:/Aladdin_data/clips_hog3d/models_run_svm_hik_eventkit1_dev1'
#
#    file_data_train = os.path.join(test_dir_data, 'med11_part1_eventkit.normhist.npy')
#    file_data_test  = os.path.join(test_dir_data, 'med11_part1_dev.normhist.npy')
#
#    file_model0 = os.path.join(test_dir_model, 'model_01.param_0.svm.model')
#    file_score0 = os.path.join(test_dir_model, 'test_01.param_0.svm.probs.txt')
#
#    data_train = easyio.load(file_data_train)
#    data_test = easyio.load(file_data_test)
#
#    model0 = svmutil.svm_load_model(file_model0)
#
#    # Parse learned nonlinear SVM model and save with all the relevant pointers
#    # - model0 is the newly trained SVM model that will be compacted (SVs will be extracted)
#    # - data_train is the training row-wise data matrix in numpy format
#    (model1, SVs) = get_compact_nonlinear_svm(model0, data_train)
#    file_model1 = os.path.join(test_dir_model, 'model_01.param_0.svm.model.compact.conf')
#    target_class = 1
#    file_svm_model = os.path.join(test_dir_model, 'model_01.param_0.svm.model.compact')
#    file_SVs = os.path.join(test_dir_model, 'model_01.param_0.svm.model.compact.SVs.npy')
#    str_kernel = 'hik'
#    write_compact_nonlinear_svm(file_model1, target_class,
#                                file_svm_model, svm_model=model1,
#                                file_SVs=file_SVs, SVs=SVs,
#                                str_kernel=str_kernel)
#
#    # load a saved model, and simply apply to the test data
#    # if test data is too big, then, apply one by one
#    model2 = parse_compact_nonlinear_svm(file_model1)
#    scores2 = apply_compact_svm(model2, data_test)
#
#    scores0 = np.loadtxt(file_score0)
#    print 'diff = %g'%np.abs((scores0-scores2)).sum()
#
#
#    print 'wow'

#def speed_test_hik():
#    """
#    hik1 ends up being much slower, not very useful
#    """
#    test_dir_data  = 'E:/Aladdin_data/clips_hog3d'
#    file_data_train  = os.path.join(test_dir_data, 'med11_part1_eventkit.normhist.npy')
#    data_train = easyio.load(file_data_train)[0:200]
#
#    import time
#
#    t00 = time.clock()
#    hik1 = kernels.compute_kernel_matrix(data_train, func_kernel=kernels.hik, verbose = 10)
#    t0d = time.clock() - t00
#
#    t10 = time.clock()
#    hik2 = kernels.compute_kernel_matrix(data_train, func_kernel=kernels.hik1, verbose = 10)
#    t1d = time.clock() - t10
#
#    hik_diff = np.max(np.abs(hik1 - hik2))
#
#    print 't0 = %g, t1 = %g, hik_diff = %g'%(t0d, t1d, hik_diff)

if __name__ == "__main__":
    #test1()
    #test2_compact_libsvm()
    #speed_test_hik()
    pass





