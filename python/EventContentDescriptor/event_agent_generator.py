"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

# generic Python libraries
import os
import sys
import numpy as np
import datetime
import shutil
import glob
import stat
import easyio

# libSVM modules
import svm
import svmutil

# SMQTK modules
import kernels
import libsvm_tools as svmtools
import deva_tools as deva
import perf_estimation as perf
import preprocess_features as prep

# Set of fixed constant values (file names etc)
FILE_EVENT_DATA   = 'data_EVENT'
FILE_EVENT_LABEL  = 'label_EVENT.txt'
FILE_EVENT_CLIPID = 'clipids_EVENT.txt'
FILE_EVENT_CV_TRAIN = 'CV_id_train_EVENT.txt'
FILE_EVENT_CV_TEST  = 'CV_id_test_EVENT.txt'
LABEL_POS = 1   # for every event kit, this fixed label is used (not, 6, 7 etc)
LABEL_NEG = 0

FILE_BG_DATA = 'data_BG'
FILE_BG_LABEL = 'label_BG.txt'
FILE_BG_CLIPID   = 'clipids_BG.txt'
FILE_BG_CV_TRAIN = 'CV_id_train_BG.txt'
FILE_BG_CV_TEST  = 'CV_id_test_BG.txt'

FILE_EAG_KERNEL         = 'kernel.txt'
FILE_KERNEL_EVENTxEVENT = 'kernel_EVENTxEVENT.npy'
FILE_KERNEL_BGxBG = 'kernel_BGxBG.npy'
FILE_KERNEL_BGxEVENT = 'kernel_BGxEVENT.npy'


FILE_PARAMETERS     = 'parameters_TRAIN_trial_list.txt'
FILE_PARAMETER_BEST = 'parameter_best.txt'
FILE_PARAMETER_BEST_JSON = 'parameter_best.json'

FILE_EA_KERNEL  = FILE_EAG_KERNEL
FILE_EA_CLIPID_ALL = 'clipids_all.txt'
FILE_EA_CV_TRAIN = 'CV_id_train.txt' # stores CV data splits
FILE_EA_CV_TEST  = 'CV_id_test.txt'  # stores CV data splits
FILE_EA_CV_IDXS  = 'CV_indices.txt'  # stores actually used CV indices

# Models trained with all the training data
FILE_EA_MODEL_LIBSVM_FULL_DIR = 'full'
FILE_EA_MODEL_LIBSVM_FULL = 'full_libSVM.model'
FILE_EA_MODEL_LIBSVM_FULL_TARGET_IDX = 'full_libSVM.model.target_idx.txt'
FILE_EA_MODEL_LIBSVM_COMPACT_DIR = 'compact'
FILE_EA_MODEL_LIBSVM_COMPACT_CONTAINER = 'compact_libSVM.model.container'
FILE_EA_MODEL_LIBSVM_COMPACT = 'compact_libSVM.model'
FILE_EA_MODEL_LIBSVM_COMPACT_SVs = 'compact_libSVM.model.SVs.npy'
FILE_EA_NAP = 'nap_cv_best.png'
FILE_EA_NAP_PARAM = 'nap_cv_params.png' # plot with all the tried parameters
#FILE_EA_DET = 'det_cv_best.png'
#FILE_EA_DET_PARAM = 'det_cv_params.png' # plot with all the tried parameters
FILE_EA_MODEL_LIBSVM_APPROX = 'approx_libSVM.model.pickle'

# Cross-validation Models
FILE_CV_EA_MODEL_LIBSVM = 'cv_libSVM.%02d.model'
FILE_CV_EA_MODEL_LIBSVM_TARGET_IDX = 'cv_libSVM.model.%02d.target_idx.txt'
FILE_CV_EA_MODEL_LIBSVM_COMPACT_DIR = 'compact_cv_%02d'
FILE_CV_EA_MODEL_LIBSVM_COMPACT_CONTAINER = FILE_EA_MODEL_LIBSVM_COMPACT_CONTAINER
FILE_CV_EA_MODEL_LIBSVM_COMPACT = FILE_EA_MODEL_LIBSVM_COMPACT
FILE_CV_EA_MODEL_LIBSVM_COMPACT_SVs = FILE_EA_MODEL_LIBSVM_COMPACT_SVs
FILE_CV_EA_MODEL_LIBSVM_APPROX = FILE_EA_MODEL_LIBSVM_APPROX
FILE_CV_EA_DET  = 'det_cv_%02d.png'
FILE_CV_PROBS   = 'probs.txt'
FILE_CV_MARGINS = 'margins.txt'
FILE_CV_RANKS   = 'ranks.txt'

# libsvm failure indicator
FILURE_POSTFIX = '.failure'
FILE_FAILURE_CV_EA_MODEL_LIBSVM = FILE_CV_EA_MODEL_LIBSVM + FILURE_POSTFIX
FILE_FAILURE_CV_EA_MODEL_LIBSVM_COMPACT_DIR = FILE_CV_EA_MODEL_LIBSVM_COMPACT_DIR + FILURE_POSTFIX
FILE_FAILURE_EA_MODEL_LIBSVM_FULL_DIR = FILE_EA_MODEL_LIBSVM_FULL_DIR + FILURE_POSTFIX
FILE_FAILURE_EA_MODEL_LIBSVM_COMPACT_DIR = FILE_EA_MODEL_LIBSVM_COMPACT_DIR + FILURE_POSTFIX
# DET curve figure size
FILE_PARA_EA_DET = 'det_%s.png'
DET_WIDTH  = 1200
DET_HEIGHT = 1000

# TEST DATA
FILE_TEST_DATA_PATTERN_FEATURES = 'part_*_data.npy' #'part_%04d_data.npy'
def FUNC_GET_ID_FROM_FILE_TEST_DATA(filename):
    tmp = os.path.basename(filename)
    return int(tmp[5:5+4])
FILE_TEST_DATA_PATTERN_CLIPIDS  = 'part_%04d_clipids.txt'
FILE_TEST_PROB_PATTERN   = 'part_%04d_probs.txt'
FILE_TEST_MARGIN_PATTERN = 'part_%04d_margins.txt'
FILE_TEST_MARGIN_MER_PATTERN = 'part_%04d_margins_mer.txt'


# log file
_logfile = None

def logfile_timestamp():
    computer_name = str.upper(os.getenv('COMPUTERNAME')) # works on windows
    now = str(datetime.datetime.now())
    now = now.replace('-', '_')
    now = now.replace(' ', '_')
    now = now.replace(':', '_')
    now = now.replace('.', '_')
    filename = 'log-%s-%s.txt'%(computer_name, now)
    return filename

def set_logfile(logfile):
    global _logfile
    _logfile = logfile
    print 'Global log file is set: %s'%_logfile

def log(str_log, logfile = None):
    print str_log
    sys.stdout.flush()
    logfile_curr = None
    if logfile is None:
        global _logfile
        if _logfile is not None:
            logfile_curr = _logfile
        logfile_curr = _logfile
    else:
        logfile_curr = logfile

    if logfile_curr is not None:
        with open(logfile_curr, 'a+') as fin:
            fin.write(str_log)
            fin.flush()

def touch(filename):
    try:
        fin = open(filename, 'wb')
        fin.close()
    except OSError:
        if os.path.exists(filename):
            pass
        else:
            raise

def parse_file_EAG_kernel(file_EAG_kernel):
    str_kernel = None
    with open(file_EAG_kernel) as fin:
        str_kernel = fin.read().strip()
    return str_kernel

def parse_file_best_parameter_file(file_best_parameter):
    return parse_file_EAG_kernel(file_best_parameter)

def parse_file_single_integer(file_single_integer):
    single_int = None
    with open(file_single_integer) as fin:
        single_int = int(fin.read().strip())
    return single_int

def preprocess_data_per_kernel(data, str_kernel):

    if str_kernel == 'L1norm_linear':
        # L1 normalize row-wise data
        data = prep.apply_norm_l1(data)

        log('event_agent_generator.preprocess_data_per_kernel (%s): data has been processed with L1-norm\n'%str_kernel)

    if str_kernel == 'ngd':

        # L1 normalize row-wise data
        data = prep.apply_norm_l1(data)
        data = np.sqrt(data)
        log('event_agent_generator.preprocess_data_per_kernel (%s): data has been processed with L1-norm + np.sqrt during loading\n'%str_kernel)

    if str_kernel == 'hik':
        log('event_agent_generator.preprocess_data_per_kernel (%s): No pre-processing\n'%str_kernel)

    if str_kernel == 'L1norm_hik':
        # L1 normalize row-wise data
        data = prep.apply_norm_l1(data)
        log('event_agent_generator.preprocess_data_per_kernel (%s): data has been processed with L1-norm\n'%str_kernel)

    return data

def get_data(file_data_npy, file_data_txt=None, str_kernel=None):
    """
    Load data from input file, and apply preprocessing if kernel is specified.
    @param file_data_npy: data in numpy format
    @param file_data_txt: (optional) txt-file based data
    @param str_kernel: name of kernel
    @return: loaded data (with preprocessing)
    @rtype: numpy.array
    """

    # if numpy format binary data is not available, create one
    if os.path.exists(file_data_npy):
        data = np.load(file_data_npy)
    elif os.path.exists(file_data_txt):
        data = np.loadtxt(file_data_txt)
        np.save(file_data_npy, data)
        log('Saved npy-converted matrix with size %s: %s\n'%(str(data.shape), os.path.basename(file_data_npy)))

    # preprocess data, based on kernel type (optional)
    if str_kernel is not None:
        data = preprocess_data_per_kernel(data, str_kernel)
    return data


def makedirs(path):
    """
    This is more robust on linux
    """
    try:
        os.makedirs(path)
#        os.chmod(path, stat.S_IRWXO)
    except OSError:
        if os.path.exists(path):
            pass
        else:
            print 'eag.makedirs: cannot create: ', path
            raise


def copy_file(src, dst):
    """
    This is more robust on linux
    """
    try:
        dirpath = os.path.dirname(dst)
        print dirpath
        print src
        if not os.path.exists(dirpath):
            makedirs(dirpath)
        shutil.copy2(src, dst)
    except OSError:
        if os.path.exists(dst):
            pass
        else:
            raise

def compute_EAG_kernels2(str_kernel,
                         dirpath_out_EVENT, dirpath_out_BG,
                         dirpath_in_EVENT=None, dirpath_in_BG=None,
                         data_EVENT=None, data_BG=None,
                         clipids_EVENT=None, clipids_BG=None,
                         flag_return_data=False,
                         flag_return_kernels=True):
    """
    Compute all the kernel matrices in advance required for SVM event agent training.
    Slightly different signature from compute_EAG_kernels.
    This compute_EAG_kernels2 is a refactored version for deployment.
    - Adds option to directly pass EventKit (using data_EVENT) and BG data (using data_BG) etc, as arguments to reduce data load requests.
    @param str_kernel: kernel name
    @param dirpath_out_EVENT: path to output the computed kernel matrices specific to EVENT Kit
    @param dirpath_out_BG: path to output the computed kernel matrices related to BG only
    @param dirpath_in_EVENT: path to input data for event kit
    @param dirpath_in_BG: path to input data for BG
    @param data_EVENT: optionally loaded data for event kits, either datapath_in_EVENT or this should be provided.
    @param data_BG: optionally loaded data for BG, either datapath_in_BG or this should be provided.
    @param clipids_EVENT:(optional) loaded EVENT clipids, to avoid re-loading
    @param clipids_BG:(optional) loaded BG clipids, to avoid re-loading
    @param flag_return_data: if True, return data as well
    @param flag_return_kernels: if True, return kernel matrices as part of outputs
    @return: dictionary with matrices 'EVENTxEVENT', 'BGxEVENT', 'BGxBG', and 'clipids_EVENT', 'clipids_BG' all in numpy format
    """

    # kernel function
    func_kernel = getattr(kernels, str_kernel)

    # save kernel names
    file_kernel_EVENT = os.path.join(dirpath_out_EVENT, FILE_EAG_KERNEL)
    file_kernel_BG = os.path.join(dirpath_out_BG, FILE_EAG_KERNEL)
    if not os.path.exists(file_kernel_EVENT):
        with open(file_kernel_EVENT, 'wb') as fout:
            fout.write(str_kernel)

    if not os.path.exists(file_kernel_BG):
        with open(file_kernel_BG, 'wb') as fout:
            fout.write(str_kernel)

    # clipids for EventKit
    file_dst_clipids_EVENTKIT = os.path.join(dirpath_out_EVENT, FILE_EVENT_CLIPID)
    if clipids_EVENT is None:
        if not os.path.exists(file_dst_clipids_EVENTKIT):
            file_src_clipids_EVENTKIT = os.path.join(dirpath_in_EVENT, FILE_EVENT_CLIPID)
            copy_file(file_src_clipids_EVENTKIT, file_dst_clipids_EVENTKIT)
        clipids_EVENT = np.loadtxt(file_dst_clipids_EVENTKIT)
    else:
        np.savetxt(file_dst_clipids_EVENTKIT, clipids_EVENT, fmt='%d')

    # clipids for BG
    file_dst_clipids_BG = os.path.join(dirpath_out_BG, FILE_BG_CLIPID)
    if clipids_BG is None:
        if not os.path.exists(file_dst_clipids_BG):
            file_src_clipids_BG = os.path.join(dirpath_in_BG, FILE_BG_CLIPID)
            copy_file(file_src_clipids_BG, file_dst_clipids_BG)
        clipids_BG = np.loadtxt(file_dst_clipids_BG)
    else:
        np.savetxt(file_dst_clipids_BG, clipids_BG, fmt='%d')

    # Below, we compute a set of kernel matrices, one by one, if not computed already

    # 1) compute EVENTxEVENT
    kernel_EVENTxEVENT = None
    file_EVENTxEVENT = os.path.join(dirpath_out_EVENT, FILE_KERNEL_EVENTxEVENT)
    if not os.path.exists(file_EVENTxEVENT):
        if data_EVENT is None:
            file_data_npy = os.path.join(dirpath_in_EVENT, FILE_EVENT_DATA+'.npy')
            data_EVENT = get_data(file_data_npy, str_kernel=str_kernel)

        t0 = datetime.datetime.now()
        log('Computing KERNEL_EVENTxEVENT...')
        _tmp_out = kernels.compute_kernel_matrix(data_EVENT, func_kernel=func_kernel, mirror=True)
        kernel_EVENTxEVENT = _tmp_out['kernel_matrix']
        np.save(file_EVENTxEVENT, kernel_EVENTxEVENT)
        dt = datetime.datetime.now() - t0
        log('Saved KERNEL_EVENTxEVENT with size %s: %s\n'%(str(kernel_EVENTxEVENT.shape), os.path.basename(file_EVENTxEVENT)))
        log('- Duration: %s\n\n' % str(dt))
        if not flag_return_kernels:
            kernel_EVENTxEVENT = None # free memory if not needed anymore
    else:
        log('Skipped existing kernel: %s\n' % (os.path.basename(file_EVENTxEVENT)))


    # 2) compute BGxBG
    kernel_BGxBG = None
    file_BGxBG = os.path.join(dirpath_out_BG, FILE_KERNEL_BGxBG)
    if not os.path.exists(file_BGxBG):
        if data_BG is None:
            file_data_npy = os.path.join(dirpath_in_BG, FILE_BG_DATA+'.npy')
            data_BG = get_data(file_data_npy, str_kernel=str_kernel)

        t0 = datetime.datetime.now()
        log('Computing KERNEL_BGxBG...')
        _tmp_out = kernels.compute_kernel_matrix(data_BG, func_kernel=func_kernel, mirror=True)
        kernel_BGxBG = _tmp_out['kernel_matrix']
        np.save(file_BGxBG, kernel_BGxBG)
        dt = datetime.datetime.now() - t0
        log('Saved KERNEL_BGxBG with size %s: %s\n'%(str(kernel_BGxBG.shape), os.path.basename(file_BGxBG)))
        log('- Duration: %s\n\n' % str(dt))
        if not flag_return_kernels:
            kernel_BGxBG = None  # free memory if not needed anymore
    else:
        log('Skipped existing kernel: %s\n' % (os.path.basename(file_BGxBG)))


    # 3) compute BGxEVENT
    kernel_BGxEVENT = None
    file_BGxEVENT = os.path.join(dirpath_out_EVENT, FILE_KERNEL_BGxEVENT)
    if not os.path.exists(file_BGxEVENT):
        if data_EVENT is None:
            file_data_npy = os.path.join(dirpath_in_EVENT, FILE_EVENT_DATA+'.npy')
            data_EVENT = get_data(file_data_npy, str_kernel=str_kernel)
        if data_BG is None:
            file_data_npy = os.path.join(dirpath_in_BG, FILE_BG_DATA+'.npy')
            data_BG = get_data(file_data_npy, str_kernel=str_kernel)

        t0 = datetime.datetime.now()
        log('Computing KERNEL_BGxEVENT...')
        _tmp_out = kernels.compute_kernel_matrix(data_BG, data_EVENT, func_kernel=func_kernel)
        kernel_BGxEVENT = _tmp_out['kernel_matrix']
        np.save(file_BGxEVENT, kernel_BGxEVENT)
        dt = datetime.datetime.now() - t0
        log('Saved KERNEL_BGxEVENT with size %s: %s\n'%(str(kernel_BGxEVENT.shape), os.path.basename(file_BGxEVENT)))
        log('- Duration: %s\n\n' % str(dt))
        if not flag_return_kernels:
            kernel_BGxEVENT = None  # free memory if not needed anymore
    else:
        log('Skipped existing kernel: %s\n'%(os.path.basename(file_BGxEVENT)))

    # output to be returned.
    outputs = dict()
    outputs['clipids_EVENT'] = clipids_EVENT.tolist()
    outputs['clipids_BG'] = clipids_BG.tolist()
    if flag_return_data:
        outputs['data_EVENT'] = data_EVENT
        outputs['data_BG'] = data_BG
    if flag_return_kernels:
        if kernel_EVENTxEVENT is None:
            kernel_EVENTxEVENT = np.load(file_EVENTxEVENT)
        outputs['kernel_EVENTxEVENT'] = kernel_EVENTxEVENT
        if kernel_BGxBG is None:
            kernel_BGxBG = np.load(file_BGxBG)
        outputs['kernel_BGxBG'] = kernel_BGxBG
        if kernel_BGxEVENT is None:
            kernel_BGxEVENT = np.load(file_BGxEVENT)
        outputs['kernel_BGxEVENT'] = kernel_BGxEVENT

    return outputs


def kernel_mat_pad(kernel_mat):
    # functions add integer identifiers to prepare libSVM input
    # pad additional column to kernel matrix
    indices = np.array([range(1, kernel_mat.shape[0] + 1)]).T
    kernel_mat = np.hstack((indices, kernel_mat))
    return kernel_mat


def run_EAG_libSVM(path_EAG, path_EA,
                   path_EVENTKIT, path_BG,
                   path_PARAMETER,
                   clipids_EVENT=None,
                   clipids_BG=None,
                   labels_EVENT=None,
                   labels_BG=None,
                   data_EVENT=None,
                   data_BG=None,
                   kernel_EVENTxEVENT=None,
                   kernel_BGxEVENT=None,
                   kernel_BGxBG=None,
                   cvids_train_EventKit=None,
                   cvids_train_BG=None,
                   cvids_test_EventKit=None,
                   cvids_test_BG=None,
                   para_list=None,
                   flag_run_cv=True, cv_indices=None,
                   flag_run_full = True,
                   flag_recompute_best_param_cv = False,
                   path_kernel_BG_check = None,
                   cv_indices_compact_model = [],
                   save_compact_model = False,
                   save_approx_model = False,
                   metric= {'metric_mode':'ap', 'cv_aggregation_mode':'cv_average'}
                   ):
    """
    Run event agent generators (EAG) and produce event agents (EA) using libSVM.
    Generates full SVM model, then, generates all the way towards (optional) compact version of SVM.
    NOTE: this function assumes that all the kernel matrics are pre-computed.
    NOTE: generation of compact_model and approx_model are very memory intensive

    @param path_EAG: path to EAG directory
    @param path_EA: path to directory where deployable libSVM EAs will be stored
    @param path_EVENTKIT: a directory with event kit data (one directory for every event kit)
    @param path_BG: a directory with sampled NONE examples
    @param path_PARAMETER: a directory to find FILE_PARAMETERS to try
    @param clipids_EVENT: (optional) clipids for event kit, in list format
    @param clipids_BG: (optional) clipids for DEV, in list format
    @param labels_EVENT: (optional) labels in numpy array format
    @param labels_BG: (optional) labels in numpy array format
    @param data_EVENT: (optional) data in numpy array format
    @param data_BG: (optional) data in numpy array format
    @param flag_run_cv: run cross-validation. If False, only the full base classifiers using all training data will be generated.
    @param cv_indices: indices of cross validation splits, used for best parameter search
    @param flag_run_full: learn a classifier with all the traning data (beyond cross validation)
    @param flag_recompute_best_param_cv: if True, over-write best parameter based on current setting
    @param path_kernel_BG_check: for kernels involving BG
           if target kernel matrix file exists at the specified path, skip re-computing them.
    @param cv_indices_compact_model: indices of cross validation splits, used to generate compact models
    @param save_compact_model: if True, save compact model with explicit SVs
    @param save_approx_model:    if True, save approximate more efficient versions (only works for additive kernels)
    @param metric: metric to use for classifier optimization. Tuple of (metric_mode, cv_aggregation_mode). Default = ('ap', 'cv_average')
    @type metric: dictionary with
                  'metric_mode' is one of 'ap' (default), 'ap_prime', or ('ap_smoothed', sigma) ('ap_prime_smoothed', sigma)
                  'cv_aggregation_mode' is one of
                            'cv_average' average #|CV| metrics,
                            'cv_concat' compute a metric from concat of CV probs results

    @TODO: add an option to pass best parameter (without requiring to read from a file)
    @TODO:
    """

    if not os.path.exists(path_EA):
        makedirs(path_EA)

    t0 = datetime.datetime.now()
    log(str(t0))

    # copy kernel function file
    file_kernel_dst = os.path.join(path_EA,  FILE_EA_KERNEL)
    if not os.path.exists(file_kernel_dst):
        file_kernel_src = os.path.join(path_EAG, FILE_EAG_KERNEL)
        copy_file(file_kernel_src, file_kernel_dst)

    # Load clipids of all the data involved and write into a single combined file
    # The ordering of scores in the cross validation outputs will match these clipids
    # Also, estimate the size of the all the data
    if clipids_EVENT is None:
        clipids_EVENT = np.loadtxt(os.path.join(path_EVENTKIT, FILE_EVENT_CLIPID)).tolist()
    if clipids_BG is None:
        clipids_BG = np.loadtxt(os.path.join(path_BG, FILE_BG_CLIPID)).tolist()
    n1 = len(clipids_EVENT)  # EventKit
    n2 = len(clipids_BG)  # BG
    n12 = n1 + n2

    # clipis of all the data in the order of EventKit, BG
    clipids_all = list(clipids_EVENT)
    clipids_all.extend(clipids_BG)
    clipids_all = np.array(clipids_all)

    # Write the clipids
    file_clipids_all = os.path.join(path_EA, FILE_EA_CLIPID_ALL)
    np.savetxt(file_clipids_all, clipids_all, fmt='%d')

    #log('n1 = %d, n2 = %d, n3 = %d\n'%(n1, n2, n3))

    # Only load when needed
    data_train = None
    def load_data_train(data_EVENT, data_BG):
        if data_EVENT is None:
            file_train_EVENTKIT = os.path.join(path_EVENTKIT, FILE_EVENT_DATA+'.npy')
            data_EVENT = np.load(file_train_EVENTKIT)
        if data_BG is None:
            file_train_BG = os.path.join(path_BG, FILE_BG_DATA+'.npy')
            data_BG = np.load(file_train_BG)
        data_train = np.vstack((data_EVENT, data_BG))
        return data_train

    # Load information about used kernel
    str_kernel = parse_file_EAG_kernel(os.path.join(path_EAG, FILE_EAG_KERNEL))

    # kernel matrix for training data, rows in the order of EventKit and DEV
    # only load, if needed
    matrix_train = None
    def load_matrix_train(kernel_EVENTxEVENT, kernel_BGxBG, kernel_BGxEVENT):
        # Load multiple modular kernel matrices
        file_matrix_train1 = os.path.join(path_EAG, FILE_KERNEL_EVENTxEVENT)
        file_matrix_train2 = os.path.join(path_EAG, FILE_KERNEL_BGxBG)
        if (not os.path.exists(file_matrix_train2)) and (path_kernel_BG_check is not None):
            file_matrix_train2 = os.path.join(path_kernel_BG_check, FILE_KERNEL_BGxBG)
        file_matrix_train3 = os.path.join(path_EAG, FILE_KERNEL_BGxEVENT)

        matrix_train = np.zeros((n12, n12))
        if kernel_EVENTxEVENT is None:
            kernel_EVENTxEVENT = np.load(file_matrix_train1)
        matrix_train[0:n1, 0:n1] = kernel_EVENTxEVENT
        if kernel_BGxBG is None:
            kernel_BGxBG = np.load(file_matrix_train2)
        matrix_train[n1:n12+1, n1:n12+1] = kernel_BGxBG

        if kernel_BGxEVENT is None:
            kernel_BGxEVENT = np.load(file_matrix_train3)
        matrix_train[n1:n12+1, 0:n1] = kernel_BGxEVENT
        matrix_train[0:n1, n1:n12+1] = kernel_BGxEVENT.T
        return matrix_train

    # load labels for training
    if labels_EVENT is None:
        labels_EVENT = np.loadtxt(os.path.join(path_EVENTKIT, FILE_EVENT_LABEL))
    if labels_BG is None:
        labels_BG = np.loadtxt(os.path.join(path_BG, FILE_BG_LABEL)) # will be all zero
    label_train = np.concatenate((labels_EVENT, labels_BG), axis=0)

    # load parameters into para_list
    para_path = os.path.join(path_PARAMETER, FILE_PARAMETERS) # temporary
    log('FILE_PARAMETERS = %s'%para_path)
    if para_list is None:
        para_list = open(para_path, 'r').readlines()
        para_list = [para.strip() for para in para_list if len(para.strip()) > 0] #remove blanks before and after

    # find the idx of the target class from the learned libSVM model
    # under current setting, target idx is always LABEL_POS(1)
    def find_target_idx(model):
        idx_target = None
        for idx in range(model.nr_class):
            if model.label[idx] == LABEL_POS:
                idx_target = idx
                break
        return idx_target

    def load_cv_splits(cvids_train_EventKit, cvids_train_BG, cvids_test_EventKit, cvids_test_BG):
        cv_splits = dict()

        if cvids_train_EventKit is None:
            file_cvids_train_EventKit = os.path.join(path_EVENTKIT, FILE_EVENT_CV_TRAIN)
            cvids_train_EventKit = np.loadtxt(file_cvids_train_EventKit)
            cv_splits['cvids_train_EventKit'] = cvids_train_EventKit
        else:
            cv_splits['cvids_train_EventKit'] = cvids_train_EventKit

        if cvids_train_BG is None:
            file_cvids_train_BG = os.path.join(path_BG, FILE_BG_CV_TRAIN)
            cvids_train_BG = np.loadtxt(file_cvids_train_BG)
            cv_splits['cvids_train_BG'] = cvids_train_BG
        else:
             cv_splits['cvids_train_BG'] = cvids_train_BG
        cvids_train = np.vstack((cvids_train_EventKit, cvids_train_BG))
        cv_splits['cvids_train'] = cvids_train

        # order in cvids_test [EventKit, BG]
        if cvids_test_EventKit is None:
            file_cvids_test_EventKit = os.path.join(path_EVENTKIT, FILE_EVENT_CV_TEST)
            cvids_test_EventKit = np.loadtxt(file_cvids_test_EventKit)
            cv_splits['cvids_test_EventKit'] = cvids_test_EventKit
        else:
            cv_splits['cvids_test_EventKit'] = cvids_test_EventKit

        if cvids_test_BG is None:
            file_cvids_test_BG = os.path.join(path_BG, FILE_BG_CV_TEST)
            cvids_test_BG = np.loadtxt(file_cvids_test_BG)
            cv_splits['cvids_test_BG'] = cvids_test_BG
        else:
            cv_splits['cvids_test_BG'] = cvids_test_BG

        cvids_test = np.vstack((cvids_test_EventKit, cvids_test_BG))
        cv_splits['cvids_test'] = cvids_test

        return cv_splits


    #######################################
    # Cross-Validation with different
    # parameter sets
    # Then, select the best parameter
    #######################################
    if flag_run_cv == True:

        # Prepare labels and kernel matrix for test folds
        # label_test: the whole list of labels in the order of [EventKit, BG]
        label_test = label_train

        # kernel matrix for testing during training (identical)
        matrix_test = None

        # Prepare cross validation splits
        # CV splits for training/testing
        cv_splits = load_cv_splits(cvids_train_EventKit, cvids_train_BG, cvids_test_EventKit, cvids_test_BG)
        cvids_train = cv_splits['cvids_train']
        cvids_test = cv_splits['cvids_test']

        # copy all the used data splits
        file_cvid_train = os.path.join(path_EA, FILE_EA_CV_TRAIN)
        np.savetxt(file_cvid_train, cvids_train, fmt='%d')
        file_cvid_test = os.path.join(path_EA, FILE_EA_CV_TEST)
        np.savetxt(file_cvid_test, cvids_test, fmt='%d')

        # Save the info regarding number of cross validation folds
        if cv_indices is None:
            cv_indices = range(0, cvids_test.shape[1])
        nCV = len(cv_indices)
        file_cv_idxs = os.path.join(path_EA, FILE_EA_CV_IDXS)
        np.savetxt(file_cv_idxs, np.array(cv_indices), fmt='%d')

        # check number of samples
        if cvids_train.shape[0] != label_train.shape[0]:
            raise Exception("#'s of Label and CVid do not agree!")

        # only the rows corresponding to the *really* tested samples based on cv_indices
        # this is important to draw DET curves + measure performance
        data_indices_tested = (cvids_test[:, cv_indices].sum(axis=1) > 0)
        print 'Total number of samples to be tested during CV: %d / %d'%(data_indices_tested.sum(),
                                                                         cvids_test.shape[0])

        # Cross-validation across multiple parameter sets
        # save results of cross validation
        # save scores of margin / probability, both of them.
        # NOTE: the implementation assumes that each file gets tested all and only once
        #       if it gets tested multiple times, the previous prob/margin is over-written (troublesome)
        #       This issue needs to be addressed.
        # TODO: we would like to parallelize this loop
        for (ip, para) in enumerate(para_list):

            str_log  = '\n\nTrying Parameter: %s (%d / %d)'%(para, (ip+1), len(para_list))
            log(str_log)

            # place to hold params/margins
            dir_para = os.path.join(path_EA, para)
            if not os.path.exists(dir_para):
                makedirs(dir_para)

            # variables to hold validation scores in various forms
            probs_all   = None
            margins_all = None
            ranks_all   = None

            probs_file  = os.path.join(dir_para, FILE_CV_PROBS)
            margin_file = os.path.join(dir_para, FILE_CV_MARGINS)
            ranks_file  = os.path.join(dir_para, FILE_CV_RANKS)

            def load_probs_all():
                if os.path.exists(probs_file):
                    probs_all = np.loadtxt(probs_file)
                else:
                    probs_all = np.zeros((cvids_test.shape[0], 1))
                return probs_all

            def load_margins_all():
                if os.path.exists(margin_file):
                    margins_all = np.loadtxt(margin_file)
                else:
                    margins_all = np.zeros((cvids_test.shape[0], 1))
                return margins_all

            def load_ranks_all():
                if os.path.exists(ranks_file):
                    ranks_all = np.loadtxt(ranks_file)
                else:
                    ranks_all = np.zeros((cvids_test.shape[0], 1))
                return ranks_all

            # Cross validation
            for cv in cv_indices:

                str_log = 'Executing CV fold: %d / Total # %d'%(cv, len(cv_indices))
                # File to save CV DET curve for each fold
                # If exists, then, skip this fold
                #dir_svm_cv_compact = os.path.join(dir_para, FILE_CV_EA_MODEL_LIBSVM_COMPACT_DIR%cv)
                file_cv_model = os.path.join(dir_para, FILE_CV_EA_MODEL_LIBSVM%cv)
                file_fail_cv = os.path.join(dir_para, FILE_FAILURE_CV_EA_MODEL_LIBSVM%cv)
                # IDX file to store the idx location for target class within to-be-learned SVMs
                idx_file = os.path.join(dir_para, FILE_CV_EA_MODEL_LIBSVM_TARGET_IDX%cv)

                if os.path.exists(file_fail_cv):
                    str_log  = '\nFAILURE: cross validation fold %d for paramter: %s'%(cv, para)
                    str_log += '\n- already failed. as indicated by: %s\n'%file_fail_cv
                    log(str_log)
                elif os.path.exists(file_cv_model) and os.path.exists(idx_file): # this line gets printed no matter whether it exists or not
                    str_log  = '\nSkipping cross validation fold %d for paramter: %s'%(cv, para)
                    str_log += '\n- already computed. Following file exists: %s'%file_cv_model
                    str_log += '\n- not repeating.. moving on..'
                    log(str_log)
                else:
                    if matrix_train is None:
                        matrix_train = load_matrix_train(kernel_EVENTxEVENT, kernel_BGxBG, kernel_BGxEVENT)

                    cv_train_flag = (cvids_train[:, cv] == 1)
                    matrix_train_cv = matrix_train[cv_train_flag, :]
                    matrix_train_cv = matrix_train_cv[:, cv_train_flag]
                    matrix_train_cv = kernel_mat_pad(matrix_train_cv)
                    print 'CV training fold %d: # USED %d /  # TOTAL %d'%(cv, cv_train_flag.sum(), cvids_train.shape[0])
                    label_train_cv  = label_train[cv_train_flag]
                    print '# POS = %d / Total %d'%(label_train_cv.sum(), label_train_cv.shape[0])

                    # train CV model
                    problem = svm.svm_problem(label_train_cv.tolist(), matrix_train_cv.tolist(), isKernel=True)
                    svm_param = svm.svm_parameter(para)
                    model_cv = svmutil.svm_train(problem, svm_param)

                    # check failure (zero SVs indicate failure)
                    # NOTE: libSVM indeed sometimes fails
                    if model_cv.l == 0:
                        touch(file_fail_cv) # simply create a failure indicator file and go.

                        str_log = 'libsvm trianing failed, FAILURE file written: %s'%file_fail_cv
                        log(str_log)
                    else: # trianing is a success
                        if matrix_test is None:
                            matrix_test = matrix_train

                        # compute score for the test fold
                        cv_test_flag = (cvids_test[:, cv] == 1)
                        matrix_test_cv = matrix_test[:, cv_train_flag]
                        matrix_test_cv = matrix_test_cv[cv_test_flag, :]
                        labels_test_cv = label_test[cv_test_flag]
                        # label_train_cv  = label_train[cv_train_flag]

                        print 'CV test fold %d: # USED %d /  # TOTAL %d' % (cv, cv_test_flag.sum(), cvids_test.shape[0])
                        print '# POS = %d / Total %d' % (labels_test_cv.sum(), labels_test_cv.shape[0])

                        outputs_cv = svmtools.apply_common_nonlinear_svm(model_cv, matrix_test_cv, model_is_compact=False)
                        probs_cv = outputs_cv['probs']
                        margins_cv = outputs_cv['margins']

                        if probs_all is None:
                            probs_all = load_probs_all()
                        probs_all[cv_test_flag, 0] = probs_cv

                        if margins_all is None:
                            margins_all = load_margins_all()
                        margins_all[cv_test_flag, 0] = margins_cv

                        # todo: add mer during learning

                        # compute ranking by sorting
                        #idxs_sorted = np.argsort(probs_cv)
                        idxs_sorted = np.argsort(margins_cv) # to check whether margin is correctly signed
                        n_idxs = len(idxs_sorted)
                        rank_step = (float(1)/n_idxs)
                        ranks = np.zeros(n_idxs)
                        for (si, _idx) in enumerate(idxs_sorted):
                            ranks[_idx] = rank_step * si
                        if ranks_all is None:
                            ranks_all = load_ranks_all()
                        ranks_all[cv_test_flag, 0] = ranks

                        # Save DET curve of each cross validation model
                        file_det_cv = os.path.join(dir_para, FILE_CV_EA_DET%cv)

                    # save the cross validation model (full version),
                    # models are saved here to indicate success of CV
                    svmutil.svm_save_model(file_cv_model, model_cv)
                    idx_target = find_target_idx(model_cv)
                    with open(idx_file, 'wb') as fin:
                        fin.write('%d'%idx_target)
                    del model_cv

            # After CVs, save all the CV results
            if probs_all is not None:
                np.savetxt(probs_file,  probs_all,   fmt='%g')
            if margins_all is not None:
                np.savetxt(margin_file, margins_all, fmt='%g')
            if ranks_all is not None:
                np.savetxt(ranks_file, ranks_all, fmt='%g')

            # Save DETs over CVs, if there was any new computation
            file_det_para = os.path.join(dir_para, FILE_PARA_EA_DET%para)
            if not os.path.exists(file_det_para):
                if probs_all is None:
                    probs_all = load_probs_all()
                    cv_inputs = []
                    for cv in cv_indices:
                        cv_test_flag = (cvids_test[:, cv] == 1)
                        cv_scores = probs_all[cv_test_flag]
                        cv_labels = label_test[cv_test_flag]
                        cv_input = {'scores': cv_scores,
                                    'labels': cv_labels
                                    }
                        cv_inputs.append(cv_input)

                    plot_parameters = dict()
                    plot_parameters['title'] = 'MED Results across CV folds'
                    plot_parameters['filename_plot'] = file_det_para
                    for (_idx, _cv_idx) in enumerate(cv_indices):
                        cv_inputs[_idx]['plot_parameters'] = dict()
                        cv_inputs[_idx]['plot_parameters']['name']  = str(_cv_idx)
                        cv_inputs[_idx]['plot_parameters']['color'] = 'b'

                    _ = perf.average_precision_R0(cv_inputs, plot_parameters)


        str_log = '\n\nFinished Cross Validation'
        log(str_log)

        #######################################
        # Select Best Parameter
        # Draw performance curves of all the parameters
        #######################################

        # File to store best paramters learned during cross-validation
        file_best_parameter = os.path.join(path_EA, FILE_PARAMETER_BEST)
        file_best_parameter_json = os.path.join(path_EA, FILE_PARAMETER_BEST_JSON)
        if os.path.exists(file_best_parameter) and (flag_recompute_best_param_cv is False):
            str_log  = '\nSkipping computing best parameter..'
            str_log += '\n- it already exists: %s' % file_best_parameter
            str_log += '\nflag_recompute_best_param_cv = %s' % flag_recompute_best_param_cv
            log(str_log)
        else:
            str_log  = '\nEstimating best parameter..'
            log(str_log)

            label_tested = label_test[data_indices_tested]

            # metric type defining best parameter
            metric_mode = metric['metric_mode']
            cv_aggregation_mode = metric['cv_aggregation_mode']

            # across different parameter settings,
            # measure performance, and find the best one
            inputs = []
            outputs = []
            for para in para_list:
                dir_para = os.path.join(path_EA, para)

                # load score inputs
                if cv_aggregation_mode == 'cv_average':
                    scores_file = os.path.join(dir_para, FILE_CV_RANKS)
                elif cv_aggregation_mode == 'cv_concat':
                    scores_file = os.path.join(dir_para, FILE_CV_PROBS)
                scores_all = np.loadtxt(scores_file)
                scores_tested = scores_all[data_indices_tested]
                _input = {'scores': scores_tested,
                          'labels': label_tested
                          }
                inputs.append(_input)

                # compute metric
                if cv_aggregation_mode == 'cv_average':
                    cv_outputs = []
                    for cv in cv_indices:
                        cv_test_flag = (cvids_test[:, cv] == 1)
                        cv_scores = scores_all[cv_test_flag]
                        cv_labels = label_test[cv_test_flag]
                        cv_input = {'scores': cv_scores,
                                    'labels': cv_labels
                                    }
                        cv_output = perf.average_precision_R0([cv_input])
                        cv_outputs.append(cv_output[0])
                    _output = dict()
                    for _key in cv_outputs[0].iterkeys():
                        _output[_key] = sum(map(lambda x: x[_key], cv_outputs)) / len(cv_indices)
                    outputs.append(_output)

                elif cv_aggregation_mode == 'cv_concat':
                    _output = perf.average_precision_R0([_input])
                    outputs.append(_output)
                else:
                    raise Exception('Unknown cv_aggregation_mode', cv_aggregation_mode)

            # save all parameter analysis
            metric_analysis_full = dict()
            metric_analysis_full['params'] = para_list
            metric_analysis_full['results'] = outputs
            easyio.write(file_best_parameter_json, metric_analysis_full)

            # find the best parameter per metric_mode
            _metrics = [output[metric_mode] for output in outputs]
            para_best = None
            para_best_idx = None
            perf_max  = None # higher is better
            for (_idx, _metric) in enumerate(_metrics):
                if perf_max is None:
                    para_best = para_list[_idx]
                    perf_max = _metric
                    para_best_idx = _idx
                else:
                    if perf_max < _metric:
                        para_best = para_list[_idx]
                        perf_max  = _metric
                        para_best_idx = _idx

            # write best parameter
            str_log  = '\nBest Parameter found: %s' % para_best
            log(str_log)
            with open(file_best_parameter, 'w') as f_para_best:
                f_para_best.write(para_best)

            # draw figure
            plot_parameters = dict()
            plot_parameters['title'] = 'MED Results Metrics (%s, %s)' % (cv_aggregation_mode, str(metric_mode))
            filename_plot_params = os.path.join(path_EA, FILE_EA_NAP_PARAM)
            plot_parameters['filename_plot'] = filename_plot_params

            for (_idx, _para) in enumerate(para_list):
                inputs[_idx]['plot_parameters'] = dict()
                inputs[_idx]['plot_parameters']['name']  = _para
                inputs[_idx]['plot_parameters']['color'] = 'b'
                if _idx == para_best_idx:
                    inputs[_idx]['plot_parameters']['color'] = 'r'
                    inputs[_idx]['plot_parameters']['linewidth'] = 4

            # TODO: update perf code to accept pre-computed metrics
            _ = perf.average_precision_R0(inputs, plot_parameters)

            str_log = '\n Performance Curve saved at:\n %s'%filename_plot_params
            log(str_log)

            # draw only the best curve
            filename_plot_best = os.path.join(path_EA, FILE_EA_NAP)
            plot_parameters['filename_plot'] = filename_plot_best
            _ = perf.average_precision_R0([inputs[para_best_idx]], plot_parameters)
            str_log = '\n Best Performance Curve saved at:\n %s'%filename_plot_best
            log(str_log)


        # For best parameter setting, compute and write compact/approximate models
        # Do not save before testing above (prediction). Then, the contents of original SVM changes
        # and prediction results will be weird
        #
        if (save_compact_model == True) or (save_approx_model == True):
            para_best = parse_file_best_parameter_file(file_best_parameter)
            dir_para_best = os.path.join(path_EA, para_best)
            for cv in cv_indices_compact_model:
                file_cv_model = os.path.join(dir_para_best, FILE_CV_EA_MODEL_LIBSVM%cv)

                # if this exists, it failed
                file_cv_fail = os.path.join(dir_para_best, FILE_FAILURE_CV_EA_MODEL_LIBSVM_COMPACT_DIR%cv)
                if os.path.exists(file_cv_fail):

                    file_cv_failure_dir = FILE_FAILURE_CV_EA_MODEL_LIBSVM_COMPACT_DIR%cv
                    touch(file_cv_failure_dir)

                    str_log  = '\nFail!: compact_model or approx model not generated'
                    str_log += '\n- libSVM Originally failed as indicated by %s'%file_cv_fail
                    str_log += '\n- created a failure indicator file: %s'%file_cv_failure_dir
                    str_log += '\n- skipping..'
                    log(str_log)
                # Generate Compact/Approx model only when original training was successful.
                else:
                    dir_svm_cv_compact = os.path.join(dir_para_best, FILE_CV_EA_MODEL_LIBSVM_COMPACT_DIR%cv)
                    if not os.path.exists(dir_svm_cv_compact):
                        makedirs(dir_svm_cv_compact)
                    file_svm_cv_compact_container = os.path.join(dir_svm_cv_compact,
                                                                 FILE_CV_EA_MODEL_LIBSVM_COMPACT_CONTAINER)
                    svm_model_compact_cv = None

                    if save_compact_model == True:
                        # target files
                        file_svm_cv_compact = FILE_CV_EA_MODEL_LIBSVM_COMPACT
                        file_svm_cv_compact_SVs = FILE_CV_EA_MODEL_LIBSVM_COMPACT_SVs

                        if os.path.exists(file_svm_cv_compact_container):
                            str_log  = '\nSkipping Compact model for fold %d with best parameter setting..'%cv
                            str_log += '\n- it already exists..'
                            log(str_log)
                        else:
                            str_log  = '\nComputing Compact model for fold %d with best parameter setting..'%cv
                            log(str_log)

                            model_cv = svmutil.svm_load_model(file_cv_model)
                            if data_train is None:
                                data_train = load_data_train(data_EVENT, data_BG)

                            cv_train_flag = (cvids_train[:, cv] == 1)
                            (svm_model_compact_cv, SVs_cv) = svmtools.get_compact_nonlinear_svm(model_cv,
                                                                                                data_train[cv_train_flag,:])
                            n_SVs_cv = SVs_cv.shape[0]
                            print '# SVs: %d'%n_SVs_cv

                            svmtools.write_compact_nonlinear_svm(file_svm_cv_compact_container,
                                                                 LABEL_POS,
                                                                 file_svm_cv_compact, svm_model=svm_model_compact_cv,
                                                                 file_SVs=file_svm_cv_compact_SVs, SVs=SVs_cv,
                                                                 str_kernel=str_kernel)
                            str_log  = '\nSaved Compact model for fold %d..'%cv
                            str_log += '\n- at : %s'%file_svm_cv_compact_container
                            log(str_log)

                    # Save approximate apporx model only for the best parameter across all CV folds
                    if (save_compact_model == True) and (save_approx_model == True):
                        file_approx_model = os.path.join(dir_svm_cv_compact, FILE_CV_EA_MODEL_LIBSVM_APPROX)
                        if os.path.exists(file_approx_model):
                            str_log = '\nSkipping Computing Existing Approximate Model: %s'%file_approx_model
                            log(str_log)
                        else:
                            str_log = '\nComputing Approximate model...'
                            str_log += '\n.. loading compact model: %s'%file_svm_cv_compact_container

                            # always reload, re-use existing memory seems to have some issue
                            svm_model_compact_cv = svmtools.parse_compact_nonlinear_svm(file_svm_cv_compact_container)

                            str_log += '\n.. loaded..'
                            log(str_log)

                            approx_model = svmtools.compute_approx_nonlinear_SVM(svm_model_compact_cv, verbose=True)
                            svmtools.write_approx_nonlinear_SVM(file_approx_model, approx_model)
                            str_log = '\nApproximate model saved at: %s'%file_approx_model
                            log(str_log)

                            # memory release
                            svm_model_compact_cv = None
                            approx_model = None
    else:
        str_log = '\nNo cross validation specified: flag_run_cv = %s\n'%flag_run_cv
        file_best_parameter = os.path.join(path_EA, FILE_PARAMETER_BEST)
        if not os.path.exists(file_best_parameter):
            para_list = open(para_path, 'r').readlines()
            para_list = [para.strip('\n') for para in para_list] #remove the newline \n
            para_best = para_list[0]
            str_log += '\nBest parameter file does not exist: %s'%file_best_parameter
            str_log += '\n- defaulted to set the best parameter file to have the first in the list: %s'%para_best
            with open(file_best_parameter, 'w') as fout:
                fout.write(para_best)
            str_log += '\nWriting best parameter (the first parameter set from list): %s\n'%para_best
            log(str_log)
        else:
            str_log  = '\nExisting Best parameter file: %s'%file_best_parameter
            str_log += '\n- Parameter set = %s\n'%parse_file_best_parameter_file(file_best_parameter)
            log(str_log)


    #######################################
    # Full SVM training with all the data
    #######################################
    # full (not-compact) nonlinear libSVM model to be generated
    # if it exists, assume that the full process has been completed, and simply pass by return.

    if flag_run_full == True:
        str_log = '\nStarting Full SVM model training...'
        log(str_log)

        model_full = None
        dir_full_model = os.path.join(path_EA, FILE_EA_MODEL_LIBSVM_FULL_DIR)
        file_full_model = os.path.join(dir_full_model, FILE_EA_MODEL_LIBSVM_FULL)
        idx_file = os.path.join(path_EA, FILE_EA_MODEL_LIBSVM_FULL_DIR,
                                FILE_EA_MODEL_LIBSVM_FULL_TARGET_IDX)
        file_full_fail = os.path.join(dir_para, FILE_FAILURE_EA_MODEL_LIBSVM_FULL_DIR)

        if os.path.exists(file_full_fail):
            str_log  = 'Full SVM training already failed,, Do not execute again..'
            str_log += '\n- failure indicator file: %s'%file_full_fail
            log(str_log)
        elif os.path.exists(file_full_model):
            str_log  = '\nSkipping computing full SVM learning with all the training data..'
            str_log += '\n-Full SVM trained with Full data exists: %s'%file_full_model
            log(str_log)
        else: # it has not failed and does not exists already
            str_log  = '\nFull SVM trianing starting....'
            log(str_log)

            # train with all the data, with the best parameters
            if matrix_train is None:
                matrix_train = load_matrix_train(kernel_EVENTxEVENT, kernel_BGxBG, kernel_BGxEVENT)
            problem = svm.svm_problem(label_train.tolist(),
                                      kernel_mat_pad(matrix_train).tolist(),
                                      isKernel=True)
            para_best = parse_file_best_parameter_file(file_best_parameter)
            str_log = 'Best paramter used: %s'%para_best
            log(str_log)

            str_log  = '\Training Full SVM....'
            svm_param = svm.svm_parameter(para_best)
            model_full = svmutil.svm_train(problem, svm_param)

            if not os.path.exists(dir_full_model):
                makedirs(dir_full_model)
            svmutil.svm_save_model(file_full_model, model_full)
            idx_target = find_target_idx(model_full)
            with open(idx_file, 'wb') as fin:
                fin.write('%d'%idx_target)

            str_log += '\n- SVM model saved at: %s'%file_full_model

            # check failure (zero SVs indicate failure)
            # NOTE: libSVM indeed sometimes fails
            if model_full.l == 0:
                touch(file_full_fail) # create failure indicator
                str_log  = '\nFAIL: Full libsvm trianing failed, FAILURE file written: %s'%file_full_fail
                log(str_log)

            # memory release, there's no more training with kernel matrices
            matrix_train = None

        # save compact/approx model, only proceed when full training was successful
        if ((save_compact_model == True) or (save_approx_model == True)):

            file_failure_dir_svm_full = os.path.join(dir_para, FILE_FAILURE_EA_MODEL_LIBSVM_FULL_DIR)
            file_failure_dir_svm_compact = os.path.join(path_EA, FILE_FAILURE_EA_MODEL_LIBSVM_COMPACT_DIR)

            if os.path.exists(file_failure_dir_svm_full):
                touch(file_failure_dir_svm_compact)
                str_log  = '\nFailure: Full libsvm training failed..'
                str_log += '\n- as indicated by: %s'%file_failure_dir_svm_full
                str_log += '\n- NOT Proceeding with Compact SVM with full training examples...'
                str_log += '\n- Failure indicator recorded at: %s'%file_failure_dir_svm_compact
                log(str_log)
            else: # full model exists
                str_log = '\nDeriving Compact SVM model from Full SVM model..'
                log(str_log)

                dir_svm_compact = os.path.join(path_EA, FILE_EA_MODEL_LIBSVM_COMPACT_DIR)
                try:
                    os.mkdir(dir_svm_compact)
                except OSError:
                    pass

                svm_model_compact_full = None
                file_svm_compact_container = os.path.join(dir_svm_compact,
                                                          FILE_EA_MODEL_LIBSVM_COMPACT_CONTAINER)
                file_svm_compact = FILE_EA_MODEL_LIBSVM_COMPACT
                file_svm_compact_SVs = FILE_EA_MODEL_LIBSVM_COMPACT_SVs

                if save_compact_model == True:
                    if os.path.exists(file_svm_compact_container):
                        str_log  = '\nSkipping Compact model for Full Data with best parameter setting..'
                        str_log += '\n- it already exists..: %s'%file_svm_compact_container
                        log(str_log)
                    else:
                        str_log  = '\nComputing Compact model for Full Data with best parameter setting..'

                        if model_full is None:
                            model_full = svmutil.svm_load_model(file_full_model)
                        if data_train is None:
                            data_train = load_data_train(data_EVENT, data_BG)
                        (svm_model_compact_full, SVs) = svmtools.get_compact_nonlinear_svm(model_full, data_train)
                        svmtools.write_compact_nonlinear_svm(file_svm_compact_container,
                                                             LABEL_POS,
                                                             file_svm_compact, svm_model = svm_model_compact_full,
                                                             file_SVs=file_svm_compact_SVs, SVs = SVs,
                                                             str_kernel=str_kernel)

                if (save_compact_model == True) and (save_approx_model == True):
                    file_approx_model = os.path.join(dir_svm_compact, FILE_EA_MODEL_LIBSVM_APPROX)
                    if os.path.exists(file_approx_model):
                        str_log  = '\nSkipping Computing Existing Approximate Model:\n %s'%file_approx_model
                        log(str_log)
                    else:
                        str_log = '\nComputing Approximate model..'
                        log(str_log)

                        # always reload, re-use existing memory seems to have some issue
                        svm_model_compact_full = svmtools.parse_compact_nonlinear_svm(file_svm_compact_container)
                        approx_model = svmtools.compute_approx_nonlinear_SVM(svm_model_compact_full, verbose=True)
                        svmtools.write_approx_nonlinear_SVM(file_approx_model, approx_model)
                        str_log = '\nApproximate model saved at: %s'%file_approx_model
                        log(str_log)
                        approx_model = None # memory release

        dt = datetime.datetime.now() - t0
        log('Full Model Learning Finished after: %s\n'%str(dt))
#        else:
#            str_log  = 'SVM full model files exist:\n'
#            str_log += '- model_file = %s\n'%file_full_model
#            str_log += '- idx_file = %s\n'%idx_file
#            str_log += 'Skipping full model computation...\n'
#            log(str_log)
    else: #flag_run_full == True:
        str_log = '\nSkipping Full Model Learning: flag_run_full = %s\n'%flag_run_full
        log(str_log)


#############################################
# Functions related to Compact SVM agents
#############################################

def generate_compact_event_agents_libSVM(path_EA, path_EVENTKIT, path_NONE):
    """
    Generates compact version of SVM from full libSVM model.

    @param path_EA:  a directory where a self-contained (non-compact & non-linear) event agent (EA) is stored
    @param path_EVENTKIT: a directory with event kit data (one directory for every event kit)
    @param path_NONE: a directory with sampled NONE examples
    """

    # output files to be generated
    dir_svm_compact = os.path.join(path_EA, FILE_EA_MODEL_LIBSVM_COMPACT_DIR)
    makedirs(dir_svm_compact)

    file_svm_compact_container = os.path.join(dir_svm_compact, FILE_EA_MODEL_LIBSVM_COMPACT_CONTAINER)
    file_svm_compact = FILE_EA_MODEL_LIBSVM_COMPACT
    file_svm_compact_SVs = FILE_EA_MODEL_LIBSVM_COMPACT_SVs

    if os.path.exists(file_svm_compact_container):
        log('Skipping: event_agent_generator.generate_compact_event_agents_libSVM\n')
        log('-- Already exists: %s\n'%file_svm_compact_container)
        return 1

    # input files
    file_svm_full    = os.path.join(path_EA, FILE_EA_MODEL_LIBSVM_FULL)
    file_data_train_eventkit = os.path.join(path_EVENTKIT, FILE_EVENT_DATA+'.npy')
    file_data_train_none     = os.path.join(path_NONE, FILE_BG_DATA+'.npy')
    file_target_idx_full     = os.path.join(path_EA, FILE_EA_MODEL_LIBSVM_FULL_TARGET_IDX)
    file_kernel_str  = os.path.join(path_EA, FILE_EA_KERNEL)

    # load inputs
    svm_model_full = svmutil.svm_load_model(file_svm_full)
    target_class_idx = parse_file_single_integer(file_target_idx_full)
    target_class = svm_model_full.label[target_class_idx]
    str_kernel = parse_file_EAG_kernel(file_kernel_str)
    data_train_orig = np.vstack((np.load(file_data_train_eventkit), np.load(file_data_train_none)))
    data_train_orig = preprocess_data_per_kernel(data_train_orig, str_kernel)


    # compute compact models and write, then, log
    # SVs to be stored are in pre-processed form.
    (svm_model_compact, SVs) = svmtools.get_compact_nonlinear_svm(svm_model_full, data_train_orig)
    n_SVs = SVs.shape[0]
    svmtools.write_compact_nonlinear_svm(file_svm_compact_container,
                                         target_class,
                                         file_svm_compact, svm_model = svm_model_compact,
                                         file_SVs = file_svm_compact_SVs, SVs = SVs,
                                         str_kernel = str_kernel)

    # critical memory release
    SVs = None

    log_str = None
    with open(file_svm_compact_container, 'r') as fin:
        log_str = fin.read()

    log_str += '# of Support vectors: %d\n'%n_SVs
    log_str += '\n'
    log_str += 'Stored the following compact model at %s:\n'%file_svm_compact_container + log_str
    log_str += 'Complete: event_agent_generator.generate_compact_event_agents_libSVM\n'
    log_str += '====================================================================\n'
    log(log_str)


def load_ECD_model_libSVM(filepath_model, use_approx_model):
    """ Load learned libSVM model from disk
    @param filepath_model: filepath to the model container
    @type filepath_model: str
    @param use_approx_model: True for models with approximate kernels, only for HIK kernel for now.
    @type use_approx_model: bool
    @return: model
    """
    model = None
    if not use_approx_model:
        model = svmtools.parse_compact_nonlinear_svm(filepath_model)
        # NOTE: this step is important to pre-process SVs based on kernel type
        # this is not correct, since SVs are already preprocessed 'generate_compact_event_agents_libSVM'
        # model['SVs'] = preprocess_data_per_kernel(model['SVs'], model['str_kernel'])
    else:
        # Use Approximate model
        model = svmtools.load_approx_nonlinear_SVM(filepath_model)
    return model

def apply_ECD_model_libSVM(model, features, use_approx_model,
                           preprocess=True, report_margins_recounting=False, verbose=False):
    """
    @param model: loaded libSVM model
    @param features: 2D numpy array with each row being different data
    @type features: numpy.array (2D)
    @param use_approx_model: indicates the model is an approximate model
    @param preprocess: pre-process features based on kernel type
    @type preprocess: bool
    @param report_margins_recounting: enables recounting using margin-based estimates
    @type report_margins_recounting: bool
    @param verbose: enable verbose messages
    @type verbose: bool
    @return: dictionary with 'probs','margins', (optional)'margins_recounting' which are each numpy array of floats
    """

    if preprocess is True:
        features = preprocess_data_per_kernel(features, model['str_kernel'])

    if use_approx_model:
        outputs = svmtools.apply_approx_nonlinear_SVM(model, features, verbose=verbose,
                                                      report_margins_recounting=report_margins_recounting)
    else:
        outputs = svmtools.apply_compact_nonlinear_svm(model, features,
                                                       report_margins_recounting=report_margins_recounting)
    return outputs



def apply_compact_event_agents_libSVM(path_compact_EA, path_TEST_DATA, path_OUTPUT,
                                      use_approx_model = False, fmt_score = '%0.8f',
                                      report_margins_recounting=True,
                                      verbose = False):
    """
    Apply compact libSVM to test data (final step), then, stores Prob/Margin(optional) files

    @param path_compact_EA: a directory with FILE_EA_MODEL_LIBSVM_COMPACT_CONTAINER
    @param path_TEST_DATA: a directory with (part_%04d_clipids.txt, part_%04d.data.npy) file pairs
    @param path_OUTPUT: output directory where all the scoring results by libSVM will be stored
    @param use_approx_model: use approximate model
    @type use_approx_model: bool
    @param fmt_score: format to store scores
    @type fmt_score: str
    @param report_margins_recounting: if True, output bin-wise contribution towards margin (higher, the more contribution)
    @type report_margins_recounting: bool
    @param verbose: if True, outputs more data
    @type verbose: bool
    """

    if not os.path.exists(path_OUTPUT):
        makedirs(path_OUTPUT)

    # Load model with SVs (with pre-processing of SVs based on kernel types)
    print 'eag.apply_compact_event_agents_libSVM'
    print 'path_compact_EA = %s'%path_compact_EA

    model = None
    if not use_approx_model:
        file_model = os.path.join(path_compact_EA, FILE_EA_MODEL_LIBSVM_COMPACT_CONTAINER)
        print 'FILE_EA_MODEL_LIBSVM_COMPACT_CONTAINER = %s'%file_model
        model = load_ECD_model_libSVM(file_model, False)
    else:
        # Use Approximate model
        file_model = os.path.join(path_compact_EA, FILE_EA_MODEL_LIBSVM_APPROX)
        print 'FILE_EA_MODEL_LIBSVM_APPROX = %s'%file_model
        model = load_ECD_model_libSVM(file_model, True)

    # Load test data, and compute scores
    files_features = sorted(glob.glob(os.path.join(path_TEST_DATA, FILE_TEST_DATA_PATTERN_FEATURES)))
    str_log = '# of found files with pattern %s: %d'%(FILE_TEST_DATA_PATTERN_FEATURES, len(files_features))
    log(str_log)

    file_ids =  map(FUNC_GET_ID_FROM_FILE_TEST_DATA, files_features)
    files_clipids = [os.path.join(path_TEST_DATA, FILE_TEST_DATA_PATTERN_CLIPIDS%_id) for _id in file_ids]
    for file_clipid in files_clipids:
        if not os.path.exists(file_clipid):
            str_log = 'Clipid file missing, Aborting: %s\n'%file_clipid
            log(str_log)
            return 0
        else:
            copy_file_clipid = os.path.join(path_OUTPUT, os.path.basename(file_clipid))
            #print 'file_clipid = %s'%file_clipid
            #print 'copy_file_clipid = %s'%copy_file_clipid
            copy_file(file_clipid, copy_file_clipid)

    # output files
    files_probs   = [os.path.join(path_OUTPUT, FILE_TEST_PROB_PATTERN%_id) for _id in file_ids]
    files_margins = [os.path.join(path_OUTPUT, FILE_TEST_MARGIN_PATTERN%_id) for _id in file_ids]
    files_margins_mer = [os.path.join(path_OUTPUT, FILE_TEST_MARGIN_MER_PATTERN%_id) for _id in file_ids]

    # tuples of processing jobs (inputs/outputs)
    tuples = zip(files_features, files_clipids, files_probs, files_margins, files_margins_mer)
    str_log = '# of subsets to score: %d'%len(tuples)
    log(str_log)

    # Process
    for (i, (file_feature, file_clipid, file_prob, file_margin, file_margin_mer)) in enumerate(tuples):
        str_log = '... processeing %d / %d'%(i+1, len(tuples))
        log(str_log)

        if not(os.path.exists(file_prob) and os.path.exists(file_margin)):
            t0 = datetime.datetime.now()

            features = np.load(file_feature)
            features = preprocess_data_per_kernel(features, model['str_kernel'])
            clipids  = np.loadtxt(file_clipid)
            n1 = features.shape[0]
            n2 = len(clipids)
            if n1 != n2:
                str_log  = '# Features (%d) != # Clip IDs (%d)\n'%(n1, n2)
                str_log += 'Feature file: %s\n'%file_feature
                str_log += 'Clip ID file: %s\n'%file_clipid
                str_log += 'Aborting...\n'
                str_log += '====================================================\n'
                log(str_log)
                return 0

            outputs = None

            # apply model on the test search data
            outputs = apply_ECD_model_libSVM(model, features,
                                             use_approx_model, preprocess=False,
                                             report_margins_recounting=report_margins_recounting,
                                             verbose=verbose)

            # probability outputs computed by SVM
            if outputs.has_key('probs'):
                probs   = outputs['probs']
                with open(file_prob, 'wb') as fin:
                    for i in range(n1):
                        fin.write(('%d '+fmt_score+'\n')%(clipids[i], probs[i]))
                log('Prob file stored: %s\n'%file_prob)

            # (optional) margins computed by SVM
            if outputs.has_key('margins'):
                margins = outputs['margins']
                with open(file_margin, 'wb') as fin:
                    for i in range(n1):
                        fin.write(('%d '+fmt_score+'\n')%(clipids[i], margins[i]))
                log('Margin file stored: %s\n'%file_margin)

            # (optional) margins computed by SVM
            if outputs.has_key('margins_recounting'):
                if outputs['margins_recounting'] is not None:
                    margins_mer = outputs['margins_recounting']
                    with open(file_margin_mer, 'wb') as fin:
                        for i in range(n1):
                            fin.write('%d '%clipids[i])
                            for v in margins_mer[i]:
                                fin.write((fmt_score + ' ')%v)
                            fin.write('\n')
                    log('Margin MER file stored: %s\n'%file_margin_mer)


            # (optional) recounting computed by SVM, such as per-bin evidence strengths
            # fill-in code here, after updating kernel computation codes

            dt = datetime.datetime.now() - t0
            log('...Duration for single subset = %s\n\n'%str(dt))
        else:
            str_log  = 'Skipped Existing Prob file: %s\n'%file_prob
            str_log += 'Skipped Existing Margin file: %s\n'%file_margin
            log(str_log)


def parse_full_SVM_model(path_EA, path_EVENTKIT, path_NONE):
    """
    load a full SVM model:
    @param path_EA: event Agent directory
    @param path_EVENTKIT: eventkit directory
    @param path_NONE: DEV kit directory
    @return: SVM model
    """

    model = dict()
    # Load SVM full model
    file_model = os.path.join(path_EA, FILE_EA_MODEL_LIBSVM_FULL)
    model['svm_model']    = svmutil.svm_load_model(file_model)
    str_log = 'Load full SVM model from %s\n\n' %file_model
    log(str_log)
    # load the target class label
    file_target_idx_full  = os.path.join(path_EA, FILE_EA_MODEL_LIBSVM_FULL_TARGET_IDX)
    model['target_class'] = parse_file_single_integer(file_target_idx_full)
    str_log = 'Load SVM target class from %s\n\n' %file_target_idx_full
    log(str_log)
    # load the kernel type and function
    file_kernel_str  = os.path.join(path_EA, FILE_EA_KERNEL)
    model['str_kernel']   = parse_file_EAG_kernel(file_kernel_str)
    model['func_kernel']  = getattr(kernels, model['str_kernel'])
    str_log = 'Load SVM kernel function from %s\n\n' %file_kernel_str
    log(str_log)
    # load the training samples
    file_data_train_eventkit = os.path.join(path_EVENTKIT, FILE_EVENT_DATA+'.npy')
    file_data_train_none     = os.path.join(path_NONE, FILE_BG_DATA+'.npy')
    data_train_orig = np.vstack((np.load(file_data_train_eventkit), np.load(file_data_train_none)))
    model['train_data'] = preprocess_data_per_kernel(data_train_orig, model['str_kernel'])
    str_log = 'Load SVM training data from %s and %s\n\n' %(file_data_train_eventkit, file_data_train_none)
    log(str_log)
    str_log = '# training samples for full SVMs: %d\n' %data_train_orig.shape[0]
    log(str_log)
    return model

def apply_full_event_agents_libSVM(path_EA, path_TEST_DATA, path_EVENTKIT, path_NONE, path_OUTPUT,
                                   fmt_score = '%0.8f',
                                   report_margins_recounting = True):
    """
    Apply full libSVM to test data (final step)
    @param path_full_EA: a directory with FILE_EA_MODEL_LIBSVM_FULL_CONTAINER
    @param path_TEST_DATA: a directory with (part_%04d_clipids.txt, part_%04d.data.npy) file pairs
    @param path_EVENTKIT: eventkit directory
    @param path_NONE: pure negative training examples

    @param path_OUTPUT: output directory where all the scoring results by libSVM will be stored

    @param fmt_score: format to store scores
    """

    if not os.path.exists(path_OUTPUT):
        makedirs(path_OUTPUT)

    # load the full SVM model
    model = parse_full_SVM_model(path_EA, path_EVENTKIT, path_NONE)

    # Load test data, and compute scores
    files_features = glob.glob(os.path.join(path_TEST_DATA, FILE_TEST_DATA_PATTERN_FEATURES))
    log('# of found files with pattern %s: %d'%(FILE_TEST_DATA_PATTERN_FEATURES, len(files_features)))

    file_ids =  map(FUNC_GET_ID_FROM_FILE_TEST_DATA, files_features)
    files_clipids = [os.path.join(path_TEST_DATA, FILE_TEST_DATA_PATTERN_CLIPIDS%_id) for _id in file_ids]
    for file_clipid in files_clipids:
        if not os.path.exists(file_clipid):
            str_log = 'Clipid file missing, Aborting: %s\n'%file_clipid
            log(str_log)
            return 0

    # output files
    files_probs   = [os.path.join(path_OUTPUT, FILE_TEST_PROB_PATTERN%_id) for _id in file_ids]
    files_margins = [os.path.join(path_OUTPUT, FILE_TEST_MARGIN_PATTERN%_id) for _id in file_ids]
    files_margins_mer = [os.path.join(path_OUTPUT, FILE_TEST_MARGIN_MER_PATTERN%_id) for _id in file_ids]

    # tuples of processing jobs (inputs/outputs)
    tuples = zip(files_features, files_clipids, files_probs, files_margins, files_margins_mer)
    str_log = '# of subsets to score: %d'%len(tuples)
    log(str_log)

    # Process
    for (file_feature, file_clipid, file_prob, file_margin, file_margin_mer) in tuples:
        if not(os.path.exists(file_prob) and os.path.exists(file_margin)):
            t0 = datetime.datetime.now()
            features = np.load(file_feature)
            features = preprocess_data_per_kernel(features, model['str_kernel'])
            clipids  = np.loadtxt(file_clipid)
            n1 = features.shape[0]
            n2 = len(clipids)
            if n1 != n2:
                str_log  = '# Features (%d) != # Clip IDs (%d)\n'%(n1, n2)
                str_log += 'Feature file: %s\n'%file_feature
                str_log += 'Clip ID file: %s\n'%file_clipid
                str_log += 'Aborting...\n'
                str_log += '====================================================\n'
                log(str_log)
                return 0

            outputs = svmtools.apply_full_nonlinear_svm(model, features,
                                                        report_margins_recounting=report_margins_recounting)

            # probability outputs computed by SVM
            if outputs.has_key('probs'):
                probs   = outputs['probs']
                with open(file_prob, 'wb') as fin:
                    for i in range(n1):
                        fin.write(('%d '+fmt_score+'\n')%(clipids[i], probs[i]))
                log('Prob file stored: %s\n'%file_prob)

            # (optional) margins computed by SVM
            if outputs.has_key('margins'):
                margins = outputs['margins']
                with open(file_margin, 'wb') as fin:
                    for i in range(n1):
                        fin.write(('%d '+fmt_score+'\n')%(clipids[i], margins[i]))
                log('Margin file stored: %s\n'%file_margin)

            # (optional) margins computed by SVM
            if outputs.has_key('margins_recounting'):
                margins_mer = outputs['margins_recounting']
                with open(file_margin_mer, 'wb') as fin:
                    for i in range(n1):
                        fin.write('%d '%clipids[i])
                        for v in margins_mer[i]:
                            fin.write((fmt_score + ' ')%v)
                        fin.write('\n')
                log('Margin MER file stored: %s\n'%file_margin_mer)

            dt = datetime.datetime.now() - t0
            log('...Duration for single subset = %s\n\n'%str(dt))
        else:
            str_log  = 'Skipped Existing Prob file: %s\n'%file_prob
            str_log += 'Skipped Existing Margin file: %s\n'%file_margin
            log(str_log)

def load_parameter_best(path_EA):
    """
    Load the best parameter estimated
    @param path_EA: path to the generated event agent
    @return: best parameter string
    """
    filename = os.path.join(path_EA, FILE_PARAMETER_BEST)
    with open(filename, 'r') as fin:
        paras = fin.readlines()
    para_best = paras[0]
    return para_best


# main
if __name__ == "__main__":

    pass
