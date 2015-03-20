"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

####################################
# Routines to pre-process features
####################################

import os
import shutil
import numpy as np
import glob

###############################################################
# Top-Level generic pre-processing function: learn, get, apply
###############################################################

def learn_parameters(mode, data_in, params_in):
    """
    Top-level generic pre-processing parameter learning function
    """

    if mode == 'whiten_positive':
        return learn_whiten_positive(data_in, params_in)
    elif mode == 'norm_sqrt_whiten_positive':
        return learn_norm_sqrt_whiten_positive(data_in, params_in)
    else:
        msg_exception = 'Unknown preprocessing type: %s'%mode
        raise Exception(msg_exception)

def get_parameters(mode, files):
    """
    Load parameters based on parameter file locations
    """
    if mode == 'whiten_positive':
        return get_parameters_whiten_positive(files)
    elif mode == 'norm_sqrt_whiten_positive':
        return get_parameters_norm_sqrt_whiten_positive(files)
    else:
        msg_exception = 'Unknown preprocessing type: %s'%mode
        raise Exception(msg_exception)

def apply_parameters(mode, data_in, params):
    """
    Apply loaded parameters to data
    """

    if mode == 'whiten_positive':
        return apply_parameters_whiten_positive(data_in, params)
    elif mode == 'norm_sqrt_whiten_positive':
        return apply_parameters_norm_sqrt_whiten_positive(data_in, params)
    elif mode == 'norm_l1':
        return apply_norm_l1(data_in)
    else:
        msg_exception = 'Unknown preprocessing type: %s'%mode
        raise Exception(msg_exception)

def apply_parameters_batch(mode, params,
                           dir_src, dir_dst,
                           glob_copy='*.txt',
                           glob_apply='*.npy',
                           verbose = True):
    """
    @param mode: preprocessing mode
    @param params: loaded preprocessing parameters
    @param dir_src: Source directory
    @param dir_dst: Destination directory
    @param glob_copy: files that match the glob pattern will be simply copied over
    @param glob_apply: files that match the pattern will be pre-processed, then, saved with same filenames
    """

    files_src_copy = sorted(glob.glob(os.path.join(dir_src, glob_copy)))
    for _file_src in files_src_copy:
        basename = os.path.basename(_file_src)
        _file_dst = os.path.join(dir_dst, basename)
        copy_file(_file_src, _file_dst)

    files_src_apply = sorted(glob.glob(os.path.join(dir_src, glob_apply)))
    for _file_src in files_src_apply:
        basename = os.path.basename(_file_src)
        _file_dst = os.path.join(dir_dst, basename)

        # this is incomplete implementation, only limited to '*.npy'
        if verbose:
            print 'processing: %s'%_file_src
        data_src  = np.load(_file_src)
        data_dst = apply_parameters(mode, data_src, params)
        np.save(_file_dst, data_dst)



######################################################
# Functions related to 'Whiten + Positive'
#######################################################

def learn_whiten_positive(data_in,
                          params_in = {'dir_param' : None,
                                       'file_means' : 'means.txt',
                                       'file_stds'  : 'stds.txt',
                                       'file_minimums' :'minimums.txt'}
                          ):
    """
    Computes the parameters to
    (1) whiten using normalization based on estimated normal distribution (mean & std), then,
    (2) make all feature dimenisons positive by subtracting the minimums for each diemnsion.
    @param data_in: row-wise numpy array
    @param params_in['dir_param']    : optional directory to save parameter files, If not None, files will be saved.
    @param params_in['file_means']   : parameter file to store 'means'
    @param params_in['file_stds']    : parameter file to store 'stds'
    @param params_in['file_minimums']: parameter file to store 'minimums'
    @return: params_out['means', 'stds', 'minimums']
    """

    means = np.mean(data_in, axis=0)
    stds  = np.std(data_in, axis=0)

    whitened = (data_in - means)/stds
    minimums = np.amin(whitened, axis=0)

    _file_means    = None
    _file_stds     = None
    _file_minimums = None

    if params_in['dir_param'] is not None:
        _file_means    = os.path.join(params_in['dir_param'], params_in['file_means'])
        _file_stds     = os.path.join(params_in['dir_param'], params_in['file_stds'])
        _file_minimums = os.path.join(params_in['dir_param'], params_in['file_minimums'])

        np.savetxt(_file_means,    means,    fmt='%g')
        np.savetxt(_file_stds,     stds,     fmt='%g')
        np.savetxt(_file_minimums, minimums, fmt='%g')

    params_out = {'means'    : means,
                  'stds'     : stds,
                  'minimums' : minimums}
    return params_out

def get_parameters_whiten_positive_default():
    files = {'dir_param' : None,
             'file_means'    : 'means.txt',
             'file_stds'     : 'stds.txt',
             'file_minimums' : 'minimums.txt'}
    return files


def get_parameters_whiten_positive(files = {'dir_param' : None,
                                            'file_means'    : 'means.txt',
                                            'file_stds'     : 'stds.txt',
                                            'file_minimums' : 'minimums.txt'}):
    """
    Load the parameters of 'whiten + positive' preprocessing
    @param dir_param: (optional) directory to find parameter files, if None, raw parameter names will be tried.
    @param file_means: file with parameters 'means'
    @param file_stds:file with parameters 'stds'
    @param file_minimums: file with parameters 'minimums'
    @return: params['means', 'stds', 'minimums']
    """

    _file_means    = None
    _file_stds     = None
    _file_minimums = None

    if files['dir_param'] is None:
        _file_means    = files['file_means']
        _file_stds     = files['file_stds']
        _file_minimums = files['file_minimums']
    else:
        _file_means    = os.path.join(files['dir_param'], files['file_means'])
        _file_stds     = os.path.join(files['dir_param'], files['file_stds'])
        _file_minimums = os.path.join(files['dir_param'], files['file_minimums'])

    params = {}
    params['means']    = np.loadtxt(_file_means)
    params['stds']     = np.loadtxt(_file_stds)
    params['minimums'] = np.loadtxt(_file_minimums)
    return params


def apply_parameters_whiten_positive(data_in, params):
    """
    Pre-process data with give parameters
    @return: processed row-wise numpy array data
    """
    data_out = ((data_in - params['means']) / params['stds']) - params['minimums']
    tmp_negs = (data_out < 0.0)
    data_out[tmp_negs] = 0

    return data_out



###########################################################
# Functions related to 'L1Norm'
###########################################################

def norm_l1(data_in):
    _data_in = np.copy(data_in)
    [n,m] = _data_in.shape
    for i in range(n):
        row_sum = _data_in[i].sum()
        if row_sum > 0:
            _data_in[i] /= row_sum
        else:
            uniform = np.ones(m) / float(m)
            _data_in[i] = uniform

    return _data_in

def apply_norm_l1(data_in):
    """
    Pre-process data with give parameters
    @return: processed row-wise numpy array data
    """

    return norm_l1(data_in)

###########################################################
# Functions related to 'L1Norm + Sqrt + Whiten + Positive'
###########################################################

def norm_sqrt(data_in):
    _data_in = norm_l1(data_in)
    _data_in = np.sqrt(_data_in) # sqrt
    return _data_in

def learn_norm_sqrt_whiten_positive(data_in,
                                    params_in = {'dir_param' : None,
                                                 'file_means' : 'means.txt',
                                                 'file_stds'  : 'stds.txt',
                                                 'file_minimums' :'minimums.txt'}
                                    ):
    """
    Computes the parameters to
    (1) whiten using normalization based on estimated normal distribution (mean & std), then,
    (2) make all feature dimenisons positive by subtracting the minimums for each diemnsion.
    @param data_in: row-wise numpy array
    @param params_in['dir_param']    : optional directory to save parameter files, If not None, files will be saved.
    @param params_in['file_means']   : parameter file to store 'means'
    @param params_in['file_stds']    : parameter file to store 'stds'
    @param params_in['file_minimums']: parameter file to store 'minimums'
    @return: params_out['means', 'stds', 'minimums']
    """

    _data_in = norm_sqrt(data_in)

    return learn_whiten_positive(_data_in, params_in)


def get_parameters_norm_sqrt_whiten_positive(files = {'dir_param' : None,
                                                      'file_means'    : 'norm_sqrt_means.txt',
                                                      'file_stds'     : 'norm_sqrt_stds.txt',
                                                      'file_minimums' : 'norm_sqrt_minimums.txt'}):
    """
    Load the parameters of 'L1norm + sqrt + whiten + positive' preprocessing
    @param dir_param: (optional) directory to find parameter files, if None, raw parameter names will be tried.
    @param file_means: file with parameters 'means'
    @param file_stds:file with parameters 'stds'
    @param file_minimums: file with parameters 'minimums'
    @return: params_out['means', 'stds', 'minimums']
    """
    return get_parameters_whiten_positive(files)


def apply_parameters_norm_sqrt_whiten_positive(data_in, params):
    """
    Pre-process data with give parameters
    @return: processed row-wise numpy array data
    """

    _data_in = norm_sqrt(data_in)

    return apply_parameters_whiten_positive(data_in, params)



#######################
# Utility functions
#######################

def copy_file(src, dst):
    """
    This is more robust on linux
    """
    try:
        shutil.copy2(src, dst)
    except OSError:
        if os.path.exists(dst):
            pass
        else:
            raise

def makedirs(path):
    """
    This is more robust on linux
    """
    try:
        os.makedirs(path)
    except OSError:
        if os.path.exists(path):
            pass
        else:
            raise

