"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import numpy as np
import math
import timeit
#import homkermap_wrapper as homkermap
#import multiprocessing as mp

__all__ = ['hik', 'compute_hik_matrix_help', 'compute_hik_matrix']

# Do not change this
IDXS_KERNELS = []
IDXS_KERNELS.append(('linear', 0))
IDXS_KERNELS.append(('ngd', 1))
IDXS_KERNELS.append(('hik', 2))


def compute_kernel_matrix(data1, data2=None, func_kernel=None,
                          recounting=False,
                          diag_zero=False, verbose=-1, mirror=False):
    """
    Given two numpy matrices as inputs (with row-wise d-dimensional data),
    compute kernel values between row-wise data1 & data2, and return the matrix.
    By default, diagonal is computed too, but, can be optionally turned off to be set to be zeros

    @param data1: n1-by-d row-wise data matrix in numpy format
    @param data2: n2-by-d row-wise data matrix in numpy format
    @param func_kernel: kernel function (e.g., functions from kernels library)
    @param report_margins_recounting: if True, when appropriate, output n1-d-DIM matrix to show contribution by each bin towards kernel computation. DIM is the feature dimension.
    @type report_margins_recounting: bool
    @param diag_zero: if True, set all the diagonal entries to zero, default = False
    @param verbose: integer, generate more message
    @param mirror: if True (should be set only when kernel matrix is square), compute lower left, then, copy to top right

    @return: a dictionary with 'kernel_matrix' (n1-by-n2), and (optional) 'kernel_matrix_recounting' (n1-by-n2-by-dim)
    """

    n1 = data1.shape[0]

    if data2 == None:
        data2 = data1

    n2 = data2.shape[0]

    dim = len(data1[0])

    #print 'Kernel Matrix size = (%d,%d)'%(n1,n2)

    mat = np.zeros((n1,n2), dtype=np.float32)
    mat_recounting = None
    if recounting:
        mat_recounting = np.zeros((n1,n2,dim))

    if mirror == True:
        for i in range(n1):
            if verbose!=-1 and (i%verbose)==0:
                print '\t row %d ..'%i

            for j in range(i+1):
                if j==i and diag_zero:
                    continue

                # consider using inc_bin_kernel here
                output = func_kernel(data1[i], data2[j], recounting = recounting)
                mat[i,j] = output['kernel']
                mat[j,i] = mat[i,j]
                if recounting:
                    mat_recounting[i,j] = output['recounting']
                    mat_recounting[j,i] = mat_recounting[i,j]
    else:
        for i in range(n1):
            if verbose!=-1 and (i%verbose)==0:
                print '\t row %d ..'%i

            for j in range(n2):
                if j==i and diag_zero:
                    continue

                # consider using inc_bin_kernel here
                output = func_kernel(data1[i], data2[j], recounting = recounting)
                mat[i,j] = output['kernel']
                if recounting:
                    mat_recounting[i,j] = output['recounting']

    outputs = dict()
    outputs['kernel_matrix'] = mat
    outputs['kernel_matrix_recounting'] = mat_recounting

    return outputs


######################################
# HIK: histogram intersection kernels
######################################

def hik(hist1, hist2, recounting=False):
    """histogram intersection kernel between two histograms
    @param hist1: vector
    @type hist1: 1D numpy.array
    @param hist2: vector
    @type hist2: 1D numpy.array
    @param recounting: if True, return bin-wise outputs as well
    @type recounting: bool
    @return: dictionary of 'kernel' value and (optional) 'recounting'
    @rtype: dict of (float, (optional) np.array)
    """

    bins = (hist1 + hist2 - np.abs(hist1 - hist2)) * 0.5
    output = dict()
    output['kernel'] = bins.sum()
    if recounting:
        output['recounting'] = bins
    return output

def hik1(hist1, hist2, recounting = False):
    """histogram intersection kernel between two histograms
    another hik implementation.append
    NOTE: much slower than hik, don't use. This is slower than hik1

    @param hist1: vector
    @type hist1: 1D numpy.array
    @param hist2: vector
    @type hist2: 1D numpy.array
    @param recounting: if True, return bin-wise outputs as well
    @type recounting: bool
    @return: dictionary of 'kernel' value and (optional) 'recounting'
    @rtype: dict of (float, (optional) np.array)
    """

    bins = np.amin(np.vstack((hist1, hist2)), axis=0)
    output = dict()
    output['kernel'] = bins.sum()
    if recounting:
        output['recounting'] = bins
    return output

def compare_hik_hik1(n=500, d=100, repeat=5):
    """
    Compare speed of hik vs hik1 implementation,
    by computing HIK matrix on randomly generated data

    @param n: number of data points
    @param d: dimension of every vector
    """

    import time

    for i in range(repeat):
        print '------------------------------------------------------'
        print 'testing %d / %d'%(i+1, repeat)
        print '------------------------------------------------------'

        data = np.random.rand(n, d)
        print '... data sampled'

        print '... computing matrix 1'
        start1 = time.clock()
        kernel_matrix1 = compute_kernel_matrix(data, func_kernel=hik)['kernel_matrix']
        time1 = time.clock() - start1

        print '... computing matrix 2'
        start2 = time.clock()
        kernel_matrix2 = compute_kernel_matrix(data, func_kernel=hik1)['kernel_matrix']
        time2 = time.clock() - start2

        print 'time by hik = ', time1
        print 'time by hik1 = ', time2
        print 'results match each other = ', np.array_equal(kernel_matrix1, kernel_matrix2)



#
#def compute_hik_matrix_help(data1, data2=None, diag_zero=False):
#    """
#    Given two numpy matrices as inputs,
#    compute HIK matrix between row-wise data1 & data2,
#    and return the matrix.
#    By default, diagonal is computed too, but, can be optionally turned off to be set to be zeros
#    """
#
#    return compute_kernel_matrix(data1, data2, func_kernel=hik, diag_zero=diag_zero)
# The code below is very slow, not worth it unless massively (>100 threads) perhaps.
#    n1 = data1.shape[0]
#
#    if data2 == None:
#        data2 = data1
#
#    n2 = data2.shape[0]
#
#    print 'HIK matrix size = (%d,%d)'%(n1,n2)
#
#    nr_processor = 7
#    pool = mp.Pool(nr_processor)
#
#    def compute_ij(job):
#        (data1_, data2_, mat_, i, j) = job
#        mat_[i,j] = hik(data1_[i], data2_[j])
#
#    mat = np.zeros((n1,n2), dtype=np.float32)
#
#    jobs = []
#    for i in range(n1):
#        for j in range(n2):
#            jobs.append((data1, data2, mat, i, j))
#
#    pool.map(compute_ij, jobs)
#    return mat



#def compute_hik_matrix(filein1, filein2, fileout):
#    """
#    Compute full HIK kernel matrix between two sets or itself
#    Possibly for pagerank algorithm
#    """
#    data1 = np.loadtxt(filein1)
#    print 'compute_hik_matrix: loaded data1 = %s'%filein1
#    data2 = None
#    if filein2 == None:
#        data2 = data1
#    else:
#        data2 = np.loadtxt(filein2)
#
#    print 'compute_hik_matrix: loaded data2 = %s'%filein2
#    mat = compute_hik_matrix_help(data1, data2, True)
#
#    # make each row into a markov transition probabilities summing to one
#    n1 = data1.shape[0]
#    for i in range(n1):
#        sumi = mat[i,:].sum()
#        if sumi > 0:
#            mat[i,:] /= sumi
#
#    np.savetxt(fileout, mat, fmt='%g')

######################################
# Linear kernel (very basic)
######################################

def linear(data1, data2=None, recounting=False):
    """
    Simple linear kernel

    @param recounting: if True, return bin-wise outputs as well
    @type recounting: bool
    @return: dictionary of 'kernel' value and (optional) 'recounting'
    @rtype: dict of (float, (optional) np.array)
    """
    if data2 == None:
        data2 = data1

    bins = data1 * data2
    kernel = bins.sum()

    output = dict()
    output['kernel'] = kernel
    if recounting:
        output['recounting'] = bins
    return output

def L1norm_linear(data1, data2=None, recounting=False):
    """
    Linear kernel computation followed by L1-normalization.
    It is assumed that L1 normalization is already conducted.
    Hence, simply 'linear' kernel is computed.

    @param recounting: if True, return bin-wise outputs as well
    @type recounting: bool
    @return: dictionary of 'kernel' value and (optional) 'recounting'
    @rtype: dict of (float, (optional) np.array)
    """
    return linear(data1, data2=data2, recounting = recounting)


######################################
# Homogeneous kernel map
# based on Andrea Vedaldi's work
######################################

#def compute_kernel_matrix_homkermap(data1, data2=None,
#                                    N=3, kernelType = homkermap.VlHomogeneousKernelIntersection,
#                                    gamma = 1.0, period = -1.0,
#                                    windowType = homkermap.VlHomogeneousKernelMapWindowRectangular,
#                                    dtype = np.float32):
#    """
#    Approximate kernel matrix computation using Homogeneous kernel map.
#    Supports HIK, Chi2, Jensen-Shannon
#    For detailed parameter specification, please look at homkermap_wrapper.homkermap
#
#    @param data1: row-wise data array in numpy array format
#    @param data2: row-wise data array in numpy array format
#    @return: kernel matrix between data1 & data2.
#    """
#
#    data1_map = homkermap.homkermap(data1, N=N, kernelType=kernelType,
#                                    gamma=gamma, period=period, windowType=windowType,
#                                    dtype = dtype)
#    data1_map = np.mat(data1_map)
#
#    if data2 == None:
#        data2_map = data1_map
#    else:
#        data2_map = homkermap.homkermap(data2, N=N, kernelType=kernelType,
#                                        gamma=gamma, period=period, windowType=windowType,
#                                        dtype = dtype)
#
#    n1 = data1_map.shape[0]
#    n2 = data2_map.shape[0]
#
#    mat = data1_map * data2_map.T
#
#
##    #print 'Kernel Matrix size = (%d,%d)'%(n1,n2)
##    mat = np.zeros((n1,n2), dtype=np.float32)
##
##    if data2 == None:
##        for i in range(n1):
##            for j in range(i+1):
##                mat[i,j] = np.inner(data1_map[i], data2_map[j])
##                mat[j,i] = mat[i,j]
##    else:
##        for i in range(n1):
##            for j in range(n2):
##                mat[i,j] = np.inner(data1[i], data2[j])
#    return mat



####################################################################################################
# NGD: negative geodesic distance kernel
# ala  Dell Zhang's SIGIR 2005 paper "Text Classification with Kernels on the Multinomial Manifold
####################################################################################################

def ngd(mult1_sqrt, mult2_sqrt, recounting=False):
    """
    negative geodesic distance Kernel for normalized histograms (equivalently multinomial)
    ala  Dell Zhang's SIGIR 2005 paper

    @param mult1_sqroot: Square root vector of L1-normalized multinomial (histogram), numpy format
    @param mult2_sqroot: same as above for mult1_sqroot
    @param recounting: if True, return bin-wise outputs as well
    @type recounting: bool
    @return: dictionary of 'kernel' value and (optional) 'recounting'
    @rtype: dict of (float, (optional) np.array)
    """

    bins = mult1_sqrt * mult2_sqrt
    tmp1 = bins.sum()
    if tmp1 > 1.0:
        tmp1 = 1.0
    kernel = (-2.0 * math.acos(tmp1))

    output = dict()
    output['kernel'] = kernel
    if recounting:
        output['recounting'] = bins
    return output


def ngd_dist(mult1_sqroot, mult2_sqrt):
    """
    negative geodesic distance! (not kernel) for normalized histograms (equivalently multinomial)
    ala  Dell Zhang's SIGIR 2005 paper

    @param mult1_sqroot: Square root vector of L1-normalized multinomial (histogram), numpy format
    @param mult2_sqroot: same as above for mult1_sqroot
    @return: ngd distance
    """
    tmp1 = np.inner(mult1_sqroot, mult2_sqrt);
    if tmp1 > 1.0:
        tmp1 = 1.0
    return math.acos(tmp1)



if __name__ == "__main__":

    # tests the speed of different HIK implementations
    compare_hik_hik1()






