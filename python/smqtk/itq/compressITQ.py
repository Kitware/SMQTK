"""code for converting data X to binary code C using ITQ"""
"""training process to learn the rotation matrix R and mean descriptor sampleMean"""

import numpy as np
import numpy.matlib
import ITQ


def compressITQ(X, bit, num_iter=50):
    """
    Code for converting data X to binary code C using ITQ

    :param X: n*d data matrix, n is number of descriptors, d is dimension
    :type X: numpy.core.multiarray.ndarray

    :param bit: number of bits to compress to
    :type bit: int

    :return: [C,R,V]
       C: 2D numpy array, n*bit binary matrix,
       R: 2D numpy array, d*bit rotation matrix found by ITQ
       V: Mean vector that needs to be subtracted from new descriptors

    """
    # center the data, VERY IMPORTANT for ITQ to work
    sampleMean = np.mean(X, 0)
    X = (X - numpy.matlib.repmat(sampleMean, X.shape[0], 1))

    # PCA
    C = np.cov(X.transpose())
    pc, l, v2 = np.linalg.svd(C)
    XX = np.dot(X, pc[:, :bit])

    # ITQ to find optimal rotation
    # default is 50 iterations
    # C is the output code
    # R is the rotation found by ITQ
    C, R = ITQ.ITQ(XX, num_iter)

    rotation = np.dot(pc[:, :bit], R)

    return C, rotation, sampleMean
