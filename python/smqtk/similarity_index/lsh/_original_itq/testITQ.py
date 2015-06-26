"""Convert data X to binary code C using ITQ"""

import numpy as np
import numpy.matlib

def testITQ(X, rotation, sampleMean):
    """
    code for converting data X to binary code C using ITQ
    Input:
    @param X: n*d data matrix, n is number of images, d is dimension
    @param bit: number of bits
    Output:
    @return C: n*bit binary code matrix
    """

    # center the data
    X = (X - numpy.matlib.repmat(sampleMean, X.shape[0], 1))

    # apply rotation
    XX = np.dot(X, rotation)
    C = np.zeros( XX.shape )
    C[ XX > 0 ] = 1

    return C
