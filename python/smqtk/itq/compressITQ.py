"""code for converting data X to binary code C using ITQ"""

import numpy as np
import numpy.matlib
import ITQ

def compressITQ(X, bit):
    """
    code for converting data X to binary code C using ITQ
    Input:
    @param X: n*d data matrix, n is number of images, d is dimension
    @param bit: number of bits
    Output:    
    @return C: n*bit binary code matrix
    """

    # center the data, VERY IMPORTANT for ITQ to work
    sampleMean = np.mean(X,0);
    X = (X - numpy.matlib.repmat(sampleMean, X.shape[0],1));
    
    # PCA
    C = np.cov( X.transpose() );
    pc, l, v2 = np.linalg.svd(C);
    XX = np.dot( X, pc[:,:bit] )
    
    #ITQ to find optimal rotation
    #default is 50 iterations
    #C is the output code
    #R is the rotation found by ITQ
    C, R = ITQ.ITQ(XX,50);
    
    return C,R
