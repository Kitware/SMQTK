"""ITQ: finds a rotation of the PCA embedded data"""

import numpy as np

def ITQ( V, n_iter ):
    """
    finds a rotation of the PCA embedded data
    @param V: 2D numpy array, n*c PCA embedded data, n is the number of images and c is the code length
    @param n_iter: max number of iterations, 50 is usually enough
    @return: [B,R]
       B: 2D numpy array, n*c binary matrix,
       RR: 2D numpy array, the d*c rotation matrix found by ITQ
    """

    #initialize with an orthogonal random rotation
    bit = V.shape[1]
    R = np.random.randn(bit,bit)
    U11, S2, V2 = np.linalg.svd( R )
    R = U11[:,:bit]

    #ITQ to find optimal rotation
    for iter in range(n_iter):
        Z = np.dot(V, R)
        UX = np.ones( Z.shape ) * (-1)
        UX[Z>=0] = 1
        RR = R

        C = np.dot( UX.transpose(), V )
        UB,sigma,UA = np.linalg.svd(C)
        R = np.dot(UA, UB.transpose())

    #make B binary
    B = UX;
    B[B<0] = 0;

    return B, RR
