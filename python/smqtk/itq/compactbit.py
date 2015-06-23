import math
import numpy as np

def compactbit(b):
    """
    converts array to compacted string
    @param b: bits array
    @return: cb: compacted string of bits (using words of 'word' bits)
    """

    nSamples, nbits = b.shape
    nwords = math.ceil( nbits/8 )
    cb = np.zeros((nSamples,nwords), dtype=np.uint8)
    cc = np.zeros((nSamples,1), dy=np.uint8)

    for j in range(nbits):

        i = j%8
        w = math.ceil(j/8)

        if i==0 and w > 0 :
            cb[:,w] = cc
            cc = 0
        else:
            cc = cc + cb[:,j] * (1<<i)

    return cb
