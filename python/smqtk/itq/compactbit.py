import math
import numpy as np


def compactbit(b):
    """
    converts array to compacted string
    :param b: bits array
    :return: cb: compacted string of bits (using words of 'word' bits)
    """
    nSamples, nbits = b.shape
    nwords = math.ceil(nbits / 8)
    cb = np.zeros((nSamples, nwords), dtype=np.uint8)
    cc = np.zeros(nSamples, dtype=np.uint8)

    for j in range(nbits):

        i = j % 8
        w = j / 8

        if i == 0:
            cc = 0

        # noinspection PyAugmentAssignment
        cc = cc + b[:, j] * (1 << (7 - i))

        if i == 7:
            cb[:, w] = cc

    return cb
