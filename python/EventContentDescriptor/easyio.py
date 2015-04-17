"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""
import os
import cPickle as pk
import numpy as np
import scipy.io as scipyio
import scipy.sparse as sparse
import json


def pickle_easy_load(filename):
    """
    Convenience function to load a pickled binary file
    """    
    with open(filename, 'rb') as fin:
        data = pk.load(fin)
    return data

#    data = None
#    with open(filename, 'rb') as fin:
#        data = pickle.load(fin)
#    return data

def pickle_easy_dump(filename, data):
    """
    Convenience function to dump(save) data to pickeld format
    """
    with open(filename, 'wb') as fout:
        pk.dump(data, fout)
#    
#    with open(filename, 'wb') as fout:
#        pickle.dump(data, fout)

def load(filename):
    """
    Convenience function to read many formats based on extension, mostly numpy/scipy
    """
    ext = os.path.splitext(filename)[1]
    if ext == '.pickle':
        return pickle_easy_load(filename)
    elif ext == '.npy':
        return np.load(filename)
    elif ext == '.mtx':
        return sparse.csc_matrix(scipyio.mmread(filename)) # adding .tolil() not working
    elif ext == '.txt':
        return np.loadtxt(filename)
    elif ext == '.json':
        with open(filename, 'rb') as fin:
            contents = json.load(fin)
        return contents
    
    else:
        print 'easy_load: format not supported: %s'%ext
        return None
    
def write(filename, data):
    """
    Convenience function to write different formats based on extension, mostly numpy/scipy
    """
    ext = os.path.splitext(filename)[1]
    if ext == '.pickle':
        return pickle_easy_dump(filename, data)
    elif ext == '.npy':
        return np.save(filename, data)
    elif ext == '.mtx':
        return scipyio.mmwrite(filename, data)
    elif ext == '.txt':
        return np.savetxt(filename, data)
    elif ext == '.json':
        with open(filename, 'wb') as fout:
            ret = json.dump(data, fout)
        return ret
    else:
        print 'easy_write: format not supported: %s'%ext
        return None