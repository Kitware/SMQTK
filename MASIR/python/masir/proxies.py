# coding=utf-8

from numpy.core.multiarray import ndarray

import masir.FeatureManager
import masir.FeatureMemory

# Where numpy matrix class is located changes in different versions
try:
    # Recent version of numpy have matrix defined here
    # noinspection PyUnresolvedReferences
    from numpy.matrixlib.defmatrix import matrix
except ImportError:
    # For older versions of numpy
    # noinspection PyUnresolvedReferences
    from numpy.core.defmatrix import matrix
    # If this doesn't import, we're either missing version support for the
    # installed version of numpy


# Wrapper around numpy ndarray class, providing pass-through functions for class
# functions as of numpy 1.8.0
# TODO: Add``method_to_typeid`` for functions that return copies of data
class ArrayProxy (masir.FeatureManager.BaseProxy2):
    __metaclass__ = masir.FeatureManager.ExposedAutoGenMeta
    _exposed_ = masir.FeatureManager.all_safe_methods(ndarray)
    _exposed_properties_ = masir.FeatureManager.all_safe_properties(ndarray)
    _method_to_typeid_ = {
        '__iter__': "Iterator"
    }


# Wrapper around numpy matrix class, providing pass-through functions for class
# functions as of numpy 1.8.0.
# TODO: Add``method_to_typeid`` for functions that return copies of data
class MatrixProxy (masir.FeatureManager.BaseProxy2):
    __metaclass__ = masir.FeatureManager.ExposedAutoGenMeta
    _exposed_ = masir.FeatureManager.all_safe_methods(matrix)
    _exposed_properties_ = masir.FeatureManager.all_safe_properties(matrix)
    _method_to_typeid_ = {
        '__iter__': "Iterator"
    }


class RWLockWithProxy (masir.FeatureManager.BaseProxy2):
    __metaclass__ = masir.FeatureManager.ExposedAutoGenMeta
    _exposed_ = ('__enter__', '__exit__')


