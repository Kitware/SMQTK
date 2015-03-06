"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

from .svm_learn_impl import SVMClassifier_learner as learn
from .svm_search_impl import SVMClassifier_searcher as search

# TODO: Should describe the learning params file format here and/or define a
#       base class for impls here.
