"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""
__author__ = 'paul.tunison'

import nose.tools as tools

from SMQTK_Backend.ECDWorkers import ECDWorkerBaseProcess
from SMQTK_Backend.ECDWorkers.classifiers import get_classifiers


class test_ECDWorker_general (object):

    def test_plugin_importer(self):
        """
        testing plugin gather method, which tests that all plugins are
        structured correctly that retrieval of a known dummy class works.
        """
        classifier_map = get_classifiers()
        tools.assert_in('svm_dummy', classifier_map.keys())
        classifier_objects = classifier_map['svm_dummy']
        tools.assert_in('search', classifier_objects.keys())
        tools.assert_in('learn', classifier_objects.keys())
        tools.assert_true(issubclass(classifier_objects['search'],
                                     ECDWorkerBaseProcess))
        tools.assert_true(issubclass(classifier_objects['learn'],
                                     ECDWorkerBaseProcess))
