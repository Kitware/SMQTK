"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import os.path as osp
import numpy as np

from SMQTK_Backend.utils.data_parsers import (
    parse_deva_detection_file
)


class test_parse_deva_detection_file (object):

    def setUp(self):
        self.data_dir = osp.join(osp.dirname(__file__), 'data')

        self.correct = np.array((
            (1,  0,  1.0),
            (2,  1,  0.9),
            (3,  10, 0.8),
            (4,  11, 0.7),
            (5, 100, 0.6),
            (6, 101, 0.5),
            (7, 110, 0.4),
            (8, 111, 0.3),
            (9,   2, 0.2),
            (10, 30, 0.1),
        ))

    def test_correct(self):
        d = parse_deva_detection_file(osp.join(self.data_dir,
                                               'deva_example-correct.csv'))

        assert d.shape == self.correct.shape, \
            "Result array shapes didn't match!"

        # noinspection PyUnresolvedReferences
        # -> return is an ndarray, which **does** have an all() method
        assert (self.correct == d).all(), \
            "read and expected didn't match"

    def test_missing_header(self):
        d = parse_deva_detection_file(
            osp.join(self.data_dir, 'deva_example-missing_header.csv')
        )

        assert d.shape != self.correct.shape,\
            "result arrays were the same shape, shouldn't have been."

        # since they should be of different shape, this should return false.
        assert self.correct != d
