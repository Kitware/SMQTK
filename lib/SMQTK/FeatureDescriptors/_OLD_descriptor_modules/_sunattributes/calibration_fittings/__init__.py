"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import numpy
import os
import os.path as osp
import re


def load_calibration_fittings():
    """
    Load and return calibration matrices contained in this module.

    :return: Dictionary of calibration fittings keyed on the string name of the
        attribute, This mappings keys will match the keys listed in the
        "attributes_list.txt" file (one level up).
    :rtype: dict of (str, numpy.ndarray)

    """
    this_dir = osp.dirname(__file__)
    fittings_re = re.compile(".*\.fitting\.txt")
    calib_map = {}
    for fittings_file in [osp.join(this_dir, f) for f in os.listdir(this_dir)]:
        if fittings_re.match(fittings_file):
            attr_name = osp.basename(fittings_file).split('.')[0]
            calib_map[attr_name] = numpy.loadtxt(fittings_file)

    return calib_map
