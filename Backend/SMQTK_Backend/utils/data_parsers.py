"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import csv
import numpy as np


def parse_deva_detection_file(file_in):
    """
    Read a DEVA detection file, and outputs a CLIPSx3 matrix
    where
    Col 1 : clipid
    Col 2 : target event
    Col 3 : score
    """
    fin_ = open(file_in, 'r')
    fin  = csv.reader(fin_, delimiter=',')
    lines = [line for line in fin]
    # get rid of the first line header (perhaps better way to do this?)
    lines = lines[1::]

    mat = np.zeros([len(lines), 3])
    count = 0
    for line in lines:
        mat[count][0] = int(line[0][0:6])
        mat[count][1] = int(line[0][-3::])
        mat[count][2] = float(line[1])
        count += 1
    return mat
