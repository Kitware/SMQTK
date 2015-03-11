"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

def load_attribute_list():
    import os.path as osp
    this_dir = osp.dirname(__file__)
    with open(osp.join(this_dir, 'attributes_list.txt')) as attr_list_file:
        return tuple(l.strip() for l in attr_list_file.readlines())
