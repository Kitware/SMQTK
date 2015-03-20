# -*- coding: utf-8 -*-
"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


Provides convenience functions for datasets, including
- list of clip ids for MED subsets

"""


def allowed_subsets():
    """
    @return: dictionry with name keys, and (filename_clipis, filname_labls) as contents
        - name: standardized allowed subset names such as 'MED11TEST' etc.
        - filename_clipids: filepath for the list of clips for the subset
        - filename_labels: optional filepath for known groundtruth in NIST MED format
    @rtype: dictionry[name] = (str, optional str)
    """

    #TODO: fill in more subset names and info
    info = dict()
    info['MED11TEST'] = ('dummy_clipids.txt', 'dummy_labels.txt')

    return info


def subset_clipids(config)
    """
    Find clipids across configs

    @param configs: list of dataset names, each config is a dictionary with 'name' and 'eid' (optionl)
        config['name']: str, on of the names in 'allowed_subsets'
        config['eid']: (optional) integer for target event
    @type configs: list of dictionary
    @return: a tuple of (clipids, labels)
    @rtype: a tuple of (int list, (optional) int list). If no known grounth, labls = None
    """

    for config in configs:



    pass
