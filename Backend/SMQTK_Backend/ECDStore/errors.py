"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

class ECDDuplicateElementError (Exception):
    """
    Tried to store an element with duplicate keys.
    """
    pass


class ECDNoElementError (Exception):
    """
    No element existed for a given query specification.
    """
    pass
