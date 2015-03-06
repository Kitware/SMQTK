"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


Various exception classes for use in the FeatureStore and related classes.

"""


class VCDStoreError (Exception):
    """
    Base, generic VCDStore exception
    """
    pass


class VCDDuplicateFeatureError(VCDStoreError):
    """
    This is thrown if there is already a feature already exists at the
    provided key combination.
    """
    pass


class VCDNoFeatureError(VCDStoreError):
    """
    This is thrown if there is no feature stored at a designated key location.
    """
    pass


class VCDStorageError(VCDStoreError):
    """
    Generic exception for an error occurring during storage.
    """
    pass
