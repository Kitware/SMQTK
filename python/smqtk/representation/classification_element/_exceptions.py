"""
Generic ClassificationElement exception classes
"""

__author__ = "paul.tunison@kitware.com"


class NoClassificationError (Exception):
    """
    When a ClassificationElement has no mapping yet set, but an operation
    required it.
    """
    pass
