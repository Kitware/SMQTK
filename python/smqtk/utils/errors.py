__author__ = "paul.tunison@kitware.com"


class ReadOnlyError (Exception):
    """
    For when an attempt at modifying an immutable container is made.
    """
