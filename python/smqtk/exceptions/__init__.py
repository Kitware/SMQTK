
class NoClassificationError (Exception):
    """
    When a ClassificationElement has no mapping yet set, but an operation
    required it.
    """
    pass


class ReadOnlyError (Exception):
    """
    For when an attempt at modifying an immutable container is made.
    """
