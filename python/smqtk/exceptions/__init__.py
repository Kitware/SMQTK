
class NoClassificationError (Exception):
    """
    When a ClassificationElement has no mapping yet set, but an operation
    required it.
    """


class ReadOnlyError (Exception):
    """
    For when an attempt at modifying an immutable container is made.
    """


class InvalidUriError (Exception):
    """
    An invalid URI was provided.
    """

    def __init__(self, uri_value, reason):
        super(InvalidUriError, self).__init__(uri_value, reason)
        self.uri = uri_value
        self.reason = reason
