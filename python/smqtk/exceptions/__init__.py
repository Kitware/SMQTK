
class NoClassificationError (Exception):
    """
    When a ClassificationElement has no mapping yet set, but an operation
    required it.
    """


class NoDetectionError (Exception):
    """
    When a DetectionElement has no stored data (paired bounding box and classification
    """


class ReadOnlyError (Exception):
    """
    For when an attempt at modifying an immutable container is made.
    """


class NoUriResolutionError (Exception):
    """
    Standard exception thrown by base DataElement from_uri method when a
    subclass does not implement URI resolution.
    """


class InvalidUriError (Exception):
    """
    An invalid URI was provided.
    """

    def __init__(self, uri_value, reason):
        super(InvalidUriError, self).__init__(uri_value, reason)
        self.uri = uri_value
        self.reason = reason


class MissingLabelError(Exception):
    """
    Raised by ClassifierCollection.classify when requested classifier labels
    are missing from collection.
    """
    def __init__(self, labels):
        """
        :param labels: The labels missing from the collection
        :type labels: set[str]
        """
        super(MissingLabelError, self).__init__(labels)
        self.labels = labels
