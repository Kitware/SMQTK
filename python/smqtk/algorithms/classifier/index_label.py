from six.moves import zip

from smqtk.algorithms import Classifier
from smqtk.representation.data_element import from_uri


class IndexLabelClassifier (Classifier):
    """
    Applies a listing of labels (new-line separated) to input "descriptor"
    values, which is actually a vector of class confidence values.
    """

    @classmethod
    def is_usable(cls):
        return True

    def __init__(self, index_to_label_uri):
        """
        Construct a new "classifier" that applies labels to input vector
        indices.

        We expect to be given a URI to a new-line separated text file where each
        line is a separate label in order and matching the dimensionality of an
        input descriptor.

        :param index_to_label_uri: URI to new-line separated sequence of labels.
        :type index_to_label_uri: str

        """
        super(IndexLabelClassifier, self).__init__()

        # load label vector
        self.index_to_label_uri = index_to_label_uri
        self.label_vector = [line.strip() for line in
                             from_uri(index_to_label_uri).to_buffered_reader()]

    def get_config(self):
        """
        Return a JSON-compliant dictionary that could be passed to this class's
        ``from_config`` method to produce an instance with identical
        configuration.

        :return: JSON type compliant configuration dictionary.
        :rtype: dict

        """
        return {
            "index_to_label_uri": self.index_to_label_uri,
        }

    def get_labels(self):
        """
        Get a copy of the sequence of class labels that this classifier can
        classify descriptors into.

        :return: Sequence of possible classifier labels.
        :rtype: collections.abc.Sequence[str]

        """
        # copying container
        return list(self.label_vector)

    def _classify_arrays(self, array_iter):
        check_dim = True
        for d_vector in array_iter:
            if check_dim:
                if len(self.label_vector) != len(d_vector):
                    raise RuntimeError(
                        "Failed to apply label vector to input descriptor of "
                        "incongruous dimensionality ({} labels != {} vector "
                        "shape)".format(len(self.label_vector), d_vector.shape)
                    )
                check_dim = False
            yield dict(zip(self.label_vector, d_vector))
