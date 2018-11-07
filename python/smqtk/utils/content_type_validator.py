import abc
import six


@six.add_metaclass(abc.ABCMeta)
class ContentTypeValidator (object):
    """
    Abstract interface for the provision of a method to provide a "valid" set of
    content types relative to the sub-class' function.
    """

    @abc.abstractmethod
    def valid_content_types(self):
        """
        :return: A set valid MIME types that are "valid" within the implementing
            class' context.
        :rtype: set[str]
        """

    def is_valid_element(self, data_element):
        """
        Check if the given DataElement instance reports a content type that
        matches one of the MIME types reported by ``valid_content_types``.

        :param smqtk.representation.DataElement data_element:
             Data element instance to check.

        :return: True if the given element has a valid content type as reported
            by ``valid_content_types``, and False if not.
        :rtype: bool
        """
        return data_element.content_type() in self.valid_content_types()
