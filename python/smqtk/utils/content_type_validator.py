import abc
import six


@six.add_metaclass(abc.ABCMeta)
class ContentTypeValidator (object):
    """
    Abstract interface for the provision of a method to provide a "valid" set of
    content types relative to the sub-class' function.
    """

    __slots__ = ()

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

    def raise_valid_element(self, data_element, exception=ValueError,
                            message=None):
        """
        Check if the given data element matches a reported valid content type,
        raising the given exception class (``ValueError`` by default) if not.

        :param smqtk.representation.DataElement data_element:
             Data element instance to check.
        :param StandardError exception:
            Exception to raise if the given element does not report as a valid
            content type.
        :param str message:
            Specific message to provide with a raise exception.

        """
        if not self.is_valid_element(data_element):
            if message is None:
                message = "Data element does not match a content type " \
                          "reported as valid. Given: \"{}\"." \
                          .format(data_element.content_type())
            # noinspection PyCallingNonCallable
            # - Leave the handling of whether or not an exception is
            # constructable to the exception class being constructed (user
            # decision repercussion).
            raise exception(message)
