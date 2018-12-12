"""
Tests for high level ``from_uri`` function, separate from the
``DataElement.from_uri`` class method.
"""
import unittest

import smqtk.exceptions
import smqtk.representation.data_element


# noinspection PyClassHasNoInit
class UnresolvableElement (smqtk.representation.data_element.DataElement):
    """ Does not implement from_uri, declaring no support for URI resolution """

    @classmethod
    def is_usable(cls):
        return True

    def __repr__(self):
        return super(UnresolvableElement, self).__repr__()

    def get_config(self):
        return {}

    def content_type(self):
        return None

    def is_empty(self):
        pass

    def get_bytes(self):
        return bytes()

    def set_bytes(self, b):
        pass

    def writable(self):
        pass


# noinspection PyClassHasNoInit
class ResolvableElement (smqtk.representation.data_element.DataElement):

    @classmethod
    def from_uri(cls, uri):
        """
        :type uri: str
        :rtype: ResolvableElement
        """
        if uri.startswith('resolvable://'):
            return ResolvableElement()

    @classmethod
    def is_usable(cls):
        return True

    def __repr__(self):
        return super(ResolvableElement, self).__repr__()

    def get_config(self):
        return {}

    def content_type(self):
        return None

    def is_empty(self):
        pass

    def get_bytes(self):
        return bytes()

    def set_bytes(self, b):
        pass

    def writable(self):
        pass


class TestDataElementHighLevelFromUri (unittest.TestCase):

    def test_no_classes(self):
        def impl_generator():
            return {}

        self.assertRaises(
            smqtk.exceptions.InvalidUriError,
            smqtk.representation.data_element.from_uri,
            'whatever',
            impl_generator
        )

    def test_no_resolvable_options(self):
        """
        when no DataElement implementations provide an implementation for
        the ``from_uri`` class method
        """
        def impl_generator():
            return {UnresolvableElement}

        self.assertRaises(
            smqtk.exceptions.InvalidUriError,
            smqtk.representation.data_element.from_uri,
            'something',
            impl_generator
        )

    def test_one_resolvable_option(self):
        """
        When at least one plugin can resolve a URI
        """
        def impl_generator():
            return {UnresolvableElement, ResolvableElement}

        # URI that can be resolved by ResolvableElement
        self.assertIsInstance(
            smqtk.representation.data_element.from_uri(
                "resolvable://data",
                impl_generator
            ),
            ResolvableElement
        )

        # bad URI even though something can resolve it
        self.assertRaises(
            smqtk.exceptions.InvalidUriError,
            smqtk.representation.data_element.from_uri,
            'not_resolvable', impl_generator
        )
