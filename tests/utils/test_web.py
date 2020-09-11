import unittest
from smqtk.utils.web import ServiceProxy


class TestServiceProxy (unittest.TestCase):
    """ Tests for the ServiceProxy helper class """

    def test_init_no_scheme_fill_in(self):
        """ Test that when no http/https scheme is present at the beginning
        of the URL provided at construction time that a "http://" scheme is
        prepended.
        """
        test_url = "foo/bar"
        expected = "http://foo/bar"
        assert ServiceProxy(test_url).url == expected

        test_url = "://foo/bar"
        expected = "http://://foo/bar"
        assert ServiceProxy(test_url).url == expected

    def test_init_url_hash_scheme(self):
        """ Test that if a scheme is present at the head of the URL that it is
        not changed.
        """
        test_url = 'http://this.site/bar'
        assert ServiceProxy(test_url).url == test_url

        test_url = 'http://http://this.site/bar'
        assert ServiceProxy(test_url).url == test_url

        test_url = 'https://this.site/bar'
        assert ServiceProxy(test_url).url == test_url
