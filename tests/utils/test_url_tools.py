from __future__ import division, print_function
import unittest

from smqtk.utils.url import (
    url_join
)


class TestUrlTools (unittest.TestCase):

    def test_url_join_simple(self):
        # One parameter
        self.assertEqual(url_join('foo'), 'foo')

        # multi-parameter, convert to str
        self.assertEqual(
            url_join('https://foo', 'bar', 1, 'six'),
            'https://foo/bar/1/six'
        )

    def test_url_join_empty_all(self):
        self.assertEqual(url_join(''), '')

        self.assertEqual(
            url_join('', ''),
            ''
        )

        self.assertEqual(
            url_join('', '', ''),
            ''
        )

    def test_url_join_empty_leading(self):
        self.assertEqual(
            url_join('', 'foo'),
            'foo'
        )

        self.assertEqual(
            url_join('', '', 'foo', 'bar'),
            'foo/bar'
        )

    def test_url_join_empty_middle(self):
        self.assertEqual(
            url_join('foo', '', 'bar'),
            'foo/bar'
        )

        self.assertEqual(
            url_join('foo', '', '', 'bar', '', 'baz'),
            'foo/bar/baz'
        )

    def test_url_join_empty_last(self):
        self.assertEqual(
            url_join('foo', ''),
            "foo/"
        )

    def test_url_join_empty_mixed(self):
        self.assertEqual(
            url_join('', '', 'b', '', 'c', '', '', 'a', '', ''),
            'b/c/a/'
        )

        self.assertEqual(
            url_join('', '', 'b', '', 'c', '', '', 'a'),
            'b/c/a'
        )

    def test_url_join_protocol_handling(self):
        self.assertEqual(
            url_join('http://'),
            'http://'
        )

        self.assertEqual(
            url_join('http://', 'https://'),
            'https://'
        )

        # not that this will probably ever be an intended result, this tests
        # documented logic.
        self.assertEqual(
            url_join('http://', 'https:/'),
            'http://https:'
        )

    def test_url_join_restart_protocol(self):
        # Test restarting url concat due to protocol header
        self.assertEqual(
            url_join('http://a.b.c', 'ftp://ba.c'),
            'ftp://ba.c'
        )

        self.assertEqual(
            url_join('', 'a', 'b', 'https://', 'bar', ''),
            'https://bar/'
        )

    def test_url_join_restart_slash(self):
        self.assertEqual(
            url_join("foo", '/bar', 'foo'),
            '/bar/foo'
        )

        self.assertEqual(
            url_join("foo", '/bar', '/foo'),
            '/foo'
        )

        self.assertEqual(
            url_join("foo", '/bar', '/'),
            '/'
        )

        self.assertEqual(
            url_join("foo", '/bar', '/', 'foo'),
            '/foo'
        )

    def test_url_join_restart_mixed(self):
        self.assertEqual(
            url_join("foo", '/bar', 'https://foo'),
            'https://foo'
        )

        self.assertEqual(
            url_join("foo", 'https://foo', '/bar'),
            '/bar'
        )
