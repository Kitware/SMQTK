from __future__ import division, print_function
import unittest

import nose.tools

from smqtk.utils.url import (
    url_join
)


class TestUrlTools (unittest.TestCase):

    def test_url_join_simple(self):
        # One parameter
        nose.tools.assert_equal(url_join('foo'), 'foo')

        # multi-parameter, convert to str
        nose.tools.assert_equal(
            url_join('https://foo', 'bar', 1, 'six'),
            'https://foo/bar/1/six'
        )

    def test_url_join_empty_all(self):
        nose.tools.assert_equal(url_join(''), '')

        nose.tools.assert_equal(
            url_join('', ''),
            ''
        )

        nose.tools.assert_equal(
            url_join('', '', ''),
            ''
        )

    def test_url_join_empty_leading(self):
        nose.tools.assert_equal(
            url_join('', 'foo'),
            'foo'
        )

        nose.tools.assert_equal(
            url_join('', '', 'foo', 'bar'),
            'foo/bar'
        )

    def test_url_join_empty_middle(self):
        nose.tools.assert_equal(
            url_join('foo', '', 'bar'),
            'foo/bar'
        )

        nose.tools.assert_equal(
            url_join('foo', '', '', 'bar', '', 'baz'),
            'foo/bar/baz'
        )

    def test_url_join_empty_last(self):
        nose.tools.assert_equal(
            url_join('foo', ''),
            "foo/"
        )

    def test_url_join_empty_mixed(self):
        nose.tools.assert_equal(
            url_join('', '', 'b', '', 'c', '', '', 'a', '', ''),
            'b/c/a/'
        )

        nose.tools.assert_equal(
            url_join('', '', 'b', '', 'c', '', '', 'a'),
            'b/c/a'
        )

    def test_url_join_protocol_handling(self):
        nose.tools.assert_equal(
            url_join('http://'),
            'http://'
        )

        nose.tools.assert_equal(
            url_join('http://', 'https://'),
            'https://'
        )

        # not that this will probably ever be an intended result, this tests
        # documented logic.
        nose.tools.assert_equal(
            url_join('http://', 'https:/'),
            'http://https:'
        )

    def test_url_join_restart_protocol(self):
        # Test restarting url concat due to protocol header
        nose.tools.assert_equal(
            url_join('http://a.b.c', 'ftp://ba.c'),
            'ftp://ba.c'
        )

        nose.tools.assert_equal(
            url_join('', 'a', 'b', 'https://', 'bar', ''),
            'https://bar/'
        )

    def test_url_join_restart_slash(self):
        nose.tools.assert_equal(
            url_join("foo", '/bar', 'foo'),
            '/bar/foo'
        )

        nose.tools.assert_equal(
            url_join("foo", '/bar', '/foo'),
            '/foo'
        )

        nose.tools.assert_equal(
            url_join("foo", '/bar', '/'),
            '/'
        )

        nose.tools.assert_equal(
            url_join("foo", '/bar', '/', 'foo'),
            '/foo'
        )

    def test_url_join_restart_mixed(self):
        nose.tools.assert_equal(
            url_join("foo", '/bar', 'https://foo'),
            'https://foo'
        )

        nose.tools.assert_equal(
            url_join("foo", 'https://foo', '/bar'),
            '/bar'
        )
