import unittest

import nose.tools

from smqtk.utils.url import (
    url_join
)


class TestUrlTools (unittest.TestCase):

    def test_url_join(self):
        # No parameters
        nose.tools.assert_equal(url_join(), '')

        # One parameter
        nose.tools.assert_equal(url_join('foo'), 'foo')

        # multi-parameter
        nose.tools.assert_equal(
            url_join('https://foo', 'bar', 1, 'six'),
            'https://foo/bar/1/six'
        )
