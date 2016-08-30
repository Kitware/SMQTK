"""
Utilities for URLs
"""


def url_join(*parts):
    """
    Join URL components with '/' separator.

    :param parts: path components to join with slashes.
    :type parts: tuple[str]
    :return: Joined string
    :rtype: str
    """
    return '/'.join([str(p) for p in parts])
