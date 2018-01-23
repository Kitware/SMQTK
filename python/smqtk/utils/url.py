"""
Utilities for URLs
"""

import collections
import re


PROTOCOL_HEADER = re.compile('\w+://')
URL_SEP = '/'


def url_join(url, *urls):
    """
    Join one or more URL components intelligently.

    The return is the concatenation of ``url`` and any members of ``*urls`` with
    exactly one slash ("/") following each non-empty part except the last,
    meaning the result will only end in a trailing slash if the last part is
    empty. If a component starts with a protocol-like string or a leading slash,
    all previous components are thrown away and joining continues from that
    part.

    We define "protocol-like" as something matching the regex "\w+://".

    Leading slashes reduced to a single leading slash if multiple are present.

    This function is different in that we allow the joining of relative URLs
    that do not start with a protocol component.

    :param url: First URL component
    :type url: str

    :param urls: path components to join with slashes.
    :type urls: tuple[str]

    :return: Joined URL string
    :rtype: str

    """
    # State-builder methods
    urls = (url,) + urls
    concat = ''
    last_empty = False
    first = True
    skip_slash_prefix = True
    for p in urls:
        p = str(p)
        if not p:
            last_empty = True
        else:
            last_empty = False

            # Check protocol header presence
            m = PROTOCOL_HEADER.search(p)
            if m and m.start() == 0:
                protocol = p[m.start():m.end()]
                tail = p[m.end():].rstrip(URL_SEP)
                # Don't add separator on next if just protocol header
                skip_slash_prefix = not tail

                concat = protocol + tail
                # print "Restarting:", concat

            # Check for leading slash
            elif p.startswith(URL_SEP):
                p = p.strip(URL_SEP)
                concat = URL_SEP + p
                # Don't add separator on next if p was just a slash
                skip_slash_prefix = not p
                # print "Restarting:", concat

            # Otherwise append
            else:
                p = p.rstrip(URL_SEP)
                if first or skip_slash_prefix:
                    concat += p
                    skip_slash_prefix = False
                else:
                    concat += URL_SEP + p
                # print "Appended:", p,  "->", concat

            first = False

    if concat and last_empty:
        concat += URL_SEP

    return concat
