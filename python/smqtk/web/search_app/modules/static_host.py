import flask


__author__ = 'paul.tunison@kitware.com'


class StaticDirectoryHost (flask.Blueprint):
    """
    Module that will host a given directory to the given URL prefix (relative to
    the parent module's prefix).

    Instances of this class will have nothing set to their static URL path, as a
    blank string is used. Please reference the URL prefix value.

    """

    def __init__(self, name, static_dir, url_prefix):
        # make sure URL prefix starts with a slash
        if not url_prefix.startswith('/'):
            url_prefix = '/' + url_prefix

        super(StaticDirectoryHost, self).__init__(name, __name__,
                                                  static_folder=static_dir,
                                                  static_url_path="",
                                                  url_prefix=url_prefix)
