#!/usr/bin/env python
"""

Start SMQTK SearchApp

"""


def main():
    import optparse
    parser = optparse.OptionParser()
    parser.add_option('-c', '--config', default=None,
                      help='Path to an SMQTK configuration extension file '
                           '(a python file).')
    parser.add_option('-a', '--application', default=None,
                      help="Name of the web application to run. Required.")
    parser.add_option('-r', '--reload', action='store_true', default=False,
                      help='Turn on server reloading.')
    parser.add_option('-d', '--debug', action='store_true', default=False,
                      help='Turn on server debugging')
    parser.add_option('--host', default=None,
                      help="Run host address specification override. This will "
                           "override all other configuration method "
                           "specifications.")
    parser.add_option('--port', default=None,
                      help="Run port specification override. This will "
                           "override all other configuration method "
                           "specifications.")
    opts, args = parser.parse_args()

    import logging
    logging.basicConfig()  # TODO: Add better message format here
    logging.getLogger().setLevel(logging.INFO)
    if opts.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    host = opts.host
    port = opts.port and int(opts.port)
    debug = opts.debug
    use_reloader = opts.reload
    application_name = opts.application

    if application_name is None:
        raise ValueError("No application name given!")

    import SMQTK.Web
    App = getattr(SMQTK.Web, application_name, None)
    if App is None:
        raise ValueError("No available application by the name of '%s'"
                         % application_name)
    app = App(opts.config)
    app.run(host=host, port=port, debug=debug, use_reloader=use_reloader)


if __name__ == "__main__":
    main()
