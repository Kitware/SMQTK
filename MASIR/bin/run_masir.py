#!/usr/bin/env python
# coding=utf-8
"""
High level run script for the MASIR web application via Flask.
"""


def main():
    import optparse
    parser = optparse.OptionParser()
    parser.add_option('-c', '--config', default=None,
                      help='Path to the MASIR configuration file '
                           '(a python file).')
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

    from masir.web import MasirApp
    app = MasirApp(opts.config)
    app.run(host=host, port=port, debug=debug, use_reloader=use_reloader)

if __name__ == "__main__":
    main()
