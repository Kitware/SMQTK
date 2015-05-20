#!/usr/bin/env python
"""

Start smqtk SearchApp

"""

import logging
from flask.ext.basicauth import BasicAuth

from smqtk.utils import bin_utils


def main():
    parser = bin_utils.SMQTKOptParser()
    parser.add_option('-c', '--config', default=None,
                      help='Path to an smqtk configuration extension file '
                           '(a python file).')
    parser.add_option('-a', '--application', default=None,
                      help="Name of the web application to run. Required.")

    parser.add_option('-r', '--reload', action='store_true', default=False,
                      help='Turn on server reloading.')
    parser.add_option('-t', '--threaded', action='store_true', default=False,
                      help="Turn on web searcher threading.")
    parser.add_option('--debug-server', action='store_true', default=False,
                      help='Turn on server debugging messages')
    parser.add_option('--debug-backend', action='store_true', default=False,
                      help='Turn on smqtk backend debugging messages')

    parser.add_option('--host', default=None,
                      help="Run host address specification override. This will "
                           "override all other configuration method "
                           "specifications.")
    parser.add_option('--port', default=None,
                      help="Run port specification override. This will "
                           "override all other configuration method "
                           "specifications.")
    parser.add_option("--use-basic-auth", action="store_true", default=False,
                      help="Use global basic authentication as configured.")
    parser.add_option('-l', '--list', default=False, action="store_true",
                      help="List currently available applications for running.")
    opts, args = parser.parse_args()

    bin_utils.initialize_logging(logging.getLogger("smqtk"),
                                logging.INFO - (10*opts.debug_backend))
    bin_utils.initialize_logging(logging.getLogger("werkzeug"),
                                logging.WARN - (20*opts.debug_server))
    log = logging.getLogger("smqtk.main")

    if opts.list:
        from smqtk.web import APPLICATIONS
        log.info("")
        log.info("Available applications:")
        log.info("")
        for e in APPLICATIONS:
            log.info("\t%s" % e.__name__)
        log.info("")
        exit(0)

    host = opts.host
    port = opts.port and int(opts.port)
    debug_server = opts.debug_server
    use_reloader = opts.reload
    use_threading = opts.threaded
    application_name = opts.application
    use_basic_auth = opts.use_basic_auth

    if application_name is None:
        raise ValueError("No application name given!")

    import smqtk.web
    # noinspection PyPep8Naming
    App = getattr(smqtk.web, application_name, None)
    if App is None:
        raise ValueError("No available application by the name of '%s'"
                         % application_name)
    app = App(opts.config)
    if use_basic_auth:
        app.config["BASIC_AUTH_FORCE"] = True
        BasicAuth(app)

    app.run(host=host, port=port, debug=debug_server, use_reloader=use_reloader,
            threaded=use_threading)


if __name__ == "__main__":
    main()
