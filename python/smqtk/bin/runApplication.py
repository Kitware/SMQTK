"""
Runs conforming SMQTK Web Applications.
"""

import logging

from flask_basicauth import BasicAuth

from smqtk.utils import bin_utils
import smqtk.web


def cli_parser():
    parser = bin_utils.basic_cli_parser(__doc__)

    # Application options
    group_application = parser.add_argument_group("Application Selection")
    group_application.add_argument('-l', '--list',
                                   default=False, action="store_true",
                                   help="List currently available applications "
                                        "for running")
    group_application.add_argument('-a', '--application', default=None,
                                   help="Label of the web application to run.")

    # Server options
    group_server = parser.add_argument_group("Server options")
    group_server.add_argument('-r', '--reload',
                              action='store_true', default=False,
                              help='Turn on server reloading.')
    group_server.add_argument('-t', '--threaded',
                              action='store_true', default=False,
                              help="Turn on server multi-threading.")
    group_server.add_argument('--host',
                              default=None,
                              help="Run host address specification override. "
                                   "This will override all other configuration "
                                   "method specifications.")
    group_server.add_argument('--port',
                              default=None,
                              help="Run port specification override. This will "
                                   "override all other configuration method "
                                   "specifications.")
    group_server.add_argument("--use-basic-auth",
                              action="store_true", default=False,
                              help="Use global basic authentication as "
                                   "configured.")

    # Other options
    group_other = parser.add_argument_group("Other options")
    group_other.add_argument('--debug-server',
                             action='store_true', default=False,
                             help='Turn on server debugging messages ONLY')
    group_other.add_argument('--debug-smqtk',
                             action='store_true', default=False,
                             help='Turn on SMQTK debugging messages ONLY')

    return parser


def main():
    parser = cli_parser()
    args = parser.parse_args()

    debug_smqtk = args.debug_smqtk or args.verbose
    debug_server = args.debug_server or args.verbose

    bin_utils.initialize_logging(logging.getLogger("__main__"),
                                 logging.INFO - (10 * debug_smqtk))
    bin_utils.initialize_logging(logging.getLogger("smqtk"),
                                 logging.INFO - (10*debug_smqtk))
    bin_utils.initialize_logging(logging.getLogger("werkzeug"),
                                 logging.WARN - (20*debug_server))
    log = logging.getLogger(__name__)

    web_applications = smqtk.web.get_web_applications()

    if args.list:
        log.info("")
        log.info("Available applications:")
        log.info("")
        for l in web_applications:
            log.info("\t" + l)
        log.info("")
        exit(0)

    application_name = args.application

    if application_name is None:
        log.error("No application name given!")
        exit(1)
    elif application_name not in web_applications:
        log.error("Invalid application label '%s'", application_name)
        exit(1)

    app_class = web_applications[application_name]

    config = bin_utils.utility_main_helper(app_class.get_default_config, args,
                                           skip_logging_init=True)

    host = args.host
    port = args.port and int(args.port)
    use_reloader = args.reload
    use_threading = args.threaded
    use_basic_auth = args.use_basic_auth

    # noinspection PyUnresolvedReferences
    #: :type: smqtk.web.SmqtkWebApp
    app = app_class.from_config(config)
    if use_basic_auth:
        app.config["BASIC_AUTH_FORCE"] = True
        BasicAuth(app)
    app.config['DEBUG'] = debug_server

    log.info("Starting application")
    app.run(host=host, port=port, debug=debug_server, use_reloader=use_reloader,
            threaded=use_threading)


if __name__ == "__main__":
    main()
