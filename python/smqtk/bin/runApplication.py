"""
Runs conforming SMQTK Web Applications.
"""

import logging

from flask_basicauth import BasicAuth
from flask_cors import CORS
import six

from smqtk.utils import cli
import smqtk.web


def cli_parser():
    parser = cli.basic_cli_parser(__doc__)

    # Application options
    group_application = parser.add_argument_group("Application Selection")
    group_application.add_argument('-l', '--list',
                                   default=False, action="store_true",
                                   help="List currently available applications "
                                        "for running. More description is "
                                        "included if SMQTK verbosity is "
                                        "increased (-v | --debug-smqtk)")
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
    group_server.add_argument('--use-simple-cors',
                              action='store_true', default=False,
                              help="Allow CORS for all domains on all routes. "
                                   "This follows the \"Simple Usage\" of "
                                   "flask-cors: https://flask-cors.readthedocs"
                                   ".io/en/latest/#simple-usage")

    # Other options
    group_other = parser.add_argument_group("Other options")
    group_other.add_argument('--debug-server',
                             action='store_true', default=False,
                             help='Turn on server debugging messages ONLY. '
                                  'This is implied when -v|--verbose is '
                                  'enabled.')
    group_other.add_argument('--debug-smqtk',
                             action='store_true', default=False,
                             help='Turn on SMQTK debugging messages ONLY. '
                                  'This is implied when -v|--verbose is '
                                  'enabled.')
    group_other.add_argument('--debug-app',
                             action='store_true', default=False,
                             help='Turn on flask app logger namespace '
                                  'debugging messages ONLY. This is '
                                  'effectively enabled if the flask app is '
                                  'provided with SMQTK and "--debug-smqtk" is '
                                  'passed. This is also implied if '
                                  '-v|--verbose is enabled.')
    group_other.add_argument('--debug-ns',
                             action='append', default=[],
                             help="Specify additional python module "
                                  "namespaces to enable debug logging for.")

    return parser


def main():
    parser = cli_parser()
    args = parser.parse_args()

    debug_smqtk = args.debug_smqtk or args.verbose
    debug_server = args.debug_server or args.verbose
    debug_app = args.debug_app or args.verbose

    debug_ns_list = args.debug_ns
    debug_smqtk and debug_ns_list.append('smqtk')
    debug_server and debug_ns_list.append('werkzeug')

    # Create a single stream handler on the root, the level passed being
    # applied to the handler, and then set tuned levels on specific namespace
    # levels under root, which is reset to warning.
    cli.initialize_logging(logging.getLogger(), logging.DEBUG)
    logging.getLogger().setLevel(logging.WARN)
    log = logging.getLogger(__name__)
    # SMQTK level always at least INFO level for standard internals reporting.
    logging.getLogger("smqtk").setLevel(logging.INFO)
    # Enable DEBUG level on applicable namespaces available to us at this time.
    for ns in debug_ns_list:
        log.info("Enabling debug logging on '{}' namespace"
                 .format(ns))
        logging.getLogger(ns).setLevel(logging.DEBUG)

    webapp_types = smqtk.web.SmqtkWebApp.get_impls()
    web_applications = {t.__name__: t for t in webapp_types}

    if args.list:
        log.info("")
        log.info("Available applications:")
        log.info("")
        for l, cls in six.iteritems(web_applications):
            log.info("\t" + l)
            if debug_smqtk:
                log.info('\t' + ('^'*len(l)) + '\n' +

                         cls.__doc__ + '\n' +
                         ('*' * 80) + '\n')
        log.info("")
        exit(0)

    application_name = args.application

    if application_name is None:
        log.error("No application name given!")
        exit(1)
    elif application_name not in web_applications:
        log.error("Invalid application label '%s'", application_name)
        exit(1)

    #: :type: smqtk.web.SmqtkWebApp
    app_class = web_applications[application_name]

    # If the application class's logger does not already report as having INFO/
    # DEBUG level logging (due to being a child of an above handled namespace)
    # then set the app namespace's logger level appropriately
    app_class_logger_level = app_class.get_logger().getEffectiveLevel()
    app_class_target_level = logging.INFO - (10 * debug_app)
    if app_class_logger_level > app_class_target_level:
        level_name = \
            "DEBUG" if app_class_target_level == logging.DEBUG else "INFO"
        log.info("Enabling '{}' logging for '{}' logger namespace."
                 .format(level_name, app_class.get_logger().name))
        app_class.get_logger().setLevel(logging.INFO - (10 * debug_app))

    config = cli.utility_main_helper(app_class.get_default_config, args,
                                     skip_logging_init=True)

    host = args.host
    port = args.port and int(args.port)
    use_reloader = args.reload
    use_threading = args.threaded
    use_basic_auth = args.use_basic_auth
    use_simple_cors = args.use_simple_cors

    # noinspection PyUnresolvedReferences
    #: :type: smqtk.web.SmqtkWebApp
    app = app_class.from_config(config)
    if use_basic_auth:
        app.config["BASIC_AUTH_FORCE"] = True
        BasicAuth(app)
    if use_simple_cors:
        log.debug("Enabling CORS for all domains on all routes.")
        CORS(app)
    app.config['DEBUG'] = debug_server

    log.info("Starting application")
    app.run(host=host, port=port, debug=debug_server, use_reloader=use_reloader,
            threaded=use_threading)


if __name__ == "__main__":
    main()
