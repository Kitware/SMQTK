import argparse
import json
import logging
import os

from flask_basicauth import BasicAuth

from smqtk.utils import bin_utils
from smqtk.utils import merge_dict
import smqtk.web


def cli_parser():
    description = "Runs conforming SMQTK Web Applications."
    parser = argparse.ArgumentParser(description=description)

    # Application options
    group_application = parser.add_argument_group("Application Selection")
    group_application.add_argument('-l', '--list',
                                   default=False, action="store_true",
                                   help="List currently available applications "
                                        "for running")
    group_application.add_argument('-a', '--application', default=None,
                                   help="Label of the web application to run.")

    # Configuration options
    group_configuration = parser.add_argument_group("Configuration")
    group_configuration.add_argument('-c', '--config', default=None,
                                     help='Path to application JSON '
                                          'configuration file.')
    group_configuration.add_argument('-g', '--generate-config',
                                     default=None,
                                     help='Optionally generate a default '
                                          'configuration file at the '
                                          'specified path. If a configuration '
                                          'file was provided, we update the '
                                          'default configuration with the '
                                          'contents of the given '
                                          'configuration.')

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
                             help='Turn on server debugging messages')
    group_other.add_argument('--debug-smqtk',
                             action='store_true', default=False,
                             help='Turn on SMQTK debugging messages')

    return parser


def main():
    parser = cli_parser()
    args = parser.parse_args()

    debug_smqtk = args.debug_smqtk
    debug_server = args.debug_server

    bin_utils.initialize_logging(logging.getLogger("smqtk"),
                                 logging.INFO - (10*debug_smqtk))
    bin_utils.initialize_logging(logging.getLogger("werkzeug"),
                                 logging.WARN - (20*debug_server))
    log = logging.getLogger("smqtk.main")

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

    # Merge loaded config with default
    config_loaded = False
    config = app_class.get_default_config()
    if args.config:
        if os.path.isfile(args.config):
            with open(args.config, 'r') as f:
                merge_dict(config, json.load(f))
            config_loaded = True
        elif not os.path.isfile(args.config):
            log.error("Configuration file path not valid.")
            exit(1)

    # Output config and exit if requested
    bin_utils.output_config(args.generate_config, config, log, True)

    # Configuration must have been loaded at this point since we can't normally
    # trust the default.
    if not config_loaded:
        log.error("No configuration provided")
        exit(1)

    host = args.host
    port = args.port and int(args.port)
    use_reloader = args.reload
    use_threading = args.threaded
    use_basic_auth = args.use_basic_auth

    # noinspection PyUnresolvedReferences
    app = app_class.from_config(config)
    if use_basic_auth:
        app.config["BASIC_AUTH_FORCE"] = True
        BasicAuth(app)
    app.config['DEBUG'] = debug_server

    app.run(host=host, port=port, debug=debug_server, use_reloader=use_reloader,
            threaded=use_threading)


if __name__ == "__main__":
    main()
