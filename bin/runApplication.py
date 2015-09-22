#!/usr/bin/env python

import json
import logging
import optparse
import os

from flask.ext.basicauth import BasicAuth

from smqtk.utils import bin_utils
from smqtk.utils.jsmin import jsmin
import smqtk.web


def setup_cli(parser):
    # Application options
    group_application = optparse.OptionGroup(parser, "Application selection "
                                                     "parameters")
    group_application.add_option('-l', '--list',
                                 default=False, action="store_true",
                                 help="List currently available applications "
                                      "for running")
    group_application.add_option('-a', '--application', default=None,
                                 help="Label of the web application to run.")
    parser.add_option_group(group_application)

    # Configuration options
    group_configuration = optparse.OptionGroup(parser, "Options dealing with "
                                                       "application "
                                                       "configuration")
    group_configuration.add_option('-c', '--config', default=None,
                                   help='Path to application JSON '
                                        'configuration file.')
    group_configuration.add_option('--output-config', default=None,
                                   help='Optional path to output default JSON '
                                        'configuration to.')

    parser.add_option_group(group_configuration)

    # Server options
    group_server = optparse.OptionGroup(parser, "Server options")
    group_server.add_option('-r', '--reload',
                            action='store_true', default=False,
                            help='Turn on server reloading.')
    group_server.add_option('-t', '--threaded',
                            action='store_true', default=False,
                            help="Turn on server multi-threading.")
    group_server.add_option('--host',
                            default=None,
                            help="Run host address specification override. "
                                 "This will override all other configuration "
                                 "method specifications.")
    group_server.add_option('--port',
                            default=None,
                            help="Run port specification override. This will "
                                 "override all other configuration method "
                                 "specifications.")
    group_server.add_option("--use-basic-auth",
                            action="store_true", default=False,
                            help="Use global basic authentication as "
                                 "configured.")
    parser.add_option_group(group_server)

    # Other options
    group_other = optparse.OptionGroup(parser, "Other options")
    group_other.add_option('--debug-server',
                           action='store_true', default=False,
                           help='Turn on server debugging messages')
    group_other.add_option('--debug-smqtk',
                           action='store_true', default=False,
                           help='Turn on SMQTK debugging messages')
    parser.add_option_group(group_other)


def main():
    parser = bin_utils.SMQTKOptParser()
    setup_cli(parser)
    opts, args = parser.parse_args()

    debug_smqtk = opts.debug_smqtk
    debug_server = opts.debug_server

    bin_utils.initialize_logging(logging.getLogger("smqtk"),
                                 logging.INFO - (10*debug_smqtk))
    bin_utils.initialize_logging(logging.getLogger("werkzeug"),
                                 logging.WARN - (20*debug_server))
    log = logging.getLogger("smqtk.main")

    web_applications = smqtk.web.get_web_applications()

    if opts.list:
        log.info("")
        log.info("Available applications:")
        log.info("")
        for l in web_applications:
            log.info("\t" + l)
        log.info("")
        exit(0)

    application_name = opts.application

    if application_name is None:
        log.error("No application name given!")
        exit(1)
    elif application_name not in web_applications:
        log.error("Invalid application label '%s'", application_name)
        exit(1)

    app_class = web_applications[application_name]

    # Output config and exit if requested
    bin_utils.output_config(opts.output_config, app_class.get_default_config(),
                            log)

    if not opts.config:
        log.error("No configuration provided")
        exit(1)
    elif not os.path.isfile(opts.config):
        log.error("Configuration file path not valid.")
        exit(1)

    with open(opts.config, 'r') as f:
        config = json.loads(jsmin(f.read()))

    host = opts.host
    port = opts.port and int(opts.port)
    use_reloader = opts.reload
    use_threading = opts.threaded
    use_basic_auth = opts.use_basic_auth

    app = app_class.from_config(config)
    if use_basic_auth:
        app.config["BASIC_AUTH_FORCE"] = True
        BasicAuth(app)
    app.config['DEBUG'] = debug_server

    app.run(host=host, port=port, debug=debug_server, use_reloader=use_reloader,
            threaded=use_threading)


if __name__ == "__main__":
    main()
