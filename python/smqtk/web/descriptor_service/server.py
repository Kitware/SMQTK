# -*- coding: utf-8 -*-
"""
LICENCE
-------
Copyright 2015 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import flask
import logging
import multiprocessing
import os
import requests
import tempfile

from smqtk.utils.configuration import IngestConfiguration
from smqtk.utils import DataFile, SimpleTimer


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class DescriptorServiceServer (flask.Flask):
    """
    Simple server that takes in a specification of the following form:

        /<model>/<descriptor_type>?uri=<file_uri>

    Computes the requested descriptor for the given file and returns that via
    a JSON structure.

    Standard return JSON:
    {
        "success": <bool>,
        "descriptor": [ <float>, ... ]
    }

    """

    # Optional environment variable that can point to a configuration file
    ENV_CONFIG = "DescriptorService_CONFIG"

    @property
    def log(self):
        return logging.getLogger('.'.join((self.__module__,
                                           self.__class__.__name__)))

    def __init__(self, config_filepath=None):
        super(DescriptorServiceServer, self).__init__(
            self.__class__.__name__,
            static_folder=os.path.join(SCRIPT_DIR, 'static'),
            template_folder=os.path.join(SCRIPT_DIR, 'templates')
        )

        #
        # Configuration setup
        #
        config_env_loaded = config_file_loaded = None

        # Load default -- This should always be present, aka base defaults
        self.config.from_object('smqtk_config')
        config_default_loaded = True

        # Load from env var if present
        if self.ENV_CONFIG in os.environ:
            self.log.info("Loading config from env var (%s)...",
                          self.ENV_CONFIG)
            self.config.from_envvar(self.ENV_CONFIG)
            config_env_loaded = True

        # Load from configuration file if given
        if config_filepath and os.path.isfile(config_filepath):
            config_file_path = os.path.expanduser(os.path.abspath(config_filepath))
            self.log.info("Loading config from file (%s)...", config_file_path)
            self.config.from_pyfile(config_file_path)
            config_file_loaded = True

        self.log.debug("Config defaults loaded : %s", config_default_loaded)
        self.log.debug("Config from env loaded : %s", config_env_loaded)
        self.log.debug("Config from file loaded: %s", config_file_loaded)
        if not (config_default_loaded or config_env_loaded or config_file_loaded):
            raise RuntimeError("No configuration file specified for loading. "
                               "(%s=%s) (file=%s)"
                               % (self.ENV_CONFIG,
                                  os.environ.get(self.ENV_CONFIG, None),
                                  config_filepath))

        #: :type: dict of ((str, str), smqtk.content_description.ContentDescriptor)
        descriptor_cache = {}
        descriptor_cache_lock = multiprocessing.RLock()

        #
        # Security
        #
        self.secret_key = self.config['SECRET_KEY']

        @self.route("/")
        def list_ingest_labels():
            return flask.jsonify({
                "labels": IngestConfiguration.available_ingest_labels()
            })

        @self.route("/<string:ingest>/")
        def list_descriptor_labels(ingest):
            ic = IngestConfiguration(ingest)
            return flask.jsonify({
                "labels": ic.get_available_descriptor_labels()
            })

        @self.route("/<string:ingest>/<string:descriptor_type>/<path:uri>")
        def compute_descriptor(ingest, descriptor_type, uri):
            success = True
            message = "Nothing has happened yet"

            # Resolve URI
            # - If "file://" prefix, look for local file by the given
            # - Otherwise, try to download from web to temp directory
            filepath = None
            remove_file = False
            if uri[:7] == "file://":
                self.log.debug("Given local disk filepath")
                filepath = uri[7:]
                if not os.path.isfile(filepath):
                    success = False
                    message = "File URI did not point to an existing file on " \
                              "disk."
            else:
                self.log.debug("Given URL")
                r = requests.get(uri)
                if r.ok:
                    ext = r.headers['content-type'].rsplit('/', 1)[1]
                    fd, filepath = tempfile.mkstemp(suffix='.'+ext)
                    self.log.debug("Saving image with type '%s' to: %s",
                                   r.headers['content-type'], filepath)
                    os.close(fd)
                    with open(filepath, 'wb') as ofile:
                        ofile.write(r.content)
                    self.log.debug("Image file written.")
                    remove_file = True
                else:
                    self.log.debug("Request failed")
                    success = False
                    message = "Web request did not yield an OK response " \
                              "(received %d :: %s)" \
                              % (r.status_code, r.reason)

            descriptor = None
            if filepath:
                df = DataFile(filepath)

                # Get the descriptor instance, creating it if necessary
                index = (ingest, descriptor_type)
                self.log.debug("Index requested: %s", index)
                if index not in descriptor_cache:
                    ic = IngestConfiguration(ingest)
                    self.log.debug("Creating descriptor for requested "
                                   "index %s", index)
                    with descriptor_cache_lock:
                        descriptor_cache[index] = \
                            ic.new_descriptor_instance(descriptor_type)
                self.log.debug("Getting descriptor for index: %s", index)
                with descriptor_cache_lock:
                    d = descriptor_cache[index]
                with SimpleTimer("Computing descriptor...", self.log.debug):
                    descriptor = d.compute_descriptor(df)
                    message = "Descriptor computed of type '%s'" \
                              % descriptor_type

                if remove_file:
                    self.log.debug("Removing downloaded file.")
                    os.remove(filepath)

            return flask.jsonify({
                "success": success,
                "message": message,
                "descriptor": descriptor.tolist(),
                "reference_uri": uri
            })

    def run(self, host=None, port=None, debug=False, **options):
        """
        Override of the run method, drawing running host and port from
        configuration by default. 'host' and 'port' values specified as argument
        or keyword will override the app configuration.
        """
        super(DescriptorServiceServer, self)\
            .run(host=(host or self.config['RUN_HOST']),
                 port=(port or self.config['RUN_PORT']),
                 **options)
